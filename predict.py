from typing import List
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import argparse
import os
from tqdm import tqdm
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
import subprocess


def transform_rot90(arr, rot90):
    """

    :param arr: (H, W, D)
    :param rot90: 3x3 rotation matrix with only -1, 1, 0 entries. and determinant of -1 or 1
    :return: It returns the array rotated and mirrored like:
        out_arr_coords = rot90 @ arr_coordinates
    """
    # We first inverse the rotation to get
    rot_inv = rot90.T

    # Finally we can permute the array
    permute = np.argmax(np.abs(rot_inv), axis=0)
    ras_90_arr = arr.transpose(permute)

    # And flip the axis which are in negative direction
    ras_90_arr = ras_90_arr[
                 ::int(np.sign(np.sum(rot_inv[permute[0], 0]))),
                 ::int(np.sign(np.sum(rot_inv[permute[1], 1]))),
                 ::int(np.sign(np.sum(rot_inv[permute[2], 2]))),
                 ].copy()

    return ras_90_arr


def get_90deg_rot_mat(m):
    # Get only rotation without scale
    U, S, Vh = np.linalg.svd(m)
    rot = U @ Vh

    # We now look at the transform and align it with 90 degree angles
    axis = np.argmax(np.abs(rot), axis=0)
    rot90 = np.zeros((3, 3))
    for i in range(3):
        rot90[axis[i], i] = np.sign(rot[axis[i], i])
    return rot90


def align_to_RASplus(in_fname, target_fname, affine_fname=None):
    """
    If the voxels are not aligned to the RASplus convention, this function permutes the axis to fit RASplus
    RAS+: https://nipy.org/nibabel/coordinate_systems.html

    :param in_fname: input nifti file
    :param target_fname: output nifti file
    :param affine_fname: output transform fname, can be used to transform back later.
    :return: transformed array
    """
    img = nib.load(in_fname)
    affine = img.affine
    original_arr = img.get_fdata().squeeze()

    # Getting the transform
    # Because the original training samples voxel coords are already aligned to the RAS+ conversion
    # We just need to look at the affine rotation from this scan, as this gives us the rotation into RAS+ space
    # Then we align the rotation to the closest 90 degree rotation so we only have to permute the array axis
    rot_to_ras_90 = get_90deg_rot_mat(affine[:3, :3])

    # Now we apply the transform to the array
    ras90_arr = transform_rot90(original_arr, rot_to_ras_90)

    # Finally calculating a new affine transform just to make visualiization better
    # as nnUNet does not use this to my knowledge
    # Given:
    # v: voxel coord
    # o: coord in RAS+ view
    # A: affine matrix
    # R: new rotation to RAS+
    # Then we start with
    # o = A * v
    # New we have
    # o = A * (R * v)
    # which is not correct anymore as A * v != A * R * v
    # So we correct it to
    # o = A * R^-1 * R * v = (A * R^-1) * (R * v) = A_new * v_new
    inv_rot_mat = np.identity(4)
    inv_rot_mat[:3, :3] = rot_to_ras_90.T
    new_affine = affine @ inv_rot_mat
    new_img = nib.Nifti1Image(ras90_arr, new_affine, img.header)
    nib.save(new_img, target_fname)

    if affine_fname is not None:
        # Saving the original affine transformation
        np.save(affine_fname, affine)

    return ras90_arr


def reverse_align_to_RASplus(in_fname, target_fname, affine_fname):
    """
    Reverses align_to_RASplus, for more detail look at that.

    :param in_fname: src nifti
    :param target_fname: target output nifty
    :param affine_fname: Original affine transformation of the original scan/image. This is also the transform output
        of the align_to_RASplus function or the original nifty filename
        If the file ends with ".nii" or ".nii.gz" it will be loaded as nifti, numpy if not
    :return: back rotated array
    """
    img = nib.load(in_fname)
    affine = nib.load(affine_fname).affine if affine_fname.endswith('.nii') or affine_fname.endswith('.nii.gz') else np.load(affine_fname)
    arr = img.get_fdata().squeeze()

    # Yes this is sad, but for some reason nibabel manages to create an image which is
    # uint8 but at the same time stores floats? ... we therefore have to check if it is uint8, because this is what 
    # nnU-Net uses to store segments, and that is the id 2 (from testing)
    # In another script I used 16 for uint16 so we also check for that just to be sure
    if img.header['datatype'] == 2 or img.header['datatype'] == 16:
        arr = arr.round().astype(int)

    # Invert the affine transform
    rot_to_original = get_90deg_rot_mat(affine[:3, :3]).T

    # Apply and save
    out_arr = transform_rot90(arr, rot_to_original)

    new_img = nib.Nifti1Image(out_arr.astype(int), affine, img.header)
    nib.save(new_img, target_fname)
    return out_arr


def save_nii_like(fname: str, arr: np.ndarray, template_fname: str):
    """
    Input arr similar to format of template_fname
    """

    file_info = nib.load(template_fname)
    new_img = nib.Nifti1Image(arr, file_info.affine, file_info.header)
    nib.save(new_img, fname) 


def create_annotator_input(mri_fname, annotator_idx):
    """
    Stores annotator nifti files in the same directory of the given mri.
    It will do this by appending the annotator index and channel to the name of the mri like this for an annotator index 0:
    "directory/mri.nii.gz" creates 2 new files "directory/mri_annotator_0_0001.nii.gz" "directory/mri_annotator_0_0002.nii.gz"

    returns list of the 2 new files' filepaths paths
    """
    # Create 2 additional input channels encoding the annotator
    if not mri_fname.endswith('.nii.gz'):
        exit(f'Unrecognized file ending fo the file {mri_fname}. It should end with ".nii.gz"!')
    
    fname_wo_type = mri_fname[:-len('.nii.gz')]
    annotators01_dst_fname = f'{fname_wo_type}_annotator_{annotator_idx}_0001.nii.gz'
    annotators23_dst_fname = f'{fname_wo_type}_annotator_{annotator_idx}_0002.nii.gz'
    mri = nib.load(mri_fname).get_fdata().squeeze()
    annotators01 = np.zeros_like(mri)
    annotators23 = np.zeros_like(mri)
    if annotator_idx == 0:
        annotators01[:] = -1
    elif annotator_idx == 1:
        annotators01[:] = 1
    elif annotator_idx == 2:
        annotators23[:] = -1
    elif annotator_idx == 3:
        annotators23[:] = 1
    else:
        print(f'Unknown annotator of index {annotator_idx}')
        print('The whole encoding is set to zero!')
    save_nii_like(annotators01_dst_fname, annotators01, mri_fname)
    save_nii_like(annotators23_dst_fname, annotators23, mri_fname)

    return [annotators01_dst_fname, annotators23_dst_fname]


def do_nnunet_prediction(nnunet_plan_results_folder, in_files, out_files, folder_with_segs_from_prev_stage=None, save_probabilities=False):
    predictor = nnUNetPredictor(
        device=torch.device('cuda:0'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        nnunet_plan_results_folder,
        use_folds=(0,1,2,3,4),
        checkpoint_name='checkpoint_final.pth',
    )

    predictor.predict_from_files(
        in_files,
        out_files,
        folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        num_parts=1,
        part_id=0
    )
    
    del predictor


def do_predictions(nnunet_plan_results_folder: str,
                   in_files: List[str], 
                   out_folder: str,
                   annotator_idx=None,
                   tmp_folder=None,
                   prev_stage_nnunet_plan_results_folder=None):

    # Setting up folders
    if tmp_folder is None:
        tmp_folder = out_folder + '_tmp'
    tmp_in_folder = os.path.join(tmp_folder, 'inputs')
    tmp_prev_stage_pred_folder = os.path.join(tmp_folder, 'lowres_predictions')
    tmp_pred_folder = os.path.join(tmp_folder, 'predictions')
    tmp_pred_annotator_folder = os.path.join(tmp_folder, 'predictions_with_annotator')

    # Make sure that all input files have a unique filename
    only_fnames = [os.path.split(f)[-1] for f in in_files]
    if len(only_fnames) != len(set(only_fnames)):
        print('Some filenames are not unique. Files need to differ in their filenames and can not only differ from their path as they will be all collected and written to the output folder!')
        exit(1)
    
    # nnUNet expects a list fo lists for for its input files
    in_files = [[f] for f in in_files]

    # -----------------
    # Create temp paths
    # -----------------
    out_files = []
    aligned_in_files = []
    aligned_inbetween_out_files = []
    aligned_out_files_with_annotator_encoding = []
    aligned_out_files = []
    post_processed_files = []
    for f in in_files:
        filename = os.path.split(f[0])[-1]
        if not filename.endswith('.gz'):
            filename += '.gz'
        out_files.append(os.path.join(out_folder, filename))
        aligned_in_files.append([os.path.join(tmp_in_folder, filename)])
        aligned_out_files.append(os.path.join(tmp_pred_folder, filename))
        aligned_out_files_with_annotator_encoding.append(os.path.join(tmp_pred_annotator_folder, filename))

        # For some reason nnunet when searching for the lowres output  in the provided folder seems to remove the "FLAIR" from the file ...
        # So we remove this for the inbetween outputs ... this is not good and is just a quick fix
        aligned_inbetween_out_files.append(os.path.join(tmp_prev_stage_pred_folder, filename.replace('FLAIR', '')))

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(tmp_in_folder, exist_ok=True)
    os.makedirs(tmp_pred_folder, exist_ok=True)

    # ------------------
    # Align with RASplus
    # ------------------
    for f_in, f_out in tqdm(list(zip(in_files, aligned_in_files)), 'Aligning scans'):
        align_to_RASplus(f_in[0], f_out[0])

    # -------------------
    # Add annotator input
    # -------------------
    if annotator_idx is not None:
        with_inputs = []
        for f in tqdm(aligned_in_files, 'Create annotator input'):
            with_inputs.append(f + create_annotator_input(f[0], annotator_idx))
        aligned_in_files = with_inputs

    # --------------
    # Do Predictions
    # --------------
    if prev_stage_nnunet_plan_results_folder is None:
        do_nnunet_prediction(
            nnunet_plan_results_folder=nnunet_plan_results_folder,
            in_files=aligned_in_files,
            out_files=aligned_out_files if annotator_idx is None else aligned_out_files_with_annotator_encoding,
        )
    else:
        # Lowres
        do_nnunet_prediction(
            nnunet_plan_results_folder=prev_stage_nnunet_plan_results_folder,
            in_files=aligned_in_files,
            out_files=aligned_inbetween_out_files,
        )
        # 3D Cascade Fullres
        do_nnunet_prediction(
            nnunet_plan_results_folder=nnunet_plan_results_folder,
            in_files=aligned_in_files,
            out_files=aligned_out_files if annotator_idx is None else aligned_out_files_with_annotator_encoding,
            folder_with_segs_from_prev_stage=tmp_prev_stage_pred_folder
        )

    # -------------------------
    # Remove Annotator Encoding
    # -------------------------
    if annotator_idx is not None:
        for in_fname, out_fname in tqdm(list(zip(aligned_out_files_with_annotator_encoding, aligned_out_files)), f'Remove Annotator Encoding from Segmentation'):
            seg = nib.load(in_fname).get_fdata().squeeze().round().astype(int)
            
            # This makes the script a little easier but is not robust to changes if the multi label scheme gets changed
            seg[seg != 0] = ((seg[seg != 0] - 1) % 3) + 1
            save_nii_like(out_fname, seg, in_fname)
    
    # --------------
    # Back transform
    # --------------
    for f_in, f_out, f_affine in tqdm(list(zip(aligned_out_files, out_files, in_files)), 'Reversing RAS+ alignmend'):
        reverse_align_to_RASplus(f_in, f_out, f_affine[0])

    print('Final predictions stored in:', out_folder)


def main():
    parser = argparse.ArgumentParser(description='Predicting the segmentation of a list of MRI scans and storing the results in a folder')
    parser.add_argument('--training_dir', type=str, help='Path to the training results. This folder contains a folder for each cross validation fold')
    parser.add_argument('--out', type=str, help='Path to the folder in which we store the results. A temporary folder with the same path but "_tmp" as postfix will be created as well.')
    parser.add_argument('--prev_stage_training_dir', type=str, default=None, help='Folder to the previous predictions training results. This is required when using the 3D Cascade strategy')
    parser.add_argument('--annotator_idx', type=int, default=-1, help='Annotator index. Used if annotator encoding was used during training. Default is -1 which disables this feature. A higher number than 3 means that it is setting the whole encoding to zero. Meaning no particular annotator encoded. It prints a warning as this is not a valid input during training')
    parser.add_argument('--data', type=str, nargs='+', help='Paths to the mri scans which should be processed. the path can be different for each file but the filename itself has to be unique, as all the MRI scans will be collected and stored in the same output folder.')
    args = parser.parse_args()

    do_predictions(
        nnunet_plan_results_folder=args.training_dir,
        in_files=args.data,
        out_folder=args.out,
        annotator_idx=args.annotator_idx if args.annotator_idx >= 0 else None,
        prev_stage_nnunet_plan_results_folder=args.prev_stage_training_dir
    )


if __name__ == '__main__':
    main()

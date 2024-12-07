import nibabel as nib
import numpy as np
import os
import argparse
from tqdm import tqdm


def seg_to_one_hot(arr, num_segments: int):
    """
    From (x, y, z) to (num_seg, x, y, z)
    """
    out = []
    for i in range(num_segments):
        out.append(arr == i)
    return np.array(out, dtype=int)


def one_hot_to_seg(arr, axis=0):
    return np.argmax(arr, axis=axis)


def save_seg_nii_gz(fname: str, arr: np.ndarray, template_fname: str):
    """
    Input dim: (x, y, z), which is an int array
    """

    file_info = nib.load(template_fname)
    header = file_info.header
    # Values are just taken from labels, so some choices might seem strange
    header['dim'][0] = 3            # How many dimensions?
    header['dim'][4] = 1
    header['dim'][5] = 1            
    header['dim'][6] = 1
    header['dim'][7] = 1
    header['intent_code'] = 0       # Type of the data is normal voxel
    header['intent_p1'] = 0.0       # No clue, whatever
    header['bitpix'] = 16           # Number of bits per pixel
    header['datatype'] = 16         # Type of data is uint16
    header['pixdim'][4] = 0.0       # Dimension sizes for each voxel I think, we only set the higher ones to 1.0, the n, x, y, z we leave as it is
    header['pixdim'][5] = 0.0
    header['pixdim'][6] = 0.0
    header['pixdim'][7] = 0.0
    header['cal_min'] = 0.0         # Display min and max intensities
    header['cal_max'] = 0.0
    new_img = nib.Nifti1Image(arr.astype(np.uint16), file_info.affine, file_info.header)
    nib.save(new_img, fname) 


def combine_segments_with_majority_vote(in_files, out_fname, template_fname=None, foreground_weight=1):
    segmentations = [nib.load(f).get_fdata().round().astype(int) for f in in_files]
    one_hots = [seg_to_one_hot(s, num_segments=np.max(segmentations) + 1) for s in segmentations]
    one_hots_sum = np.sum(one_hots, axis=0)
    one_hots_sum[1:] *= foreground_weight
    out = one_hot_to_seg(one_hots_sum)
    save_seg_nii_gz(out_fname, out, template_fname if template_fname is not None else in_files[0])


def merge_segmentations(out_folder, segmentations_folders, foreground_weight=0):
    """
    All files directly in the segmentatoin_folders which end with .nii.gz are going to be merged
    (Same names are merged)
    """
    os.makedirs(out_folder, exist_ok=True)
    files = [f for f in os.listdir(segmentations_folders[0]) if f.endswith('.nii.gz')]
    
    inputs = []
    outputs = []
    for f in files:
        check_in_files = [os.path.join(seg_folder, f) for seg_folder in segmentations_folders]

        in_files = []
        for cf in check_in_files:
            if os.path.exists(cf):
                in_files.append(cf)
            else:
                print('Could not find file:', cf)

        inputs.append(in_files)
        outputs.append(os.path.join(out_folder, f))

    for in_files, out_file in tqdm(list(zip(inputs, outputs)), 'Merging Annotator Predictions'):
        combine_segments_with_majority_vote(
            in_files=in_files,
            out_fname=out_file,
            template_fname=in_files[0],
            foreground_weight=foreground_weight,
        )


def main():
    parser = argparse.ArgumentParser(description='Predicting from folder with realigning axis if voxels are not in correct format')
    parser.add_argument('--out_folder', type=str, help='Output folder')
    parser.add_argument('--annotations0', type=str, help='Folder containing segments in style of annotator 0')
    parser.add_argument('--annotations1', type=str, help='Folder containing segments in style of annotator 1')
    parser.add_argument('--annotations2', type=str, help='Folder containing segments in style of annotator 2')
    parser.add_argument('--annotations3', type=str, help='Folder containing segments in style of annotator 3')
    parser.add_argument('--foreground_weight', type=int, default=3, help='Constant multiplied to all foreground segments for the majority voting.')
    args = parser.parse_args()

    merge_segmentations(
        out_folder=args.out_folder,
        segmentations_folders=[
            args.annotations1,
            args.annotations2,
            args.annotations3,
            args.annotations4
        ],
        foreground_weight=args.foreground_weight
    )


if __name__ == '__main__':
    main()

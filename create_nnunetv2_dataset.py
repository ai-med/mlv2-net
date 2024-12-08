"""
Create a nnUnetv2 dataset from main dataset of this thesis
"""

import argparse
from math import sqrt
import os
import random
import numpy as np
import subprocess
import json
import nibabel as nib
from tqdm import tqdm
from shutil import copyfile


def load_labels(fname=None):
    """
    Returns dictionary of label names as key and their id as value
    """
    with open('data.json', 'r') as file:
        labels = json.load(file)
    return labels


def add_point(volume, coord, heat=1.0):
    """

    :param volume:
    :param coord:
    :param heat: Either -1 or 1
    :return:
    """

    sx, sy, sz = volume.shape

    window = 8

    for dx in range(-window, window+1):
        x = coord[0] + dx
        for dy in range(-window, window+1):
            y = coord[1] + dy
            for dz in range(-window, window+1):
                z = coord[2] + dz
                if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
                    vox_distance_sq = dx**2 + dy**2 + dz**2
                    strength = 1.0 / (1 + vox_distance_sq * 0.4)
                    strength *= 0.9999999 * (1.0 - min(sqrt(vox_distance_sq) / (window + 1), 1.0))
                    volume[x, y, z] = np.tanh(np.arctanh(strength * heat) + np.arctanh(volume[x, y, z]))
                    # volume[x, y, z] = np.clip(volume[x, y, z] + strength * heat, -1.0, 1.0)


def create_random_hints_heatmap(gt_seg, num_points=10):
    hints = np.zeros(gt_seg.shape, dtype=float)
    seg_coords = np.array(np.where(gt_seg > 0), dtype=int).T
    sx, sy, sz = gt_seg.shape

    for i in range(num_points):
        if random.random() < 0.1:
            # For small chance we click somewhere randomly in the volume
            x = random.randint(0, gt_seg.shape[0] - 1)
            y = random.randint(0, gt_seg.shape[1] - 1)
            z = random.randint(0, gt_seg.shape[2] - 1)
        else:
            while True:
                x, y, z = random.choice(seg_coords)
                # Apply some random noise to the points
                random_scale = random.random() * random.random()
                x += random.randint(int(-60 * random_scale), int(60 * random_scale) + 1)  # Segmentations are sometimes more or less wide, so here bigger noise
                y += random.randint(int(-30 * random_scale), int(30 * random_scale) + 1)
                z += random.randint(int(-30 * random_scale), int(30 * random_scale) + 1)

                if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
                    break

        add_point(hints, (x, y, z), 1.0 if gt_seg[x, y, z] > 0 else -1.0)

    return hints


def save_nii_like(fname: str, arr: np.ndarray, template_fname: str):
    """
    Input arr similar to format of template_fname
    """

    file_info = nib.load(template_fname)
    new_img = nib.Nifti1Image(arr, file_info.affine, file_info.header)
    nib.save(new_img, fname) 


def execute_command(command, **kwargs):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True,
                               shell=True,
                               **kwargs)
    stdout, stderr = process.communicate()

    if stderr != '':
        print(f'\nError while processing command:\n{command}\n\nOutput and error:')
        print(stdout)
        print(stderr)
        print('\n Canceling dataset creation!')
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Creates a nnUnetv2 dataset out of existing data')
    parser.add_argument('--src_mri_files', type=str, nargs='+', help='MRI scan files')
    parser.add_argument('--src_ann_files', type=str, nargs='+', help='MRI annotation files')
    parser.add_argument('--annotators', type=int, nargs='+', help='Annotator indices of the four anntoators from 0 to 3.')
    parser.add_argument('--labels', type=str, help='Path to a JSON file describing the segmentation classes')
    parser.add_argument('--dest', type=str, help='Path to the nnU-Net "<path>/nnUNet_raw" folder')
    args = parser.parse_args()


    # Select samples to prepare and copy
    samples_mri = args.src_mri_files
    samples_ann = args.src_ann_files
    annotator_indices = args.annotators

    # Search for free dataset id
    os.makedirs(args.dest, exist_ok=True)
    existing_sets = os.listdir(args.dest)
    existing_ids = [int(s[7:10]) for s in existing_sets]
    set_id = -1
    for i in range(11, 1000):
        if i not in existing_ids:
            set_id = i
            break
    if set_id == -1:
        exit('Could not find an available id from 11 to 999!')
    new_dataset_name = f'Dataset{set_id:03}_LearnAnnotatorStyles'
    dest_dataset_folder = os.path.join(args.dest, new_dataset_name)

    # create new labels based on a combination of annotator and segmentation
    labels_wo_background = list(load_labels(args.labels).items())[1:]       # Ignoring the background
    labels = dict()
    labels['background'] = 0
    for annotator_idx in range(4):
        for name, val in labels_wo_background:
            labels[f'{name}_annIdx_{annotator_idx}'] = val + annotator_idx * len(labels_wo_background)
    
    # Doing the actual conversion
    json_content = {
        'channel_names':
        {
            '0': 'FLAIR',
            '1': 'noNorm',   # We don't want any scaling for the annotator channels as the default z scoring is per image and would remove the constant value completely
            '2': 'noNorm'
        },
        'labels': labels,
        'numTraining': len(samples_mri),
        'file_ending': '.nii.gz'
    }
    print(f'Storing new dataset with {len(samples_mri)} samples in:\n{dest_dataset_folder}\n')
    os.makedirs(os.path.join(dest_dataset_folder, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(dest_dataset_folder, 'labelsTr'), exist_ok=True)
    with open(os.path.join(dest_dataset_folder, 'dataset.json'), "w") as outfile:
        json.dump(json_content, outfile, indent=4)
    for i, (mri_src_fname, labels_src_fname, annotator_idx) in enumerate(tqdm(list(zip(samples_mri, samples_ann, annotator_indices)), f'Creating {new_dataset_name}')):

        # For the imges sometimes we have compressed files sometimes we don't have them
        mri_dst_fname = os.path.join(dest_dataset_folder, 'imagesTr', f'mri_{(i+1):03}_0000.nii.gz')
        copyfile(mri_src_fname, mri_dst_fname)

        # Labels we load and then change to include annotator information
        labels_dst_fname = os.path.join(dest_dataset_folder, 'labelsTr', f'ann_{(i+1):03}.nii.gz')
        labels_arr = nib.load(labels_src_fname).get_fdata().astype(int)
        labels_arr[labels_arr != 0] += annotator_idx * len(labels_wo_background)
        save_nii_like(labels_dst_fname, labels_arr, labels_src_fname)

        # Now we create 2 additional input layers encoding the annotator
        annotators01_dst_fname = os.path.join(dest_dataset_folder, 'imagesTr', f'mri_{(i+1):03}_0001.nii.gz')
        annotators23_dst_fname = os.path.join(dest_dataset_folder, 'imagesTr', f'mri_{(i+1):03}_0002.nii.gz')
        mri = nib.load(mri_dst_fname).get_fdata()
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
            exit(f'Unknown annotator of index {annotator_idx}')
        save_nii_like(annotators01_dst_fname, annotators01, mri_dst_fname)
        save_nii_like(annotators23_dst_fname, annotators23, mri_dst_fname)
    else:
        print('Done!')


if __name__ =='__main__':
    main()
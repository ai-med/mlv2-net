# MLV<sup>2</sup>-Net
Official implementation of paper MLV<sup>2</sup>-Net: Rater-Based Majority-Label Voting for Consistent Meningeal Lymphatic Vessel Segmentation - ML4H 2024


# Installation
1. Install and setup `nnU-Net v2` as described in the [Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).
2. Install additional python packages from the `requirements.txt` as follows:
```commandline
pip install -r /path/to/requirements.txt
```

# Training
## 1. Create nnU-Net v2 Dataset
```bash
python create_nnunetv2_dataset.py \
--src_mri_files <mri_file1> <mri_file2> ... \
--src_ann_files <annotation_file1> <annotation_file2> ... \
--annotators <annotator_idx1> <annotator_idx2> ... \
--labels ./data/mlv_example_labels.json \
--dest <path to nnUNet_raw folder>
```

## 2. Train nnU-Net model
Please follow the instructions of the official [Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md) to first preprocess and then perform training on each of the 5 folds:
```commandline
nnUNetv2_plan_and_preprocess -d <dataset_id> --verify_dataset_integrity
nnUNetv2_train <dataset_id> 3d_fullres <fold_idx> --npz
```

# Inference
## 1. Create Predictions for each individual Annotator
```bash
python predict.py \
--training_dir <path_to_nnunet_training_folder> \
--out <path_to_store_annotators_predictions_in> \
--annotator_idx <annotator_idx> \
--data <unlabeled_mri_file1> <unlabeled_mri_file2> ...
```
It is important to store each annotators predictions in different output folder to prevent them from being overwritten.

## 2. Merging individual Predictions
```bash
python merge_annotations.py \
--out_folder <path_to_output_folder> \
--annotations0 <annotator_0_prediction1> <annotator_0_prediction2> ... \
--annotations1 <annotator_1_prediction1> <annotator_1_prediction2> ... \
--annotations2 <annotator_2_prediction1> <annotator_2_prediction2> ... \
--annotations3 <annotator_3_prediction1> <annotator_3_prediction2> ... \
```
# Citation
If you find this code to be useful, please cite
```
@InProceedings{bongratz25-mlv2-net,
  title = 	 {MLV2-Net: Rater-Based Majority-Label Voting for Consistent Meningeal Lymphatic Vessel Segmentation},
  author =       {Bongratz, Fabian and Karmann, Markus and Holz, Adrian and Bonhoeffer, Moritz and Neumaier, Viktor and Deli, Sarah and Schmitz-Koep, Benita and Zimmer, Claus and Sorg, Christian and Thalhammer, Melissa and Hedderich, Dennis M and Wachinger, Christian},
  booktitle = 	 {Proceedings of the 4th Machine Learning for Health Symposium},
  pages = 	 {143--153},
  year = 	 {2025},
  volume = 	 {259},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {15--16 Dec},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v259/main/assets/bongratz25a/bongratz25a.pdf},
  url = 	 {https://proceedings.mlr.press/v259/bongratz25a.html},
}
```

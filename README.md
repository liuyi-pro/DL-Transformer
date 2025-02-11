# DL-Transformer

This repository contains the implementation of our paper "Dynamic Window Transformer with Curriculum Learning for Remote Sensing Image Classification".  We are actively maintaining this repository and will continue to update it with additional resources.

## Overview
This work proposes an improved vision transformer with dynamic window shifting mechanism for remote sensing image classification, incorporating curriculum learning strategy to enhance model performance.
The current repository includes the main network architecture and training scripts. Additional resources and pretrained weights will be updated in the future.

## Dataset Preparation

Organize your dataset in the following structure:
```mipsasm
dataset_root/
├── train/
│   ├── class_0/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class_n/
│       ├── image1.jpg
│       └── image2.jpg
└── test/
    ├── class_0/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── class_1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── class_n/
        ├── image1.jpg
        └── image2.jpg
```
## Main Components

- `modelV3.py`: Our improved vision transformer implementation, built upon Swin Transformer. We express our gratitude to the original Swin Transformer authors for their outstanding work.

- `cal_difficult_score.py`: Core implementation of curriculum learning strategy. This script divides the dataset into multiple subsets based on difficulty, enabling the model to train progressively from easier to harder samples.

- `train.py`: Standard training script without curriculum learning.

- `train_CL.py`: Training script with our proposed curriculum learning strategy.

- `ours_with_trans_optuna.py`: Implements Optuna hyperparameter optimization. Note: Manual switching between curriculum stages and loading of previous stage model weights is required.

## Evaluation Tools

- `cls_acc_macro.py`: Implements macro-averaged accuracy metric for better evaluation on long-tailed datasets.

- `create_confusion_matrix_new.py`: Generates confusion matrix for trained models.

- `predict.py`: Script for single sample prediction.

- `tail_class_acc.py`: Reports classification accuracy for tail classes.

## Usage

1. First, prepare your dataset following the structure shown above.

2. To implement curriculum learning:
   ```bash
   # Generate difficulty scores and dataset division
   python cal_difficult_score.py

   # Train with curriculum learning
   python train_CL.py
3. For standard training without curriculum learning:
   ```bash
      python train.py
4. For hyperparameter optimization:
   ```bash
     python ours_with_trans_optuna.py
5. To evaluate model performance:
   ```bash

   # Generate confusion matrix
   python create_confusion_matrix_new.py

   # Check tail class performance
   python tail_class_acc.py

# MyProject

# UNet-Based Nuclei Segmentation Project

This repository contains the implementation of a modified UNet architecture for nuclei segmentation using deep learning techniques. The project leverages TensorFlow and Keras libraries for training, and includes data augmentation techniques to improve the robustness of the model.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Data Augmentation](#data-augmentation)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Work](#future-work)

---

## Project Overview
This project explores the segmentation of cell nuclei from images using a modified UNet architecture. The modifications aim to improve segmentation accuracy by enhancing the encoder and decoder blocks while using advanced data augmentation techniques to address the variability in pathological images. The project demonstrates two main scenarios:

1. **UNet Original**: Baseline implementation of the UNet architecture.
2. **UNet Modified**: Includes added regularization, batch normalization, and deeper encoder-decoder layers.

## Features
- **Data Augmentation**: Incorporates techniques like elastic transformations, brightness adjustment, and geometric transformations.
- **Modified UNet**: Improved encoder and decoder architecture with batch normalization and dropout.
- **Custom Loss Functions**: Utilizes Jaccard loss and Dice loss for unbalanced dataset optimization.
- **Evaluation Metrics**: Implements IoU, Dice coefficient, precision, recall, and F1 score.
- **Flexible Training Framework**: Adaptable for various segmentation tasks with minimal adjustments.

## Prerequisites
- Python 3.8 or above
- Libraries: TensorFlow, Keras, NumPy, OpenCV, albumentations, Matplotlib
- A GPU-enabled environment (Google Colab Pro or Kaggle recommended)

   ```
 Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
1. Prepare your dataset with images and corresponding masks in the following structure:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   ├── masks/
   ├── valid/
       ├── images/
       ├── masks/
   ```
2. Ensure images are in `.bmp` format and masks are in `.tif` format.

## Data Augmentation
Use the provided [DataAugmentation.ipynb](./DataAugmentation.ipynb) notebook to augment your dataset. The augmentation techniques include:
- Elastic transformations
- Grid distortions
- Brightness and contrast adjustments
- Horizontal and vertical flips

The augmented data will be saved in a new directory:
```
Augmented_Dataset/
├── images/
├── masks/
```

## Model Training
Use the [TrainingModelCode.ipynb](./TrainingModelCode.ipynb) notebook to train the model. Key hyperparameters:
- **Batch Size**: 2
- **Learning Rate**: 3e-4
- **Epochs**: 200

### Training Steps:
1. Load the dataset with the specified directory structure.
2. Configure the modified UNet model.
3. Train the model using the `fit` function with the following callbacks:
   - EarlyStopping
   - ModelCheckpoint
   - ReduceLROnPlateau

## Evaluation Metrics
1. **IoU (Intersection over Union)**: Measures the overlap between predicted and ground truth masks.
2. **Dice Coefficient**: Evaluates the similarity between predicted and ground truth masks.
3. **Pixel Accuracy**: Calculates the ratio of correctly predicted pixels.
4. **F1 Score**: Harmonic mean of precision and recall.

## Results
- **UNet Original**:
  - Dice Coefficient: 80%
  - IoU: 50%
- **UNet Modified**:
  - Dice Coefficient: 76%
  - IoU: 66%

The results demonstrate the effectiveness of the modified UNet architecture, with better segmentation performance observed in challenging cases.

## How to Run
1. Perform data augmentation using `DataAugmentation.ipynb`.
2. Train the model using `TrainingModelCode.ipynb`.
3. Evaluate the model and visualize segmentation results.

## Future Work
- **Enhancing the Dataset**: Acquire more pathological images to improve model generalization.
- **Transfer Learning**: Replace the encoder of the UNet with pre-trained architectures like VGG16.
- **Hybrid Approaches**: Combine CNN-based segmentation with traditional methods.
- **Validation by Pathologists**: Ensure clinical relevance of segmentation results.

---

## Acknowledgments
- Frameworks: TensorFlow, Keras
- Platforms: Google Colab Pro, Kaggle
- Libraries: albumentations, OpenCV

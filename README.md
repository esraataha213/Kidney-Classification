# CT Kidney Project

## Overview

The CT Kidney Project aims to leverage deep learning techniques to analyze CT images of kidneys for medical insights. This project focuses on utilizing convolutional neural networks (CNNs) for image processing and classification tasks. The ultimate goal is to support healthcare professionals with accurate and efficient kidney condition analysis.

## Features
- Data Preprocessing: Includes normalization of CT scan images for better model generalization.
- Deep Learning Architecture: Utilizes CNN layers, including Conv2D, MaxPooling2D, Flatten, Dense, and Dropout.
- Prediction Pipelines: Implements both scratch and VGG16-based models for binary and multiclass classification.
- Medical Focus: Specifically designed to assist in analyzing kidney-related conditions.
- Deployment Ready: The model can be integrated into healthcare applications via APIs.

## Dataset

The project uses a dataset of CT kidney images, preprocessed for consistency and improved training performance. The dataset includes:

- Total Data: 12,446 unique images.
- Normal: 5,077 images.
- Cyst: 3,709 images.
- Stone: 1,377 images.
- Tumor: 2,283 images.

 ## Model Architecture

The project employs two types of deep learning architectures:

1- Scratch Models:

- Binary classification model for distinguishing between normal and abnormal conditions.
- Multiclass classification model for categorizing abnormal conditions into 'Cyst', 'Stone', or 'Tumor'.

2- VGG16 Pretrained Models:

- Binary classification model using transfer learning with VGG16.
- Multiclass classification model using transfer learning with VGG16.

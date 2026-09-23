# Hybrid Image Classification using InceptionV3 + MobileNetV2 with PCA and XGBoost

## Overview

A hybrid image classification project for Horse vs Human using two pretrained CNN models:

- InceptionV3

- MobileNetV2

Both models process the same image independently. Their features are combined and reduced using PCA, followed by XGBoost for final classification.

## Model Architecture

Input Image → InceptionV3 + MobileNetV2 → Feature Fusion → StandardScaler → PCA → XGBoost → Horse / Human

## Dataset

Horses or Humans dataset from TensorFlow Datasets.

## Workflow

- Load the dataset.

- Extract features using InceptionV3 and MobileNetV2.

- Combine features from both models.

- Apply StandardScaler.

- Apply PCA for dimensionality reduction.

- Train XGBoost classifier.

- Predict Horse or Human for new images.

## Technologies

Python, TensorFlow, Keras, InceptionV3, MobileNetV2, Scikit-learn, PCA, XGBoost, NumPy, Jupyter Notebook.

## Project Goal

To explore a hybrid approach that combines feature representations from two pretrained CNN models for image classification.

Author

Vrushali V. Khedkar


<img src="confusion_Matrix_grph.png" width="500">


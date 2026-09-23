# Hybrid Image Classification using InceptionV3 + MobileNetV2 with PCA and XGBoost

## Project Overview

This project implements a hybrid deep learning approach for Horse vs Human image classification.

Two pretrained CNN models are used in parallel to extract features from the same input image:

- InceptionV3

- MobileNetV2

The extracted features are combined and passed through StandardScaler, PCA, and XGBoost for final classification.

## Objective

The objective is to classify an input image into one of two classes:

- Horse

- Human

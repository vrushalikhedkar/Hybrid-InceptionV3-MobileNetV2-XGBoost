Hybrid Image Classification using InceptionV3 + MobileNetV2 with PCA and XGBoost

📌 Project Overview

This project implements a hybrid deep learning approach for Horse vs Human image classification.

Two pretrained CNN models are used in parallel to extract features from the same input image:

InceptionV3

MobileNetV2

The extracted features are fused and then passed through StandardScaler, PCA, and XGBoost for final classification.

🎯 Objective

The objective of this project is to classify an input image into:

Horse

Human

The project uses two pretrained CNN architectures as feature extractors instead of relying on a single CNN model.

🏗️ Model Architecture

flowchart TD
    A[Input Image] --> B[InceptionV3]
    A --> C[MobileNetV2]

    B --> D[2048 Features]
    C --> E[1280 Features]

    D --> F[Feature Fusion]
    E --> F

    F --> G[3328 Features]
    G --> H[StandardScaler]
    H --> I[PCA]
    I --> J[98 Components]
    J --> K[XGBoost]
    K --> L[Horse / Human]

Architecture Flow

Input Image
     │
     ├──────────────► InceptionV3 ──► 2048 Features
     │
     └──────────────► MobileNetV2 ──► 1280 Features

              2048 + 1280
                   │
                   ▼
            Feature Fusion
             3328 Features
                   │
                   ▼
             StandardScaler
                   │
                   ▼
                  PCA
             98 Components
                   │
                   ▼
               XGBoost
                   │
                   ▼
             Horse / Human

Both InceptionV3 and MobileNetV2 receive the same input image independently. Their extracted feature vectors are concatenated before scaling, PCA, and XGBoost classification.

🧠 Models Used

1. InceptionV3

InceptionV3 is used as a pretrained feature extractor with ImageNet weights.

inception_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

Output feature size:

2048

2. MobileNetV2

MobileNetV2 is used as the second pretrained feature extractor with ImageNet weights.

mobilenet_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

Output feature size:

1280

🔗 Feature Fusion

Features extracted from both pretrained models are concatenated:

X = np.concatenate(
    [inception_features, mobilenet_features],
    axis=1
)

Feature dimensions:

InceptionV3  = 2048
MobileNetV2  = 1280
-------------------
Combined     = 3328

📉 StandardScaler

The combined feature vector is standardized using StandardScaler.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

📊 PCA

Principal Component Analysis (PCA) is used to reduce the dimensionality of the fused feature representation.

pca = PCA(n_components=98)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

The fused representation is reduced from 3328 features to 98 components.

🌳 XGBoost Classification

XGBoost is used as the final classifier.

params = {
    "objective": "binary:logistic",
    "use_label_encoder": True,
    "max_depth": 50,
    "learning_rate": 0.1,
    "n_estimators": 50,
    "tree_method": "gpu_hist"
}

clf = XGBClassifier(**params)
clf.fit(X_train_pca, y_train)

The classifier predicts the final Horse/Human class.

📂 Dataset

The project uses the Horses or Humans dataset from TensorFlow Datasets.

data, info = tfds.load(
    "horses_or_humans",
    split=["train", "test"],
    shuffle_files=False,
    as_supervised=True,
    with_info=True
)

🔄 Complete Workflow

Load the Horses or Humans dataset.

Load pretrained InceptionV3 and MobileNetV2.

Pass the same image independently through both models.

Extract 2048 features from InceptionV3.

Extract 1280 features from MobileNetV2.

Concatenate both feature vectors into 3328 features.

Apply StandardScaler.

Apply PCA and reduce the representation to 98 components.

Train XGBoost.

Evaluate the model using accuracy, classification report, and confusion matrix.

Use the same pipeline for prediction on a new image.

🖼️ New Image Prediction

For a new image, the same trained pipeline is used:

New Image
    │
    ├──► InceptionV3
    │
    └──► MobileNetV2
             │
             ▼
       Feature Fusion
             │
             ▼
       StandardScaler
             │
             ▼
            PCA
             │
             ▼
          XGBoost
             │
             ▼
       Horse / Human

📈 Evaluation

The model can be evaluated using:

Accuracy

Classification Report

Confusion Matrix

Example:

y_pred = clf.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

🛠️ Technologies Used

Python

TensorFlow

TensorFlow Datasets

Keras

InceptionV3

MobileNetV2

Scikit-learn

StandardScaler

PCA

XGBoost

NumPy

Matplotlib

Seaborn

Jupyter Notebook

📁 Project Structure

Hybrid-InceptionV3-MobileNetV2-XGBoost/
│
├── Hybrid_CNN_XGBoost.ipynb
├── README.md
├── requirements.txt
└── .gitignore

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/Hybrid-InceptionV3-MobileNetV2-XGBoost.git

Install the required libraries:

pip install -r requirements.txt

Run Jupyter Notebook:

jupyter notebook

📦 Requirements

tensorflow
tensorflow-datasets
numpy
scikit-learn
xgboost
matplotlib
seaborn
jupyter

🔮 Future Improvements

Compare individual CNN models with the hybrid model.

Optimize XGBoost hyperparameters.

Experiment with different PCA components.

Add a web interface for image prediction.

Deploy the model as an API or web application.

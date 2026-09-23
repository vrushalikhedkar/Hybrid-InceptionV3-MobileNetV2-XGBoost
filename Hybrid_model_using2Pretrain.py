import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import InceptionV3, MobileNetV2



data, info = tfds.load("horses_or_humans", split=['train', 'test'], shuffle_files=False, as_supervised=True, with_info=True)



data



train_data = data[0]
test_data = data[1]



info



print("Train data:", info.splits["train"].num_examples)
print("Test data :", info.splits["test"].num_examples)



inception_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

mobilenet_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)



print(" InceptionV3 ")
inception_model.summary()

print(" MobileNetV2 ")
mobilenet_model.summary()



IMG_SIZE = (224,224)



def extract_features(dataset):

    inception_features = []
    mobilenet_features = []
    labels = []

    for image, label in dataset:

        image = tf.image.resize(image, IMG_SIZE)
        image = tf.expand_dims(image, axis=0)

        # InceptionV3
        inception_input = tf.keras.applications.inception_v3.preprocess_input(image)
        f1 = inception_model.predict(inception_input, verbose=0)

        # MobileNetV2
        mobilenet_input = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        f2 = mobilenet_model.predict(mobilenet_input, verbose=0)

        inception_features.append(f1[0])
        mobilenet_features.append(f2[0])
        labels.append(label.numpy())

    return (
        np.array(inception_features),
        np.array(mobilenet_features),
        np.array(labels)
    )



X_train_inception, X_train_mobile, y_train = extract_features(train_data)

X_test_inception, X_test_mobile, y_test = extract_features(test_data)



print(X_train.shape)
print(X_test.shape)



X_train = np.concatenate(
    [X_train_inception, X_train_mobile],
    axis=1
)

X_test = np.concatenate(
    [X_test_inception, X_test_mobile],
    axis=1
)



print("Train:", X_train.shape)
print("Test:", X_test.shape)



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)



print("Train scaled:", X_train_scaled.shape)
print("Test scaled :", X_test_scaled.shape)



from sklearn.decomposition import PCA

pca = PCA(n_components=0.98)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Train PCA:", X_train_pca.shape)
print("Test PCA :", X_test_pca.shape)



print("Original features:", X_train.shape[1])
print("PCA features:", X_train_pca.shape[1])
print("Explained variance:", pca.explained_variance_ratio_.sum())



from xgboost import XGBClassifier



params = {
    'objective': 'binary:logistic',
    'use_label_encoder': True,
    'max_depth': 50,
    'learning_rate': 0.1,
    'n_estimators': 50,
}



xgb_model = XGBClassifier(**params)



xgb_model.fit(
    X_train_pca,
    y_train
)



y_pred = xgb_model.predict(X_test_pca)



y_pred



from sklearn.metrics import accuracy_score



accuracy = accuracy_score(y_test, y_pred)



print("Accuracy:", accuracy)



from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred))



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)



print(cm)



img = tf.keras.utils.load_img(
    "horse_image.jpg", target_size=(224,224)
)



img



img_array = tf.keras.utils.img_to_array(img)



img_array.shape



img_array = tf.expand_dims(img_array, axis=0)



inception_input = tf.keras.applications.inception_v3.preprocess_input(img_array)
inception_features = inception_model.predict(inception_input, verbose=0)



mobilenet_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
mobilenet_features = mobilenet_model.predict(mobilenet_input, verbose=0)



img_features = np.concatenate(
    [inception_features, mobilenet_features],
    axis=1
)



print("Combined features:", img_features.shape)



scaled_feat = scaler.transform(img_features)



reduced_features = pca.transform(scaled_feat)



print("Reduced features:", reduced_features.shape)



prediction = clf.predict(reduced_features)



print("Prediction:", prediction)










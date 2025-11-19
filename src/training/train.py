import os
from typing import Dict
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans

from src.data.dataset import load_images_as_array, train_val_test_split
from src.models.cnn import build_cnn
from tensorflow.keras import models


class TrainingArtifacts:
    def __init__(self, cnn=None, ada=None, kmeans=None, paths=None):
        self.cnn = cnn
        self.ada = ada
        self.kmeans = kmeans
        self.paths = paths or {}


def train_all(data_dir: str = "data", epochs: int = 5, output_dir: str = "artifacts") -> TrainingArtifacts:
    X, y = load_images_as_array(data_dir)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y)

    os.makedirs(output_dir, exist_ok=True)
    artifacts = TrainingArtifacts()

    if X.ndim == 4:  # image data
        cnn = build_cnn(input_shape=X_train.shape[1:])
        cnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=16, verbose=0)
        y_pred_cnn = (cnn.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        print("CNN Report:\n", classification_report(y_test, y_pred_cnn))
        artifacts.cnn = cnn
        cnn_path = os.path.join(output_dir, "cnn_model.h5")
        cnn.save(cnn_path)
        artifacts.paths["cnn"] = cnn_path
        # Feature extraction for tabular models using named embedding layer
        embedding_layer = cnn.get_layer('embedding')
        feature_extractor = models.Model(inputs=cnn.input, outputs=embedding_layer.output)
        feats = feature_extractor.predict(X, verbose=0)
        X_tab = feats
    else:
        X_tab = X

    # Split again for tabular models
    (X_train_t, y_train_t), (_, _), (X_test_t, y_test_t) = train_val_test_split(X_tab, y)

    ada = AdaBoostClassifier(n_estimators=50, random_state=42)
    ada.fit(X_train_t, y_train_t)
    ada_pred = ada.predict(X_test_t)
    print("AdaBoost Report:\n", classification_report(y_test_t, ada_pred))
    artifacts.ada = ada
    ada_path = os.path.join(output_dir, "ada_model.pkl")
    joblib.dump(ada, ada_path)
    artifacts.paths["ada"] = ada_path

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train_t)
    cluster_labels = kmeans.predict(X_test_t)
    print("KMeans Cluster label distribution:", np.bincount(cluster_labels))
    artifacts.kmeans = kmeans
    kmeans_path = os.path.join(output_dir, "kmeans.pkl")
    joblib.dump(kmeans, kmeans_path)
    artifacts.paths["kmeans"] = kmeans_path

    return artifacts

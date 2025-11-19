import os
import glob
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image

SUPPORTED_IMG_EXT = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]


def load_image_paths(data_dir: str) -> List[str]:
    paths = []
    for ext in SUPPORTED_IMG_EXT:
        paths.extend(glob.glob(os.path.join(data_dir, ext)))
    return sorted(paths)


def load_images_as_array(data_dir: str, target_size=(224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """Load retinal images into arrays.

    Expected directory structure:
        data_dir/
            healthy/  (label 0)
            risk/     (label 1)

    If no images are found, falls back to simulated tabular data (shape (500,10)).

    Returns:
        X: np.ndarray
            Image mode -> (N, H, W, C) float32 in [0,1]
            Fallback   -> (N, 10)
        y: np.ndarray shape (N,) with 0/1 labels
    """
    X = []
    y = []
    for label_name, label in [("healthy", 0), ("risk", 1)]:
        class_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for img_path in load_image_paths(class_dir):
            try:
                img = image.load_img(img_path, target_size=target_size)
                arr = image.img_to_array(img) / 255.0
                X.append(arr)
                y.append(label)
            except Exception as e:
                print(f"[WARN] Failed to load {img_path}: {e}")
    if not X:
        # fallback to simulated tabular for now
        X_sim = np.random.rand(500, 10).astype("float32")
        y_sim = np.random.randint(0, 2, 500).astype("int32")
        return X_sim, y_sim
    return np.array(X, dtype="float32"), np.array(y, dtype="int32")


def train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Create train/val/test splits with stratification.

    Args:
        X (np.ndarray): Features/images.
        y (np.ndarray): Labels.
        test_size (float): Fraction for test.
        val_size (float): Fraction for validation.
        random_state (int): RNG seed.
    Returns:
        ((X_train,y_train),(X_val,y_val),(X_test,y_test))
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size+val_size, random_state=random_state, stratify=y
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-relative_val, random_state=random_state, stratify=y_temp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute inverse-frequency class weights for imbalanced datasets.

    Args:
        y: label array with classes 0/1
    Returns:
        dict mapping class->weight
    """
    unique, counts = np.unique(y, return_counts=True)
    total = y.shape[0]
    weights: Dict[int, float] = {}
    for cls, cnt in zip(unique, counts):
        weights[int(cls)] = total / (len(unique) * cnt)
    return weights

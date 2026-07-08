import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


IMG_SIZE = 48
NUM_CLASSES = 7


def preprocess_image(img, use_clahe=False):
    """
    Preprocess a single grayscale image.
    """

    img = img.astype(np.uint8)

    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)

    return img


def load_data(csv_path="dataset/fer2013.csv", use_clahe=False):

    print("Loading FER2013 dataset...")

    df = pd.read_csv(csv_path)

    images = []

    for pixel_sequence in df["pixels"]:

        image = np.fromstring(
            pixel_sequence,
            dtype=np.uint8,
            sep=" "
        ).reshape(48, 48)

        image = preprocess_image(
            image,
            use_clahe=use_clahe
        )

        images.append(image)

    images = np.array(images, dtype=np.float32)

    labels = to_categorical(
        df["emotion"],
        NUM_CLASSES
    )

    train_mask = df["Usage"] == "Training"
    val_mask = df["Usage"] == "PublicTest"
    test_mask = df["Usage"] == "PrivateTest"

    X_train = images[train_mask]
    y_train = labels[train_mask]

    X_val = images[val_mask]
    y_val = labels[val_mask]

    X_test = images[test_mask]
    y_test = labels[test_mask]

    print("\nDataset Loaded Successfully")
    print("-" * 40)
    print("Training   :", X_train.shape)
    print("Validation :", X_val.shape)
    print("Testing    :", X_test.shape)
    print("-" * 40)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


if __name__ == "__main__":

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        use_clahe=False
    )
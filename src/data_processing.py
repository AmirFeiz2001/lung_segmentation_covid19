import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, img_size=128):
    """
    Load and preprocess images from a folder.

    Args:
        folder (str): Path to folder containing images.
        img_size (int): Target image size (default: 128).

    Returns:
        list: List of preprocessed images.
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (img_size, img_size))
            images.append(img)
    return images

def prepare_data(ct_dir, mask_dir, img_size=128, num_cases=20):
    """
    Prepare CT and mask data for training.

    Args:
        ct_dir (str): Directory with CT scan PNGs.
        mask_dir (str): Directory with mask PNGs.
        img_size (int): Target image size.
        num_cases (int): Number of cases to process.

    Returns:
        tuple: Normalized CT and mask arrays.
    """
    ct_images = load_images_from_folder(ct_dir)
    mask_images = load_images_from_folder(mask_dir)

    CT, Mask = [], []
    for case in range(min(num_cases, len(ct_images))):
        ct, mask = ct_images[case], mask_images[case]
        for slice_idx in range(ct.shape[2]):
            ct_img = cv2.resize(ct[..., slice_idx], (img_size, img_size), interpolation=cv2.INTER_AREA).astype('float64')
            mask_img = cv2.resize(mask[..., slice_idx], (img_size, img_size), interpolation=cv2.INTER_AREA).astype('float64')
            CT.append(ct_img[..., np.newaxis])
            Mask.append(mask_img[..., np.newaxis])

    CT, Mask = np.array(CT), np.array(Mask)
    mins = CT.min(axis=(1, 2, 3), keepdims=True)
    maxs = CT.max(axis=(1, 2, 3), keepdims=True)
    norm_CT = (CT - mins) / (maxs - mins + 1e-8)  # Avoid division by zero

    return norm_CT, Mask

def split_data(CT, Mask, test_size=0.1):
    """
    Split data into train and test sets.

    Args:
        CT (np.ndarray): Normalized CT images.
        Mask (np.ndarray): Mask images.
        test_size (float): Fraction of data for testing.

    Returns:
        tuple: Train and test sets for CT and Mask.
    """
    return train_test_split(CT, Mask, test_size=test_size, random_state=42)

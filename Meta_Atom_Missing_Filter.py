"""
Meta-Atom Missing Filter
Identifies missing vs present pillars using Hu-moment features extracted
from the largest adaptive-threshold object in each tile.
DBSCAN is run in PCA space; the densest cluster is treated as GOOD.
Everything else (other clusters + noise) is BAD (missing/defective).
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


# ------------------------------
# Image Processing
# ------------------------------

def adaptive_binary(img):
    """Adaptive thresholding to isolate bright meta-atom region."""
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 3
    )


def extract_largest_object(img_bin):
    """Extract mask of largest connected component."""
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, 0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    mask = np.zeros_like(img_bin)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    return mask, area


def hu_moments(mask):
    """Compute 7 Hu moments from mask."""
    M = cv2.moments(mask)
    return cv2.HuMoments(M).flatten()


# ------------------------------
# Missing Pillar Classification
# ------------------------------

def count_missing(array_dir, eps=1.0, min_samples=10, visualize=False):
    """
    Count missing/defective pillars in a 21×21 array of tiles.

    GOOD pillars = densest DBSCAN cluster
    BAD pillars = everything else
    """
    array_dir = Path(array_dir)
    files = sorted(array_dir.glob("*.bmp"))

    if len(files) == 0:
        print(f"Warning: No .bmp files found in {array_dir}")
        return 0

    features, imgs, paths = [], [], []

    for p in files:
        img = cv2.imread(str(p), 0)
        if img is None:
            continue

        img_bin = adaptive_binary(img)
        mask, area = extract_largest_object(img_bin)

        if mask is None:
            hu = np.zeros(7)
        else:
            hu = hu_moments(mask)

        features.append(hu)
        imgs.append(img)
        paths.append(p)

    features = np.array(features)

    # Standardize → PCA(2)
    X = StandardScaler().fit_transform(features)
    X_pca = PCA(n_components=2).fit_transform(X)

    # DBSCAN in PCA space
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_pca)
    raw = db.labels_
    unique = np.unique(raw)

    # Identify good cluster = densest cluster (most members)
    valid_clusters = [c for c in unique if c != -1]

    if len(valid_clusters) == 0:
        # Degenerate, cannot cluster → everything is "present"
        return 0

    cluster_sizes = {c: np.sum(raw == c) for c in valid_clusters}
    good_cluster = max(cluster_sizes, key=cluster_sizes.get)

    # Everything not in the densest cluster is BAD
    final = np.zeros_like(raw)
    final[raw == good_cluster] = 1

    n_missing = np.sum(final == 0)

    # Optional visualization
    if visualize:
        visualize_clusters(imgs, paths, final, array_dir.name, X_pca)

    return n_missing


# ------------------------------
# Visualization of clusters
# ------------------------------

def visualize_clusters(images, paths, labels, name, X_pca):
    """Display 12 examples of GOOD and BAD pillars + PCA scatter."""
    # PCA scatter
    plt.figure(figsize=(7,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="coolwarm", s=10)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"{name} – PCA Space (1=Good, 0=Bad)")
    plt.show()

    # Helper to show images
    def show(cid, title):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            print(f"No samples for {title}")
            return

        idx = np.random.choice(idx, min(12, len(idx)), replace=False)

        fig, ax = plt.subplots(3, 4, figsize=(10,8))
        ax = ax.flatten()
        for a, i in zip(ax, idx):
            a.imshow(images[i], cmap="gray")
            a.set_title(paths[i].name, fontsize=7)
            a.axis("off")
        plt.suptitle(f"{name}: {title} ({len(idx)} tiles)")
        plt.tight_layout()
        plt.show()

    show(0, "Bad / Missing")
    show(1, "Good")


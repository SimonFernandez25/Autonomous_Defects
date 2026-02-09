"""
Layered Clustering for Meta-Atom Defect Classification

A hierarchical approach for sequential defect separation in metasurface arrays.

Layers:
    1. Missing Detection (Center Intensity)
    2. Collapsed Detection (Darkness + Area)
    3. Stitching Error Detection (Sharp Contrast Lines)
    4. Contextual Irregularity (Multi-feature LOF)

Author: Auto-generated from Feature_Experiments.ipynb
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from scipy import ndimage

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MetaAtom:
    """Represents a single meta-atom tile."""
    array: str
    row: int
    col: int
    filename: str
    filepath: str
    image: np.ndarray
    defect_type: str = 'Unknown'
    layer_assignments: Dict[str, int] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame export."""
        d = {
            'array': self.array,
            'row': self.row,
            'col': self.col,
            'filename': self.filename,
            'defect_type': self.defect_type,
        }
        d.update({f'layer_{k}': v for k, v in self.layer_assignments.items()})
        d.update(self.features)
        return d


@dataclass 
class ClusterResult:
    """Result from a clustering layer."""
    layer_name: str
    defect_name: str
    extracted_indices: List[int]
    remaining_indices: List[int]
    labels: np.ndarray
    features: np.ndarray
    cluster_counts: Dict[int, int]


# =============================================================================
# Tile Loading
# =============================================================================

def load_tiles(meta_atoms_dir: Path, array_names: List[str] = None) -> List[MetaAtom]:
    """Load all meta-atom tiles from directory."""
    if array_names is None:
        array_names = ["Array_1Crop", "Array_2Crop", "Array_3Crop"]
    
    tiles = []
    for array_name in array_names:
        array_dir = meta_atoms_dir / array_name
        if not array_dir.exists():
            continue
        
        for fpath in array_dir.glob("*.bmp"):
            img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            fname = fpath.stem
            try:
                coords = fname.split('_')[-1]
                row, col = map(int, coords.split(','))
            except:
                row, col = -1, -1
            
            tiles.append(MetaAtom(
                array=array_name,
                row=row,
                col=col,
                filename=fpath.name,
                filepath=str(fpath),
                image=img
            ))
    
    return tiles


def segment_arrays(base_dir: Path, output_dir: Path, grid_size: int = 21, 
                   tile_size: int = 32) -> None:
    """Segment array images into individual tiles."""
    output_dir.mkdir(exist_ok=True)
    
    for arr in ["Array_1Crop.bmp", "Array_2Crop.bmp", "Array_3Crop.bmp"]:
        arr_path = base_dir / arr
        if not arr_path.exists():
            continue
        
        base_name = arr.replace('.bmp', '')
        array_out = output_dir / base_name
        if array_out.exists():
            continue
        
        img = cv2.imread(str(arr_path))
        if img is None:
            continue
        
        H, W = img.shape[:2]
        array_out.mkdir(exist_ok=True)
        
        x_spacing, y_spacing = W / grid_size, H / grid_size
        for r in range(grid_size):
            for c in range(grid_size):
                cx = int((c + 0.5) * x_spacing)
                cy = int((r + 0.5) * y_spacing)
                x1, y1 = max(0, cx - tile_size), max(0, cy - tile_size)
                x2, y2 = min(W, cx + tile_size), min(H, cy + tile_size)
                tile = img[y1:y2, x1:x2]
                cv2.imwrite(str(array_out / f"{base_name}_{r+1},{c+1}.bmp"), tile)


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_center_intensity(tiles: List[MetaAtom]) -> np.ndarray:
    """Layer 1: Center vs edge intensity features for MISSING detection."""
    features = []
    for tile in tiles:
        img = tile.image
        h, w = img.shape
        ch, cw = h // 2, w // 2
        start_h, start_w = h // 4, w // 4
        center = img[start_h:start_h+ch, start_w:start_w+cw]
        
        mean_c = np.mean(center)
        std_c = np.std(center)
        min_c = np.min(center)
        max_c = np.max(center)
        
        edge_mask = np.ones_like(img, dtype=bool)
        edge_mask[start_h:start_h+ch, start_w:start_w+cw] = False
        mean_e = np.mean(img[edge_mask])
        contrast = mean_c - mean_e
        
        tile.features['center_mean'] = mean_c
        tile.features['center_std'] = std_c
        tile.features['center_edge_contrast'] = contrast
        
        features.append([mean_c, std_c, min_c, max_c, mean_e, contrast])
    
    return np.array(features)


def extract_darkness_area(tiles: List[MetaAtom]) -> np.ndarray:
    """Layer 2: Darkness and area features for COLLAPSED detection."""
    features = []
    for tile in tiles:
        img = tile.image
        mean_int = np.mean(img)
        dark_ratio = np.sum(img < 80) / img.size
        very_dark_ratio = np.sum(img < 50) / img.size
        
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(max(contours, key=cv2.contourArea)) if contours else 0
        norm_area = area / img.size
        
        tile.features['mean_intensity'] = mean_int
        tile.features['dark_ratio'] = dark_ratio
        tile.features['contour_area'] = norm_area
        
        features.append([mean_int, dark_ratio, very_dark_ratio, area, norm_area])
    
    return np.array(features)


def extract_stitching_features(tiles: List[MetaAtom]) -> np.ndarray:
    """Layer 3: Sharp contrast line features for STITCHING detection."""
    features = []
    for tile in tiles:
        img = tile.image.astype(float)
        h, w = img.shape
        
        # Row/column profiles
        row_profile = np.mean(img, axis=1)
        col_profile = np.mean(img, axis=0)
        row_grad = np.abs(np.diff(row_profile))
        col_grad = np.abs(np.diff(col_profile))
        
        max_row_jump = np.max(row_grad) if len(row_grad) > 0 else 0
        max_col_jump = np.max(col_grad) if len(col_grad) > 0 else 0
        max_jump = max(max_row_jump, max_col_jump)
        
        # Half splits
        top_half = np.mean(img[:h//2, :])
        bottom_half = np.mean(img[h//2:, :])
        left_half = np.mean(img[:, :w//2])
        right_half = np.mean(img[:, w//2:])
        vertical_split = np.abs(top_half - bottom_half)
        horizontal_split = np.abs(left_half - right_half)
        max_split = max(vertical_split, horizontal_split)
        
        # Edge-center difference
        border = 5
        strips = [
            np.mean(img[:border, :]),
            np.mean(img[-border:, :]),
            np.mean(img[:, :border]),
            np.mean(img[:, -border:])
        ]
        center_region = np.mean(img[h//4:3*h//4, w//4:3*w//4])
        edge_center_diff = max(np.abs(s - center_region) for s in strips)
        
        # Sobel edges
        sobel_h = cv2.Sobel(img.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
        sobel_v = cv2.Sobel(img.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
        max_horiz_edge = np.max(np.abs(sobel_h).mean(axis=1))
        max_vert_edge = np.max(np.abs(sobel_v).mean(axis=0))
        
        stitching_score = max_jump + max_split + edge_center_diff
        tile.features['stitching_score'] = stitching_score
        tile.features['max_intensity_jump'] = max_jump
        
        features.append([
            max_row_jump, max_col_jump, max_jump,
            vertical_split, horizontal_split, max_split,
            edge_center_diff, max_horiz_edge, max_vert_edge, stitching_score
        ])
    
    return np.array(features)


def extract_rotation_symmetry(tiles: List[MetaAtom]) -> np.ndarray:
    """Layer 4a: Rotation self-similarity."""
    features = []
    for tile in tiles:
        img = tile.image.astype(float)
        h, w = img.shape
        s = min(h, w)
        img_sq = img[:s, :s]
        
        rotations = [img_sq, np.rot90(img_sq, 1), np.rot90(img_sq, 2), np.rot90(img_sq, 3)]
        
        l2_diffs = []
        ssim_scores = []
        for i in range(4):
            for j in range(i+1, 4):
                l2_diffs.append(np.sqrt(np.mean((rotations[i] - rotations[j])**2)))
                if ssim is not None:
                    try:
                        ssim_scores.append(ssim(rotations[i], rotations[j], data_range=255))
                    except:
                        ssim_scores.append(1.0)
                else:
                    ssim_scores.append(1.0)
        
        mean_l2 = np.mean(l2_diffs)
        tile.features['rotation_asymmetry'] = mean_l2
        
        features.append([mean_l2, np.max(l2_diffs), 1 - np.mean(ssim_scores), 1 - np.min(ssim_scores)])
    
    return np.array(features)


def extract_neighbor_deviation(tiles: List[MetaAtom]) -> np.ndarray:
    """Layer 4b: Deviation from local neighbors."""
    def base_features(img):
        mean_int, std_int = np.mean(img), np.std(img)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            circ = (4 * np.pi * area) / (perim**2) if perim > 0 else 0
        else:
            area, circ = 0, 0
        return np.array([mean_int, std_int, area / img.size, circ])
    
    # Build index
    tile_features = {}
    for t in tiles:
        key = (t.array, t.row, t.col)
        tile_features[key] = base_features(t.image)
    
    features = []
    for t in tiles:
        arr, r, c = t.array, t.row, t.col
        my_feat = tile_features[(arr, r, c)]
        
        neighbor_feats = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                key = (arr, r + dr, c + dc)
                if key in tile_features:
                    neighbor_feats.append(tile_features[key])
        
        if neighbor_feats:
            deviation = my_feat - np.mean(neighbor_feats, axis=0)
            deviation_norm = np.linalg.norm(deviation)
        else:
            deviation = np.zeros(4)
            deviation_norm = 0
        
        t.features['neighbor_deviation'] = deviation_norm
        features.append(np.concatenate([deviation, [deviation_norm]]))
    
    return np.array(features)


def extract_anisotropy(tiles: List[MetaAtom]) -> np.ndarray:
    """Layer 4c: Structure tensor anisotropy."""
    features = []
    for tile in tiles:
        img = tile.image.astype(float)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        Ixx = ndimage.gaussian_filter(gx * gx, sigma=2)
        Iyy = ndimage.gaussian_filter(gy * gy, sigma=2)
        Ixy = ndimage.gaussian_filter(gx * gy, sigma=2)
        
        Ixx_sum, Iyy_sum, Ixy_sum = np.sum(Ixx), np.sum(Iyy), np.sum(Ixy)
        trace = Ixx_sum + Iyy_sum
        det = Ixx_sum * Iyy_sum - Ixy_sum**2
        
        if trace > 0:
            discriminant = max(0, trace**2 - 4*det)
            lambda1 = (trace + np.sqrt(discriminant)) / 2
            lambda2 = (trace - np.sqrt(discriminant)) / 2
            aniso = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
        else:
            aniso, lambda1, lambda2 = 0, 0, 0
        
        orientation = 0.5 * np.arctan2(2 * Ixy_sum, Ixx_sum - Iyy_sum)
        tile.features['anisotropy'] = aniso
        
        features.append([aniso, aniso**2, np.abs(orientation), lambda1/(lambda2+1e-10)])
    
    return np.array(features)


def extract_curvature_irregularity(tiles: List[MetaAtom]) -> np.ndarray:
    """Layer 4d: Signed curvature tail metrics."""
    features = []
    for tile in tiles:
        img = tile.image
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(max(contours, key=cv2.contourArea)) >= 10:
            cnt = max(contours, key=cv2.contourArea).squeeze()
            if len(cnt.shape) == 1:
                cnt = cnt.reshape(-1, 2)
            n = len(cnt)
            
            if n >= 10:
                dx = np.gradient(cnt[:, 0].astype(float))
                dy = np.gradient(cnt[:, 1].astype(float))
                ddx, ddy = np.gradient(dx), np.gradient(dy)
                denom = (dx**2 + dy**2)**1.5 + 1e-10
                kappa = (dx * ddy - dy * ddx) / denom
                
                kappa_abs = np.abs(kappa)
                kurtosis = np.mean(kappa_abs**4) / (np.mean(kappa_abs**2)**2 + 1e-10) - 3
                max_curv = np.max(kappa_abs)
                p95_curv = np.percentile(kappa_abs, 95)
                zero_crossings = np.sum(np.diff(np.sign(kappa)) != 0) / n
                curv_var = np.var(kappa)
                
                tile.features['curvature_irregularity'] = kurtosis
                features.append([kurtosis, max_curv, p95_curv, zero_crossings, curv_var])
                continue
        
        tile.features['curvature_irregularity'] = 0
        features.append([0, 0, 0, 0, 0])
    
    return np.array(features)


def extract_radial_deviation(tiles: List[MetaAtom], n_bins: int = 8) -> np.ndarray:
    """Layer 4e: Deviation from global mean radial profile."""
    def radial_profile(img):
        h, w = img.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_r = np.sqrt(cx**2 + cy**2)
        bin_edges = np.linspace(0, max_r, n_bins + 1)
        profile = []
        for i in range(n_bins):
            mask = (r >= bin_edges[i]) & (r < bin_edges[i+1])
            profile.append(np.mean(img[mask]) if np.sum(mask) > 0 else 0)
        return np.array(profile)
    
    profiles = np.array([radial_profile(t.image.astype(float)) for t in tiles])
    mean_profile = np.mean(profiles, axis=0)
    
    features = []
    for i, t in enumerate(tiles):
        deviation = profiles[i] - mean_profile
        dev_norm = np.linalg.norm(deviation)
        t.features['radial_deviation'] = dev_norm
        features.append([dev_norm, np.max(np.abs(deviation)), np.abs(deviation[0]), np.abs(deviation[-1])])
    
    return np.array(features)


# =============================================================================
# Clustering Functions
# =============================================================================

def cluster_layer(tiles: List[MetaAtom], features: np.ndarray, layer_name: str,
                  n_clusters: int = 3, extract_smallest: bool = True) -> ClusterResult:
    """Run K-Means clustering and extract defect cluster."""
    # Clean and scale
    X = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    n_comp = min(5, X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    # Store labels
    for i, tile in enumerate(tiles):
        tile.layer_assignments[layer_name] = int(labels[i])
    
    # Find smallest cluster
    cluster_counts = Counter(labels)
    if extract_smallest:
        defect_cluster = min(cluster_counts, key=cluster_counts.get)
    else:
        defect_cluster = None
    
    # Split indices
    extracted = [i for i, l in enumerate(labels) if l == defect_cluster] if extract_smallest else []
    remaining = [i for i, l in enumerate(labels) if l != defect_cluster] if extract_smallest else list(range(len(labels)))
    
    return ClusterResult(
        layer_name=layer_name,
        defect_name=layer_name.replace('layer_', ''),
        extracted_indices=extracted,
        remaining_indices=remaining,
        labels=labels,
        features=X,
        cluster_counts=dict(cluster_counts)
    )


def compute_lof_scores(tiles: List[MetaAtom], features: np.ndarray, 
                       n_neighbors: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Local Outlier Factor scores."""
    X = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
    labels = lof.fit_predict(X_norm)
    scores = -lof.negative_outlier_factor_
    
    for i, tile in enumerate(tiles):
        tile.features['lof_score'] = scores[i]
        tile.features['is_lof_outlier'] = int(labels[i] == -1)
    
    return scores, labels


# =============================================================================
# Pipeline
# =============================================================================

def run_layered_pipeline(tiles: List[MetaAtom], verbose: bool = True) -> Dict[str, ClusterResult]:
    """Run the complete layered clustering pipeline."""
    results = {}
    current_tiles = tiles.copy()
    current_indices = list(range(len(tiles)))
    
    # Layer 1: Missing
    if verbose:
        print("Layer 1: Missing Detection (Center Intensity)...")
    X1 = extract_center_intensity(current_tiles)
    result1 = cluster_layer(current_tiles, X1, 'layer_missing', n_clusters=3)
    
    for idx in result1.extracted_indices:
        tiles[current_indices[idx]].defect_type = 'Missing'
    results['missing'] = result1
    
    # Update for next layer
    remaining_mask = result1.remaining_indices
    current_tiles = [current_tiles[i] for i in remaining_mask]
    current_indices = [current_indices[i] for i in remaining_mask]
    
    if verbose:
        print(f"  Extracted: {len(result1.extracted_indices)} | Remaining: {len(current_tiles)}")
    
    # Layer 2: Collapsed
    if verbose:
        print("Layer 2: Collapsed Detection (Darkness + Area)...")
    X2 = extract_darkness_area(current_tiles)
    result2 = cluster_layer(current_tiles, X2, 'layer_collapsed', n_clusters=3)
    
    for idx in result2.extracted_indices:
        tiles[current_indices[idx]].defect_type = 'Collapsed'
    results['collapsed'] = result2
    
    remaining_mask = result2.remaining_indices
    current_tiles = [current_tiles[i] for i in remaining_mask]
    current_indices = [current_indices[i] for i in remaining_mask]
    
    if verbose:
        print(f"  Extracted: {len(result2.extracted_indices)} | Remaining: {len(current_tiles)}")
    
    # Layer 3: Stitching
    if verbose:
        print("Layer 3: Stitching Detection (Sharp Contrast)...")
    X3 = extract_stitching_features(current_tiles)
    result3 = cluster_layer(current_tiles, X3, 'layer_stitching', n_clusters=3)
    
    for idx in result3.extracted_indices:
        tiles[current_indices[idx]].defect_type = 'Stitching'
    results['stitching'] = result3
    
    remaining_mask = result3.remaining_indices
    current_tiles = [current_tiles[i] for i in remaining_mask]
    current_indices = [current_indices[i] for i in remaining_mask]
    
    if verbose:
        print(f"  Extracted: {len(result3.extracted_indices)} | Remaining: {len(current_tiles)}")
    
    # Layer 4: Contextual Features
    if verbose:
        print("Layer 4: Contextual Features...")
    
    X_rot = extract_rotation_symmetry(current_tiles)
    X_neigh = extract_neighbor_deviation(current_tiles)
    X_aniso = extract_anisotropy(current_tiles)
    X_curv = extract_curvature_irregularity(current_tiles)
    X_rad = extract_radial_deviation(current_tiles)
    
    X_contextual = np.hstack([X_rot, X_neigh, X_aniso, X_curv, X_rad])
    
    lof_scores, lof_labels = compute_lof_scores(current_tiles, X_contextual)
    n_outliers = np.sum(lof_labels == -1)
    
    # Mark remaining as Good or Irregular based on LOF
    for i, idx in enumerate(current_indices):
        if lof_labels[i] == -1:
            tiles[idx].defect_type = 'Irregular'
        else:
            tiles[idx].defect_type = 'Good'
    
    if verbose:
        print(f"  LOF Outliers (Irregular): {n_outliers}")
        print(f"  Good: {len(current_tiles) - n_outliers}")
    
    return results


# =============================================================================
# Export Functions
# =============================================================================

def export_dataset(tiles: List[MetaAtom], output_path: Path) -> pd.DataFrame:
    """Export tile data to CSV."""
    data = [t.to_dict() for t in tiles]
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df


def get_summary_stats(tiles: List[MetaAtom]) -> Dict[str, int]:
    """Get defect type counts."""
    types = [t.defect_type for t in tiles]
    return dict(Counter(types))

import cv2
import numpy as np
import os
import glob
import math
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

BASE_DIR = r"c:\Users\srfdyz\Downloads\Defect\Meta_Atoms"
OUTPUT_DIR = r"c:\Users\srfdyz\Downloads\Defect\debug_output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def adaptive_binary(img):
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 3
    )

def extract_features(img):
    # 0. Stitching Feature (Gradient)
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    max_gradient = np.max(mag)
    
    # 1. Shape / Hu Methods (for Missing Clustering)
    img_bin = adaptive_binary(img)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hu = np.zeros(7)
    area = 0
    perimeter = 0
    solidity = 0
    
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        perimeter = cv2.arcLength(largest_cnt, True)
        
        # Solidity
        hull = cv2.convexHull(largest_cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / hull_area
        
        mask = np.zeros_like(img_bin)
        cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
        M = cv2.moments(mask)
        hu = cv2.HuMoments(M).flatten()

    # 2. Defect Features
    mean_intensity = np.mean(img)
    if perimeter == 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
    return {
        'Hu': hu,
        'Intensity': mean_intensity,
        'Area': area,
        'Circularity': circularity,
        'Solidity': solidity,
        'MaxGradient': max_gradient
    }

def save_debug_montage(records, category_name, prefix, max_images=30):
    if not records: return
    sample = records[:max_images]
    images = []
    target_size = None
    for rec in sample:
        img = cv2.imread(rec['FilePath'])
        if img is not None:
            if target_size is None: target_size = (img.shape[1], img.shape[0])
            if (img.shape[1], img.shape[0]) != target_size: img = cv2.resize(img, target_size)
            images.append(img)
    if not images: return
    n = len(images)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    montage = np.zeros((rows * target_size[1], cols * target_size[0], 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        montage[r*target_size[1]:(r+1)*target_size[1], c*target_size[0]:(c+1)*target_size[0]] = img
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_final_debug_{category_name.lower()}.jpg"), montage)

def process_array(array_name):
    # Reuse process logic
    dir_path = os.path.join(BASE_DIR, array_name)
    if not os.path.exists(dir_path): return

    files = glob.glob(os.path.join(dir_path, "*.bmp"))
    print(f"Processing {array_name}: {len(files)} files")
    
    records = []
    for fpath in files:
        fname = os.path.basename(fpath)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        try:
            base = os.path.splitext(fname)[0] 
            coords = base.split('_')[-1] 
            row, col = map(int, coords.split(',')) if ',' in coords else (-1, -1)
        except: row, col = -1, -1
            
        feats = extract_features(img)
        records.append({
            'Array': array_name, 'Filename': fname, 'FilePath': fpath,
            'Row': row, 'Col': col, 'Category': 'Pending',
            **feats
        })

    if not records: return

    # --- 1. STITCHING ERROR (Max Gradient) ---
    grads = [r['MaxGradient'] for r in records]
    # Stitching errors have unusually high gradients (sharp lines).
    # Normal atoms are smooth or have standard edges.
    # Let's say top 1% or 2% are stitching? Or Z-score?
    # User said "inflating the missing tab".
    # Let's use a robust outlier detection.
    p95 = np.percentile(grads, 95)
    p99 = np.percentile(grads, 99)
    # If outlier is WAY higher.
    # Heuristic: Threshold = 1.5 * p99? Or just > p98?
    # Let's try 98th percentile as candidate.
    thresh_grad = np.percentile(grads, 98) 
    
    # Actually, a visible line is severe.
    # Let's look at the distribution stats first.
    print(f"  Gradient Stats: Max={np.max(grads):.1f}, Mean={np.mean(grads):.1f}, 99%={p99:.1f}")
    
    # Set threshold to catch extrema.
    # If normal max is 200, stitching might be 400? Or just saturated 255 changes?
    # Sobel on 0-255 image: Max possible is high.
    
    # I'll conservatively mark top 1% as Stitching for now, 
    # OR if it exceeds a hard physical limit of "sharp contrast".
    thresh_grad = np.percentile(grads, 99) # Top 1% are most likely stitching
    
    for r in records:
        if r['MaxGradient'] > thresh_grad:
            r['Category'] = 'Stitching'

    # --- 2. MISSING (Hybrid) ---
    # Only check 'Pending'
    pending_indices = [i for i, r in enumerate(records) if r['Category'] == 'Pending']
    if pending_indices:
        # Clustering on Pending
        hu_matrix = np.nan_to_num(np.array([records[i]['Hu'] for i in pending_indices]))
        if len(hu_matrix) > 0:
            X_std = StandardScaler().fit_transform(hu_matrix)
            X_pca = PCA(n_components=2).fit_transform(X_std)
            db = DBSCAN(eps=0.7, min_samples=10).fit(X_pca)
            labels = db.labels_
            
            # Good cluster
            unique = set(labels) - {-1}
            good_lbl = max({c: np.sum(labels==c) for c in unique}, key=lambda k: np.sum(labels==k)) if unique else -1
            
            median_area = np.median([records[i]['Area'] for i in pending_indices])
            thresh_area = median_area * 0.6
            
            for idx, label in zip(pending_indices, labels):
                r = records[idx]
                is_noise = (label != good_lbl)
                is_small = (r['Area'] < thresh_area)
                if is_noise or is_small:
                    r['Category'] = 'Missing'
    
    # --- 3. COLLAPSED (Solidity + Intensity) ---
    present = [r for r in records if r['Category'] == 'Pending']
    if present:
        # Collapsed: Solid Black.
        # High Solidity (~1.0) AND Low Intensity.
        # Good/Irregular: Loose Gold -> Lower solidity, Higher intensity.
        
        ints = [r['Intensity'] for r in present]
        sols = [r['Solidity'] for r in present]
        
        thresh_int = np.percentile(ints, 10) # Expanded range slightly as suggested "missed some"
        # Solidity thresh: Collapsed are "more complete". 
        # Good pillars are "loose circles". 
        # So Collapsed Solidity > Good Solidity.
        # Let's check median solidity.
        med_sol = np.median(sols)
        print(f"  Solidity Median: {med_sol:.3f}")
        
        # If Collapsed are solid, they should be > 0.9?
        # Let's say: Intensity < Low AND Solidity > Median?
        
        for r in present:
            # Logic: Dark AND Solid
            if r['Intensity'] < thresh_int and r['Solidity'] > med_sol:
                r['Category'] = 'Collapsed'

    # --- 4. IRREGULAR (Circularity) ---
    present = [r for r in records if r['Category'] == 'Pending']
    if present:
        circs = [r['Circularity'] for r in present]
        thresh_circ = np.percentile(circs, 5)
        for r in present:
            if r['Circularity'] < thresh_circ:
                r['Category'] = 'Irregular'
                
    # --- 5. GOOD ---
    for r in records:
        if r['Category'] == 'Pending': r['Category'] = 'Good'

    # Reporting
    cat_counts = {}
    for r in records: cat_counts[r['Category']] = cat_counts.get(r['Category'], 0) + 1
    print(f"  Result: {cat_counts}")

    csv_path = os.path.join(BASE_DIR, f"{array_name}_results.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['Array','Row','Col','Category','Filename','Intensity','Area','Circularity','Solidity','MaxGradient','Hu','FilePath']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    for cat in ['Missing', 'Collapsed', 'Irregular', 'Stitching', 'Good']:
        save_debug_montage([r for r in records if r['Category']==cat], cat, array_name)

def main():
    for arr in ["Array_1Crop", "Array_2Crop", "Array_3Crop"]:
        process_array(arr)

if __name__ == "__main__":
    main()

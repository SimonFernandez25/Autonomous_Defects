import cv2
import numpy as np
import os
import glob
import math
import csv

BASE_DIR = r"c:\Users\srfdyz\Downloads\Defect\Meta_Atoms"
OUTPUT_DIR = r"c:\Users\srfdyz\Downloads\Defect\debug_output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # 1. Intensity
    mean_intensity = np.mean(img)
    
    # 2. Segmentation (Otsu)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'Intensity': mean_intensity,
            'Area': 0,
            'Perimeter': 0,
            'Circularity': 0
        }
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = calculate_circularity(area, perimeter)
    
    return {
        'Intensity': mean_intensity,
        'Area': area,
        'Perimeter': perimeter,
        'Circularity': circularity
    }

def save_debug_montage(records, category_name, prefix, max_images=25):
    """Saves a montage of images from the records list."""
    if not records:
        return

    sample = records[:max_images]
    images = []
    target_size = None

    for rec in sample:
        img = cv2.imread(rec['FilePath'])
        if img is not None:
            if target_size is None:
                target_size = (img.shape[1], img.shape[0])
            if (img.shape[1], img.shape[0]) != target_size:
                img = cv2.resize(img, target_size)
            images.append(img)
    
    if not images:
        return

    n = len(images)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    w, h = target_size
    montage = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        montage[r*h:(r+1)*h, c*w:(c+1)*w] = img
        
    out_path = os.path.join(OUTPUT_DIR, f"{prefix}_debug_{category_name.lower()}.jpg")
    cv2.imwrite(out_path, montage)
    # print(f"Saved debug image: {out_path}")

def classify_array(array_name):
    dir_path = os.path.join(BASE_DIR, array_name)
    if not os.path.exists(dir_path):
        return []
        
    files = glob.glob(os.path.join(dir_path, "*.bmp"))
    print(f"Processing {array_name}: {len(files)} files")
    
    array_data = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            base = os.path.splitext(fname)[0] 
            parts = base.split('_') 
            coords = parts[-1] 
            if ',' in coords:
                r_str, c_str = coords.split(',')
                row = int(r_str)
                col = int(c_str)
            else:
                row = -1
                col = -1
        except:
            row, col = -1, -1

        feats = extract_features(fpath)
        if feats:
            record = {
                'Array': array_name,
                'Filename': fname,
                'FilePath': fpath,
                'Row': row,
                'Col': col,
                'Intensity': feats['Intensity'],
                'Area': feats['Area'],
                'Circularity': feats['Circularity'],
                'Category': 'Pending'
            }
            array_data.append(record)
            if fname in ["Array_1Crop_1,14.bmp", "Array_1Crop_1,13.bmp"]:
                label = "MISSING" if "1,14" in fname else "GOOD"
                print(f"    DEBUG [{label}]: {fname} -> I:{feats['Intensity']:.2f}, A:{feats['Area']:.2f}, C:{feats['Circularity']:.3f}")

    if not array_data:
        return []

    # Local Stats & thresholds
    get_col = lambda col: np.array([r[col] for r in array_data])
    
    areas = get_col('Area')
    ints = get_col('Intensity')
    circs = get_col('Circularity')
    
    print(f"  Stats {array_name}:")
    print(f"    Area Median: {np.median(areas):.2f}")
    print(f"    Int Median: {np.median(ints):.2f}")
    print(f"    Circ Median: {np.median(circs):.2f}")

    # 1. Missing (Area)
    # If 1,14 (Missing) has Area 1521, and Good has 3000.
    # 1521 is ~50% of 3000.
    # Let's set thresh at 60% of Median (if Median is Good).
    median_area = np.median(areas)
    thresh_missing = median_area * 0.6
    
    for r in array_data:
        if r['Area'] < thresh_missing:
            r['Category'] = 'Missing'
            
    # 2. Collapsed (Intensity)
    pending = [r for r in array_data if r['Category'] == 'Pending']
    if pending:
        p_ints = np.array([r['Intensity'] for r in pending])
        # Bottom 5% or Z-score?
        # Let's stick to percentile for robustness against outliers
        thresh_collapsed = np.percentile(p_ints, 5)
        for r in pending:
            if r['Intensity'] < thresh_collapsed:
                r['Category'] = 'Collapsed'

    # 3. Irregular (Circularity)
    pending = [r for r in array_data if r['Category'] == 'Pending']
    if pending:
        p_circs = np.array([r['Circularity'] for r in pending])
        thresh_irregular = np.percentile(p_circs, 5)
        for r in pending:
            if r['Circularity'] < thresh_irregular:
                r['Category'] = 'Irregular'
    
    # 4. Good
    for r in array_data:
        if r['Category'] == 'Pending':
            r['Category'] = 'Good'
            
    # Counts
    counts = {}
    for r in array_data:
        counts[r['Category']] = counts.get(r['Category'], 0) + 1
    print(f"  Counts: {counts}")
    
    # Save CSV
    fieldnames = ['Array', 'Row', 'Col', 'Category', 'Filename', 'Intensity', 'Area', 'Circularity', 'FilePath']
    csv_path = os.path.join(BASE_DIR, f"{array_name}_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(array_data)
    print(f"  Saved CSV: {csv_path}")
    
    # Save Debug Images (Per Array)
    for cat in ['Missing', 'Collapsed', 'Irregular', 'Good']:
        cat_recs = [r for r in array_data if r['Category'] == cat]
        save_debug_montage(cat_recs, cat, array_name) # Prefix with array name
    
    return array_data

def main():
    arrays = ["Array_1Crop", "Array_2Crop", "Array_3Crop"]
    
    for arr in arrays:
        classify_array(arr)

if __name__ == "__main__":
    main()

import numpy as np
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.Clustering.layered_clustering import MetaAtom, extract_stitching_features

def test_stitching_refinement():
    # Create two dummy images:
    # 1. Image with stitching hitting the blob
    # 2. Image with stitching only in the background
    
    img_size = 32
    # Background: ~100
    # Blob: ~200 (center 10x10)
    # Stitching: +50 jump
    
    def create_dummy(stitch_pos, is_row=True):
        img = np.full((img_size, img_size), 100, dtype=np.uint8)
        # Blob
        img[11:21, 11:21] = 200
        # Stitch
        if is_row:
            img[stitch_pos:, :] += 50
        else:
            img[:, stitch_pos:] += 50
        return img

    # Case 1: Stitch hits the blob (row 15)
    img_hit = create_dummy(15, is_row=True)
    tile_hit = MetaAtom(array="Array_TEST", row=0, col=0, filename="hit.png", filepath="none", image=img_hit)
    
    # Case 2: Stitch misses the blob (row 5)
    img_miss = create_dummy(5, is_row=True)
    tile_miss = MetaAtom(array="Array_TEST", row=0, col=1, filename="miss.png", filepath="none", image=img_miss)
    
    print("Extracting features...")
    feats = extract_stitching_features([tile_hit, tile_miss])
    
    print(f"Tile Hit - hits_blob: {tile_hit.features['hits_blob']}, score: {tile_hit.features['stitching_score']:.2f}")
    print(f"Tile Miss - hits_blob: {tile_miss.features['hits_blob']}, score: {tile_miss.features['stitching_score']:.2f}")
    
    # Logic: Score with hit should be higher than score with miss (ceteris paribus)
    # Both have same contrast jump (+50)
    if tile_hit.features['hits_blob'] == 1.0 and tile_miss.features['hits_blob'] == 0.0:
        print("SUCCESS: Blob intersection logic correctly distinguished the cases.")
    else:
        print("FAILURE: Blob intersection logic did not distinguish the cases.")
        sys.exit(1)

if __name__ == "__main__":
    test_stitching_refinement()


import numpy as np
import sys
import os
from pathlib import Path
from dataclasses import dataclass

# Define dummy MetaAtom for testing
@dataclass
class MetaAtom:
    array: str
    row: int
    col: int
    image: np.ndarray = None
    defect_type: str = 'Unknown'

# Mocking the ClusterResult
class ClusterResult:
    def __init__(self, extracted_indices, remaining_indices):
        self.extracted_indices = extracted_indices
        self.remaining_indices = remaining_indices

def test_stitching_logic():
    print("Testing refined stitching logic...")
    
    # Setup mock tiles (21x21 grid)
    tiles = []
    for r in range(1, 22):
        for c in range(1, 22):
            tiles.append(MetaAtom(array="Test", row=r, col=c))
    
    # Current indices for the layer (let's say all tiles)
    current_indices = list(range(len(tiles)))
    current_tiles = [tiles[i] for i in current_indices]
    
    # Mock some detections
    # Row 5 has 6 hits (>= 5) -> systemic
    # Col 10 has 5 hits (>= 5) -> systemic
    # Isolated hit at (15, 15) -> should become Irregular
    detected_local_indices = []
    
    # Row 5 (indices for row 5 are 4*21 + 0..5)
    for c in range(1, 7):
        # find index in current_tiles where row=5, col=c
        idx = next(i for i, t in enumerate(current_tiles) if t.row == 5 and t.col == c)
        detected_local_indices.append(idx)
        
    # Col 10 (indices for col 10)
    for r in range(10, 15):
        idx = next(i for i, t in enumerate(current_tiles) if t.row == r and t.col == 10)
        detected_local_indices.append(idx)
        
    # Isolated
    iso_idx = next(i for i, t in enumerate(current_tiles) if t.row == 15 and t.col == 15)
    detected_local_indices.append(iso_idx)
    
    # --- The Logic (Copied from layered_clustering.py for local unit test) ---
    detected_indices = set(detected_local_indices)
    rows_with_hits = {}
    cols_with_hits = {}
    
    for idx in detected_indices:
        t = current_tiles[idx]
        rows_with_hits[t.row] = rows_with_hits.get(t.row, []) + [idx]
        cols_with_hits[t.col] = cols_with_hits.get(t.col, []) + [idx]
        
    final_stitching_indices = set()
    systemic_indices = set()
    
    for r, idxs in rows_with_hits.items():
        if len(idxs) >= 5:
            for i, t in enumerate(current_tiles):
                if t.row == r:
                    final_stitching_indices.add(i)
                    systemic_indices.add(i)
                    
    for c, idxs in cols_with_hits.items():
        if len(idxs) >= 5:
            for i, t in enumerate(current_tiles):
                if t.col == c:
                    final_stitching_indices.add(i)
                    systemic_indices.add(i)
                    
    final_irregular_indices = set()
    for idx in detected_indices:
        if idx not in systemic_indices:
            final_irregular_indices.add(idx)
            
    # Apply to mock tiles
    for idx in final_stitching_indices:
        tiles[current_indices[idx]].defect_type = 'Stitching'
    for idx in final_irregular_indices:
        tiles[current_indices[idx]].defect_type = 'Irregular'
        
    # --- Verification ---
    # Check Row 5
    row5_tiles = [t for t in tiles if t.row == 5]
    assert all(t.defect_type == 'Stitching' for t in row5_tiles), "Row 5 should be all Stitching"
    
    # Check Col 10
    col10_tiles = [t for t in tiles if t.col == 10]
    assert all(t.defect_type == 'Stitching' for t in col10_tiles), "Col 10 should be all Stitching"
    
    # Check Isolated at (15, 15)
    iso_tile = next(t for t in tiles if t.row == 15 and t.col == 15)
    assert iso_tile.defect_type == 'Irregular', f"(15, 15) should be Irregular, got {iso_tile.defect_type}"
    
    # Check some other tile
    other_tile = next(t for t in tiles if t.row == 1 and t.col == 1)
    assert other_tile.defect_type == 'Unknown', "Other tiles should be untouched"

    print("Verification SUCCESS: threshold-based stitching logic working correctly.")

if __name__ == "__main__":
    test_stitching_logic()

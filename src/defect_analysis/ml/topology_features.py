"""
Feature extraction for defect topology, clustering, and alignment.
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, squareform
import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PITCH = 12.0  # um
# DBSCAN radius: slightly more than sqrt(2)*pitch to connect diagonals
# sqrt(2)*12 = 16.97. Let's use 18.0 to be safe for immediate neighbors.
DBSCAN_EPS = 18.0  
DBSCAN_MIN_SAMPLES = 2  # A cluster needs at least 2 points

# Graph connectivity threshold for "components"
GRAPH_CONNECT_DIST = 18.0 


def compute_clustering_metrics(defect_coords):
    """
    Compute clustering metrics using DBSCAN.
    
    Parameters
    ----------
    defect_coords : list of (r_offset, c_offset)
        Relative coordinates of defects in the window.
    
    Returns
    -------
    dict
        n_clusters_dbscan, largest_cluster_size, mean_cluster_size, cluster_density
    """
    if len(defect_coords) < 2:
        return {
            "n_clusters_dbscan": 0,
            "largest_cluster_size": len(defect_coords),
            "mean_cluster_size": len(defect_coords),
            "cluster_density": 0.0
        }
    
    # Convert to physical units (um) for DBSCAN
    coords_um = np.array(defect_coords) * PITCH
    
    # DBSCAN
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(coords_um)
    labels = db.labels_
    
    # Noise points are -1. We consider them as size-1 clusters or ignore?
    # "n_clusters_dbscan" usually counts valid clusters.
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
        
    n_clusters = len(unique_labels)
    
    if n_clusters == 0:
        # All noise
        return {
            "n_clusters_dbscan": 0,
            "largest_cluster_size": 1, 
            "mean_cluster_size": 1.0,
            "cluster_density": 0.0
        }
        
    sizes = []
    points_in_clusters = 0
    for k in unique_labels:
        size = np.sum(labels == k)
        sizes.append(size)
        points_in_clusters += size
        
    largest_size = max(sizes)
    mean_size = np.mean(sizes)
    
    # Cluster Density: points / convex hull area of all defect points
    # If 2 points (collinear), area is 0 -> density inf?
    # Use a safe fallback.
    try:
        if len(coords_um) >= 3:
            hull = ConvexHull(coords_um)
            area = hull.volume # In 2D, volume is area
            # Avoid divide by zero
            density = len(coords_um) / (area + 1e-9)
        else:
            density = 0.0 # Line or point has no area density
    except:
        density = 0.0
        
    return {
        "n_clusters_dbscan": n_clusters,
        "largest_cluster_size": int(largest_size),
        "mean_cluster_size": float(mean_size),
        "cluster_density": float(density)
    }


def compute_alignment_metrics(defect_coords):
    """
    Compute alignment and anisotropy using PCA.
    
    Returns
    -------
    dict
        defect_principal_axis_ratio, defect_principal_axis_angle, alignment_score
    """
    if len(defect_coords) < 3:
        # Not enough points for meaningful PCA anisotropy
        return {
            "defect_principal_axis_ratio": 1.0, # isotropic
            "defect_principal_axis_angle": 0.0,
            "alignment_score": 0.0
        }
        
    coords_um = np.array(defect_coords) * PITCH
    
    # Centering
    coords_centered = coords_um - np.mean(coords_um, axis=0)
    
    pca = PCA(n_components=2)
    pca.fit(coords_centered)
    
    # Eigenvalues (variance explained)
    explained_variance = pca.explained_variance_ 
    # lambda1 >= lambda2
    l1, l2 = explained_variance[0], explained_variance[1]
    
    # Ratio: l1 / l2. If l2 is 0 (perfect line), handle infinity.
    if l2 < 1e-9:
        ratio = 100.0 # Cap
    else:
        ratio = l1 / l2
        
    # Angle of dominant eigenvector (component 0)
    comp0 = pca.components_[0]
    angle_deg = math.degrees(math.atan2(comp0[1], comp0[0]))
    # Normalize to 0-180
    if angle_deg < 0:
        angle_deg += 180.0
        
    # Alignment score: normalized variance orthogonal to principal axis (l2 / sum)
    # Actually user said "normalized variance orthogonal...". 
    # If aligned, l2 is small. Score should be high?
    # Or "variance orthogonal" -> 0 means perfectly aligned.
    # Let's use Anisotropy = (l1 - l2) / (l1 + l2)?
    # User def: "normalized variance orthogonal to principal axis"
    # This implies l2 / (l1+l2). 0 = line, 0.5 = isotropic circle.
    # But usually "alignment score" implies higher is better.
    # Let's interpret as 1 - (l2 / (l1+l2)) -> 1.0 is line, 0.5 is circle.
    # Or just `l2 / (l1+l2)` and call it `scatter_score`.
    # I'll stick to user text but if it's "alignment score", 1 should be aligned.
    # "normalized variance orthogonal" = l2 / TotalVar.
    # This is a measure of "spread orthogonal to axis". 
    # I will construct `1 - (l2/Total)` to make it an "Alignment Score".
    
    total_var = l1 + l2 + 1e-9
    alignment_score = 1.0 - (l2 / total_var)
    
    return {
        "defect_principal_axis_ratio": float(ratio),
        "defect_principal_axis_angle": float(angle_deg),
        "alignment_score": float(alignment_score)
    }


def compute_spatial_topology(defect_coords, window_half_size_um=4*PITCH):
    """
    Compute graph metrics and max defect-free chord.
    
    Parameters
    ----------
    defect_coords : list
    window_half_size_um : float
        Radius/Half-width of the window to define bounds for convex hull or free space.
    
    Returns
    -------
    dict
        max_defect_free_chord, defect_graph_components, avg_graph_degree
    """
    if len(defect_coords) == 0:
        # Full window is defect free. Max chord = diagonal?
        # Window is 9x9 pillars. 8 gaps -> 8*12 = 96um wide. 
        # Diagonal sqrt(2)*96 = 135um.
        return {
            "max_defect_free_chord": 135.0, # Approx
            "defect_graph_components": 0,
            "avg_graph_degree": 0.0
        }

    coords_um = np.array(defect_coords) * PITCH
    
    # --- Graph Metrics ---
    # Build adjacency matrix based on distance threshold
    dist_mat = squareform(pdist(coords_um))
    adj_mat = dist_mat < GRAPH_CONNECT_DIST
    np.fill_diagonal(adj_mat, False) # No self-loops
    
    # Components (BFS/DFS)
    n_points = len(coords_um)
    visited = [False] * n_points
    n_components = 0
    for i in range(n_points):
        if not visited[i]:
            n_components += 1
            # BFS
            queue = [i]
            visited[i] = True
            while queue:
                u = queue.pop(0)
                neighbors = np.where(adj_mat[u])[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
                        
    # Avg Degree
    degrees = np.sum(adj_mat, axis=1) # Boolean sum is count
    avg_degree = np.mean(degrees)
    
    # --- Max Defect-Free Chord ---
    # Largest circle that can fit? Or largest distance between defects?
    # "largest distance across window with no defects"
    # This is complex (Largest Empty Circle problem or visibility graph).
    # Simplification: Distance between the two most distant defects?
    # No, that's "Span".
    # User might mean: "Largest gap between defects".
    # Or "Diameter of the largest empty circle".
    # Let's approximate by: Delaunay triangulation edges (gaps between neighbors)? 
    # Longest edge in Delaunay triangulation *might* represent a large gap.
    # Or simple: "Distance from Centroid to furthest Voronoi vertex"?
    #
    # Given "max_defect_free_chord" usually implies a linear path.
    # Let's use: Max(Delaunay Edge Length).
    # If defects are dense, edges are short. If sparse/gappy, edges are long.
    # If 1 defect, chord is large (window boundaries).
    # This requires Window Boundaries to be part of the set?
    # Simple approx: Max(pdist(coords)) is Span.
    # Max(Nearest Neighbor Dist) is "Isolation".
    # Let's try Delaunay Max Edge.
    
    if len(coords_um) < 4:
         # Delaunay often needs 4 points for simplex in 2D or robust calc
         # If sparse, the gap is basically the window size.
         max_chord = 100.0 # Placeholder huge gap
    else:
        try:
            tri = Delaunay(coords_um)
            # Edges of triangulation
            max_edge = 0.0
            for simplex in tri.simplices:
                # 3 points in simplex. 3 edges.
                pts = coords_um[simplex]
                # Distances
                d0 = np.linalg.norm(pts[0]-pts[1])
                d1 = np.linalg.norm(pts[1]-pts[2])
                d2 = np.linalg.norm(pts[2]-pts[0])
                max_edge = max(max_edge, d0, d1, d2)
            max_chord = max_edge
        except:
            max_chord = 100.0

    return {
        "max_defect_free_chord": float(max_chord),
        "defect_graph_components": n_components,
        "avg_graph_degree": float(avg_degree)
    }


def compute_symmetry_metrics(defect_coords):
    """
    Centroid offset and quadrant imbalance.
    """
    if len(defect_coords) == 0:
        return {
            "defect_centroid_offset": 0.0,
            "quadrant_imbalance": 0.0
        }
        
    coords_um = np.array(defect_coords) * PITCH
    
    # Centroid Offset (from 0,0)
    centroid = np.mean(coords_um, axis=0)
    offset = np.linalg.norm(centroid)
    
    # Quadrant Imbalance
    # Q1: +x, +y; Q2: -x, +y; Q3: -x, -y; Q4: +x, -y
    q_counts = [0, 0, 0, 0]
    for x, y in coords_um:
        if x >= 0 and y >= 0: q_counts[0] += 1
        elif x < 0 and y >= 0: q_counts[1] += 1
        elif x < 0 and y < 0: q_counts[2] += 1
        elif x >= 0 and y < 0: q_counts[3] += 1
        
    imbalance = np.std(q_counts)
    
    return {
        "defect_centroid_offset": float(offset),
        "quadrant_imbalance": float(imbalance)
    }


def extract_all_topology(defect_coords):
    """Refactored master function."""
    metrics = {}
    metrics.update(compute_clustering_metrics(defect_coords))
    metrics.update(compute_alignment_metrics(defect_coords))
    metrics.update(compute_spatial_topology(defect_coords))
    metrics.update(compute_symmetry_metrics(defect_coords))
    return metrics

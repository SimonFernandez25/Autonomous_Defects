def save_grid_images_simple(image_path, output_dir, grid_size=21, tile_size=50, show_grid=True):
    """
    Extracts a perfect grid of tiles from a CLEAN, manually cropped array.
    Displays the grid overlay before saving tiles.

    Parameters:
    -----------
    image_path : str
        Path to manually cropped array image (21×21 grid).
    output_dir : str
    grid_size : int
        Number of grid rows/cols (default 21).
    tile_size : int
        Half-width of tile crop.
    show_grid : bool
        If True, display image with grid overlay.
    """
    import os, cv2, numpy as np
    import matplotlib.pyplot as plt

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read {image_path}")

    H, W = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Output folder
    array_dir = os.path.join(output_dir, base_name)
    os.makedirs(array_dir, exist_ok=True)

    # Compute grid spacing
    x_spacing = W / grid_size
    y_spacing = H / grid_size

    # ===============================
    # SHOW GRID BEFORE SAVING TILES
    # ===============================
    if show_grid:
        img_disp = img.copy()

        # Draw vertical lines
        for c in range(grid_size + 1):
            x = int(c * x_spacing)
            cv2.line(img_disp, (x, 0), (x, H), (0, 255, 0), 1)

        # Draw horizontal lines
        for r in range(grid_size + 1):
            y = int(r * y_spacing)
            cv2.line(img_disp, (0, y), (W, y), (0, 255, 0), 1)

        # Draw grid centers
        for r in range(grid_size):
            for c in range(grid_size):
                cx = int((c + 0.5) * x_spacing)
                cy = int((r + 0.5) * y_spacing)
                cv2.circle(img_disp, (cx, cy), 3, (255, 0, 0), -1)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
        plt.title(f"Grid Overlay: {base_name}")
        plt.axis("off")
        plt.show()

    # ===============================
    # EXTRACT & SAVE 441 TILES
    # ===============================
    saved = 0

    for r in range(grid_size):
        for c in range(grid_size):

            cx = int((c + 0.5) * x_spacing)
            cy = int((r + 0.5) * y_spacing)

            x1 = max(0, cx - tile_size)
            y1 = max(0, cy - tile_size)
            x2 = min(W, cx + tile_size)
            y2 = min(H, cy + tile_size)

            tile = img[y1:y2, x1:x2]

            fname = f"{base_name}_{r+1},{c+1}.bmp"
            cv2.imwrite(os.path.join(array_dir, fname), tile)
            saved += 1

    print(f"✓ Saved {saved} tiles to {array_dir}")

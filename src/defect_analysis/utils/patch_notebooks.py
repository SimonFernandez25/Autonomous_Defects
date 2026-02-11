import nbformat
import os
import sys
from pathlib import Path

def patch_notebook(filepath):
    filename = os.path.basename(filepath)
    print(f"Patching {filename}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        modified = False
        
        # New robust project root detection and path injection
        path_block = (
            "import sys\n"
            "import os\n"
            "from pathlib import Path\n\n"
            "# Add src directory to path\n"
            "current_dir = Path(os.getcwd())\n"
            "root_dir = current_dir.parent if current_dir.name == 'notebooks' else current_dir\n"
            "src_dir = str(root_dir / \"src\")\n"
            "if src_dir not in sys.path:\n"
            "    sys.path.append(src_dir)\n"
        )

        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                
                # Case 1: 1_Segmentation_and_Preprocessing.ipynb
                if filename == "1_Segmentation_and_Preprocessing.ipynb" and "def preprocess_image" in source:
                    new_source = (
                        "\"\"\"\n"
                        "Array Segmentation using Watershed + DBSCAN\n"
                        "Segments objects arranged in a rectangular array pattern\n"
                        "\"\"\"\n\n"
                    )
                    new_source += path_block + "\n"
                    new_source += (
                        "import numpy as np\n"
                        "import cv2\n"
                        "import matplotlib.pyplot as plt\n"
                        "from defect_analysis.utils.segmentation import segment_array, batch_process, visualize_results\n"
                    )
                    cell.source = new_source
                    modified = True
                    continue

                # Case 2: General migration from scripts.* to defect_analysis.*
                if "scripts." in source or "from scripts" in source:
                    cell.source = cell.source.replace("from scripts.Clustering", "from defect_analysis.clustering")
                    cell.source = cell.source.replace("from scripts.ML", "from defect_analysis.ml")
                    cell.source = cell.source.replace("import scripts.ML", "import defect_analysis.ml")
                    cell.source = cell.source.replace("from scripts.Utilities", "from defect_analysis.utils")
                    modified = True

                # Inject path_block if it's an import cell but missing path logic
                if ("from defect_analysis" in cell.source) and ("sys.path.append" not in cell.source):
                    cell.source = path_block + "\n" + cell.source
                    modified = True

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            print(f"Successfully patched {filename}")
        else:
            print(f"No changes needed for {filename}")
            
    except Exception as e:
        print(f"Error patching {filename}: {e}")

if __name__ == "__main__":
    notebooks_dir = r'c:\Users\srfdyz\OneDrive - University of Missouri\Desktop\Defects\Autonomous_Defects\notebooks'
    if not os.path.exists(notebooks_dir):
        notebooks_dir = 'notebooks'
        
    for filename in os.listdir(notebooks_dir):
        if filename.endswith(".ipynb"):
            patch_notebook(os.path.join(notebooks_dir, filename))

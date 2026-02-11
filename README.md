# Autonomous Defects Analysis Pipeline

This repository contains the codebase for analyzing fabrication defects in meta-atom arrays and correlating their morphology with spectral perturbations.

## Repository Structure

The project is organized into a clean, modular structure:

-   `src/defect_analysis/`: Core Python package containing the analysis logic.
    -   `clustering/`: Logic for layered defect classification.
    -   `ml/`: Machine learning models and spectral analysis utilities.
    -   `utils/`: General image processing and segmentation utilities.
-   `notebooks/`: Sequential pipeline steps in Jupyter Notebook format.
    -   `1_Segmentation_and_Preprocessing.ipynb`: Image segmentation and cluster identification.
    -   `2_Defect_Analysis_and_Classification.ipynb`: Multi-layered defect detection (Missing, Collapsed, Stitching, Irregular).
    -   `3_Spectral_Physics_Analysis.ipynb`: Correlating morphology features with peak shifts and full spectra.
    -   `4_Locality_and_Variance_Study.ipynb`: Advanced physics analysis (locality scale, variance decomposition).
-   `data/`: Input datasets including image tiles and FTIR spectra.
-   `results/`: Intermediate analysis results and classification outputs.
-   `figures/`: Generated publication-quality figures.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-image`
- `opencv-python`
- `matplotlib`
- `seaborn`
- `nbformat`

### Running the Pipeline

The pipeline is designed to be run sequentially through the numbered notebooks in the `notebooks/` directory.

1.  **Segmentation**: Run `1_Segmentation_and_Preprocessing.ipynb` to process raw images and identify meta-atom locations.
2.  **Classification**: Run `2_Defect_Analysis_and_Classification.ipynb` to classify defects based on physical features.
3.  **Spectral Analysis**: Run `3_Spectral_Physics_Analysis.ipynb` to build models that predict spectral responses from defect morphology.
4.  **Physics Study**: Run `4_Locality_and_Variance_Study.ipynb` for deep dives into locality length scales and variance components.

## Development

The core logic is housed in the `src/` directory to facilitate code reuse and maintainability. Notebooks use relative path injection to ensure the `defect_analysis` package is discoverable from within the `notebooks/` folder.

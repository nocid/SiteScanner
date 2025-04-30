# Installation and Usage

This page provides detailed instructions for installing and running SiteScanner.

## Installation

### Prerequisites

*   **Python:** Version 3.8 or higher.
*   **Pip:** Python package installer.
*   **(Optional) Conda:** For managing environments and simplifying PyTorch/PyG installation.
*   **(Optional) CUDA:** For GPU acceleration (requires compatible NVIDIA drivers and CUDA Toolkit).

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/nocid/SiteScanner 
    cd SiteScanner
    ```

2.  **(Recommended) Create and Activate a Virtual Environment:**
    Using Conda:
    ```bash
    conda create -n sitescanner python=3.9 # Or your preferred Python 3.8+ version
    conda activate sitescanner
    ```
    Using venv:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install PyTorch:**
    Install PyTorch according to your system (CPU or CUDA version). Follow the official instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).
    *Example (CPU version):*
    ```bash
    pip install torch torchvision torchaudio
    # Or using conda
    # conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```
    *Example (CUDA 11.x version - check PyTorch site for current commands):*
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Or using conda
    # conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

4.  **Install PyTorch Geometric (PyG) and Dependencies:**
    PyG installation can be complex due to external libraries (`torch-scatter`, `torch-sparse`, etc.). Follow the official [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) carefully, selecting the command that matches your PyTorch version and CUDA version (if applicable).

5.  **Install Remaining Dependencies:**
    Install the other packages listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    This includes `biopython`, `numpy`, `e3nn`, `scikit-learn`, `scipy`, etc.

6.  **(Optional) Install SiteScanner Package:**
    Installing the package makes the `sitescanner` command available system-wide (within the activated environment).
    ```bash
    pip install -e .
    ```
    The `-e` flag installs it in "editable" mode, meaning changes to the source code are immediately reflected without needing reinstallation.

## Running SiteScanner

SiteScanner is run from the command line.

### Basic Command

```bash
# If installed via pip install -e .:
sitescanner [OPTIONS] <pdb_file>

# Alternatively, run directly from the project root:
python sitescanner/cli.py [OPTIONS] <pdb_file>
```

### Arguments

*   **`<pdb_file>` (Required):** Path to the input PDB file for which to predict binding sites.

### Options

*   **`-o OUTPUT_DIR`, `--output_dir OUTPUT_DIR` (Required):**
    Directory where the prediction results (text file and modified PDB) will be saved.
*   **`-m MODEL_PATH`, `--model_path MODEL_PATH`:**
    Path to a custom trained model file (`.pt` or `.pth`). If not specified, the default model included with the package (`sitescanner/weights/model_weights.pt`) will be used.
*   **`-d {cpu,cuda}`, `--device {cpu,cuda}`:**
    Device to run the model inference on. (Default: `cpu`). Choose `cuda` if you have a compatible NVIDIA GPU and installed the CUDA-enabled PyTorch/PyG versions.
*   **`-t THRESHOLD`, `--threshold THRESHOLD`:**
    Probability threshold for classifying a residue as part of a binding site. Must be between 0.0 and 1.0. (Default: `0.9`).
*   **`--postprocess`:**
    Enable DBSCAN clustering post-processing to refine raw predictions into the largest spatial cluster. Recommended for potentially noisy predictions.
*   **`--dbscan_eps EPS`:**
    DBSCAN epsilon parameter (maximum distance in Angstroms between C-alpha atoms for neighborhood). Used only if `--postprocess` is enabled. (Default: `10.0`).
*   **`--dbscan_min_samples MIN_SAMPLES`:**
    DBSCAN min_samples parameter (minimum number of neighboring predicted residues within `eps` distance to form a cluster core). Used only if `--postprocess` is enabled. (Default: `4`).
*   **`-v`, `--version`:**
    Show the program's version number and exit.
*   **`-h`, `--help`:**
    Show the help message describing arguments and options, then exit.

### Examples

*   **Basic Prediction (CPU, default threshold 0.9):**
    ```bash
    sitescanner my_protein.pdb -o ./results
    ```
*   **Prediction with DBSCAN Refinement:**
    ```bash
    sitescanner my_protein.pdb -o ./results --postprocess
    ```
*   **Prediction on GPU with Custom Threshold and Refinement Parameters:**
    ```bash
    sitescanner structure.pdb -o ./output_refined -d cuda -t 0.8 --postprocess --dbscan_eps 8.0 --dbscan_min_samples 5
    ```
*   **Using a Custom Model:**
    ```bash
    sitescanner complex.pdb -o ./custom_model_results -m /path/to/my_model.pth
    ```

## Output Files

For each input PDB file (e.g., `input.pdb`), SiteScanner generates two files in the specified output directory:

1.  **`input_predictions.txt`:**
    *   A plain text file listing the residue identifiers (Format: `ChainID_ResName_ResSeq`, e.g., `A_GLY_101`)
    *   If `--postprocess` is used, this file contains only the residues belonging to the largest spatial cluster found by DBSCAN.
    *   If `--postprocess` is *not* used, this file contains all residues whose prediction probability met the `--threshold`.

2.  **`input_predicted_sites.pdb`:**
    *   A copy of the original PDB file.
    *   The B-factor column (columns 61-66) for all atoms belonging to the predicted residues (as listed in the corresponding `.txt` file) is set to a high value (default: 99.00).
    *   All other atoms have their B-factor set to a low value (default: 10.00).
    *   This allows for easy visualization of the predicted binding site in molecular viewers (e.g., PyMOL, Chimera) by coloring based on B-factor.

## Demo Script

The `demo/` directory contains example PDB files and a script `run_demo.sh`. This script demonstrates running SiteScanner on the sample files with DBSCAN post-processing enabled. See the main [README.md](../README.md) or the [Index](./index.md) for instructions on running the demo.

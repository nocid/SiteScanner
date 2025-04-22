# SiteScanner

Predict protein binding sites from PDB structures.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/sitescanner # Replace with actual URL
    cd SiteScanner
    ```
2.  **(Optional but Recommended) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional) Install PyTorch Geometric dependencies:**
    Follow the specific instructions for your OS/CUDA version on the [PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install `torch-scatter`, `torch-sparse`, etc., if not handled correctly by the base `torch_geometric` install.

5.  **(Optional) Install the package:** For easier command-line access and testing.
    ```bash
    pip install -e . # Installs in editable mode
    ```

## Basic Usage

Run the binding site prediction on a single PDB file:

```bash
# If installed via pip install -e .:
sitescanner path/to/your/protein.pdb -o path/to/output_directory

# Alternatively, run directly:
python sitescanner/cli.py path/to/your/protein.pdb -o path/to/output_directory
```

Use the `--postprocess` flag to enable DBSCAN refinement of predictions:
```bash
sitescanner path/to/your/protein.pdb -o path/to/output_directory --postprocess
```

See `sitescanner --help` or `python sitescanner/cli.py --help` for all options.

## Demo

A simple demo script is provided to run predictions on example PDB files.

1.  **Prerequisites**: Ensure you have installed the package and its dependencies (see Installation).
2.  **Ensure Demo Data Exists**: The demo expects PDB files (`1a2k.pdb`, `1hsg.pdb`, `3pqr.pdb`) to be present in the `demo/demo_data/` directory. If they are missing, you can download them or run a setup step (if provided).
3.  **Make Script Executable (if necessary)**: You might need to give the script execute permissions:
    ```bash
    chmod +x demo/run_demo.sh
    ```
4.  **Run the Demo Script**: Execute the script from the project root directory:
    ```bash
    ./demo/run_demo.sh
    ```
    This script runs the predictions *with* DBSCAN post-processing enabled by default.
5.  **Check Results**: The script will create a `demo/demo_results/` directory. Inside, you'll find subdirectories for each processed PDB file (e.g., `demo/demo_results/1a2k/`). Each subdirectory contains:
    *   `*_predictions.txt`: A text file listing residues in the largest identified spatial cluster.
    *   `*_predicted_sites.pdb`: A PDB file where residues in the largest cluster are highlighted (e.g., by modifying the B-factor field).

## Testing

To run the unit tests:

1.  **Install test dependencies:** Make sure `pytest` and `pytest-mock` are installed (they are included in `requirements.txt`).
    ```bash
    pip install -r requirements.txt
    ```
    Ensure the `sitescanner` package itself is accessible (e.g., by running `pip install -e .` from the project root).

2.  **Run pytest:** Execute pytest from the project root directory:
    ```bash
    pytest
    ```

## Model

(Details about the model architecture, training data, and default weights location would go here.)

The default model weights are expected to be located at `sitescanner/weights/model_weights.pt`. You can specify a different model using the `-m` or `--model_path` argument.

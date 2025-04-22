# SiteScanner

Predict protein binding sites from PDB structures.

**For detailed documentation, please see the [Documentation Index](./docs/index.md).**

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

### Default Weights
The default pre-trained model weights are packaged with the library and expected to be located at `sitescanner/weights/model_weights.pt`. You can specify a path to a different model checkpoint file using the `-m` or `--model_path` command-line argument.

### Architecture
The model used is a Geometric Neural Network based on the `MultiScaleEquivariantResidualNet` architecture defined in `sitescanner/model_definition.py`. It utilizes PyTorch Geometric and the `e3nn` library to process protein structures as graphs and perform SE(3)-equivariant operations, allowing it to learn features sensitive to 3D geometry and rotation. Key parameters are defined within the model definition file and used when loading the model in `sitescanner/core.py`.

### Training Data
To train the model, we used binding site data from the PDBbind dataset that curates pdb structure containing a ligand and a clear binding site.

# --- Contents for sitescanner/core.py ---
# Incorporates elements and concepts from SiteScanner/sbi_model1.py

import torch
import os
# Import necessary components from your other package modules
from .data_utils import pdb_to_graph
from .model_definition import MultiScaleEquivariantResidualNet # Import your model class

try:
    # Python 3.9+
    from importlib.resources import files as importlib_files
except ImportError:
    # Fallback for Python < 3.9
    from importlib.resources import path as importlib_path
    import contextlib

# --- Function to get the path to the weights file ---
def get_default_weights_path():
    """Gets the path to the default model weights packaged with the library."""
    try:
        # Use files() API if available (Python 3.9+)
        # Creates a traversable object representing the package data
        ref = importlib_files('sitescanner').joinpath('weights/model_weights.pth')
        # Check if the resource exists before returning its path as a string
        if ref.is_file():
            return str(ref)
        else:
             raise FileNotFoundError("Default weights file not found in package data.")
    except NameError:
         # Fallback for Python < 3.9 using deprecated path() context manager
        try:
            with contextlib.ExitStack() as stack:
                 # path() returns a context manager yielding a Path object
                 p = stack.enter_context(importlib_path('sitescanner.weights', 'model_weights.pth'))
                 return str(p)
        except Exception as e:
             raise FileNotFoundError(f"Default weights file not found using importlib.resources.path: {e}")



def load_trained_model(model_path, device='cpu'):
    """Loads a pre-trained model state dict."""
    # Define model architecture (ensure parameters match the trained model)
    # These might need to be configurable or saved with the model
    node_dim = 23 # Example value, MUST match trained model
    edge_dim = 19 # Example value, MUST match trained model
    hidden_dim = 64 # Example value, MUST match trained model
    num_layers = 2 # Example value, MUST match trained model

    model = MultiScaleEquivariantResidualNet(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ) #
    # Load the learned weights
    model.load_state_dict(torch.load(model_path, map_location=device)) #
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Loaded trained model from {model_path}")
    return model

def predict_binding_sites(pdb_file_path: str, model_path: str = None, device: str = 'cpu', probability_threshold: float = 0.8):
    """
    Predicts binding site residues for a given PDB file using a trained model.

    Args:
        pdb_file_path: Path to the input PDB file.
        model_path: Path to the trained model's state_dict (.pt or .pth file).
                    If None, the default model included in the package will be used.
        device: Device to run inference on ('cpu' or 'cuda').
        probability_threshold: Threshold for classifying a residue as binding site.

    Returns:
        A tuple containing:
            - A list of residue IDs predicted to be part of the binding site.
            - The full list of residue IDs corresponding to the probabilities.
        Returns None if graph creation fails.
    """
    if not os.path.exists(pdb_file_path):
        raise FileNotFoundError(f"Input PDB not found: {pdb_file_path}")
    
    # If no model path provided, use the default weights
    if model_path is None:
        try:
            model_path = get_default_weights_path()
            print(f"Using default model weights: {model_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find default model weights: {str(e)}")
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    # 1. Load the trained model
    model = load_trained_model(model_path, device=device)

    # 2. Process the input PDB into a graph
    # This function needs to be available, likely from data_utils.py
    graph_data, residue_ids = pdb_to_graph(pdb_file_path) #

    # Check if graph creation or residue ID extraction failed
    if graph_data is None or residue_ids is None:
        print(f"Could not process PDB file into graph or extract residue IDs: {pdb_file_path}")
        return None # Indicate failure

    # Ensure graph_data has a batch attribute for single graph processing
    # Required by some layers like e3nn GatePointsNetwork
    if not hasattr(graph_data, 'batch') or graph_data.batch is None:
        num_nodes = graph_data.num_nodes
        graph_data.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    graph_data = graph_data.to(device) # Move graph data to the correct device

    # 3. Perform inference
    with torch.no_grad(): # Disable gradient calculations for inference
        # The model's forward method from sbi_model1.py returns (out, proj)
        # We only need 'out' for classification.
        # Assuming model returns only one value (logits) based on error
        # out_logits, _ = model(graph_data) # ERROR: too many values to unpack
        out_logits = model(graph_data)

    # 4. Process predictions using adapted logic from snippet
    # probabilities = torch.sigmoid(out_logits).cpu().numpy().flatten() # Apply sigmoid and move to CPU
    # predictions = (probabilities >= probability_threshold).astype(int) # Perform binary classification here

    

    # Apply sigmoid and threshold to get a boolean tensor
    preds_tensor = torch.sigmoid(out_logits) > probability_threshold

    # 5. Map boolean predictions back to residue IDs
    # Convert boolean tensor elements to Python booleans for the condition
    binding_site_residues = [
        res_id for res_id, pred_val in zip(residue_ids, preds_tensor.cpu().numpy()) if bool(pred_val)
    ] # Generate the list of positive residue IDs

    # Return only the list of binding site residues and all residue IDs
    return binding_site_residues, residue_ids

# --- End of sitescanner/core.py ---
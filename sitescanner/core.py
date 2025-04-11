# --- Contents for sitescanner/core.py ---
# Incorporates elements and concepts from SiteScanner/sbi_model1.py

import torch
import os
# Import necessary components from your other package modules
from .data_utils import pdb_to_graph
from .model_definition import MultiScaleEquivariantResidualNet # Import your model class

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

def predict_binding_sites(pdb_file_path: str, model_path: str, device: str = 'cpu', probability_threshold: float = 0.5):
    """
    Predicts binding site residues for a given PDB file using a trained model.

    Args:
        pdb_file_path: Path to the input PDB file.
        model_path: Path to the trained model's state_dict (.pt or .pth file).
        device: Device to run inference on ('cpu' or 'cuda').
        probability_threshold: Threshold for classifying a residue as binding site.

    Returns:
        A list of residue IDs predicted to be part of the binding site.
        Returns None if graph creation fails.
    """
    if not os.path.exists(pdb_file_path):
        raise FileNotFoundError(f"Input PDB not found: {pdb_file_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    # 1. Load the trained model
    model = load_trained_model(model_path, device=device)

    # 2. Process the input PDB into a graph
    # This function needs to be available, likely from data_utils.py
    graph_data, residue_ids = pdb_to_graph(pdb_file_path) #

    if graph_data is None:
        print(f"Could not process PDB file into graph: {pdb_file_path}")
        return None # Indicate failure

    graph_data = graph_data.to(device) # Move graph data to the correct device

    # 3. Perform inference
    with torch.no_grad(): # Disable gradient calculations for inference
        # The model's forward pass expects a batch, even if it's a single graph.
        # We don't need batching here if data_utils prepares a single Data object.
        # If using DataLoader later, batching would be handled there.
        # The model's forward method from sbi_model1.py returns (out, proj)
        # We only need 'out' for classification.
        out_logits, _ = model(graph_data) #

    # 4. Process predictions
    probabilities = torch.sigmoid(out_logits).cpu().numpy().flatten() # Apply sigmoid and move to CPU
    predictions = (probabilities >= probability_threshold).astype(int) #

    # 5. Map predictions back to residue IDs
    binding_site_residues = [
        res_id for res_id, pred in zip(residue_ids, predictions) if pred == 1
    ] #

    return binding_site_residues

# --- End of sitescanner/core.py ---
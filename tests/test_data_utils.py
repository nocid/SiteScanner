import pytest
import os
import torch
from torch_geometric.data import Data
from sitescanner import data_utils

# Define the path to the sample PDB file relative to the tests directory
SAMPLE_PDB_PATH = os.path.join(os.path.dirname(__file__), 'sample.pdb')

# Mock external dependencies (SASA and Dihedrals) to isolate pdb_to_graph logic
@pytest.fixture(autouse=True)
def mock_external_computations(mocker):
    # Mock SASA computation - return dummy values per residue
    # Adjust the number of dummy values if sample.pdb changes
    mock_sasa = {
        ('A', (' ', 1, ' ')): 0.5,
        ('A', (' ', 2, ' ')): 0.6,
        ('A', (' ', 3, ' ')): 0.7,
    }
    mocker.patch('sitescanner.data_utils.compute_sasa', return_value=mock_sasa)

    # Mock Phi/Psi computation - return dummy angles
    mock_phi = { ('A', (' ', 2, ' ')): -60.0, ('A', (' ', 3, ' ')): -70.0 }
    mock_psi = { ('A', (' ', 1, ' ')): 140.0, ('A', (' ', 2, ' ')): 150.0 }
    mocker.patch('sitescanner.data_utils.compute_phi_psi_from_pdb', return_value=(mock_phi, mock_psi))

    # Mock extract_residue_features just to return a vector of expected size (23)
    # This avoids needing to test the complex feature extraction logic here
    mocker.patch('sitescanner.data_utils.extract_residue_features', return_value=[0.0] * 23)

def test_pdb_to_graph_success():
    """Test successful graph creation from the sample PDB."""
    assert os.path.exists(SAMPLE_PDB_PATH), f"Test PDB not found at {SAMPLE_PDB_PATH}"

    graph, residue_ids = data_utils.pdb_to_graph(SAMPLE_PDB_PATH, distance_threshold=5.0)

    # Basic checks
    assert graph is not None
    assert isinstance(graph, Data)
    assert isinstance(residue_ids, list)

    # Check residue IDs (sample.pdb has ALA-1, GLY-2, SER-3 in chain A)
    assert residue_ids == ['A_ALA_1', 'A_GLY_2', 'A_SER_3']
    num_nodes = len(residue_ids)
    assert graph.num_nodes == num_nodes

    # Check tensor shapes and types
    assert isinstance(graph.x, torch.Tensor)
    assert graph.x.shape == (num_nodes, 23) # Node features
    assert graph.x.dtype == torch.float

    assert isinstance(graph.pos, torch.Tensor)
    assert graph.pos.shape == (num_nodes, 3) # C-alpha positions
    assert graph.pos.dtype == torch.float

    assert isinstance(graph.edge_index, torch.Tensor)
    assert graph.edge_index.shape[0] == 2 # Shape [2, num_edges]
    assert graph.edge_index.dtype == torch.long
    # Add check for index validity if edges exist
    if graph.edge_index.numel() > 0:
        assert graph.edge_index.max() < num_nodes

    assert isinstance(graph.edge_attr, torch.Tensor)
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1] # num_edges
    assert graph.edge_attr.shape[1] == 19 # Edge features (16 radial + 3 SH)
    assert graph.edge_attr.dtype == torch.float

    # Simple check for non-zero edges with a small distance threshold
    # ALA-1 to GLY-2 C-alpha distance is ~2.9 Angstroms in sample.pdb
    assert graph.edge_index.shape[1] > 0, "Expected edges were not found with threshold 5.0"

def test_pdb_to_graph_file_not_found():
    """Test FileNotFoundError when PDB does not exist."""
    with pytest.raises(FileNotFoundError):
        data_utils.pdb_to_graph("nonexistent_file.pdb")

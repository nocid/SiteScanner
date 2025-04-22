import pytest
import torch
from unittest.mock import patch, MagicMock
from sitescanner import core
from torch_geometric.data import Data

# Mock object for the model
class MockModel:
    def __init__(self, output_value):
        self.output_value = output_value
        self.eval_called = False

    def __call__(self, graph_data):
        # Return a tensor with the specified shape and value
        num_nodes = graph_data.num_nodes
        return torch.full((num_nodes, 1), self.output_value, dtype=torch.float)

    def to(self, device):
        return self # Mock device placement

    def eval(self):
        self.eval_called = True
        return self # Mock eval call

    def load_state_dict(self, state_dict):
        pass # Mock state dict loading

# Mock object for graph data
def create_mock_graph(num_nodes=5):
    return Data(x=torch.randn(num_nodes, 23), # Match node_dim
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 19)), # Match edge_dim
                pos=torch.randn(num_nodes, 3),
                num_nodes=num_nodes)

@pytest.fixture
def mock_dependencies(mocker):
    """Fixture to mock functions called by predict_binding_sites."""
    # Mock load_trained_model
    mock_model_instance = MockModel(output_value=2.5) # Logit > sigmoid threshold
    mocker.patch('sitescanner.core.load_trained_model', return_value=mock_model_instance)

    # Mock pdb_to_graph
    mock_graph = create_mock_graph(num_nodes=10)
    mock_residue_ids = [f"A_RES_{i}" for i in range(10)]
    mocker.patch('sitescanner.core.pdb_to_graph', return_value=(mock_graph, mock_residue_ids))

    # Mock get_default_weights_path (needed if model_path is None)
    mocker.patch('sitescanner.core.get_default_weights_path', return_value='dummy/path/weights.pth')

    # Mock os.path.exists
    mocker.patch('os.path.exists', return_value=True)

    return {
        'mock_model': mock_model_instance,
        'mock_graph': mock_graph,
        'mock_residue_ids': mock_residue_ids
    }

def test_predict_binding_sites_success(mock_dependencies):
    """Test successful prediction path."""
    pdb_file = "fake.pdb"
    threshold = 0.8 # Sigmoid(2.5) is approx 0.92, so should be > threshold

    binding_sites, all_ids = core.predict_binding_sites(
        pdb_file_path=pdb_file,
        probability_threshold=threshold
    )

    # Check return types
    assert isinstance(binding_sites, list)
    assert isinstance(all_ids, list)

    # Check content
    assert all_ids == mock_dependencies['mock_residue_ids']
    # Since mock logit 2.5 -> prob ~0.92 > 0.8, all residues should be predicted
    assert len(binding_sites) == len(all_ids)
    assert binding_sites == all_ids

def test_predict_binding_sites_threshold_filter(mock_dependencies):
    """Test that the threshold correctly filters residues."""
    # Update mock model to output a lower logit
    mock_dependencies['mock_model'].output_value = 0.5 # Sigmoid(0.5) approx 0.62
    pdb_file = "fake.pdb"
    threshold = 0.7 # Now threshold is higher than probability

    binding_sites, all_ids = core.predict_binding_sites(
        pdb_file_path=pdb_file,
        probability_threshold=threshold
    )

    assert all_ids == mock_dependencies['mock_residue_ids']
    # Since prob ~0.62 < 0.7, no residues should be predicted
    assert len(binding_sites) == 0

def test_predict_binding_sites_graph_fail(mock_dependencies, mocker):
    """Test the case where pdb_to_graph returns None."""
    mocker.patch('sitescanner.core.pdb_to_graph', return_value=(None, None))
    pdb_file = "fake.pdb"

    result = core.predict_binding_sites(pdb_file_path=pdb_file)

    assert result is None

def test_predict_binding_sites_pdb_not_found(mocker):
    """Test FileNotFoundError for PDB."""
    mocker.patch('os.path.exists', return_value=False)
    with pytest.raises(FileNotFoundError, match="Input PDB not found"):
        core.predict_binding_sites(pdb_file_path="nonexistent.pdb")

# TODO: Add tests for load_trained_model and get_default_weights_path (may require more complex mocking)

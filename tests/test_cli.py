import pytest
from sitescanner import cli
import argparse

# Test successful parsing with required arguments
def test_cli_basic_args():
    args_list = ['dummy.pdb', '-o', 'output_dir']
    args = cli.main.__closure__[0].cell_contents.parse_args(args_list) # Access parser inside main
    assert args.pdb_file == 'dummy.pdb'
    assert args.output_dir == 'output_dir'
    assert args.threshold == 0.9 # Check default
    assert args.device == 'cpu' # Check default
    assert not args.postprocess # Check default

# Test parsing with optional arguments
def test_cli_optional_args():
    args_list = [
        'protein.pdb',
        '-o', 'results',
        '-m', 'model.pth',
        '-d', 'cuda',
        '-t', '0.7',
        '--postprocess',
        '--dbscan_eps', '8.0',
        '--dbscan_min_samples', '5'
    ]
    args = cli.main.__closure__[0].cell_contents.parse_args(args_list)
    assert args.pdb_file == 'protein.pdb'
    assert args.output_dir == 'results'
    assert args.model_path == 'model.pth'
    assert args.device == 'cuda'
    assert args.threshold == 0.7
    assert args.postprocess is True
    assert args.dbscan_eps == 8.0
    assert args.dbscan_min_samples == 5

# Test invalid threshold value
def test_cli_invalid_threshold():
    with pytest.raises(SystemExit):
         # Accessing the parser via closure is a bit hacky but avoids redefining it
         # A better way would be to refactor cli.py to expose the parser
         parser = cli.main.__closure__[0].cell_contents
         parser.parse_args(['dummy.pdb', '-o', 'out', '-t', '1.5'])

# Test missing required argument
def test_cli_missing_output_dir():
    with pytest.raises(SystemExit):
         parser = cli.main.__closure__[0].cell_contents
         parser.parse_args(['dummy.pdb'])

# Test invalid DBSCAN params when postprocess is enabled
def test_cli_invalid_dbscan_params():
    parser = cli.main.__closure__[0].cell_contents
    # Invalid eps
    with pytest.raises(SystemExit):
        parser.parse_args(['dummy.pdb', '-o', 'out', '--postprocess', '--dbscan_eps', '-1.0'])
    # Invalid min_samples
    with pytest.raises(SystemExit):
        parser.parse_args(['dummy.pdb', '-o', 'out', '--postprocess', '--dbscan_min_samples', '0'])

# Note: Testing the full main() function execution requires more complex setup
# (mocking file system, core functions, subprocesses, etc.)
# These tests focus solely on the argument parsing setup within cli.main.

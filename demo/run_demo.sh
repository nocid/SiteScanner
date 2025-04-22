#!/bin/bash

# Script to run the SiteScanner demo with DBSCAN post-processing
# Assumes execution from the project root directory

# --- Configuration ---
# Relative path to the SiteScanner CLI script
SCANNER_CLI="sitescanner/cli.py"
# Relative path to the directory containing demo PDB files
DEMO_DATA_DIR="demo/demo_data"
# Relative path to the directory where results will be saved
RESULTS_DIR="demo/demo_results"
# PDB files to process
PDB_FILES=("1a2k.pdb" "1hsg.pdb" "3pqr.pdb")
# SiteScanner options (adjust as needed)
DEVICE="cpu" # Use 'cuda' if GPU is available and configured
THRESHOLD="0.9"
# DBSCAN is enabled via the --postprocess flag below
# Default DBSCAN params (eps=10.0, min_samples=4) will be used unless specified here

# --- Script Execution ---

# Check if the scanner CLI script exists
if [ ! -f "$SCANNER_CLI" ]; then
    echo "Error: SiteScanner CLI not found at '$SCANNER_CLI'. Make sure you are running this script from the project root."
    exit 1
fi

# Create the results directory if it doesn't exist
echo "Creating results directory: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Process each PDB file
echo "Starting SiteScanner demo with DBSCAN post-processing..."
for pdb_file in "${PDB_FILES[@]}"; do
    input_path="$DEMO_DATA_DIR/$pdb_file"
    base_name=$(basename "$pdb_file" .pdb)
    output_subdir="$RESULTS_DIR/$base_name"

    if [ ! -f "$input_path" ]; then
        echo "Warning: PDB file not found at '$input_path'. Skipping."
        continue
    fi

    echo "-----------------------------------------------------"
    echo "Processing $pdb_file (with DBSCAN refinement)..."
    echo "Input: $input_path"
    echo "Output Subdirectory: $output_subdir"
    echo "-----------------------------------------------------"

    # Create specific output subdirectory for this PDB file
    mkdir -p "$output_subdir"

    # Construct the command with post-processing enabled
    command="python $SCANNER_CLI \
        "$input_path" \
        -o "$output_subdir" \
        --device "$DEVICE" \
        --threshold "$THRESHOLD" \
        --postprocess" # Add flag to enable DBSCAN
        # To override DBSCAN defaults, add e.g.: --dbscan_eps 8.0 --dbscan_min_samples 5

    # Print the command being executed (optional, for debugging)
    # echo "Executing: $command"

    # Execute the command
    eval $command

    if [ $? -eq 0 ]; then
        echo "Successfully processed $pdb_file. Refined results saved in $output_subdir."
    else
        echo "Error processing $pdb_file. Check the output above for details."
        # Decide if you want the script to exit on first error or continue
        # exit 1 # Uncomment to stop on first error
    fi
    echo "" # Add a newline for readability
done

echo "-----------------------------------------------------"
echo "Demo finished. Check the '$RESULTS_DIR' directory for prediction outputs."
echo "Outputs include refined predictions based on the largest spatial cluster found by DBSCAN:"
echo "  - *_predictions.txt: List of residues in the largest cluster."
echo "  - *_predicted_sites.pdb: Original PDB with residues in the largest cluster highlighted."
echo "-----------------------------------------------------"

exit 0

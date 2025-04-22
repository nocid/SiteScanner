# SiteScanner Documentation

Welcome to the documentation for **SiteScanner**, a tool designed to predict protein binding sites from PDB structure files using geometric deep learning.

## Overview

SiteScanner leverages the power of graph neural networks (GNNs), specifically utilizing equivariant GNN layers (via the `e3nn` library), to analyze the 3D structure and chemical properties of proteins. It processes an input PDB file, converts it into a graph representation, and feeds this graph to a pre-trained model to predict which residues are likely part of a functional binding site.

## Key Features

*   **Input:** Accepts standard Protein Data Bank (PDB) files.
*   **Prediction:** Uses a pre-trained equivariant GNN model to predict binding site residues.
*   **Output:**
    *   A text file (`*_predictions.txt`) listing the predicted binding site residue IDs.
    *   A modified PDB file (`*_predicted_sites.pdb`) where predicted residues have their B-factor column set to a high value (e.g., 99.0) for easy visualization.
*   **Refinement:** Includes an optional post-processing step using DBSCAN clustering to refine scattered predictions into the most plausible, spatially coherent binding pocket.
*   **Command-Line Interface:** Easy-to-use CLI for running predictions and configuring parameters.

## Navigation

*   **[Installation and Usage](./usage.md):** Detailed instructions on how to install SiteScanner, run predictions, and understand the command-line options.
*   **[Methodology](./theory.md):** Explanation of the underlying methods, including graph construction, model architecture overview, and the DBSCAN post-processing approach.
*   **[Analyzing Results](./analyzing_results.md):** Examples of how to perform basic structural analysis on the output files.

## Getting Started

1.  **Installation:** Follow the steps in the [Installation and Usage](./usage.md) guide.
2.  **Run Prediction:**
    ```bash
    sitescanner path/to/your/protein.pdb -o path/to/output_directory --postprocess
    ```
    (The `--postprocess` flag enables DBSCAN refinement).

## Resources

*   **Source Code:** [GitHub Repository](https://github.com/nocid/SiteScanner) 
*   **Issue Tracker:** Report bugs or suggest features via the GitHub Issues page.

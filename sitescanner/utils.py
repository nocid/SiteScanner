# --- Contents for sitescanner/utils.py ---

import os
import numpy as np # Added
from typing import List # Added for type hinting
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
try: # Added
    from sklearn.cluster import DBSCAN # Added
except ImportError: # Added
    DBSCAN = None # Added - Handle optional dependency

def save_predictions_to_txt(binding_site_residues: List[str], output_filepath: str):
    """
    Saves the predicted binding site residues (provided as a list)
    to a human-readable text file.

    Args:
        binding_site_residues (List[str]): A list of residue identifiers (e.g., "A_GLY_10")
                                           that are predicted as binding sites.
        output_filepath (str): Path to the output text file.
    """
    # if len(predictions) != len(residue_ids):
    #     raise ValueError("Length of predictions must match length of residue_ids.")

    # predicted_sites = [res_id for res_id, pred in zip(residue_ids, predictions) if pred == 1]
    # Use the input list directly
    predicted_sites = binding_site_residues

    try:
        with open(output_filepath, 'w') as f:
            f.write("# Predicted Binding Site Residues (SiteScanner)\n")
            if predicted_sites:
                for res_id in predicted_sites:
                    f.write(f"{res_id}\n")
            else:
                f.write("No binding site residues predicted.\n")
        print(f"Predicted binding sites saved to: {output_filepath}")
    except IOError as e:
        print(f"Error writing prediction file {output_filepath}: {e}")


def color_pdb_by_prediction(input_pdb_path: str, output_pdb_path: str, binding_site_residues: List[str], predicted_bfactor: float = 99.00, default_bfactor: float = 10.00):
    """
    Creates a copy of the input PDB file with the B-factor column modified
    to indicate predicted binding site residues based on the provided list.

    Args:
        input_pdb_path (str): Path to the original PDB file.
        output_pdb_path (str): Path where the modified PDB file will be saved.
        binding_site_residues (List[str]): A list of residue identifiers (e.g., "A_GLY_10")
                                           predicted as binding sites.
        predicted_bfactor (float): B-factor value to assign to predicted binding site residues.
        default_bfactor (float): B-factor value to assign to non-binding site residues.
    """
    if not os.path.exists(input_pdb_path):
        raise FileNotFoundError(f"Input PDB file not found: {input_pdb_path}")
    # if len(predictions) != len(residue_ids):
    #     raise ValueError("Length of predictions must match length of residue_ids.")

    # Create a set of predicted residue IDs for efficient lookup
    # predicted_residue_set = {res_id for res_id, pred in zip(residue_ids, predictions) if pred == 1}
    predicted_residue_set = set(binding_site_residues)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb_path)

    # Modify B-factors
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if residue is an amino acid and format its ID correctly
                # Note: PDB residue ID is a tuple (hetfield, resseq, icode)
                if PDB.is_aa(residue, standard=True): # Consider standard AAs
                    res_id_tuple = residue.get_id()
                    # Format ID as used in residue_ids list (e.g., "A_GLY_10")
                    # Assuming residue_ids format is ChainID_ResName_ResSeq
                    current_res_id_str = f"{chain.id}_{residue.get_resname()}_{res_id_tuple[1]}"

                    is_predicted = current_res_id_str in predicted_residue_set

                    # Set B-factor for all atoms in the residue
                    for atom in residue:
                        if is_predicted:
                            atom.set_bfactor(predicted_bfactor)
                        else:
                            atom.set_bfactor(default_bfactor)
                else:
                     # Handle non-AA residues if necessary, e.g., set default B-factor
                     for atom in residue:
                         atom.set_bfactor(default_bfactor)


    # Save the modified structure
    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(output_pdb_path)
        print(f"PDB with colored B-factors saved to: {output_pdb_path}")
    except Exception as e:
        print(f"Error saving modified PDB file {output_pdb_path}: {e}")


# --- Added DBSCAN refinement function ---

def refine_predictions_with_dbscan(predicted_residue_ids: List[str], pdb_file_path: str, eps: float = 10.0, min_samples: int = 4) -> List[str]:
    """
    Refines a list of predicted binding site residues using DBSCAN clustering
    on their C-alpha coordinates.

    Args:
        predicted_residue_ids (List[str]): List of residue IDs predicted by the model.
        pdb_file_path (str): Path to the original PDB file.
        eps (float): DBSCAN epsilon parameter (max distance between samples).
        min_samples (int): DBSCAN min_samples parameter (number of neighbors for core point).

    Returns:
        List[str]: A refined list containing only residues from the largest spatial cluster.
                   Returns the original list if clustering fails or finds no clusters.
    """
    if DBSCAN is None:
        print("Warning: scikit-learn not installed. Skipping DBSCAN post-processing.")
        return predicted_residue_ids

    if not predicted_residue_ids or len(predicted_residue_ids) < min_samples:
        print(f"Warning: Not enough predicted residues ({len(predicted_residue_ids)}) for DBSCAN clustering (min_samples={min_samples}). Returning original predictions.")
        return predicted_residue_ids

    predicted_residue_set = set(predicted_residue_ids)
    coords = []
    ids_for_coords = []

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file_path)

        for chain in structure.get_chains():
            for residue in chain.get_residues():
                if PDB.is_aa(residue, standard=True):
                    res_id_tuple = residue.get_id()
                    current_res_id_str = f"{chain.id}_{residue.get_resname()}_{res_id_tuple[1]}"

                    if current_res_id_str in predicted_residue_set:
                        try:
                            ca_coord = residue["CA"].get_coord()
                            coords.append(ca_coord)
                            ids_for_coords.append(current_res_id_str)
                        except KeyError:
                            print(f"Warning: Could not find C-alpha atom for predicted residue {current_res_id_str}. Skipping.")
                            continue # Skip residues without C-alpha

    except FileNotFoundError:
        print(f"Error: PDB file not found at {pdb_file_path} during post-processing. Returning original predictions.")
        return predicted_residue_ids
    except Exception as e:
        print(f"Error parsing PDB for post-processing: {e}. Returning original predictions.")
        return predicted_residue_ids

    if len(coords) < min_samples:
        print(f"Warning: Not enough C-alpha coordinates ({len(coords)}) extracted for DBSCAN (min_samples={min_samples}). Returning original predictions.")
        return predicted_residue_ids

    np_coords = np.array(coords)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np_coords)
    labels = clustering.labels_

    # Find the largest cluster (excluding noise points labeled -1)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

    if counts.size == 0:
        print("DBSCAN: No clusters found (all points considered noise). Returning original predictions.")
        return predicted_residue_ids # Or maybe return empty list? Returning original for now.
    else:
        largest_cluster_label = unique_labels[counts.argmax()]
        print(f"DBSCAN: Found {len(unique_labels)} cluster(s). Largest is label {largest_cluster_label} with {counts.max()} residues.")

        # Filter residue IDs belonging to the largest cluster
        refined_residue_ids = [
            res_id for i, res_id in enumerate(ids_for_coords)
            if labels[i] == largest_cluster_label
        ]
        return refined_residue_ids

# --- End of sitescanner/utils.py ---
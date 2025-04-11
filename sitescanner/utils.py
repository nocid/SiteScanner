# --- Contents for sitescanner/utils.py ---

import os
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

def save_predictions_to_txt(predictions, residue_ids, output_filepath):
    """
    Saves the predicted binding site residues to a human-readable text file.

    Args:
        predictions (list or np.array): A list/array of 0s and 1s, same length as residue_ids.
                                        1 indicates a predicted binding site residue.
        residue_ids (list): A list of residue identifiers (e.g., "A_GLY_10") corresponding
                            to the predictions.
        output_filepath (str): Path to the output text file.
    """
    if len(predictions) != len(residue_ids):
        raise ValueError("Length of predictions must match length of residue_ids.")

    predicted_sites = [res_id for res_id, pred in zip(residue_ids, predictions) if pred == 1]

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


def color_pdb_by_prediction(input_pdb_path: str, output_pdb_path: str, predictions, residue_ids, predicted_bfactor: float = 99.00, default_bfactor: float = 10.00):
    """
    Creates a copy of the input PDB file with the B-factor column modified
    to indicate predicted binding site residues.

    Args:
        input_pdb_path (str): Path to the original PDB file.
        output_pdb_path (str): Path where the modified PDB file will be saved.
        predictions (list or np.array): A list/array of 0s and 1s corresponding to residue_ids.
        residue_ids (list): A list of residue identifiers (e.g., "A_GLY_10") generated
                            during graph creation, matching the order of predictions.
        predicted_bfactor (float): B-factor value to assign to predicted binding site residues.
        default_bfactor (float): B-factor value to assign to non-binding site residues.
    """
    if not os.path.exists(input_pdb_path):
        raise FileNotFoundError(f"Input PDB file not found: {input_pdb_path}")
    if len(predictions) != len(residue_ids):
        raise ValueError("Length of predictions must match length of residue_ids.")

    # Create a set of predicted residue IDs for efficient lookup
    predicted_residue_set = {res_id for res_id, pred in zip(residue_ids, predictions) if pred == 1}

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


# --- End of sitescanner/utils.py ---
# --- Contents for sitescanner/data_utils.py ---


import os
import re
import numpy as np
import torch
from torch_geometric.data import Data
from Bio import PDB
from Bio.PDB import PDBParser, ShrakeRupley, is_aa
from scipy.spatial import distance_matrix, KDTree
from e3nn import o3
from typing import Dict, Tuple

# --- Feature Calculation Functions ---

def compute_phi_psi_from_pdb(pdb_file: str):
    """Computes phi (ϕ) and psi (ψ) dihedral angles from a given PDB file.""" #
    # Parse the PDB file
    parser = PDB.PDBParser(QUIET=1)
    structure = parser.get_structure("protein", pdb_file)

    # Convert to internal coordinates
    structure.atom_to_internal_coordinates()

    # Extract first chain
    chain = list(structure.get_chains())[0]
    ic_chain = chain.internal_coord

    # Get dihedral angles
    d: Dict[
        Tuple[PDB.internal_coords.AtomKey, ...], PDB.internal_coords.Dihedron
    ] = ic_chain.dihedra # type hint simplified for brevity

    phi_angles = {}
    psi_angles = {}

    for key, dihedral in d.items():
        atom1, atom2, atom3, atom4 = key  # Each is an AtomKey object
        try:
            # Simplified splitting assuming standard AtomKey format
            res1_info, atom_name1 = str(atom1).split("_")
            res2_info, atom_name2 = str(atom2).split("_")
            res3_info, atom_name3 = str(atom3).split("_")
            res4_info, atom_name4 = str(atom4).split("_")

            res1 = int(re.findall(r'\d+', res1_info)[0])
            res2 = int(re.findall(r'\d+', res2_info)[0])
            res3 = int(re.findall(r'\d+', res3_info)[0])
            res4 = int(re.findall(r'\d+', res4_info)[0]) #

        except (ValueError, IndexError): # Handle both ValueError and potential IndexError
            continue # Skip this dihedral if it has an unexpected format

        # Identify PHI (ϕ): (C(i-1), N(i), CA(i), C(i))
        if atom_name1 == "C" and atom_name2 == "N" and atom_name3 == "CA" and atom_name4 == "C":
            phi_angles[res2] = dihedral.angle # Phi belongs to residue `i` (res2)

        # Identify PSI (ψ): (N(i), CA(i), C(i), N(i+1))
        if atom_name1 == "N" and atom_name2 == "CA" and atom_name3 == "C" and atom_name4 == "N":
            psi_angles[res1] = dihedral.angle # Psi belongs to residue `i` (res1)

    return phi_angles, psi_angles

def compute_sasa(pdb_file):
    """Compute Solvent Accessible Surface Area (SASA) for the protein.""" #
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Use Shrake-Rupley method to calculate SASA
    sr = ShrakeRupley()
    sr.compute(structure) # This will compute the SASA values for the structure

    sasa = {}
    # Loop through all models, chains, and residues
    for model in structure:
        for chain in model:
            for residue in chain:
                 # Only amino acids
                if PDB.is_aa(residue):
                    # Access SASA through the residue's attribute `sasa`
                     # Check if SASA is computed for this residue
                    if hasattr(residue, 'sasa') and residue.sasa is not None:
                        residue_id = f"{chain.id}_{residue.get_resname()}_{residue.get_id()[1]}" #
                        sasa_value = residue.sasa # The SASA value computed by Shrake-Rupley
                        sasa[residue_id] = sasa_value # Store SASA per residue

    return sasa

def extract_residue_features(residue, sasa_values, phi, psi):
    """Extracts structural features from a residue""" #
    aa_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'] #

    aa_one_hot = np.zeros(len(aa_types)) #
     # One-hot encoding
    if residue.get_resname() in aa_types:
        aa_one_hot[aa_types.index(residue.get_resname())] = 1

    # Get residue ID for lookup in dictionaries
    residue_id = f"{residue.get_parent().id}_{residue.get_resname()}_{residue.get_id()[1]}" #

    # Access SASA, phi, and psi using the residue ID
    sasa = sasa_values.get(residue_id, 0) # Use 0 as default if not found
    phi_angle = phi.get(residue.get_id()[1], 0) # Use 0 as default if not found
    psi_angle = psi.get(residue.get_id()[1], 0) # Use 0 as default if not found

    # Concatenate all features into a single vector
    features = np.concatenate([aa_one_hot, [sasa, phi_angle, psi_angle]]) #

    return features

def radial_basis(distances, num_radial=16, max_radius=5.0):
    """
    Encode distances using radial basis functions.
    Args: ... Returns: ...
    """ #
    # Create centers for radial basis functions
    centers = torch.linspace(0, max_radius, num_radial)

    # Compute width of Gaussian functions
    width = (max_radius / (num_radial - 1)) * 2

    # Expand dimensions for broadcasting
    distances = distances.unsqueeze(-1)
    centers = centers.unsqueeze(0)

    # Compute Gaussian radial basis functions
    rbf = torch.exp(-(distances - centers)**2 / width**2)

    return rbf

# --- Graph Creation Function ---

def pdb_to_graph(pdb_file, distance_threshold=8.0):
    """Loads a PDB file and converts it into a PyTorch Geometric graph."""
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file {pdb_file} not found!") #

    parser = PDBParser(QUIET=True) #
    structure = parser.get_structure("protein", pdb_file) #

    # Precompute SASA and dihedrals to avoid recomputing them for each residue
    sasa_values = compute_sasa(pdb_file) #
    phi_angles, psi_angles = compute_phi_psi_from_pdb(pdb_file) #

    nodes = [] # List of residue features
    positions = [] # C-alpha positions for graph edges
    residue_ids = [] # Store residue identifiers (e.g., "A_GLY_10")
    residue_map = {} # Map residue object to its index

    idx = 0
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            if is_aa(residue): #
                try:
                    ca = residue["CA"].get_coord() # Extract C-alpha coordinates
                    positions.append(ca) #
                    residue_map[residue] = idx #
                    nodes.append(extract_residue_features(residue, sasa_values, phi_angles, psi_angles)) #
                    res_id_str = f"{chain.id}_{residue.get_resname()}_{residue.get_id()[1]}" #
                    residue_ids.append(res_id_str) #
                    idx += 1
                except KeyError:
                    # Skip residues without C-alpha (e.g., missing density)
                    continue

    # Convert positions to NumPy array for faster processing
    positions = np.array(positions) #

    if not positions.size: # Handle cases with no valid residues
        return None, []

    # Use KDTree to find edges within the distance threshold
    tree = KDTree(positions) #
    pairs = tree.query_pairs(distance_threshold) #

    # Convert pairs to edge list
    edge_index = np.array(list(pairs)).T #
    if edge_index.size == 0: # Handle cases with no edges
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = torch.empty((0, 19), dtype=torch.float) # 19 = num_radial + sh_dim
    else:
        edge_index = edge_index.astype(np.int64) # Ensure correct dtype

        # Validate indices before tensor conversion
        if np.any(edge_index >= len(positions)):
            invalid = edge_index[edge_index >= len(positions)] #
            print(f"Warning: Invalid edge indices found: {invalid}. Max node index: {len(positions)-1}")
            # Filter out invalid edges (consider raising an error depending on desired strictness)
            valid_mask = (edge_index[0] < len(positions)) & (edge_index[1] < len(positions))
            edge_index = edge_index[:, valid_mask]

        # Calculate edge attributes (distance, radial basis, spherical harmonics)
        edge_src, edge_dst = edge_index #
        edge_vec = torch.tensor(positions[edge_dst] - positions[edge_src], dtype=torch.float) #
        distances = torch.norm(edge_vec, dim=1) #

        # Generate radial basis features
        basis = radial_basis(distances, num_radial=16, max_radius=distance_threshold) # Use threshold as max_radius

        # Add spherical harmonics for direction (degree=1 → 3D)
        edge_attr_sh = o3.spherical_harmonics([1], edge_vec, normalize=True, normalization='component') #

        edge_attr = torch.cat([basis, edge_attr_sh], dim=1) # → [num_edges, 16+3=19]

    # --- Create PyG Data object ---
    x = torch.tensor(np.array(nodes), dtype=torch.float) #
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long) #
    pos_tensor = torch.tensor(positions, dtype=torch.float) #

    # Add validation check
    if edge_index_tensor.numel() > 0:
        max_node_idx = x.shape[0] - 1 # Since indices start at 0
        if edge_index_tensor.max() > max_node_idx:
            raise ValueError(f"Invalid edge index {edge_index_tensor.max()} detected for {max_node_idx+1} nodes in {pdb_file}") #

    graph_data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr, pos=pos_tensor) #

    return graph_data, residue_ids # Return graph and the list of residue IDs


# --- Functions below were for training data prep in the original script ---
# --- They might be moved to a separate training script/module later ---

def extract_binding_residues(pocket_pdb_file):
    """Extracts binding residues IDs from a pocket PDB file.""" #
    # (Implementation remains the same as in sbi_model1.py)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pocket_pdb_file)
    binding_residues = set()
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            if PDB.is_aa(residue): # Ensure it's an amino acid
                res_id = f"{chain.id}_{residue.get_resname()}_{residue.get_id()[1]}" #
                binding_residues.add(res_id)
    return binding_residues

def process_complex(complex_dir):
    """Processes a directory containing protein and pocket PDBs for training.""" #
    # (Implementation remains the same, but uses pdb_to_graph)
    pdb_protein_file = None
    pdb_pocket_file = None
    # ... find files ...
    for file in os.listdir(complex_dir):
        if file.endswith(".pdb"):
            if "pocket" in file.lower(): #
                pdb_pocket_file = os.path.join(complex_dir, file)
            elif "protein" in file.lower(): # Assuming protein file name contains 'protein'
                 pdb_protein_file = os.path.join(complex_dir, file)

    # Fallback if 'protein' isn't in the name
    if not pdb_protein_file:
         for file in os.listdir(complex_dir):
             if file.endswith(".pdb") and "pocket" not in file.lower():
                 pdb_protein_file = os.path.join(complex_dir, file)
                 break # Take the first non-pocket PDB

    if not pdb_protein_file or not pdb_pocket_file:
        print(f"Skipping {complex_dir}: Missing required PDB files.") #
        return None

    protein_graph, residue_ids = pdb_to_graph(pdb_protein_file) #
    if protein_graph is None: # Handle case where graph creation failed
        print(f"Skipping {complex_dir}: Could not create graph from {pdb_protein_file}.")
        return None

    binding_residues = extract_binding_residues(pdb_pocket_file) #

    # Create labels
    y = torch.tensor(
        [1 if rid in binding_residues else 0 for rid in residue_ids],
        dtype=torch.float
    ).view(-1, 1) #

    protein_graph.y = y # Add labels to graph
    return protein_graph # Return labeled graph for training


# --- End of sitescanner/data_utils.py ---
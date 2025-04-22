import argparse
import os
import sys

# Add the parent directory to sys.path to allow sibling imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitescanner import core, utils, __version__ # Assuming __version__ is defined in __init__.py
from sitescanner.utils import refine_predictions_with_dbscan

def main():
    parser = argparse.ArgumentParser(
        description=f"SiteScanner v{__version__}: Predict protein binding sites from PDB files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output Arguments
    parser.add_argument(
        "pdb_file",
        help="Path to the input PDB file."
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory to save the prediction results."
    )

    # Model and Inference Arguments
    parser.add_argument(
        "-m", "--model_path",
        default=None,
        help="Path to the trained model file (.pt or .pth). Uses default if not specified."
    )
    parser.add_argument(
        "-d", "--device",
        default='cpu',
        choices=['cpu', 'cuda'],
        help="Device to run inference on ('cpu' or 'cuda')."
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.9,
        help="Probability threshold for classifying a residue as a binding site (0.0 to 1.0)."
    )

    # Post-processing Arguments
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Enable DBSCAN clustering post-processing to refine predictions."
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=10.0,
        help="DBSCAN epsilon parameter (distance in Angstroms) for post-processing."
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=4,
        help="DBSCAN min_samples parameter for post-processing."
    )

    # Other Arguments
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    args = parser.parse_args()

    # --- Argument Validation ---
    if not 0.0 <= args.threshold <= 1.0:
        parser.error(f"Threshold must be between 0.0 and 1.0, got: {args.threshold}")
    if args.postprocess and args.dbscan_eps <= 0:
         parser.error(f"DBSCAN epsilon must be positive, got: {args.dbscan_eps}")
    if args.postprocess and args.dbscan_min_samples <= 0:
         parser.error(f"DBSCAN min_samples must be positive, got: {args.dbscan_min_samples}")

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- SiteScanner Run ---")
    print(f"PDB file: {args.pdb_file}")
    print(f"Output directory: {args.output_dir}")
    if args.model_path:
        print(f"Using model: {args.model_path}")
    else:
        print("Using default model.")
    print(f"Device: {args.device}")
    print(f"Probability threshold: {args.threshold}")
    print(f"Post-processing enabled: {args.postprocess}")
    if args.postprocess:
        print(f"  DBSCAN eps: {args.dbscan_eps}")
        print(f"  DBSCAN min_samples: {args.dbscan_min_samples}")
    print("-----------------------")


    try:
        # --- Run Prediction ---
        prediction_results = core.predict_binding_sites(
            pdb_file_path=args.pdb_file,
            model_path=args.model_path,
            device=args.device,
            probability_threshold=args.threshold
        )

        if prediction_results is None:
            print("Prediction failed. Could not process PDB into graph.")
            sys.exit(1)

        binding_site_residues, all_residue_ids = prediction_results
        print(f"Initial prediction successful. Found {len(binding_site_residues)} potential binding site residues.")

        # --- Apply Post-processing (Optional) ---
        if args.postprocess:
            print("Applying DBSCAN post-processing...")
            refined_residues = refine_predictions_with_dbscan(
                predicted_residue_ids=binding_site_residues,
                pdb_file_path=args.pdb_file,
                eps=args.dbscan_eps,
                min_samples=args.dbscan_min_samples
            )
            print(f"Refined {len(binding_site_residues)} predictions down to {len(refined_residues)} residues in the largest spatial cluster.")
            binding_site_residues = refined_residues # Update the list to be saved
        else:
            print("Skipping post-processing.")


        # --- Save Results ---
        base_name = os.path.splitext(os.path.basename(args.pdb_file))[0]
        txt_output_path = os.path.join(args.output_dir, f"{base_name}_predictions.txt")
        pdb_output_path = os.path.join(args.output_dir, f"{base_name}_predicted_sites.pdb")

        # Use the (potentially refined) binding_site_residues list
        utils.save_predictions_to_txt(
            binding_site_residues=binding_site_residues,
            output_filepath=txt_output_path
        )

        utils.color_pdb_by_prediction(
            input_pdb_path=args.pdb_file,
            output_pdb_path=pdb_output_path,
            binding_site_residues=binding_site_residues
        )

        print("\nProcessing complete.")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        print("\n--- Full Traceback ---", file=sys.stderr)
        traceback.print_exc() # Print the full traceback to stderr
        print("--- End Traceback ---", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

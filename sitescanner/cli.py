import argparse
import os
import sys

# Add the parent directory to sys.path to allow sibling imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitescanner import core, utils, __version__ # Assuming __version__ is defined in __init__.py

def main():
    parser = argparse.ArgumentParser(
        description=f"SiteScanner v{__version__}: Predict protein binding sites from PDB files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "pdb_file",
        help="Path to the input PDB file."
    )
    parser.add_argument(
        "-m", "--model_path",
        required=True,
        help="Path to the trained model file (.pt or .pth)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory to save the prediction results."
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
        default=0.5,
        help="Probability threshold for classifying a residue as a binding site (0.0 to 1.0)."
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    args = parser.parse_args()

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        parser.error(f"Threshold must be between 0.0 and 1.0, got: {args.threshold}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing PDB file: {args.pdb_file}")
    print(f"Using model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Probability threshold: {args.threshold}")

    try:
        # Run prediction
        prediction_results = core.predict_binding_sites(
            pdb_file_path=args.pdb_file,
            model_path=args.model_path,
            device=args.device,
            probability_threshold=args.threshold
        )

        if prediction_results is None:
            print("Prediction failed. Could not process PDB into graph.")
            sys.exit(1)

        # Unpack results (modified in core.py)
        binding_site_residues, raw_predictions, all_residue_ids = prediction_results

        print(f"Prediction successful. Found {len(binding_site_residues)} potential binding site residues.")

        # Define output file names
        base_name = os.path.splitext(os.path.basename(args.pdb_file))[0]
        txt_output_path = os.path.join(args.output_dir, f"{base_name}_predictions.txt")
        pdb_output_path = os.path.join(args.output_dir, f"{base_name}_predicted_sites.pdb")

        # Save results using utils functions
        # Note: save_predictions_to_txt internally filters based on the raw_predictions array
        utils.save_predictions_to_txt(
            predictions=raw_predictions,
            residue_ids=all_residue_ids,
            output_filepath=txt_output_path
        )

        utils.color_pdb_by_prediction(
            input_pdb_path=args.pdb_file,
            output_pdb_path=pdb_output_path,
            predictions=raw_predictions,
            residue_ids=all_residue_ids
            # Can add arguments for predicted_bfactor, default_bfactor if needed
        )

        print("\nProcessing complete.")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        # Consider adding more specific error handling based on potential issues
        # from model loading, inference, or file saving.
        sys.exit(1)

if __name__ == "__main__":
    main()

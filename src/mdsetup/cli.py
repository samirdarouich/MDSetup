import argparse
import sys

from mdsetup.analysis.umbrella_sampling import find_frames


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Select frames from a CSV file.")

    # Define the expected arguments
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file containing 'frame' and 'cv' columns.",
    )
    parser.add_argument(
        "start", type=float, help="The starting value of the desired CV range."
    )
    parser.add_argument(
        "end", type=float, help="The ending value of the desired CV range."
    )
    parser.add_argument("num_frames", type=int, help="The number of frames to select.")
    parser.add_argument(
        "--show_histogram",
        action="store_true",
        help="If set, show histogram of CV values.",
    )

    return parser.parse_args()


def select_frames():
    # If input is piped, read it and split into arguments
    if not sys.stdin.isatty():
        piped_input = sys.stdin.read().strip().split()
        if len(piped_input) < 4:
            print(
                "Error: Insufficient arguments provided through pipe.", file=sys.stderr
            )
            sys.exit(1)
        # Inject piped input into `sys.argv` for argparse to handle
        sys.argv.extend(piped_input)

    # Parse arguments from the command line
    args = parse_args()

    # Call the select_frames function with the parsed arguments
    selected_files = find_frames(
        args.csv_file,
        args.start,
        args.end,
        args.num_frames,
        args.show_histogram,
    )

    print(" ".join(f"{item}" for item in selected_files))

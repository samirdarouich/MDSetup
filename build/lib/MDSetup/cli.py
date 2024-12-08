import argparse

from MDSetup.analysis.umbrella_sampling import find_frames


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
    parser.add_argument(
        "--system_name",
        type=str,
        default="conf%d.gro",
        help="The name of the system files, with '%d' as a placeholder for frame number.",
    )

    return parser.parse_args()


def select_frames() -> str:
    # Parse arguments from the command line
    args = parse_args()

    # Call the select_frames function with the parsed arguments
    selected_files = find_frames(
        args.csv_file,
        args.start,
        args.end,
        args.num_frames,
        args.show_histogram,
        args.system_name,
    )

    return " ".join(f"'{item}'" for item in selected_files)

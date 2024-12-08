import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_context("poster")


def find_frames(
    csv_file: str,
    start: float,
    end: float,
    num_frames: int,
    show_histogram: bool = False,
    system_name: str = "conf%d.gro",
) -> List[str]:
    """
    Finds frames from a CSV file that are closest to the desired collective variable
    (CV) values.

    Args:
        csv_file (str): Path to the CSV file containing 'frame' and 'cv' columns.
        start (float): The starting value of the desired CV range.
        end (float): The ending value of the desired CV range.
        num_frames (int): The number of frames to select.
        show_histogram (bool, optional): If True, a histogram of the CV values will be
            saved.
        system_name (str, optional): The name of the system files (.gro). Should contain
            a '%d' placeholder for the frame number.

    Returns:
        List[str]: List containing the path to the selected frames.
    """
    # Assert that system_name contains a '%d' placeholder
    assert (
        "%d" in system_name
    ), "The system_name must contain a '%d' placeholder for the frame number!"

    # Read in frame and cv values from the csv file
    df = pd.read_csv(csv_file)

    # Generate the desired cv values
    desired_cvs = np.linspace(start, end, num_frames)

    if show_histogram:
        plot_step = (end - start) / (num_frames - 1) * 2
        sns.histplot(df["cv"], kde=False, bins=num_frames)
        plt.xticks(np.arange(start, end + plot_step, plot_step).round(2))
        plt.xlabel("CV Value")
        plt.ylabel("Frequency")
        plt.savefig(
            os.path.dirname(csv_file) + "/cv_histogram.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    # Initialize an empty list to store the selected frames
    selected_frames = []

    # Set to track frames we've already selected
    selected_frame_indices = set()

    # Loop over each desired cv and find the closest unique frame
    for desired_cv in desired_cvs:
        # Find the closest frame
        closest_frame_idx = np.abs(df["cv"] - desired_cv).argmin()
        closest_frame = df.iloc[closest_frame_idx]

        # If the frame hasn't been selected yet, add it to the list
        if closest_frame["frame"] not in selected_frame_indices:
            selected_frames.append(closest_frame)
            selected_frame_indices.add(closest_frame["frame"])

    # Convert the list of selected frames into a DataFrame
    selected_frames_df = pd.DataFrame(selected_frames)

    # Ensure the frames are sorted by their cv values in ascending order
    selected_frames_df = selected_frames_df.sort_values(by="cv")

    # Assert that the selected frames' cv values are in ascending order
    assert (
        selected_frames_df["cv"].is_monotonic_increasing
    ), "The selected frames' cv values are not in ascending order!"

    return [system_name % frame for frame in selected_frames_df["frame"]]

import os
import subprocess
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Template

from mdsetup.tools.submission import submit_and_wait

sns.set_context("poster")


def create_index_file(
    template_index: str,
    structure: str,
    index_file: str,
    selections: List[str],
    names: List[str],
):
    """
    Creates an index file for molecular dynamics simulations based on a template.

    Args:
        template_index (str): Path to the template index file.
        structure (str): Path to the structure file.
        index_file (str): Path where the generated index file will be saved.
        selections (List[str]): List of selection strings for the index file.
        names (List[str]): List of names corresponding to the selections.

    Returns:
        str: Path to the generated index file.

    Raises:
        FileNotFoundError: If the template index file does not exist.
        subprocess.CalledProcessError: If the bash script execution fails.
    """
    index_file_settings = {
        "index_file": index_file,
        "structure": structure,
        "selections": selections,
        "names": names,
    }

    with open(template_index) as f:
        template = Template(f.read())

    # Write template
    os.makedirs(os.path.dirname(index_file), exist_ok=True)

    # Write new file
    rendered = template.render(**index_file_settings)

    index_bash = index_file.replace(".ndx", ".sh")
    with open(index_bash, "w") as f:
        f.write(rendered)

    exe = subprocess.run(["bash", index_bash], capture_output=True, text=True)

    if exe.returncode != 0:
        raise subprocess.CalledProcessError(
            exe.returncode, exe.args, exe.stdout, exe.stderr
        )

    return index_file


def find_frames(
    template_frames: str,
    trajectory_folder: str,
    index_file: str,
    selection: str,
    start: float,
    end: float,
    num_frames: int,
    csv_file: Optional[str] = None,
    show_histogram: bool = False,
) -> List[str]:
    """
    Finds frames from a CSV file that are closest to the desired collective variable
    (CV) values.

    Args:
        template_frames (str): Path to the template file for generating the bash script.
        trajectory_folder (str): Path to the folder containing the trajectory files.
        index_file (str): Path to the index file used in the analysis.
        selection (str): Selection string for the atoms of interest.
        start (float): The starting value of the desired CV range.
        end (float): The ending value of the desired CV range.
        num_frames (int): The number of frames to select.
        csv_file: Path to the CSV file containing 'frame' and 'cv' columns.
        show_histogram: If True, a histogram of the CV values will be
            saved.

    Returns:
        List[str]: List containing the path to the selected frames.
    """
    # create extra folder in the trajectory folder
    ensemble = "../" + "_".join(trajectory_folder.split("/")[-1].split("_")[1:])
    folder = trajectory_folder + "/trajectory"

    frames_file_settings = {
        "folder": folder,
        "ensemble": ensemble,
        "index_file": index_file,
        "selection": selection,
    }

    with open(template_frames) as f:
        template = Template(f.read())

    # Write template
    os.makedirs(folder, exist_ok=True)

    # Write new file
    rendered = template.render(**frames_file_settings)
    frames_bash = folder + "/get_cv.sh"
    with open(frames_bash, "w") as f:
        f.write(rendered)

    exe = subprocess.run(["bash", frames_bash], capture_output=True, text=True)

    if exe.returncode != 0:
        raise subprocess.CalledProcessError(
            exe.returncode, exe.args, exe.stdout, exe.stderr
        )

    # Read in frame and cv values from the csv file
    if csv_file is None:
        csv_file = folder + "/collective_variable.csv"
    df = pd.read_csv(csv_file)

    # Generate the desired cv values
    desired_cvs = np.linspace(start, end, num_frames)

    if show_histogram:
        plot_step = (end - start) / 5
        sns.histplot(df["cv"], kde=False, bins=num_frames)
        plt.xticks(np.arange(start, end + plot_step, plot_step).round(2))
        plt.xlabel("CV Value")
        plt.ylabel("Frequency")
        plt.savefig(
            os.path.dirname(csv_file) + "/cv_histogram.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.show()
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
    assert len(selected_frames_df) == num_frames, (
        "Number of selected frames is not equal to the desired number of frames. "
        "Check input!"
    )

    print("Selected frames with CV values:")
    for frame, cv in zip(selected_frames_df["frame"], selected_frames_df["cv"]):
        print(f"Frame: {frame:.0f}, CV: {cv}")
    print("\n")

    return [
        os.path.dirname(csv_file) + "/conf%d.gro" % frame
        for frame in selected_frames_df["frame"]
    ]


def gmx_wham_analysis(
    wham_template: str,
    temperature: float,
    folder: str,
    tpr_files: str,
    pullx_files: str,
    output_prefix: str,
    unit: str = "kCal",
    on_cluster: bool = False,
    submission_command: str = "qsub",
    **wham_kwargs,
) -> pd.DataFrame:
    """
    Perform WHAM (Weighted Histogram Analysis Method) analysis using GROMACS.

    Args:
        wham_template (str): Path to the WHAM template file.
        temperature (float): Temperature at which the analysis is performed.
        folder (str): Directory where the analysis will be performed.
        tpr_files (str): Path to the TPR files.
        pullx_files (str): Path to the pullx files.
        output_prefix (str): Prefix for the output files.
        unit (str, optional): Unit of the PMF (Potential of Mean Force).
        on_cluster (bool, optional): Whether to run the analysis on a cluster.
        submission_command (str, optional): Command to submit jobs to the cluster.
        **wham_kwargs: Additional keyword arguments to be passed to the wham analysis.

    Returns:
        Dataframe with potential of mean force (PMF) values.
    """
    with open(wham_template) as f:
        template = Template(f.read())

    wham_settings = {
        "temperature": temperature,
        "folder": folder,
        "tpr_files": os.path.relpath(tpr_files, folder),
        "pullx_files": os.path.relpath(pullx_files, folder),
        "output_prefix": output_prefix,
        "unit": unit,
        "kwargs": wham_kwargs,
    }

    # Write template
    os.makedirs(folder, exist_ok=True)

    # Write new file
    rendered = template.render(**wham_settings)
    wham_bash = folder + f"/wham_analysis_{temperature}.sh"
    with open(wham_bash, "w") as f:
        f.write(rendered)

    # Execute analysis
    if on_cluster:
        print("\nSubmit WHAM to cluster and wait untils it is finished.\n")
        submit_and_wait(job_files=[wham_bash], submission_command=submission_command)
    else:
        print("\nPerform WHAM locally! Wait until it is finished.\n")
        # Call the bash to build the box. Write output to file.
        with open(f"{folder}/wham_analysis_{temperature}_output.txt", "w") as f:
            exe = subprocess.run(
                ["bash", wham_bash],
                stdout=f,
                stderr=f,
            )
            if exe.returncode != 0:
                raise subprocess.CalledProcessError(
                    exe.returncode, exe.args, exe.stdout, exe.stderr
                )

    # Read and plot PMF and Histogram
    xvg_file = f"{folder}/{output_prefix}_pmf.xvg"
    df_pmf = pd.read_csv(
        xvg_file, sep="\s+", names=["cv", "PMF"], skiprows=13, comment="@"
    )
    plt.plot(df_pmf["cv"], df_pmf["PMF"])
    plt.xlabel("CV")
    plt.ylabel(f"PMF ({unit}/mol)")
    plt.savefig(
        f"{folder}/pmf_{temperature}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    plt.close()

    xvg_file = f"{folder}/{output_prefix}_hist.xvg"
    df_histo = pd.read_csv(xvg_file, skiprows=13, comment="@", sep="\s+", header=None)
    columns = ["cv"] + [f"umbrella_{i}" for i in range(0, len(df_histo.columns) - 1)]
    df_histo.columns = columns
    for umbrella in df_histo.columns[1:]:
        plt.plot(df_histo["cv"], df_histo[umbrella])
    plt.xlabel("CV")
    plt.ylabel("count")
    plt.savefig(
        f"{folder}/histo_{temperature}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    plt.close()

    return df_pmf

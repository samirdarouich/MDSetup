import multiprocessing
import os
import re
import subprocess
from typing import List

import numpy as np
import pandas as pd
from jinja2 import Template

from ..tools.general import add_nan_if_no_brackets, flatten_list, generate_series
from ..tools.submission import submit_and_wait


def read_gromacs_xvg(file_path: str, time_fraction: float = 0.0):
    """
    Reads data from a Gromacs XVG file and returns a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the XVG file.
    - time_fraction (float, optional): The time_fraction of data to select. Defaults to 0.0.

    Returns:
    - pandas.DataFrame: A DataFrame containing the selected data.

    Description:
    This function reads data from a Gromacs XVG file specified by 'file_path'. It extracts the data columns and their corresponding properties from the file. The data is then filtered based on the 'time_fraction' parameter, selecting only the data points that are within the specified time_fraction of the maximum time value. The selected data is returned as a pandas DataFrame, where each column represents a property and each row represents a data point.

    Example:
    read_gromacs_xvg('data.xvg', time_fraction=0.5)
    """
    data = []
    properties = []
    units = []
    special_properties = []
    special_means = []
    special_stds = []
    special_units = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("@") and "title" in line:
                title = line.split('"')[1]
                continue
            if line.startswith("@") and "xaxis  label" in line:
                properties.append(line.split('"')[1].split()[0])
                units.append(line.split('"')[1].split()[1])
                continue
            if line.startswith("@") and "yaxis  label" in line:
                for u in line.split('"')[1].split(","):
                    # Check if this is a special file with several properties
                    if len(u.split("(")) > 1 and u.split("(")[0].strip():
                        properties.append(u.split()[0])
                        units.append(
                            u.split()[1].replace(r"\\N", "").replace(r"\\S", "^")
                        )
                    else:
                        units.append(
                            "("
                            + u.split("(")[1].replace(")", "").replace(" ", ".")
                            + ")"
                        )
                continue
            if line.startswith("@") and ("s" in line and "legend" in line):
                if "=" in line:
                    special_properties.append(
                        line.split('"')[1].split("=")[0].replace(" ", "")
                    )
                    mean, std, unit = [
                        a.replace(")", "").replace(" ", "").replace("+/-", "")
                        for a in line.split('"')[1].split("=")[1].split("(")
                    ]
                    special_means.append(mean)
                    special_stds.append(std)
                    special_units.append(f"({unit})")
                else:
                    properties.append(line.split('"')[1])
                continue
            elif line.startswith("@") or line.startswith("#"):
                continue  # Skip comments and metadata lines
            parts = line.split()
            data.append([float(part) for part in parts])

    # Create column wise array with data
    data = np.array([np.array(column) for column in zip(*data)])

    # In case special properties are delivered, there is just one regular property, which is given for every special property.
    if special_properties:
        properties[-1:] = [
            properties[-1] + "[" + re.search(r"\[(.*?)\]", sp).group(1) + "]"
            for sp in special_properties
        ]
        units[-1:] = [units[-1] for _ in special_properties]

    # Only select data that is within (time_fraction,1)*t_max
    idx = data[0] >= time_fraction * data[0][-1]

    # Save data
    property_dict = {}

    for p, d, u in zip(properties, data[:, idx], units):
        property_dict[f"{p} {u}"] = d

    # As special properties only have a mean and a std. Generate a series that exactly has the mean and the standard deviation of this value.
    for sp, sm, ss, su in zip(
        special_properties, special_means, special_stds, special_units
    ):
        property_dict[f"{sp} {su}"] = generate_series(
            desired_mean=float(sm), desired_std=float(ss), size=len(data[0, idx])
        )

    return pd.DataFrame(property_dict)


def read_lammps_output(
    file_path: str,
    time_fraction: float = 0.0,
    header: int = 2,
    header_delimiter: str = ",",
):
    """
    Reads a LAMMPS output file and returns a pandas DataFrame containing the data.

    Parameters:
        file_path (str): The path to the LAMMPS output file.
        time_fraction (float, optional): The time_fraction of data to keep based on the maximum value of the first column. Defaults to 0.0.
        header (int, optional): The number of header lines from which to extract the keys for the reported values. Defaults to 2.
        header_delimiter (str, optional): The delimiter used in the header line. Defaults to ",".

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the LAMMPS output file.

    Raises:
        KeyError: If the LAMMPS output file does not have enough titles.

    Note:
        - The function assumes that the LAMMPS output file has a timestamp in the first line.
        - If the timestamp is not present, the provided time_fraction parameter will be ignored.
        - The function expects the LAMMPS output file to have a specific format, with titles starting with '#'.

    """
    with open(file_path) as file:
        titles = [file.readline() for _ in range(3)]

    titles = [t.replace("#", "").strip() for t in titles if t.startswith("#")]

    if len(titles) < header:
        raise KeyError(
            f"LAMMPS output file has only '{len(titles)}' titles. Cannot use title n°'{header}' !"
        )
    else:
        lammps_header = [h.strip() for h in titles[header - 1].split(header_delimiter)]

    # Check if for every key a unit is provided, if not add NaN.
    lammps_header = add_nan_if_no_brackets(lammps_header)

    df = pd.read_csv(file_path, comment="#", delimiter=r"\s+", names=lammps_header)

    if any(key in df.columns[0].lower() for key in ["time", "step"]):
        idx = df.iloc[:, 0] >= df.iloc[:, 0].max() * time_fraction
        return df.loc[idx, :]
    else:
        print(
            f"\nNo timestamp provided in first line of LAMMPS output. Hence, can't discard the provided time_fraction '{time_fraction}'"
        )
        return df


def extract_from_gromacs(
    files: List[str],
    extracted_properties: List[str],
    time_fraction: float = 0.0,
    command: str = "energy",
    args: List[str] = [""],
    ensemble_name: str = "",
    output_name: str = "properties",
    on_cluster: bool = False,
    extract: bool = False,
    submission_command: str = "",
    extract_template: str = "",
    **kwargs,
):
    """
    Extract properties from GROMACS files.

    Parameters:
    - files (List[str]): A list of file paths to the GROMACS edr or xvg files.
    - extracted_properties (List[str]): A list of strings representing the properties to extract from the output files.
    - time_fraction (float, optional): The time_fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
    - command (str, optional): GROMACS command to use for extraction. Defaults to "energy".
    - args (List[str], optional): Additional arguments for the GROMACS command. Defaults to [""].
    - ensemble_name (str, optional): Name of the ensemble file. Defaults to "".
    - output_name (str, optional): Name of the output file. Defaults to "properties".
    - on_cluster (bool, optional): Flag indicating if extraction should be done on a cluster. Defaults to False.
    - extract (bool, optional): Flag indicating if extraction should be performed. Defaults to False.
    - submission_command (str, optional): Command for submitting extraction to a cluster. Defaults to "".
    - extract_template (str, optional): Path to template for extraction. Defaultst to "".
    - **kwargs (Any): Arbitrary keyword arguments,

    Returns:
    - List[pd.DataFrame]: List of extracted dataframes.

    """
    if extract:
        if not os.path.exists(extract_template):
            raise FileExistsError(
                f"Provided extract template does not exists: {extract_template}"
            )
        else:
            with open(extract_template) as f:
                template = Template(f.read())

        bash_files = []

        # Iterate through the files and write extraction bash file
        for path in files:
            rendered = template.render(
                folder=os.path.dirname(path),
                extracted_properties=extracted_properties,
                gmx_command=f"{command} -f {ensemble_name} -s {ensemble_name} {' '.join(args)} -o {output_name}",
            )

            bash_file = f"{os.path.dirname( path )}/extract_properties.sh"

            with open(bash_file, "w") as f:
                f.write(rendered)

            bash_files.append(bash_file)

    if on_cluster and extract:
        print("Submit extraction to cluster:\n")
        print("\n".join(bash_files), "\n")
        print("Wait until extraction is done...")
        submit_and_wait(bash_files, submission_command=submission_command)
    elif not on_cluster and extract:
        print("Extract locally\n")
        print("\n".join(bash_files), "\n")
        print("Wait until extraction is done...")
        num_processes = multiprocessing.cpu_count()
        # In case there are multiple CPU's, leave one without task
        if num_processes > 1:
            num_processes -= 1

        # Create a pool of processes
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        # Execute the tasks in parallel
        pool.map(
            lambda file: subprocess.run(["bash", file], capture_output=True), bash_files
        )

        # Close the pool to free up resources
        pool.close()
        pool.join()

    if extract:
        print("Extraction finished!\n\n")

    files = [os.path.dirname(file) + f"{output_name}.xvg" for file in files]

    extracted_df_list = [
        read_gromacs_xvg(file_path=file, time_fraction=time_fraction) for file in files
    ]

    return extracted_df_list


def extract_from_lammps(
    files: List[str],
    extracted_properties: List[str],
    time_fraction: float = 0.0,
    header: int = 2,
    header_delimiter: str = ",",
    **kwargs,
):
    """
    Extracts specified properties from LAMMPS output files in parallel and returns a list of dataframes containing the extracted data.

    This function reads multiple LAMMPS output files, extracts specified properties, and returns them as a list of pandas DataFrames. The function utilizes multiprocessing to handle multiple files in parallel, improving performance on systems with multiple CPUs.

    Parameters:
    - files (List[str]): A list of file paths to the LAMMPS output files.
    - extracted_properties (List[str]): A list of strings representing the properties to extract from the output files.
    - time_fraction (float, optional): The time_fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
    - header (int, optional): The number of header lines from which to extract the keys for the reported values. Defaults to 2.
    - header_delimiter (str, optional): The delimiter used in the header of the files. Defaults to ",".
    - **kwargs (Any): Arbitrary keyword arguments.

    Returns:
    - List[pd.DataFrame]: A list of pandas DataFrames, each containing the extracted properties from the corresponding LAMMPS output file.

    Raises:
    - KeyError: If no data is extracted or if the specified keys cannot be extracted from the files.

    """

    for file in files:
        if not os.path.exists(file):
            raise FileExistsError(f"Output file '{file}' does not exist!")

    # In case there are multiple CPU's, leave one without task
    num_processes = multiprocessing.cpu_count()
    if num_processes > 1:
        num_processes -= 1

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=num_processes)

    inputs = [(file, time_fraction, header, header_delimiter) for file in files]

    # Execute the tasks in parallel
    data_list = pool.starmap(read_lammps_output, inputs)

    # Close the pool to free up resources
    pool.close()
    pool.join()

    if len(data_list) == 0:
        raise KeyError("No data was extracted!")

    # Get the columns of one of the extracted data frames
    df_keys = {
        key.split("(")[0].strip(): i for i, key in enumerate(data_list[0].columns)
    }

    # Get the index of the keys to extract
    key_idx = flatten_list(df_keys.get(key, []) for key in extracted_properties)

    if len(key_idx) == 0:
        raise KeyError(
            f"Specified keys '{', '.join(extracted_properties) }' could not be extracted! Valid keys are: '{', '.join(df_keys.keys())}'"
        )

    # Drop all nonrelevant columns from the data frames
    extracted_df_list = [df.iloc[:, key_idx] for df in data_list]

    return extracted_df_list

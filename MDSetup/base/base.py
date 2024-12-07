import os

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

from ..tools.general import DISTANCE, FOLDER_PRECISION, UNITS, load_yaml, update_paths


class BaseSetup:
    def __init__(
        self,
        system_setup: str,
        simulation_default: str,
        simulation_ensemble: str,
        submission_command: str,
        simulation_sampling: str = "",
    ) -> None:
        """Initialize a new instance of the MDsetup class.

        Args:
          system_setup (str):
            Path to the system setup YAML file. Containing all system settings.
          simulation_default (str):
            Path to the simulation default YAML file. Containing all default MD
            settings.
          simulation_ensemble (str):
            Path to the simulation ensemble YAML file. Containing all MD ensemble
            settings.
          submission_command (str):
            Command to submit jobs to cluster.
          simulation_sampling (str,optional):
            Path to the sampling YAML file. Containing all sampling settings. This is
            only needed for LAMMPS.

        Returns:
            None
        """
        # Open the yaml files and extract the necessary information
        self.system_setup = load_yaml(system_setup)
        self.simulation_default = load_yaml(simulation_default)
        self.simulation_ensemble = load_yaml(simulation_ensemble)
        self.simulation_sampling = (
            load_yaml(simulation_sampling) if simulation_sampling else {}
        )

        # Check for input length
        lengths = [
            len(self.system_setup[key])
            for key in ["temperature", "pressure", "density"]
        ]
        assert all(length == lengths[0] for length in lengths), (
            "Make sure that the same number of state points are provided for "
            "temperature, pressure and density"
        )

        # Define state folder name
        self.state_folder = "_".join(
            [
                f"{folder_attribute[:4]}_%.{FOLDER_PRECISION}f"
                for folder_attribute in self.system_setup["folder_attributes"]
            ]
        )

        # Convert all paths provided in system setup to absolute paths
        main_path = os.path.dirname(os.path.abspath(system_setup))
        self.system_setup["folder"] = update_paths(
            self.system_setup["folder"], main_path
        )
        update_paths(self.system_setup["paths"], main_path)

        # Check for all necessary keys

        # Print software
        print(f"MD input will be generated for '{self.system_setup['software']}'!")

        # Save molecules in the system (sort out molecules that are not present in system)
        self.system_molecules = [
            mol for mol in self.system_setup["molecules"] if mol["number"] > 0
        ]

        # Get the name (residue) list
        self.residues = [mol["name"] for mol in self.system_molecules]

        # Get molecular mass and number for each molecule
        self.molar_masses = [
            MolWt(Chem.MolFromSmiles(mol["smiles"])) for mol in self.system_molecules
        ]
        self.molecule_numbers = [mol["number"] for mol in self.system_molecules]

        # Get conversion from AA to nm/AA
        self.distance_conversion = DISTANCE[self.system_setup["software"]]

        # Submission command for the cluster
        self.submission_command = submission_command

        # Create an analysis dictionary containing all files
        self.analysis_dictionary = {}

        # Define project folder
        self.project_folder = (
            f"{self.system_setup['folder']}/{self.system_setup['name']}"
        )

    def define_state_cond(self, **state_attributes):
        return self.state_folder % tuple(
            state_attributes[folder_attribute]
            for folder_attribute in self.system_setup["folder_attributes"]
        )

    def define_state_cond_list(self):
        return [ self.define_state_cond(**state) for state in self.loop_through_states()]

    def define_state_text(self, **state_attributes):
        return ", ".join(
            (
                f"{folder_attribute.replace('_',' ')}: "
                f"{state_attributes[folder_attribute]:.{FOLDER_PRECISION}f} "
                f"{UNITS[folder_attribute]}"
            )
            for folder_attribute in self.system_setup["folder_attributes"]
        )

    def loop_through_states(self):
        for i, (temperature, pressure, density) in enumerate(
            zip(
                self.system_setup["temperature"],
                self.system_setup["pressure"],
                self.system_setup["density"],
            )
        ):
            # Compute mole fraction of component 1
            mole_fraction = self.molecule_numbers[0] / (sum(self.molecule_numbers))

            yield {
                "i": i,
                "temperature": temperature,
                "pressure": pressure,
                "density": density,
                "mole_fraction": mole_fraction,
            }

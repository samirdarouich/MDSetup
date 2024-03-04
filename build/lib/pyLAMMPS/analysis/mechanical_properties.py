import os
import numpy as np
from typing import Dict, List
from jinja2 import Template

from ..tools.general_utils import work_json
from .general_analysis import mean_properties, plot_data

def get_stiffness_tensor( pressure: List[List[List[float]]], deformation_magnitudes: List[float]  ):
    """
    Calculate the stiffness tensor based on the given pressure and deformation magnitudes.

    Parameters:
        pressure (List[List[List[float]]]): A 3D list representing the pressure values for each deformation direction.
            The innermost list contains the pressure values for one deformation magnitude in the following order:
            [pxx, pyy, pzz, pyz, pxz, pxy].
            The middle list contains the pressure values for each deformation magnitude.
            The outermost list contains the pressure values for each deformation magnitude in each deformation direction (x,y,z,yz,xz,xy).
        deformation_magnitudes (List[float]): A list of deformation magnitudes.

    Returns:
        numpy.ndarray: The stiffness tensor, a 6x6 numpy array.
            The entries of the stiffness tensor are calculated as follows:
            - C11: elongation in x direction and slope of pressure in x direction
            - C21: elongation in x direction and slope of pressure in y direction
            - C31: elongation in x direction and slope of pressure in z direction
            - C41: elongation in x direction and slope of pressure in yz direction
            - C51: elongation in x direction and slope of pressure in xz direction
            - C61: elongation in x direction and slope of pressure in xy direction

    Note:
        - The stiffness tensor is calculated by performing a linear fit for each pressure along the deformation in each direction.
        - The stiffness tensor is symmetric, so the resulting tensor is rounded to ensure symmetry.

    """
    ## Entries of stiffness tensor ##

    # C11: elongation in x direction and slope of pressure in x direction
    # C21: elongation in x direction and slope of pressure in y direction
    # C31: elongation in x direction and slope of pressure in z direction 
    # C41: elongation in x direction and slope of pressure in yz direction 
    # C51: elongation in x direction and slope of pressure in xz direction 
    # C61: elongation in x direction and slope of pressure in xy direction 

    stiffness_tensor = np.zeros((6,6))

    for i in range(6):
        pxx,pyy,pzz = [p[0] for p in pressure[i]],[p[1] for p in pressure[i]],[p[2] for p in pressure[i]]
        pxy,pxz,pyz = [p[3] for p in pressure[i]],[p[4] for p in pressure[i]],[p[5] for p in pressure[i]]

        ## (linear) Fit for each pressure along the deformation in xx,yy,zz,... direction ##

        stiffness_tensor[0][i] = -np.polyfit( deformation_magnitudes, pxx, 1 )[0]
        stiffness_tensor[1][i] = -np.polyfit( deformation_magnitudes, pyy, 1 )[0]
        stiffness_tensor[2][i] = -np.polyfit( deformation_magnitudes, pzz, 1 )[0]
        stiffness_tensor[3][i] = -np.polyfit( deformation_magnitudes, pyz, 1 )[0]
        stiffness_tensor[4][i] = -np.polyfit( deformation_magnitudes, pxz, 1 )[0]
        stiffness_tensor[5][i] = -np.polyfit( deformation_magnitudes, pxy, 1 )[0] 

    # Ensure symmetry
    stiffness_tensor = np.round( ( stiffness_tensor + stiffness_tensor.T ) / 2 , 3 )

    return stiffness_tensor

def compute_VRH( stiffness_tensor: np.ndarray ):
    """
    Compute the Voigt-Reuss-Hill (VRH) averages of the elastic constants.

    Parameters:
        stiffness_tensor (np.ndarray): The 6x6 stiffness tensor representing the elastic constants.

    Returns:
        tuple: A tuple containing the VRH averages of the elastic constants in the following order:
            - K (float): Bulk modulus.
            - G (float): Shear modulus.
            - E (float): Young's modulus.
            - nu (float): Poisson's ratio.

    Notes:
        - The stiffness tensor should be in Voigt notation.
        - The stiffness tensor should be symmetric.
        - The stiffness tensor should be positive definite.

    References:
        - Voigt, W. (1928). Lehrbuch der Kristallphysik. Teubner.
        - Reuss, A. (1929). Berechnung der Fließgrenze von Mischkristallen auf Grund der Plastizitätsbedingung für Einkristalle. ZAMM - Journal of Applied Mathematics and Mechanics / Zeitschrift für Angewandte Mathematik und Mechanik, 9(1), 49–58.
        - Hill, R. (1952). The Elastic Behaviour of a Crystalline Aggregate. Proceedings of the Physical Society. Section A, 65(5), 349–354.
    """
    c_sq = ( stiffness_tensor[0][0] + stiffness_tensor[0][1] ) * stiffness_tensor[2][2] - 2 * stiffness_tensor[0][2]**2
    M    = stiffness_tensor[0][0] + stiffness_tensor[0][1] + 2 * stiffness_tensor[2][2] - 4 * stiffness_tensor[0][2]

    K_V  = ( 1 / 9 * ( stiffness_tensor[0][0] + stiffness_tensor[1][1] + stiffness_tensor[2][2] ) + 
            2 / 9 * ( stiffness_tensor[0][1] + stiffness_tensor[0][2] + stiffness_tensor[1][2] ) )

    K_R  = c_sq / M

    G_V  = 1 / 30 * ( M + 12 * stiffness_tensor[3][3] + 12 * stiffness_tensor[5][5] ) 

    G_R  = ( 5 / 2 * c_sq * stiffness_tensor[3][3] * stiffness_tensor[5][5] / 
            ( c_sq * ( stiffness_tensor[3][3] + stiffness_tensor[5][5] ) + 3 * K_V * stiffness_tensor[3][3] * stiffness_tensor[5][5] ) )

    K    = 0.5 * ( K_V + K_R )
    G    = 0.5 * ( G_V + G_R )
    E    = 9 * K * G / ( 3 * K + G )
    nu   = (3 * K - 2 * G ) / ( 6 * K + 2 * G )

    return K, G, E, nu

def plot_main( deformation_magnitudes: List[float], p_main_e_main: List[List[List[float]]], output_folder: str, ax_limits: List[List[float]]=[[],[]]  ):
    """
    Plot the main stress and strain components as a function of deformation magnitude.

    Parameters
    ----------
    deformation_magnitudes : List[float]
        List of deformation magnitudes.
    p_main_e_main : List[List[List[float]]]
        A 3D list representing the pressure values for each deformation direction.
        The innermost list contains the pressure values for one deformation direction for all deformation magnitudes. 
        The middle list contains the pressure values with all deformation magnitudes for all principal deformation directions.
        The outermost list contains the pressure values for each deformation magnitude in each principal deformation direction (x,y,z).
    output_folder : str
        Path to the output folder where the plots will be saved.
    ax_limits : List[List[float]], optional
        List of lists containing the limits for the x-axis and y-axis of the plot.
        The first element of the outer list should be a list containing the limits for the x-axis,
        and the second element should be a list containing the limits for the y-axis.
        If not provided, the limits will be automatically determined.

    Returns
    -------
    None

    """
    plot_deformation_directions = ["xx","yy","zz"]

    for i in range(3):
        pxx, std_pxx = np.mean( p_main_e_main[i][0], axis = 0 ), np.std( p_main_e_main[i][0], axis = 0 )
        pyy, std_pyy = np.mean( p_main_e_main[i][1], axis = 0 ), np.std( p_main_e_main[i][1], axis = 0 )
        pzz, std_pzz = np.mean( p_main_e_main[i][2], axis = 0 ), np.std( p_main_e_main[i][2], axis = 0 )

        data = [ [-np.array(deformation_magnitudes), pxx, None, std_pxx], [-np.array(deformation_magnitudes), pyy, None, std_pyy], [-np.array(deformation_magnitudes), pzz, None, std_pzz] ]
        labels = [  r"$\sigma_\mathrm{xx}$", r"$\sigma_\mathrm{yy}$", r"$\sigma_\mathrm{zz}$", r"$\epsilon_\mathrm{%s}$"%plot_deformation_directions[i], "stress [GPa]" ]
        colors = [ "tab:blue", "tab:green", "tab:orange"]
        linestyle = ["-", ":", "--"]
        marker = [".","^","x"]

        save_path = f"{output_folder}/eps_{plot_deformation_directions[i]}"

        x_ticks = deformation_magnitudes

        plot_data( data,labels,colors,path_out=save_path,ax_lim=ax_limits,ticks=[x_ticks],
                   linestyle=linestyle,markerstyle=marker,lr=True,label_size=30, legend_size=24)

class mechanical_properties:
    """
    Class representing the mechanical properties analysis of a system.

    Attributes:
        destination_folder (str): The destination folder for the analysis results.
        input_template (str): The path to the input template file.
        project_name (str): The name of the project.
        temperature (float): The temperature of the system [K].
        pressure (float): The pressure of the system [bar].
        time (Dict[str, float]): A dictionary containing the time parameters for the simulations.
            The keys are: "equib", "prod", "thermo", "sampling_frequency", "sampling_number".
        equilibration_folder (str): The folder path for the equilibration simulations.
        deformation_folder (str): The folder path for the deformation simulations.

    Methods:
        __init__(self, destination_folder: str, input_template: str, project_name: str, temperature: float, pressure: float, time: Dict[str, float]={}) -> None:
            Initializes the mechanical_properties object.
        setup_equilibration(self, path: Dict[str, str], restart: bool=False, copies: int=1 ):
            Sets up the equilibration simulations.
        setup_deformation(self, deformation_magnitudes: List[float], path: Dict[str, str], restart: bool=True, copies: int=1 ):
            Sets up the deformation simulations.
        analysis_lattice(self, supercell_dim: List[int], copies: int=-1, fraction: float=0.0, save_json: bool=True):
            Performs lattice analysis on the equilibration simulations.
        analysis_mechanical(self, deformation_magnitudes: List[float], copies: int=-1, fraction: float=0.0, save_json: bool=True, verbose: bool=False):
            Performs mechanical analysis on the deformation simulations.

    """
    def __init__(self, destination_folder: str, input_template: str, project_name: str, temperature: float, pressure: float, time: Dict[str, float]={} ) -> None:
        """
        Initializes the mechanical_properties object.

        Parameters:
            destination_folder (str): The destination folder for the simulation and analysis results. An "equilibration" and a "deformation" folder will be made.
            input_template (str): The path to the input template file.
            project_name (str): The name of the project.
            temperature (float): The temperature of the system.
            pressure (float): The pressure of the system.
            time (Dict[str, float], optional): A dictionary containing the time parameters for the simulations. Defaults to {}.

        Returns:
            None
        """
        self.destination_folder   = destination_folder
        self.input_template       = input_template
        self.project_name         = project_name
        self.temperature          = temperature
        self.pressure             = round( pressure / 1.01325, 3 )
        self.equilibration_folder = f"{self.destination_folder}/equilibration/"
        self.deformation_folder   = f"{self.destination_folder}/deformation/"

        if not time:
            time = { "equib": int(2e6), "prod": int(1e6), "thermo": int(1e5), "sampling_frequency": 10, "sampling_number": 1000 }

        self.time           = time
        
        # Check if the template exists
        if not os.path.exists( input_template ):
            return FileNotFoundError(f"Input template do not exist: {input_template}")
        
        # Create destination folder
        os.makedirs( destination_folder, exist_ok = True )

    def setup_equilibration(self, path: Dict[str, str], restart: bool=False, copies: int=1 ):
        """
        Sets up the equilibration simulations.

        Parameters:
            path (Dict[str, str]): A dictionary containing the paths to the required files for the simulations. These are 
                                    "parameter" and "data" or "restart" if restart is wanted.
            restart (bool, optional): Flag indicating whether to use a restart file for the simulations. Defaults to False.
            copies (int, optional): The number of copies to create for the simulations. Defaults to 1.

        Returns:
            None
        """
        self.equilibration_copies = copies

        # Generate absolute paths
        path = { key: os.path.abspath(item) for key,item in path.items() }
        
        with open( self.input_template ) as f:
            template = Template( f.read() )
        
        for copy in range(copies+1):
            
            output_folder = f"{self.destination_folder}/equilibration/0{copy}"

            relative_path = { key: os.path.relpath(item,output_folder) for key,item in path.items() }

            # Render template
            render = template.render( path = relative_path, restart = restart, project_name = self.project_name, temperature = self.temperature,
                                      pressure = self.pressure, time = self.time, deformation = {"direction": None, "rate": 0.0 } )
            
            # Create simulation folder
            os.makedirs( output_folder, exist_ok = True )

            with open( f"{output_folder}/{self.project_name}.in", "w" ) as f:
                f.write( render )

    def setup_deformation(self, deformation_magnitudes: List[float], path: Dict[str, str], restart: bool=True, copies: int=1 ):
        """
        Sets up the deformation simulations.

        Parameters:
            deformation_magnitudes (List[float]): A list of deformation magnitudes to simulate.
            path (Dict[str, str]): A dictionary containing the paths to the required files for the simulations. These are "parameter" and "data" or "restart" if restart is wanted.
            restart (bool, optional): Flag indicating whether to use a restart file for the simulations. Defaults to True.
            copies (int, optional): The number of copies to create for the simulations. Defaults to 1.

        Returns:
            None
        """
        self.deformation_directions = [ "x", "y", "z", "yz", "xz", "xy", "undeformed" ]
        self.deformation_magnitudes = deformation_magnitudes
        self.deformation_copies     = copies

        # Generate absolute paths
        path = { key: os.path.abspath(item) for key,item in path.items() }

        with open( self.input_template ) as f:
            template = Template( f.read() )
        
        for deformation_direction in self.deformation_directions:

            if deformation_direction != "undeformed":
                for deformation_magnitude in deformation_magnitudes:
                    # Don't simulate the undeformed system in each deformation direction, as this would simulate the undeformed system 6*no_copies
                    if deformation_magnitude == 0.0:
                        continue
                    else:
                        for copy in range(copies+1):
                            
                            output_folder = f"{self.destination_folder}/deformation/{deformation_direction}/{deformation_magnitude}/0{copy}"

                            restart_file  = f"{self.equilibration_folder}/0{copy}/{self.project_name}.restart"

                            relative_path = { "parameter": os.path.relpath( path["parameter"], output_folder ), "restart": os.path.relpath( restart_file, output_folder ) }

                            # Render template
                            render = template.render( path = relative_path, restart = restart, project_name = self.project_name, temperature = self.temperature,
                                                    pressure = self.pressure, time = self.time, deformation = {"direction": deformation_direction, "rate": deformation_magnitude }  )
                            
                            # Create simulation folder
                            os.makedirs( output_folder, exist_ok = True )

                            with open( f"{output_folder}/{self.project_name}.in", "w" ) as f:
                                f.write( render )

            else:
                for copy in range(copies+1):
                    
                    output_folder = f"{self.destination_folder}/deformation/{deformation_direction}/0{copy}"

                    restart_file  = f"{self.equilibration_folder}/0{copy}/{self.project_name}.restart"

                    relative_path = { "parameter": os.path.relpath( path["parameter"], output_folder ), "restart": os.path.relpath( restart_file, output_folder ) }

                    # Render template
                    render = template.render( path = relative_path, restart = restart, project_name = self.project_name, temperature = self.temperature,
                                            pressure = self.pressure, time = self.time, deformation = {"direction": deformation_direction, "rate": 0.0 }  )
                    
                    # Create simulation folder
                    os.makedirs( output_folder, exist_ok = True )

                    with open( f"{output_folder}/{self.project_name}.in", "w" ) as f:
                        f.write( render )


    def analysis_lattice(self, supercell_dim: List[int], copies: int=-1, fraction: float=0.0, save_json: bool=True):
        """
        Performs lattice analysis on the equilibration simulations.

        Parameters:
            supercell_dim (List[int]): A list of three integers representing the dimensions of the supercell in the x, y, and z directions, respectively.
            copies (int, optional): The number of equilibration copies to analyze. If set to -1, all equilibration copies will be analyzed. Defaults to -1.
            fraction (float, optional): The fraction of data to use for analysis. Defaults to 0.0, which means all data will be used.
            save_json (bool, optional): Flag indicating whether to save the analysis results as a JSON file. Defaults to True.

        Returns:
            dict: A dictionary containing the mean and standard deviation of the lattice parameters and density. The keys are:
                - "a": [mean_a/fa, std_a/fa]
                - "b": [mean_b/fb, std_b/fb]
                - "c": [mean_c/fc, std_c/fc]
                - "alpha": [mean_alpha, std_alpha]
                - "beta": [mean_beta, std_beta]
                - "gamma": [mean_gamma, std_gamma]
                - "density": [mean_density, std_density]

        Note:
            - The lattice parameters are normalized by the dimensions of the supercell in the x, y, and z directions.
            - The mean and standard deviation are calculated based on the equilibration copies specified or all equilibration copies if copies=-1.
            - The fraction parameter can be used to analyze only a fraction of the data. If set to 0.0, all data will be used.
            - If save_json is True, the analysis results will be saved as a JSON file in the equilibration folder.
        """
        if copies == -1:
            copies = self.equilibration_copies

        a_list = []
        b_list = []
        c_list = []
        alpha_list = []
        beta_list  = []
        gamma_list = []
        dens_list  = []

        fa, fb, fc = supercell_dim

        paths = [ f"{self.equilibration_folder}/0{copy}/{self.project_name}.lattice" for copy in range(copies+1) ] 

        for path in paths:
            _, a,b,c,alpha,beta,gamma,dens = mean_properties( path, keys=["v_a", "v_b", "v_c", "v_alpha" ,"v_beta", "v_gamma" ,"v_density"], fraction = fraction )
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
            alpha_list.append(alpha)
            beta_list.append(beta)
            gamma_list.append(gamma)
            dens_list.append(dens)

        results = { "a": [ np.mean(a_list) / fa, np.std( np.array(a_list) / fa ) ], "b": [ np.mean(b_list) / fb, np.std( np.array(b_list) / fb ) ],
                    "c": [ np.mean(c_list) / fc, np.std( np.array(c_list) / fc ) ], "alpha": [ np.mean(alpha_list), np.std(alpha_list) ],
                    "beta": [ np.mean(beta_list), np.std(beta_list) ], "gamma": [ np.mean(gamma_list), np.std(gamma_list) ], "density": [ np.mean(dens_list), np.std(dens_list) ] }
        
        if save_json:
            work_json( file_path = f"{self.equilibration_folder}/lattice.json", data = results, to_do = "write" )

        return results
    
    def analysis_mechanical(self, deformation_magnitudes: List[float], copies: int=-1, fraction: float=0.0, save_json: bool=True, verbose: bool=False):
        """
        Performs mechanical analysis on the deformation simulations.

        Parameters:
            deformation_magnitudes (List[float]): A list of deformation magnitudes to be analyzed.
            copies (int, optional): The number of deformation copies to analyze. If set to -1, all deformation copies will be analyzed. Defaults to -1.
            fraction (float, optional): The fraction of data to use for analysis. Defaults to 0.0.
            save_json (bool, optional): Flag indicating whether to save the analysis results as a JSON file. Defaults to True.
            verbose (bool, optional): Flag indicating whether to print verbose output. Defaults to False.

        Returns:
            dict: A dictionary containing the analysis results. The keys are:
                - "bulk_modulus": A list containing the mean and standard deviation of the bulk modulus.
                - "shear_modulus": A list containing the mean and standard deviation of the shear modulus.
                - "youngs_modulus": A list containing the mean and standard deviation of the Young's modulus.
                - "poission_ratio": A list containing the mean and standard deviation of the Poisson's ratio.
                - "stiffness_tensor": A list containing the mean and standard deviation of the stiffness tensor.

        """
        if copies == -1:
            copies = self.deformation_copies

        # Remove the undeformed state from all deformation directions
        deformation_directions = [ "x", "y", "z", "yz", "xz", "xy", ]

        # Result lists
        p_main_e_main          = [[[],[],[]],[[],[],[]],[[],[],[]]]
        stiffness_tensor_list  = []
        K_list                 = []
        G_list                 = []
        E_list                 = []
        nu_list                = []
        results                = {}

        for copy in range(copies+1):

            #### Get data ####
            
            # Get the simulation paths
            simulation_paths = [ [ f"{self.deformation_folder}/{deformation_direction}/{deformation_magnitude}/0{copy}/{self.project_name}.pressure" 
                                   if deformation_magnitude != 0.0 else
                                   f"{self.deformation_folder}/undeformed/0{copy}/{self.project_name}.pressure" 
                                   for deformation_magnitude in deformation_magnitudes ] for deformation_direction in deformation_directions 
                               ]

            # Pre define pressure list
            pressure = [ [ [] for _ in range( len( deformation_magnitudes ) ) ] for _ in range(6) ]

            # Sample in each deformation direction (xx,yy,zz,yz,xz,xy) for each deformation (-0.05, ..., 0.00, ..., 0.05) 
            # the resulting stress tensor (pxx,pyy,pzz,pxy,pxz,pyz)
            for i,path in enumerate(simulation_paths):
                # Convert pressure from atm in GPa ( GPa = 10^-9 Pa = 10^-9 * 101325 * atm )
                for k,def_path in enumerate(path):
                    press_array    = mean_properties( def_path, keys = [ "c_thermo_press[%d]"%j for j in range(1,7) ], fraction = fraction )
                    # As the time average is also returned
                    press_array    = press_array[1:]
                    pressure[i][k] = press_array * 101325 / 1e9
            
            # Save the main stressses in the main derformation directions for visualisation purpose
            for i in range(3):
                p_main_e_main[i][0].append( [ p[0] for p in pressure[i] ] )
                p_main_e_main[i][1].append( [ p[1] for p in pressure[i] ] )
                p_main_e_main[i][2].append( [ p[2] for p in pressure[i] ] )
            
            #### Get slope of stress-strain curve at low deformation ####

            stiffness_tensor = get_stiffness_tensor( pressure = pressure, deformation_magnitudes = deformation_magnitudes )

            # Text ouput
            txt  = f"\nCopy n°{copy}:\nResulting Stiffness Tensor\n"
            txt += "\n"
            for i in range(1,7): txt += "  ".join( [ "C%d%d"%(i,j) for j in range(1,7) ] ) + "\n"
            txt += "\n"
            for i in range(0,6): txt += "  ".join( [ str(np.round(st,2)) for st in stiffness_tensor[i,:] ] ) + "\n"
            txt += "\n"

            #### Computation of mechanical properties ####

            #Voigt Reuss Hill
            K, G, E, nu = compute_VRH( stiffness_tensor )

            txt += "\nMechanical properties with Voigt Reuss Hill: \n"
            txt += "Bulk modulus K = %.0f GPa\n"%K
            txt += "Shear modulus G = %.0f GPa\n"%G
            txt += "Youngs modulus E = %.0f GPa\n"%E
            txt += "Poission ratio nu = %.3f\n"%nu

            if verbose: print(txt)

            stiffness_tensor_list.append( stiffness_tensor )
            K_list.append( K )
            G_list.append( G )
            E_list.append( E )
            nu_list.append( nu )

        stiffness_tensor_list = np.stack( stiffness_tensor_list, axis=0 )

        txt  = "\nAveraged Stiffness Tensor\n"
        txt += "\n"
        for i in range(1,7): txt += "  ".join( [ "C%d%d"%(i,j) for j in range(1,7) ] ) + "\n"
        txt += "\n"
        for i in range(0,6): txt += "  ".join( [ "%.2f ± %.2f"%(st,std) for st,std in zip( np.mean( stiffness_tensor_list, axis=0 )[i,:], np.std( stiffness_tensor_list, axis=0 )[i,:] ) ] ) + "\n"
        txt += "\n"
        txt += "\nAveraged mechanical properties with Voigt Reuss Hill: \n"
        txt += "\n"
        txt += "Bulk modulus K = %.0f ± %.0f GPa \n"%( np.mean( K_list ), np.std( K_list ) ) 
        txt += "Shear modulus G = %.0f ± %.0f GPa \n"%( np.mean( G_list ), np.std( G_list ) ) 
        txt += "Youngs modulus E = %.0f ± %.0f GPa \n"%( np.mean( E_list ), np.std( E_list ) ) 
        txt += "Poission ratio nu = %.3f ± %.3f \n"%( np.mean( nu_list ), np.std( nu_list ) ) 

        print(txt)

        # Plot principal deformation plots
        plot_main( deformation_magnitudes = deformation_magnitudes, p_main_e_main = p_main_e_main, output_folder = self.deformation_folder )
        
        # Save results in dictionary
        results["bulk_modulus"]     = [ np.mean( K_list ), np.std( K_list ) ]
        results["shear_modulus"]    = [ np.mean( G_list ), np.std( G_list ) ]
        results["youngs_modulus"]   = [ np.mean( E_list ), np.std( E_list ) ]
        results["poission_ratio"]   = [ np.mean( nu_list ), np.std( nu_list ) ]
        results["stiffness_tensor"] = [ np.mean( stiffness_tensor_list, axis=0 ).tolist(), np.std( stiffness_tensor_list, axis=0 ).tolist() ]

        if save_json:
            work_json( file_path = f"{self.deformation_folder}/mechanical_properties.json", data = results, to_do = "write" )

        return results
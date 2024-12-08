from .stiffness import get_stiffness_tensor
from .visualize import plot_deformation, show_stiffness_tensor
from .voigt_reuss_hill import compute_VRH

# Properties to extract
PROPERTIES = ["pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]

# Define deformation directions (order used to compute stiffness tensor)
DEFORMATION_DIRECTIONS = ["xx", "yy", "zz", "yz", "xz", "xy"]

METHOD = {"VRH": compute_VRH}

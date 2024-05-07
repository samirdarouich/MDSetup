
from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'This repository enables users to perform molecular dynamics simulations utilizing LAMMPS. \
               It can either start from given parameter and data file, or create them itself by using moleculegraph, PLAYMOL and supplementary Python code. \
               YAML files are utilized to parse simulation settings, such as ensemble definitions, sampling properties and system specific input.'

# Setting up
setup(
    name="pyLAMMPS",
    version=VERSION,
    description=DESCRIPTION,
    url='https://github.com/samirdarouich/pyLAMMPS',
    author="Samir Darouich",
    author_email="samir.darouich@itt.uni-stuttgart.de",
    license_files = ('LICENSE'),
    packages=find_packages(),
    install_requires=['numpy',
                      'pandas',
                      'seaborn',
                      'scipy',
                      'toml',
                      'PyYAML',
                      'jinja2',
                      'rdkit',
                      'Pubchempy',
                      'alchemlyb',
                      'moleculegraph @ git+https://github.com/maxfleck/moleculegraph.git'
                      ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Users",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
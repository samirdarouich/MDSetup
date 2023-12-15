
from setuptools import setup

VERSION = '0.0.1'
DESCRIPTION = 'This module enables users to perform molecular dynamics simulations utilizing LAMMPS with \
               any force field. The process begins with a SMILES and graph representation of each component, \
               where PLAYMOL constructs initial systems at specific densities. The moleculegraph software and supplementary Python code \
               are then used to generate a LAMMPS data and input file via jinja2 templates.'

# Setting up
setup(
    name="pyLAMMPS",
    version=VERSION,
    description=DESCRIPTION,
    url='https://github.com/samirdarouich/pyLAMMPS',
    author="Samir Darouich",
    author_email="samir.darouich@itt.uni-stuttgart.de",
    license_files = ('LICENSE'),
    packages=["pyLAMMPS"],
    install_requires=['numpy',
                      'toml',
                      'jinja2',
                      'scipy',
                      'Pubchempy'
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
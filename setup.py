from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="DerivadosUNAL",
    version="0.0.1",
    description="Valoracion de Opciones (SMC)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juanquintero00/DerivadosUNAL",
    author="Juan Camilo Quintero Reina",
    author_email="juquinterore@unal.edu.co",
    license="MIT",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=[
        "Click==7.1.2",
        "matplotlib>=3.3.4",
        "numpy>=1.18.4",
        "pandas>=1.3.3",
        "seaborn>=0.10.1"
    ]
)
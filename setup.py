from setuptools import setup, find_packages

setup(
    name="xcoll-toolkit",
    version="0.1.0.dev1",
    author="Giacomo Broggi, Andrey Abramov",
    author_email="giacomo.broggi@cern.ch",
    description="A toolkit for collimation simulations with the Xsuite-BDSIM(Geant4) coupling via Xcoll.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gbrogginess/xcoll-toolkit",
    packages=find_packages(),
    # TODO: implement install_requires
    # install_requires=[
    #     "numpy",
    #     "pandas",
    #     "xpart",
    #     "xtrack",
    #     "xcoll",
    #     "xgas",
    # ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "run_collimation=xcoll_toolkit.scripts.run_collimation:main",
            "run_beamgas=xcoll_toolkit.scripts.run_beamgas:main",
        ],
    },
)
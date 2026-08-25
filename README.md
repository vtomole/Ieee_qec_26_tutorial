# qrennd-article
This is a repository containing the final datasets that support the figures in "Neural network decoder for near-term surface-code experiments" as well as the scripts that perform any analysis/fitting and generate the figures as they appear in the paper.

This also serves as a Python package, which in addition contains some of the code used for plotting the surface code layouts and stabilizer circuits used in simulation.

# Requirements
To install and use this package, you will need to have Python 3.10+.

# Package management
This package uses [Poetry](https://python-poetry.org/) to manage the project dependencies. These can then be installed into a [virtual environment](https://docs.python.org/3/library/venv.html#module-venv) that will allow you to run these scripts.

We note that Poetry uses a `pyproject.toml` to define the project dependenices. This file format is also supported by PIP, so it is possible for this package to be built using `pip install`. However, we would recommend installing `poetry` on your system and install the projecting using that instead. For more information, refer to the poetry documentation.

# Dependencies
The dependencies of this project (and the version we have used for the analsis) are defined in the `pyproject.toml` file, with the specific version of each package also locked into `poetry.lock`. This should enable the reproducability of the results. 

To install all dependencies using poetry, you can simply run `poetry install --all-extras`.

# Licensing
The code here is licensed by the Apache License, version 2.0. See the `LICENSE` file in the project root directory.

# Project structure
The code here contains two main modules.

The first is a `src` module, which contains the source code for this package. Inside of it there is only a single python module, `util`, which implement the utility functions that are used for generating the plots.

The `util` submodule contains two submodules: `circuits` and `layouts`. The code layout class, methods for plotting a 2D layout, and methods to generate a distance-d rotated surface code layout and contained in the `layout` submodule.

`circuits` on the other hand contains code for plotting a circuit, consisting of layers of single-qubit or two-qubit gates.

Outside of the soure code, there is also the `figures` directory, which contains the data and scripts used to generate the figures in the article.

There are several folder, each of which contains the data for one of the figures in the paper. Below we provide a list of each figure number and the corresponding folder:
1. `code_layout`
2. `network_schematic`
3. `y_bias_performance`
4. `google_performance`
5. `google_decoder_comparison`
6. `error_suppression`
7. `assign_performance`
8. `qec_circuit`

Each folder typically contains a `data` subdirectory, where the datasets required to reproduce the figure can be found. There is an `img` directory where we save the generated figures in a `pdf` and `png` format. Finally, each folder contains a jupyter notebook and a python script that should be the same (with the python script being more useful to track changes in GitHub).

The general exceptions are figures, which are schematics  (namely, Figure 1 and 2) and do not require any data.

The datasets are typically saved using the [NetCDF5](https://www.unidata.ucar.edu/software/netcdf/) file format and can be loaded using the `xarray` Python package. Each array is self-labeled and should be readible. For completeness, we describe below the data contained in each dataset

1. `y_bias_performance`: contains 3 datasets for whether the NN made a logical error or not (boolean data) when decoding the evaluation data. The 3 datasets are named `qrennd_BY{train_bias}_train_errors.nc` and correspond to the performance of a network trained at a Y-bias of `train_bias`, which is either 0, 1 or 100. The dimensions of these datasets include the number of QEC rounds executed, the prepared logical state and the shot index. In additon, `mwpm_errors` contains the logical errors done by a MWPM decoder. 
2. `google_performance` contains two datasets named `d3_nn_errors.nc` and `d5_nn_errors.nc`, which are arrays containing the logical errors done by the NN decoder when being evaluated on the data from Google's recent experiment (either the experimental data or the simulation that used the error model they provide). The two arrays correspond to the results for the d=3 and d=5 codes respectively.The dimensions are once again the shots, QEC rounds and initial states, basis in which the experiment was done, but the distance 3 datasets also contains the center coordinates of the qubits, which define which d=3 patch in particular we are looking at.
3. `google_decoder_comparison` contains two datasets called `d3_decoder_errors.nc` and `d5_decoder_errors.nc`, corresponding to the logical errors using 5 different decoder for the d=3 codes and the d=5 one. The coordinates are the same as the datasets above: initial states, shots, QEC rounds, basis in which the experiment was done and center qubit index for the d=3 code.
4. `error_suppression` contains three datasets called `p0.0005_decoder_errors.nc`, `p0.001_decoder_errors.nc`, `p0.005_decoder_errors.nc`, which correspond to the NN decoder (trained for an error rate of 0.001) being evaluated on datasets with error rates 0.0005, 0.001 and 0.005 respectively. The dimensions of each dataset are the initial states, shots and QEC rounds.
5. `assign_performance`: contains a single dataset `decoder_error_arr.nc`, which contains the logical errors done by a NN decoder trained and evaluated on a datasets containing an assignment error probability of 0, 0.001, 0.01 or 0.1. The dimensions of the dataset are the initial states, shots, QEC rounds, assignment error probability and decoder (soft NN, hard NN or MWPM).

Note that the initial states above are either specific data-qubit states or a boolean (corresponding to all qubits prepared in 0 or 1). Most simulations were done only in the Z-basis - in this case the basis is not one of the dimensions of the data arrays.

The scripts each define all the modules and code necessary to run them. They import the data and perform any fitting required to generate the final figure. Each notebook is commented and should show how one can open these datasets.
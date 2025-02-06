# zenoh_fl
 
## Structure

This repository has the following structure:
```
├── data_processing/
|── datasets/
├── docker_old
├── MPI/
├── zcomm/
├── ZENOH/
├── config.py
├── federated_learning.py
├── generate_table.py
├── install.sh
├── requirements.txt
├── run_experiments.sh
├── run.sh
└── Util.py
```

For the main structure:
- data_processing contains the python scripts for dataset preprocessing and division for N workers.
- datasets contains the folders for the datasets and their scripts for loading, preprocessing and dividing data.
- docker_old contains an old docker image which was used to validate the MPI and ZENOH implementations altough it has not been maintained recently
- MPI/ contains the implementation of the federated learning algorithms using the MPI communication middleware, which are used in federated_learning.py.
- zcomm contains the source code of the PoC of zenoh in Rust
- ZENOH contains the same scripts as MPI/ although these ones use the zenoh PoC middleware 
- config.py defines the datasets available to use in the experiments.
- federated_learning.py is the main python executable responsible for lauching the experiments.
- generate_table.py provides the script to extract the values presented in the tables
- install.sh install the necessary ubunto packages, creates a python env, installs the python requirements, and compiles the zcomm PoC and installs it.
- requirements.txt provides the requirements packages for pyhton.
- run_experiments.sh provides an all in one script to run the same experiments as the ones performed in the paper
- run.sh is the script used when running MPI in multiple machines as it launches a specific environment before running the federated_learning.py script.
- Util.py contains a class which all datasets must inherit from.
- All python files have a --help to see the options for running them.

For the dataset structure:
- data/ contains the data files, including the train, test, validation and specific workers' data.
- [DATASET].py contains the implementation of the dataset, including the data loading and the data division.
- It is necessary to place the raw data for every dataset inside the data/ subfolder of said dataset before runnign the scripts
- The README.md shows where the datasets can be downloaded.

## Setup

First, run install.sh, if the environment does no run ubuntu, the first lines might not work.

Download the datasets and put them in the data/ folder of each dataset.

Repeat the process for each worker and for the parameter server.
Ensure that you use alias in the parameter server for each worker as  worker[i] and start the scripts from the parameter server.

Ensure you have a priv/pub key for ssh in the parameter server.

## Running the experiments

- Define the number of workers, seed and datasets to run  in the run_experiments.sh

- The seeds used for the experiments were 13, 17, 42

If everything was set correctly the script should start by sending its ssh pub key for each worker.

Then perform preprocessing and division.

Finally it will run the training for zenoh and mpi.

## Results


## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)
* **Gabriele Baldoni** - [gabrik](https://github.com/gabrik)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

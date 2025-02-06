# zenoh_fl

This repository is the code base of the paper titled "Leveraging decentralized communication for privacy-preserving federated learning in 6G Networks".
If the code is used for any other project please reference the work as described in the end of the readme.
 
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

### Execution time
| **Performance** | | **Execution time (seconds)** | | |
|---|---|---| ---- |--- |
||Synchronous|| Asynchronous | |
| MCC | MPI | Zenoh | MPI | Zenoh |
| | | **Slicing5G** | | |
| 0.50 | --   | --   | --   | --   |      
| 0.60 | --   | --   | --   | --   |     
| 0.70 | 11.5 | 10.9 | 10.9 | 10.6 |      
| 0.80 | 12.4 | 11.9 | 11.8 | --   |      
| 0.90 | 13.4 | 13.0 | 13.7 | 11.6 |      
| 0.99 | 35.2 | 36.7 | 34.5 | 36.3 |    
| | | **NetSlice5G** | | |
| 0.50 | 5.0  | 6.3  | 3.0 | 2.0  |      
| 0.60 | 5.2  | 6.5  | 3.3 | 2.3  |     
| 0.70 | 6.0  | 7.4  | 4.2 | 3.3  |      
| 0.80 | 6.7  | 8.3  | 5.5 | 5.1  |      
| 0.90 | 8.6  | 10.6 | 7.2 | 7.1  |      
| 0.99 | 10.6 | 12.8 | 9.1 | 9.6  | 
| | | **IOT_DNL** | | |
| 0.50 | --   | --   | --   | --   |      
| 0.60 | --   | --   | --   | --   |     
| 0.70 | 10.5 | 9.8  | --   | --   |      
| 0.80 | --   | --   | --   | --   |      
| 0.90 | 11.7 | 11.1 | 6.5  | 5.6  |      
| 0.99 | 24.8 | 25.0 | 24.3 | 24.9 | 
| | | **UNSW** | | |
| 0.50 | --   | --   | --   | --   |      
| 0.60 | --   | --   | --   | --   |     
| 0.70 | 24.3 | 25.4 | 19.9 | 21.0 |      
| 0.80 | 37.7 | 39.4 | 46.4 | 49.4 |      
| 0.90 | --   | --   | --   | --   |      
| 0.99 | --   | --   | --   | --   | 

## Paper reference

Rafael Teixeira, Gabriele Baldoni, Mário Antunes, Diogo Gomes, Rui L. Aguiar,
Leveraging decentralized communication for privacy-preserving federated learning in 6G Networks,
Computer Communications,
Volume 233,
2025,
108072,
ISSN 0140-3664,
https://doi.org/10.1016/j.comcom.2025.108072.
(https://www.sciencedirect.com/science/article/pii/S0140366425000295)


## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)
* **Gabriele Baldoni** - [gabrik](https://github.com/gabrik)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

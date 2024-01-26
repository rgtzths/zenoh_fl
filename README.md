# zenoh_fl

## Structure

This repository has the following structure:
```
├── MPI/
├── [DATASET]/
├── config.py
├── data_division.py
├── federated_learning.py
├── model_eval.py
├── single_training.py
└── Util.py
```

Other folders contain data and results for specific datasets, which have the following structure:
```
├── data/
├── mpi/
└── [DATASET].py
```

For the main structure:
- MPI/ contains the implementation of the federated learning algorithms, which are used in federated_learning.py. 
- config.py defines the datasets to be used in the experiments.
- Util.py contains a class which all datasets must inherit from.
- The other files are used to run experiments, use --help to see the options.

For the dataset structure:
- data/ contains the data files, including the train, test, validation and specific workers' data.
- mpi/ contains the results of the federated learning algorithms, including the models and the training logs, for each experiment.
- [DATASET].py contains the implementation of the dataset, including the data loading and the data division.

## Setup

First, create a virtual environment and install the requirements:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the datasets and put them in the data/ folder of each dataset (links found in the data/README.md files of each dataset).

Finally, run the data_processing.py and data_division.py files for each dataset, to process the data and divide it into train, test, validation and workers' data.

## Running the experiments

- Run single_training.py for each dataset
- Run federated_learning.py for the experiments you want to run

To run on docker using a single command use `./run-tests.sh` that will run selected tests based on its parametes.
Usage:
```
Usage: ./run-tests.sh 
   -a async centralized
   -S sync decentralized
   -A async decentralized
   -s sync centralized	
   -h help
```

You can also run the experiments via Docker compose

```
COMM=[mpi|zenoh] TEST=[1|2|3|4] docker compose up 
```

## Results


## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)
* **Gabriele Baldoni** - [gabrik](https://github.com/gabrik)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

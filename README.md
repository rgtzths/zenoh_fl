# zenoh_fl

## Installing

To run the scenarios locally please install the requirements

 `pip install -r requirements.txt`

If you want to run the experiments in RPIs please follow the `rpi_setup_comands.md`

If you want to run the experiments in docker please run `docker compose up`.

mpi_decentralized_assync

## Running

To run locally you only need the following commands

`mpirun -np 4 python mpi_decentralized_assync.py -d dataset/one_hot_encoding/`

`mpirun -np 4 python mpi_decentralized_sync.py -d dataset/one_hot_encoding/`

`mpirun -np 4 python mpi_centralized_sync.py -d dataset/one_hot_encoding/`

`mpirun -np 4 python mpi_centralized_assync.py -d dataset/one_hot_encoding/`

If you want to run the single_host setting you can run

`python single_host.py`

To run on docker you will need to connect to the master container

`docker exec -it --user mpiuser zenoh_fl-master-1 bash`

Connect one time to every worker to confirm the fingerprint of the server.

`ssh worker1` and then `exit`

After this you can run the command

`mpirun -np 4 python mpi_decentralized_assync.py -d dataset`

`mpirun -np 4 python mpi_decentralized_sync.py -d dataset`

`mpirun -np 4 python mpi_centralized_sync.py -d dataset`

`mpirun -np 4 python mpi_centralized_assync.py -d dataset`

inside the `code` folder.

To change the hyperparameters please confirm the available options in the files.


## Results


## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

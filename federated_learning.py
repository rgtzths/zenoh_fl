import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from MPI.decentralized_async import run as mpi_decentralized_async
from MPI.decentralized_sync import run as mpi_decentralized_sync
from MPI.centralized_async import run as mpi_centralized_async
from MPI.centralized_sync import run as mpi_centralized_sync
from ZENOH.decentralized_async import run as zenoh_decentralized_async
from ZENOH.decentralized_sync import run as zenoh_decentralized_sync
from ZENOH.centralized_async import run as zenoh_centralized_async
from ZENOH.centralized_sync import run as zenoh_centralized_sync
import asyncio

from config import DATASETS, OPTIMIZERS

parser = argparse.ArgumentParser()
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default="IOT_DNL")
parser.add_argument("-c", help="Communication protocol (zenoh, mpi)", default="mpi")
parser.add_argument("-m", help="Method [1:decentralized async, 2:decentralized, sync]", default=1, type=int)
parser.add_argument("-o", help=f"Optimizer {list(OPTIMIZERS.keys())}", default="Adam")
parser.add_argument("-lr", help="Learning rate", default=0.001, type=float)
parser.add_argument("-b", help="Batch size", default=1024, type=int)
parser.add_argument("-s", help="MCC score to achieve", default=1, type=float)
parser.add_argument("-ge", help="Max Global epochs", default=100, type=int)
parser.add_argument("-le", help="Local epochs", default=3, type=int)
parser.add_argument("-p", help="Patience", default=10, type=int)
parser.add_argument("-md", help="Minimum Delta", default=0.001, type=float)
parser.add_argument("-a", help="Updated bound for the master/worker", default=0.2, type=float)
parser.add_argument("-nw", help="Number of workers (for zenoh)", default=4, type=int)
parser.add_argument("-wid", help="Worker id (for zenoh)", default=0, type=int)
parser.add_argument("-f", help="Output folder", default="results", type=str)
parser.add_argument("--seed", help="Seed used for reproducibility", default=42, type=int)
args = parser.parse_args()

if args.d not in DATASETS.keys():
    raise ValueError(f"Dataset name must be one of {list(DATASETS.keys())}")
if args.o not in OPTIMIZERS.keys():
    raise ValueError(f"Optimizer name must be one of {list(OPTIMIZERS.keys())}")

if args.c == "mpi":
    match args.m:
        case 1:
            mpi_decentralized_async(DATASETS[args.d](args.seed), OPTIMIZERS[args.o], args.s, args.lr, args.b, args.ge, args.le, args.a, args.p, args.md, args.f)
        case 2:
            mpi_decentralized_sync(DATASETS[args.d](args.seed), OPTIMIZERS[args.o], args.s, args.lr, args.b, args.ge, args.le, args.p, args.md, args.f)
        case 3:
            mpi_centralized_async(DATASETS[args.d](args.seed), OPTIMIZERS[args.o], args.s, args.lr, args.b, args.ge, args.p, args.md, args.f)
        case 4:
            mpi_centralized_sync(DATASETS[args.d](args.seed), OPTIMIZERS[args.o], args.s, args.lr, args.b, args.ge, args.p, args.md, args.f)
else:
    match args.m:
        case 1:
            asyncio.run(zenoh_decentralized_async(DATASETS[args.d](args.seed), OPTIMIZERS[args.o], args.s, args.lr, args.b, args.ge, args.le, args.a,args.p, args.md,  args.nw, args.wid, args.f, "tcp/127.0.0.1:7447"))
        case 2:
            asyncio.run(zenoh_decentralized_sync(DATASETS[args.d](args.seed), OPTIMIZERS[args.o], args.s, args.lr, args.b, args.ge, args.le, args.p, args.md, args.nw, args.wid, args.f, "tcp/127.0.0.1:7447"))
        case 3:
            asyncio.run(zenoh_centralized_async(DATASETS[args.d](args.seed), OPTIMIZERS[args.o], args.s, args.lr, args.b, args.ge, args.p, args.md, args.nw, args.wid, args.f, "tcp/127.0.0.1:7447"))
        case 4:
            asyncio.run(zenoh_centralized_sync(DATASETS[args.d](args.seed), OPTIMIZERS[args.o], args.s, args.lr, args.b, args.ge, args.p, args.md, args.nw, args.wid, args.f, "tcp/127.0.0.1:7447"))
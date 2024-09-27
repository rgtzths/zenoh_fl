import argparse
import tensorflow as tf
import sys
sys.path.append('.')

from config import DATASETS

parser = argparse.ArgumentParser()
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default="IOT_DNL")
parser.add_argument("--seed", help="Seed used for reproducibility", default=42, type=int)
args = parser.parse_args()

if args.d not in DATASETS.keys():
    raise ValueError(f"Dataset name must be one of {list(DATASETS.keys())}")

tf.keras.utils.set_random_seed(args.seed)

dataset_util = DATASETS[args.d](args.seed)
dataset_util.data_processing()

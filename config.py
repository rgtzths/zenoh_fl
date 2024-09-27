import tensorflow as tf

from datasets.IOT_DNL.IOT_DNL import IOT_DNL
from datasets.Slicing5G.Slicing5G import Slicing5G
from datasets.UNSW.UNSW import UNSW
from datasets.NetSlice5G.NETSLICE5G import NetSlice5G


DATASETS = {
    "IOT_DNL": IOT_DNL,
    "Slicing5G": Slicing5G,
    "NetSlice5G" : NetSlice5G,
    "UNSW" : UNSW,
}

OPTIMIZERS = {
    "SGD": tf.keras.optimizers.SGD,
    "Adam": tf.keras.optimizers.Adam
}

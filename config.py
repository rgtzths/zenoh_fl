import tensorflow as tf

from IOT_DNL.IOT_DNL import IOT_DNL
from Slicing5G.Slicing5G import Slicing5G
from FLASH.FLASH import FLASH

DATASETS = {
    "IOT_DNL": IOT_DNL(),
    "Slicing5G": Slicing5G(),
    "FLASH": FLASH()
}

OPTIMIZERS = {
    "SGD": tf.keras.optimizers.SGD,
    "Adam": tf.keras.optimizers.Adam
}

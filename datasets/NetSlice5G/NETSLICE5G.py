import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf

from Util import Util

class NetSlice5G(Util):

    def __init__(self, seed):
        super().__init__("NetSlice5G", seed)


    def data_processing(self):
        df_train = pd.read_csv(f"datasets/{self.name}/data/train_dataset.csv")
        #df_test = pd.read_csv(f"datasets/{self.name}/data/test_dataset.csv")

        output = f"datasets/{self.name}/data/{self.seed}"

        Path(output).mkdir(parents=True, exist_ok=True)

        x_train = df_train.values[:,:-1]
        y_train = df_train.values[:,-1] -1

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=self.seed, test_size=0.2)
        x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, random_state=self.seed, test_size=0.2)


        print(f"\nTotal samples {df_train.values.shape[0]}")
        print(f"Shape of the train data: {x_train.shape}")
        print(f"Shape of the validation data: {x_cv.shape}")
        print(f"Shape of the test data: {x_test.shape}\n")
        
        np.savetxt(f"{output}/X_train.csv", x_train, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/X_val.csv", x_cv, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/X_test.csv", x_test, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/y_train.csv", y_train, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/y_val.csv", y_cv, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/y_test.csv", y_test, delimiter=",", fmt="%d")
    
    def create_model(self):
        return tf.keras.models.Sequential([
                # flatten layer
                tf.keras.layers.Input(shape=(16,)),
                # hidden layers
                tf.keras.layers.Dense(73, activation='relu'),
                tf.keras.layers.Dropout(0.5),

                # output layer
                tf.keras.layers.Dense(3, activation="softmax")
            ])
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
import tensorflow as tf

from Util import Util

class UNSW(Util):

    def __init__(self, seed=42):
        super().__init__("UNSW", seed)


    def data_processing(self):
        dataset = f"datasets/{self.name}/data/NF-UNSW-NB15-v2.csv"
        output = f"datasets/{self.name}/data/{self.seed}"

        Path(output).mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(dataset)
        data.dropna()
        y = data['Attack']

        le = LabelEncoder()

        y = pd.Series(le.fit_transform(y), name='target')

        x = data.drop(columns=['Label', "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "Attack"])

        n_samples = x.shape[0]

        scaler = QuantileTransformer(output_distribution='normal')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.seed, shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=self.seed, shuffle=True)

        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

        print(f"\nTotal samples {n_samples}")
        print(f"Shape of the train data: {x_train.shape}")
        print(f"Shape of the validation data: {x_val.shape}")
        print(f"Shape of the test data: {x_test.shape}\n")

        # Save the data
        x_train.to_csv(f"{output}/X_train.csv", index=False)
        x_val.to_csv(f"{output}/X_val.csv", index=False)
        x_test.to_csv(f"{output}/X_test.csv", index=False)
        y_train.to_csv(f"{output}/y_train.csv", index=False)
        y_val.to_csv(f"{output}/y_val.csv", index=False)
        y_test.to_csv(f"{output}/y_test.csv", index=False)


    def create_model(self):
        # https://www.semanticscholar.org/reader/00f44fb92b02552208729e228f0f3a6e06cbdadd table II
        return tf.keras.models.Sequential([
            # input layer
            tf.keras.layers.Input(shape=(39,)),
            # hidden layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(96, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            # output layer
            tf.keras.layers.Dense(10, activation='softmax')
        ])

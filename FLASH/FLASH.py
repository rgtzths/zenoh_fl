from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Input, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import numpy as np

from Util import Util

class FLASH(Util):

    def __init__(self):
        super().__init__("FLASH")


    def data_processing(self):
        return



    def data_division(self, n_workers):
        return
    


    def load_training_data(self):
        return
    


    def load_validation_data(self):
        x_val = np.load(f"{self.name}/data/x_test.npz")["data"]
        y_val = np.load(f"{self.name}/data/y_test.npz")["data"]
        y_val, _ = self.custom_label(y_val)
        return x_val, y_val
    


    def load_test_data(self):
       return



    def load_worker_data(self, n_workers, worker_id):
        x_train = np.load(f"{self.name}/data/{n_workers}_workers/x_train_subset_{worker_id}.npz")["data"]
        y_train = np.load(f"{self.name}/data/{n_workers}_workers/y_train_subset_{worker_id}.npz")["data"]
        y_train, _ = self.custom_label(y_train)

        return x_train, y_train



    def create_model(self):
        x_val = np.load(f"{self.name}/data/x_test.npz")["data"]
        y_val = np.load(f"{self.name}/data/y_test.npz")["data"]
        _, num_classes = self.custom_label(y_val)
        input_shape = x_val.shape[1:]
        return self.create_lidar_model(input_shape, num_classes)
    


    def custom_label(self, y):
        num_classes = y.shape[1]
        y_final = []
        for i in range(y.shape[0]):
            y_final.append(y[i,:].argmax())

        return np.array(y_final), num_classes



    def create_lidar_model(self, input_shape, num_classes):
        drop_prob = 0.3
        channel = 32  # 32 now is the best, better than 64, 16
        input_lid = Input(shape=input_shape, name='lidar_input')
        a = layer = Conv2D(channel, kernel_size=(3, 3),
                            activation='relu', padding="SAME", input_shape=input_shape, name='lidar_conv1')(input_lid)
        layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv2')(layer)
        layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv3')(layer)  # + a
        layer = Add(name='lidar_add1')([layer, a])  # DR
        layer = MaxPooling2D(pool_size=(2, 2), name='lidar_maxpool1')(layer)
        b = layer = Dropout(drop_prob, name='lidar_dropout1')(layer)

        layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv4')(layer)
        layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv5')(layer)  # + b
        layer = Add(name='lidar_add2')([layer, b])  # DR
        layer = MaxPooling2D(pool_size=(2, 2), name='lidar_maxpool2')(layer)
        c = layer = Dropout(drop_prob, name='lidar_dropout2')(layer)

        layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv6')(layer)
        layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv7')(layer)  # + c
        layer = Add(name='lidar_add3')([layer, c])  # DR
        layer = MaxPooling2D(pool_size=(1, 2), name='lidar_maxpool3')(layer)
        d = layer = Dropout(drop_prob, name='lidar_dropout3')(layer)

        # # if add this layer, need 35 epochs to converge
        layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu',name='lidar1_added')(layer)
        layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu',name='lidar1_added2')(layer) #+ d
        layer = Add(name='lidar_addbetween')([layer, d])  # DR
        layer = MaxPooling2D(pool_size=(1, 2),name='lidar_maxpool3between')(layer)
        e = layer = Dropout(drop_prob, name='lidar_dropout3between')(layer)

        layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv8')(layer)
        layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv9')(layer)  # + d
        layer = Add(name='lidar_add4')([layer, e])  # DR

        layer = Flatten(name='lidar_flatten')(layer)
        layer = Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name="lidar_dense1")(layer)  #was 512
        layer = Dropout(0.2, name='lidar_dropout4')(layer)  # 0.25 is similar ... could try more values
        layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name="lidar_dense2")(layer)
        layer = Dropout(0.2, name='lidar_dropout5')(layer)  # 0.25 is similar ... could try more values
        out = Dense(num_classes, activation='softmax',name = 'lidar_output')(layer)
        
        return Model(inputs=input_lid, outputs=out)
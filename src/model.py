# coding: utf-8
# pylint:
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn

import os

import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from utils import parse_args, save2pickle, tictoc

np.random.seed(1)
tf.compat.v1.set_random_seed(13)


# ------------------------------------------------------------------------------
#                                 MODEL MODULE
# ------------------------------------------------------------------------------


class Constrain(tf.keras.layers.Layer):
    """Constrain layer Type 1

        combine center cropping & space to channel & average pooling
        to constrain the info from space into channel.

        input:
            - 5D: (batch_size, timesteps, height, width, channels)

        output:
            - 5D: (batch_size, timesteps, height/2, width/2, 2*channels)
    """

    def __init__(self):
        super().__init__()

    def build(self, input_shape):

        # average pooling
        self.avg_pool_1 = tf.keras.layers.AveragePooling2D()
        # center cropping
        self.crop_1 = tf.keras.layers.Cropping2D(
            cropping=((int(input_shape[-3]) // 4, int(input_shape[-3]) // 4),
                      (int(input_shape[-2]) // 4, int(input_shape[-2]) // 4)))
        # space to channel (4 parts)
        self.space2channel_1 = tf.keras.layers.Cropping2D(
            cropping=((0, int(input_shape[-3]) // 2),
                      (0, int(input_shape[-2]) // 2)))
        self.space2channel_2 = tf.keras.layers.Cropping2D(
            cropping=((0, int(input_shape[-3]) // 2),
                      (int(input_shape[-2]) // 2, 0)))
        self.space2channel_3 = tf.keras.layers.Cropping2D(
            cropping=((int(input_shape[-3]) // 2, 0),
                      (0, int(input_shape[-2]) // 2)))
        self.space2channel_4 = tf.keras.layers.Cropping2D(
            cropping=((int(input_shape[-3]) // 2, 0),
                      (int(input_shape[-2]) // 2, 0)))

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. merge dimension sample and timestep
                   1. global average pool for spatial
                   2. cropping for spatial
                   3. space to channel
                   4. concat
                   5. spread dimensions
        """
        # get shape 5D
        _, T, H, W, C = inputs.get_shape().as_list()
        # 5D to 4D
        x = tf.reshape(inputs, [-1, H, W, C])
        # avg
        x_avg = self.avg_pool_1(x)
        # crop
        x_crop = self.crop_1(x)
        # space to channel
        x_space2channel = tf.concat((self.space2channel_1(x),
                                     self.space2channel_2(x),
                                     self.space2channel_3(x),
                                     self.space2channel_4(x)), axis=-1)
        # concat
        x = tf.concat([x_avg, x_crop], -1)
        # reshape
        x = tf.reshape(x, [-1, T, H // 2, W // 2, 2*C])  # 6*C
        return x


class Downsampling(tf.keras.layers.Layer):
    """Constrain layer Type 2.

        construct by convolutional & pooling layers to exact info
        between each channels.

    Input shape:
        - 5D: (batch_size, timesteps, height, width, channels)

     Output shape:
        - 5D: (batch_size, timesteps, height/2, width/2, channels)
    """

    def __init__(self):
        super().__init__()

    def build(self, input_shape):

        self.conv_1 = tf.keras.layers.Conv2D(
            filters=2*int(input_shape[-1]),
            kernel_size=(3, 3),
            padding='same')
        self.pool_1 = tf.keras.layers.MaxPooling2D()

        self.conv_2 = tf.keras.layers.Conv2D(
            filters=2*int(input_shape[-1]),
            kernel_size=(3, 3),
            padding='same')
        self.pool_2 = tf.keras.layers.MaxPooling2D(strides=(2, 2))

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. merge dimension sample and timestep
                   1. convolutional layer
                   2. pooling
                   5. spread dimensions
        """
        # get shape 5D
        _, T, H, W, C = inputs.get_shape().as_list()
        # 5D to 4D
        x = tf.reshape(inputs, [-1, H, W, C])
        #
        x = self.conv_1(x)
        x = self.pool_1(x)
        #
        x = tf.reshape(x, [-1, T, H // 2, W // 2, 2 * C])
        return x


class Channel_Attention(tf.keras.layers.Layer):
    """Channel Attention layers.

    Parameters
    ----------
    activation: str, optional (default 'sigmoid')
    dense_ratio: float, optional (default 1)

    input:
        could be applied for both input shape
        - 4D: (batch_size, height, weight, channels) &
        - 5D: (batch_size, timesteps, height, weight, channels)

    output:
        - channel attentioned inputs: the same dims with inputs
        - channel attention weight:
            - 4D: (batch_size, 1, 1, channels) &
            - 5D: (batch_size, timesteps, 1, 1, channels)

    References:
        - Contains the implementation of Squeeze-and-Excitation(SE) block.
        As described in https://arxiv.org/abs/1709.01507.
    """

    def __init__(self,
                 dense_ratio=1,
                 activation='sigmoid'):
        super().__init__()

        self.dense_ratio = dense_ratio
        self.activation = activation

    def build(self, input_shape):

        self.dense_avg_pool_1 = tf.keras.layers.Dense(
            units=int(input_shape[-1]) //
            self.dense_ratio,
            activation='relu',
            name='MLP_AVG_POOL_1')
        self.dense_avg_pool_2 = tf.keras.layers.Dense(
            units=int(input_shape[-1]),
            name='MLP_AVG_POOL_2')
        self.activation = tf.keras.layers.Activation(
            activation='sigmoid',
            name='CHANNEL_ATTENTION_SIGMOID')

        super().build(input_shape)

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. global average pooling & max pooling
                   1. two same dense layers for both pooling
                   2. concat
        """
        # avg pooling
        avg_pool = tf.compat.v1.reduce_mean(
            inputs, axis=[-3, -2], keep_dims=True)
        x_avg_pool = self.dense_avg_pool_1(avg_pool)
        x_avg_pool = self.dense_avg_pool_2(x_avg_pool)
        # max pooling
        max_pool = tf.compat.v1.reduce_max(
            inputs, axis=[-3, -2], keep_dims=True)
        x_max_pool = self.dense_avg_pool_1(max_pool)
        x_max_pool = self.dense_avg_pool_2(x_max_pool)
        # concat
        scale = self.activation(x_avg_pool + x_max_pool)
        return inputs*scale


class Spatial_Attention(tf.keras.layers.Layer):
    """Spatial Attention layers.

    Parameters
    ----------
    kernel_size:

    input:
        - 4D: (batch_size, height, weight, channels) &
        - 5D: (batch_size, timesteps, height, weight, channels)

    output shape:
        - channel attentioned inputs: the same dims with inputs
        - channel attention weight:
            - 4D: (batch_size, 1, 1, channels) &
            - 5D: (batch_size, timesteps, 1, 1, channels)

    References:
        - Contains the implementation of Convolutional Block Attention Module
        (CBAM) block. As described in https://arxiv.org/abs/1807.06521.
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        self.kernel_size = kernel_size

    def build(self, input_shape):

        self.dims = len(input_shape)
        self.Conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=[1, 1],
            padding='same',
            name='CONV')
        self.activation = tf.keras.layers.Activation(
            activation='sigmoid',
            name='SPATIAL_ATTENTION_SIGMOID')

        super().build(input_shape)

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. global average pooling & max pooling for channel
                   1. concat
                   2. convolutional and activate
                   3. spread dimensions
        """
        # dense channel dims
        avg_pool = tf.compat.v1.reduce_mean(
            inputs, axis=[-1], keep_dims=True)
        max_pool = tf.compat.v1.reduce_max(
            inputs, axis=[-1], keep_dims=True)
        # concat
        concat = tf.concat([avg_pool, max_pool], -1)

        # reshape
        if self.dims == 5:
            _, T, H, W, C = concat.get_shape().as_list()
            concat = tf.reshape(concat, [-1, H, W, C])
        else:
            _, H, W, C = concat.get_shape().as_list()

        # contract info
        x = self.Conv(concat)
        # activate
        scale = self.activation(x)

        # reshape
        if self.dims == 5:
            scale = tf.reshape(scale, [-1, T, H, W, 1])
        else:
            scale = tf.reshape(scale, [-1, H, W, 1])

        return inputs*scale


class Self_Attention(tf.keras.layers.Layer):
    """Self Attention layers.

    Parameters
    ----------
    kernel_size:

    input shape:
        could be applied for both input shape
        - 4D: (batch_size, height, weight, channels) &

    output shape:
        - channel attentioned inputs: the same dims with inputs
        - channel attention weight:
            - 4D: (batch_size, 1, 1, channels) &
            - 5D: (batch_size, timesteps, 1, 1, channels)

    References:
        - Attention is all you need.
    """

    def __init__(self, dropout_rate=0.1):
        super().__init__()

        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        self.dims = len(input_shape)

        self.H, self.W, self.C = \
            int(input_shape[-3]), \
            int(input_shape[-2]), \
            int(input_shape[-1])

        if self.dims == 5:
            self.T = int(input_shape[-4])

        self.q = tf.keras.layers.Conv2D(
            filters=int(input_shape[-1]), kernel_size=1)
        self.k = tf.keras.layers.Conv2D(
            filters=int(input_shape[-1]), kernel_size=1)
        self.v = tf.keras.layers.Conv2D(
            filters=int(input_shape[-1]), kernel_size=1)

        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.bn = tf.keras.layers.BatchNormalization()

        super().build(input_shape)

    def call(self, inputs):
        """Exec"""
        if self.dims == 5:
            inputs = tf.reshape(inputs, [-1, self.H, self.W, self.C])

        # get q,k,v
        Q = self.q(inputs)
        K = self.k(inputs)
        V = self.v(inputs)
        # get shape
        _, H, W, C = Q.get_shape().as_list()
        # reshape 4D to 3D
        Q_ = tf.reshape(Q, [-1, H*W, C])
        K_ = tf.reshape(K, [-1, H*W, C])
        V_ = tf.reshape(V, [-1, H*W, C])
        # multiply
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        # scale
        outputs = outputs / (C ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)
        # Dropouts
        outputs = self.dropout_1(outputs)
        # weighted sum
        outputs = tf.matmul(outputs, V_)
        # reshape 3D to 4D
        outputs = tf.reshape(outputs, [-1, H, W, C])
        # residual connection
        outputs += inputs
        # normalization
        outputs = self.bn(outputs)

        if self.dims == 5:
            outputs = tf.reshape(outputs, [-1, self.T, self.H, self.W, self.C])

        return outputs


class ED_ConvLSTM(tf.keras.layers.Layer):
    """encoder-decoder ConvLSTM"""

    def __init__(self, output_len):
        super().__init__()

        self.output_len = output_len

        self.convlstm_1 = tf.keras.layers.ConvLSTM2D(
            32, (3, 3), activation='tanh',
            #kernel_initializer='he_normal',
            padding='same', return_sequences=True)
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.convlstm_2 = tf.keras.layers.ConvLSTM2D(
            32, (3, 3), activation='tanh',
            padding='same', return_sequences=True)
        self.bn_2 = tf.keras.layers.BatchNormalization()

        self.convlstm_5 = tf.keras.layers.ConvLSTM2D(
            32, (3,3), activation='tanh', padding='same', return_sequences=True)
        self.bn_5 = tf.keras.layers.BatchNormalization()


        self.convlstm_7 = tf.keras.layers.ConvLSTM2D(32, (3,3), activation='tanh', padding='same',return_sequences=False)
        self.bn_7 = tf.keras.layers.BatchNormalization()

        self.convlstm_3 = tf.keras.layers.ConvLSTM2D(
            32, (3, 3), activation='tanh',
            padding='same', return_sequences=True)
        self.bn_3 = tf.keras.layers.BatchNormalization()



        
        self.convlstm_4 = tf.keras.layers.ConvLSTM2D(
            1, (3, 3), activation='tanh',
            padding='same', return_sequences=False)

        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. convlstm & bn (encoder)
                   1. transform last state to decoder
                   2. decoder
        """
        x = self.convlstm_1(inputs)
        x = self.bn_7(x)


        #x = tf.keras.layers.Lambda(
        #    lambda x: tf.keras.backend.concatenate(
        #        [x[:, np.newaxis, :, :, :]]*self.output_len, axis=1))(x)

        #x = self.convlstm_3(x)
        #x = self.bn_3(x)

        x = self.convlstm_4(x)
        x = self.bn_3(x)
        x = self.dense(x)
        return x


class ConvLSTM(tf.keras.layers.Layer):
    """ConvLSTM"""

    def __init__(self):
        super().__init__()

    def build(self, input_shape):

        self.convlstm_1 = tf.keras.layers.ConvLSTM2D(
            filters=32, kernel_size=(3, 3),
            padding='same', return_sequences=True,
            activation='tanh', recurrent_activation='hard_sigmoid',
            kernel_initializer='glorot_uniform', unit_forget_bias=True,
            dropout=0.3, recurrent_dropout=0.3, go_backwards=True)
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.convlstm_2 = tf.keras.layers.ConvLSTM2D(
            filters=32, kernel_size=(3, 3),
            padding='same', return_sequences=True,
            activation='tanh', recurrent_activation='hard_sigmoid',
            kernel_initializer='glorot_uniform', unit_forget_bias=True,
            dropout=0.3, recurrent_dropout=0.3, go_backwards=True)
        self.bn_2 = tf.keras.layers.BatchNormalization()

        self.convlstm_3 = tf.keras.layers.ConvLSTM2D(
            filters=16, kernel_size=(3, 3),
            padding='same', return_sequences=False,
            activation='tanh', recurrent_activation='hard_sigmoid',
            kernel_initializer='glorot_uniform', unit_forget_bias=True,
            dropout=0.3, recurrent_dropout=0.3, go_backwards=True)
        self.bn_3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        """Exec

        .. rubic:: process loop
                   0. convlstm  & bn
        """
        x = self.convlstm_1(inputs)
        x = self.bn_1(x)
        x = self.convlstm_2(x)
        x = self.bn_2(x)
        x = self.convlstm_3(x)
        x = self.bn_3(x)

        return x


def smnet():
    """SMNET

    Parameters
    ----------
    config. (detail see utils.py)
        0. timestep
        1. height
        2. width
        3. channel
        4. Downsampling
        5. Constrain
        6. ConvLSTM
        7. nums_input_attention
        8. Channel_Attention
        9. Spatial_Attention
        10. nums_self_attention

    .. rubic:: process loop
               0. constrain and downsampling
               1. channel attention and spatial attention
               2. encoder-decoder convlstm
    """
    # config
    config = parse_args()
    # inputs
    inputs = tf.keras.layers.Input(shape=(config.len_inputs,
                                          8,8,#config.height_inputs,
                                          #config.width_inputs,
                                          config.channel_inputs))
    x = inputs
    # layers
    if config.downsampling:
        #x = Constrain()(x)
        #x = Downsampling()(x)
        print('2')
    for _ in range(config.nums_input_attention):
        #if config.channel_attention:
        #    x = Channel_Attention()(x)
        #if config.spatial_attention:
        #    x = Spatial_Attention()(x)
        print('1')

    if config.convlstm:
        x = ConvLSTM()(x)  # 5D

        if config.self_attention:
            for i in range(config.nums_self_attention):
                x = Self_Attention()(x)

        x = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(x)
        x = tf.expand_dims(x, 1)
    else:
        x = ED_ConvLSTM(config.len_outputs)(x)

    # build
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    # summary
    model.summary()

    return model


def lstm():

    config = parse_args()
    # inputs
    inputs = tf.keras.layers.Input(shape=(config.len_inputs, 3))
    x = tf.keras.layers.LSTM(units=32, kernel_initializer='zeros',
                              recurrent_initializer='zeros',
                             return_sequences=False)(inputs)
    #x = tf.keras.layers.LSTM(units=128, activation='tanh',
    #                         return_sequences=True)(x)
    #x = tf.keras.layers.LSTM(units=128, activation='tanh',
    #                         return_sequences=False)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    #x = tf.keras.layers.LSTM(units=1, activation='relu')(x)
    # build
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    # summary
    model.summary()

    return model


def dnn():

    config = parse_args()
    inputs = tf.keras.layers.Input(shape=(config.len_inputs, 64*3))
    #x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LSTM(units=16, activation='tanh', return_sequences=True)(inputs)
    #x = tf.keras.layers.LSTM(units=64, activation='tanh', return_sequences=True)(x)
    #x = tf.keras.layers.LSTM(units=32, activation='tanh', return_sequences=False)(x)
    #x = tf.keras.layers.LSTM(units=1, activation='relu')(x)
    x = tf.keras.layers.LSTM(64)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.summary()

    return model



class ML_Models():
    """A class of traditional machine learning models (need improved)"""

    def __init__(self, inputs):
        self.inputs = inputs
        x_train = inputs['x_train']
        y_train = inputs['y_train']
        N = x_train.shape[0]
        self.x_train = x_train[:int(N*0.8)]
        self.y_train = y_train[:int(N*0.8)]
        self.x_validate = x_train[int(N*0.8):]
        self.y_validate = y_train[int(N*0.8):]
        self.x_test = inputs['x_valid']
        self.y_test = inputs['y_valid']

    def Ridge(self, inputs, alpha=.1):

        model = Ridge(alpha=alpha)
        model.fit(inputs['x_train'], inputs['y_train'])
        pred_valid = model.predict(inputs['x_valid'])
        log = dict()
        log['y_valid'] = inputs['y_valid']
        log['pred_valid'] = pred_valid
        return log

    def SVR(self, inputs):
        """
        score = []
        func = []
        for i in [1, 10, 100]:
            model = svm.SVR(C=i)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_validate)
            func.append(model)
            score.append(mean_squared_error(self.y_validate, y_pred))

        model = func[score.index(min(score))]
        pred_valid = model.predict(self.x_test)



        log = dict()
        log['y_valid'] = self.y_test
        log['pred_valid'] = pred_valid
        return log
        """

        model = svm.SVR()

        N = inputs['x_train'].shape[0]
        N_train = int(N * 0.8)
        model.fit(inputs['x_train'][:N_train], inputs['y_train'][:N_train])

        pred_valid = model.predict(inputs['x_valid'])

        print('MSE on train dataset of SVR is {}'.format(
            mean_squared_error(inputs['y_train'][:N_train], model.predict(
                inputs['x_train'][:N_train]))))

        print('MSE on validate dataset of SVR is {}'.format(
            mean_squared_error(inputs['y_train'][N_train:], model.predict(
                inputs['x_train'][N_train:]))))

        print('MSE on test dataset of SVR is {}'.format(
            mean_squared_error(inputs['y_valid'], pred_valid)))
        log = dict()
        log['y_valid'] = inputs['y_valid']
        log['pred_valid'] = pred_valid
        return log

    def RF(self, inputs):
        """Train keras rf"""
        """
        score = []
        func = []
        for i in [1,10,100]:
            model = RandomForestRegressor()

            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_validate)
            func.append(model)
            score.append(mean_squared_error(self.y_validate,y_pred))

        model = func[score.index(min(score))]
        pred_valid = model.predict(self.x_test)

        log = dict()
        log['y_valid'] = self.y_test
        log['pred_valid'] = pred_valid
        return log
        """

        model = RandomForestRegressor()

        N = inputs['x_train'].shape[0]
        N_train = int(N * 0.8)
        model.fit(inputs['x_train'][:N_train], inputs['y_train'][:N_train])

        pred_valid = model.predict(inputs['x_valid'])

        print('MSE on train dataset of RF is {}'.format(
            mean_squared_error(inputs['y_train'][:N_train], model.predict(
                inputs['x_train'][:N_train]))))

        print('MSE on validate dataset of RF is {}'.format(
            mean_squared_error(inputs['y_train'][N_train:], model.predict(
                inputs['x_train'][N_train:]))))

        print('MSE on test dataset of RF is {}'.format(
            mean_squared_error(inputs['y_valid'], pred_valid)))
        log = dict()
        log['y_valid'] = inputs['y_valid']
        log['pred_valid'] = pred_valid
        return log

    def __call__(self):

        log = {}
        log['ridge'] = self.Ridge(self.inputs)
        log['SVR'] = self.SVR(self.inputs)
        log['RF'] = self.RF(self.inputs)

        return log


"""

def grid_search(func, params, train_data, validate_data):

    for i in params:
        func.fit(train_data)
        func.predict(validate_data)
        func.append(func)
        score.append(validate_data, ...)

    return func[min(score)]


def grid_search(func, params, x_train, y_train):

    from sklearn.model_selection import train_test_split
    x_t, x_v, y_t, y_v = train_test_split(
        x_train, y_train, shuffle=False, test_size=0.2)

    for i in params:

        model = RandomForestRegressor(**i)
        model.fit(x_t, y_t)
        score.append(r2_score(y_v, model.predict(x_v)))
"""

# ------------------------------------------------------------------------------
#                                 TRAIN MODULE
# ------------------------------------------------------------------------------


class Train():
    """train for SMNET"""

    def __init__(self, inputs):
        # config
        self.config = parse_args()
        # generate callback for model
        self.gen_callback()
        # train and save
        log = self.train(inputs)
        # save
        save2pickle(log,
                    out_path=self.config.path_outputs,
                    out_file=str(self.config.num_jobs)+".pickle")  # + '_'+str(self.config.batch_size) + ".pickle")

    def gen_callback(self):

        # mkdir
        if not os.path.exists(self.config.path_log):
            os.mkdir(self.config.path_log)

        # output file path
        checkpoint_path = self.config.path_log+str(self.config.num_jobs) + \
            '/'+str(self.config.num_jobs)+'.ckpt'

        # Tensorboard, earlystopping, Modelcheckpoint
        self.callbacks = [
            # tf.keras.callbacks.TensorBoard(self.config.path_log),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                save_weights_only=True)]#, monitor='mse')]  # ,
        # tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

    def train(self, inputs):

        print('[SMNET][MODEL] building SMNET \n')

        # model
        self.model = smnet()
        # compile
        self.model.compile(optimizer=tf.keras.optimizers.Adam
                           (self.config.learning_rate),
                           loss=self.config.loss,
                           metrics=self.config.metrics)

        print('[SMNET][TRAIN] training SMNET \n')

        # fit
        self.model.fit(inputs['x_train'][:,:,12:20,12:20,:], inputs['y_train'][:,-1],
                       batch_size=self.config.batch_size,
                       epochs=self.config.epoch,
                       validation_split=self.config.split_ratio,
                       callbacks=self.callbacks)

        model = smnet()
        model.load_weights(self.config.path_log+str(self.config.num_jobs) +
                           '/'+str(self.config.num_jobs)+'.ckpt')

        pred_train = model.predict(inputs['x_train'][:,:,12:20,12:20,:])
        pred_valid = model.predict(inputs['x_valid'][:,:,12:20,12:20,:])

        print('[SMNET][SAVE] saving SMNET \n')

        log = dict()
        log['y_train'] = inputs['y_train'][:,-1]
        log['y_valid'] = inputs['y_valid'][:,-1]
        log['pred_train'] = pred_train
        log['pred_valid'] = pred_valid

        return log


def trainL(inputs):
    """Train keras LSTM"""
    config = parse_args()
    model = lstm()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=config.loss,
        metrics=config.metrics
    )
    history = model.fit(inputs['x_train'], inputs['y_train'],
                        batch_size=config.batch_size,
                        epochs=config.epoch,)
                        #validation_split=config.split_ratio)
    pred_valid = model.predict(inputs['x_valid'])
    log = dict()
    log['y_valid'] = inputs['y_valid']#.reshape(-1,8,8)
    log['pred_valid'] = pred_valid#.reshape(-1,8,8)
    return log


if __name__ == "__main__":
    smnet()

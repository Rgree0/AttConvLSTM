# coding: utf-8
# pylint:
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn

import datetime as dt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
from utils import parse_args, save2pickle
np.random.seed(1)


def _check_4_dimensions(obj, obj_name='obj'):
    """Check obj is not tuple and have four dimensions"""
    if obj.ndim != 4:
        raise TypeError('%s must be 4 dimensions numpy matrix.' % obj_name)


def _print_4_dimensions(obj, obj_name='obj'):
    """Print 4 dimension length of obj"""
    print('The dimensions of %s \
           timestep is:%d, latitude is:%d, longitude is:%d, feature is:%d'
          % (obj_name, obj.shape[0], obj.shape[1], obj.shape[2], obj.shape[3]))


def _check_all_nan(obj, obj_name='obj'):
    """Check all nan in numpy matrix"""
    if np.sum(np.isnan(obj)) == obj.size:
        raise TypeError('Elements in %s all is nan.' % obj_name)


def _check_numpy(obj):
    """Check obj is numpy matrix"""
    if type(obj) != np.ndarray:
        obj = np.array(obj)
    return obj


def load():
    """Load SMAP_L4 data and concat as inputs"""
    # config
    config = parse_args()
    # load
    s = np.load(config.path_rawinputs + 'SM_US.npy')
    p = np.load(config.path_rawinputs + 'P_US.npy')
    t = np.load(config.path_rawinputs + 'ST_US.npy')

    """
    # init
    s_ = np.full((s.shape[0] // 4, s.shape[1], s.shape[2], 1), np.nan)
    p_ = np.full((s.shape[0] // 4, s.shape[1], s.shape[2], 1), np.nan)
    t_ = np.full((s.shape[0] // 4, s.shape[1], s.shape[2], 1), np.nan)

    for i in range(s.shape[0] // 4):
        s_[i, :, :, :] = np.nanmean(s[i * 4:i * 4 + 4, :, :, :], axis=0)
        p_[i, :, :, :] = np.nanmean(p[i * 4:i * 4 + 4, :, :, :], axis=0)
        t_[i, :, :, :] = np.nanmean(t[i * 4:i * 4 + 4, :, :, :], axis=0)

    np.save(config.path_rawinputs+'SM_US_12HH.npy', s_)
    np.save(config.path_rawinputs+'P_US_12HH.npy', p_)
    np.save(config.path_rawinputs+'ST_US_12HH.npy', t_)

    # init
    s_ = np.full((s.shape[0] // 2, s.shape[1], s.shape[2], 1), np.nan)
    p_ = np.full((s.shape[0] // 2, s.shape[1], s.shape[2], 1), np.nan)
    t_ = np.full((s.shape[0] // 2, s.shape[1], s.shape[2], 1), np.nan)

    for i in range(s.shape[0] // 2):
        s_[i, :, :, :] = np.nanmean(s[i * 2:i * 2 + 2, :, :, :], axis=0)
        p_[i, :, :, :] = np.nanmean(p[i * 2:i * 2 + 2, :, :, :], axis=0)
        t_[i, :, :, :] = np.nanmean(t[i * 2:i * 2 + 2, :, :, :], axis=0)

    np.save(config.path_rawinputs+'SM_US_6HH.npy', s_)
    np.save(config.path_rawinputs+'P_US_6HH.npy', p_)
    np.save(config.path_rawinputs+'ST_US_6HH.npy', t_)
    """
    inputs = np.concatenate((s_, p_, t_), axis=-1)
    # check
    _check_4_dimensions(inputs, obj_name='inputs')
    _print_4_dimensions(inputs, obj_name='inputs')
    _check_all_nan(inputs, obj_name='inputs')
    inputs = _check_numpy(inputs)
    # clean memory cache
    del s, p, t  # , s_, p_, t_
    return inputs


# ------------------------------------------------------------------------------
#                     Time variables generating module                       
# ------------------------------------------------------------------------------

class Time():
    """Generate time terms, including month, year, hour and fourier series"""

    def __init__(self,  begin_date, end_date):
        """input

        Parameters
        ----------
        begin_date, end date: str, optional
            type as 'YYYY-MM-DD'
        resolution: str, optional
            'DD': daily data
            '3HH': 3 hourly data
            'HH': hourly

        Attributes
        ----------
        time: float
            array-like [timestep, 1, 1, 1]
        """
        self.begin_date = begin_date
        self.end_date = end_date

    def gen_dates_jd_fourier(self, NANNUAL=6, NSEASON=5):

        # ----------------------------dates-------------------------------------
        # Initialize the list from begin_date to end_date
        dates = []
        # Initialize the timeindex for append in dates array.
        _dates = dt.datetime.strptime(self.begin_date, "%Y-%m-%d")
        # initialized the timeindex for decide whether break loop
        _date = self.begin_date[:]
        # main loop
        while _date <= self.end_date:
            # pass date in the array
            dates.append(_dates)
            # refresh date by step 1
            _dates = _dates + dt.timedelta(1)
            # changed condition by step 1
            _date = _dates.strftime("%Y-%m-%d")
        # generate dates
        _dates = []
        for idx, i in enumerate(dates):
            for j in range(8):
                _dates.append(i)
        # ----------------------------jd----------------------------------------
        # changed DatatimeIndex array in time format
        _dates = pd.to_datetime(_dates, format='%Y-%m-%d')
        # create DatatimeIndex array of 1st day of each year
        new_year_day = pd.to_datetime([pd.Timestamp(year=i, month=1, day=1)
                                       for i in _dates.year])
        jd = np.array(list((_dates - new_year_day).days + 1))
        # ----------------------------fourier-----------------------------------
        # get
        N = np.arange(1, len(jd)+1)
        # init
        sin_annual_series = np.full((len(N), NANNUAL), np.nan)
        cos_annual_series = np.full((len(N), NANNUAL), np.nan)
        sin_seasonal_series = np.full((len(N), NSEASON), np.nan)
        cos_seasonal_series = np.full((len(N), NSEASON), np.nan)
        # run
        for i in range(NANNUAL):
            sin_annual_series[:, i] = np.sin(
                (2*N*(i + 1)*math.pi) / len(jd))
            cos_annual_series[:, i] = np.cos(
                (2*N*(i + 1)*math.pi) / len(jd))
        for j in range(NSEASON):
            sin_seasonal_series[:, j] = np.sin(
                2 * (jd) * (j + 1) * math.pi / 365)
            cos_seasonal_series[:, j] = np.cos(
                2*(jd)*(j + 1)*math.pi / 365)
        # concat
        fourier = np.concatenate([sin_annual_series,
                                  cos_annual_series,
                                  sin_seasonal_series,
                                  cos_seasonal_series], axis=1)
        return _dates, jd, fourier

    def __call__(self):
        #
        dates, jd, fourier = self.gen_dates_jd_fourier()
        return dates, jd, fourier


class Preprocessing():
    """a class for preprocessing data."""

    def __init__(self, data):
        """inputs.

        Parameters
        ----------
        data: float, must be 4 dimension.
        config: dict, parser
            config contain all hyperparmaters for SMNET, includes:
            0. train_valid_ratio: float
                split ratio of train and valid dataset.
            1. num_height: int
                the number of inputs height
            2. num_width: int
                the number of inputs width

        .. notes::
            For soil moisture prediction task, inputs must be 4 dimensions
            and array-like of shape = [time, lat, lon, feature]

        Attributes
        ----------
        inputs: float
            The inputs of model
        outputs: float
            The target of model
        """
        _check_4_dimensions(data, obj_name='inputs')
        self.data = data
        self.config = parse_args()
        self.Nt, self.Nlat, self.Nlon, self.Nfeature = data.shape

    def __str__(self):
        return "inputs"

    def train_valid_seq_split(self, data, train_valid_ratio):
        """Split data into train and valid dataset by order"""
        # check
        _check_4_dimensions(data, obj_name='inputs')
        # get train sets length
        train, valid = train_test_split(data,
                                        test_size=train_valid_ratio,
                                        shuffle=False)
        return train, valid

    def normalization(self, train, valid, num_lat, num_lon):
        """Normalization data using MinMaxScaler

        .. Notes:: Instead normalize on the whole data, we first normalize
                   train data, and then normalize valid data by traind scaler
        """
        # check
        _check_4_dimensions(train, obj_name='train')
        _check_4_dimensions(train, obj_name='valid')
        # scaler class
        scaler = MinMaxScaler()

        # TODO:
        data = np.concatenate((train, valid), axis=0)

        # normalization
        for lat in range(num_lat):
            for lon in range(num_lon):
                if np.isnan(train[:, lat, lon, :]).any():
                    pass
                    # print('This pixel is all NaN.')
                else:
                    data[:, lat, lon, :] = scaler.fit_transform(
                        data[:, lat, lon, :])
                    """
                    train[:, lat, lon, :] = scaler.fit_transform(
                        train[:, lat, lon, :])
                    valid[:, lat, lon, :] = scaler.transform(
                        valid[:, lat, lon, :])
                    """
        # concat
        # data = np.concatenate((train, valid), axis=0)
        return data

    def avg_interplot(self, data, num_lat, num_lon, num_height, num_width):
        """Interplot data with average value of region

        .. Notes:: We interplot nan value with average of this region.
        """
        # check
        _check_4_dimensions(data, obj_name='inputs')

        for i in range(0, num_lat - num_height + 1, num_height):
            for j in range(0, num_lon - num_width + 1, num_width):
                for n in range(data.shape[-1]):

                    # cropping target data
                    crop = data[:, i: i + num_height, j: j + num_width, n]
                    mean_input = np.mean(crop, axis=0)

                    if np.sum(np.isnan(mean_input)) == \
                            num_height * num_width:
                        pass
                        # print('This pixel is all NaN.')
                    else:
                        # index for nan
                        index = np.isnan(mean_input)

                        for k in range(num_height):
                            for l in range(num_width):

                                if index[k, l]:
                                    data[:, i + k:i + k + 1,
                                         j + l: j + l + 1, n] = \
                                        np.nanmean(crop, axis=(
                                            1, 2), keepdims=True)
        return data

    def __call__(self):
        """Exec preprocessing process

        .. rubic:: process loop
                   0. fill default value to nan
                   1. split data by order
                   2. normalization
                   3. interplot
                   4. seperate into inputs and outputs
        """
        # turn fill value to nan
        self.data[self.data == self.config.fillvalue] = np.nan
        # split data
        train, valid = self.train_valid_seq_split(
            data=self.data,
            train_valid_ratio=self.config.train_test_ratio)
        """normalization, plot, split"""
        # normalization
        data = self.normalization(train=train, valid=valid,
                                  num_lat=self.Nlat, num_lon=self.Nlon)
        # interplot
        inputs = self.avg_interplot(data=data,
                                    num_lat=self.Nlat, num_lon=self.Nlon,
                                    num_height=self.config.height_inputs,
                                    num_width=self.config.width_inputs)
        outputs = inputs[:, :, :, 0][:, :, :, np.newaxis]
        # outputs = self.data[:, :, :, 0][:, :, :, np.newaxis]
        # outputs = self.avg_interplot(data=outputs,
        #                             num_lat=self.Nlat, num_lon=self.Nlon,
        #                             num_height=self.config.height_inputs,
        #                             num_width=self.config.width_inputs)

        return inputs, outputs


class Generate_grid():
    """A class for generating inputs of grid data"""

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.config = parse_args()
        self.Nt, self.Nlat, self.Nlon, self.Nfeature = inputs.shape

    def gen_lagged_series(self, inputs, len_inputs=10, window_size=8):
        """Generate lagged array"""
        # init
        lag = np.full((self.Nt, self.Nlat, self.Nlon, len_inputs), np.nan)
        # generate
        for i, window_idx in enumerate(range(window_size, window_size+len_inputs)):
            res_size = self.Nt - window_idx
            lag[:, :, :, i] = np.concatenate(
                (inputs[res_size:, :, :, 0], inputs[:res_size, :, :, 0]), axis=0)
        # exclude first sm and add lagged series
        inputs = np.concatenate((inputs[:, :, :, 1:], lag), axis=-1)
        return inputs

    def gen_time_series(self, inputs):
        dates, jd, fourier = Time(begin_date='2015-03-31',
                                  end_date='2019-01-24')()

        period = np.concatenate(
            (dates.year[:, np.newaxis], dates.month[:, np.newaxis],
             dates.day[:, np.newaxis], fourier), axis=-1)[:, np.newaxis, np.newaxis, :]
        period = np.tile(period, (1, inputs.shape[1], inputs.shape[2], 1))
        inputs = np.concatenate((inputs, period), axis=-1)
        return inputs

    def __call__(self):
        # config
        config = parse_args()
        # add lagged series
        inputs = self.gen_lagged_series(self.inputs)
        # add time series
        inputs = self.gen_time_series(inputs)
        # inputs
        x_train, x_valid = Preprocessing(inputs).train_valid_seq_split(
            inputs, config.train_test_ratio)
        y_train, y_valid = Preprocessing(self.outputs).train_valid_seq_split(
            self.outputs, config.train_test_ratio)
        # package into a dict
        data = {}
        data['x_train'] = x_train
        data['x_valid'] = x_valid
        data['y_train'] = y_train
        data['y_valid'] = y_valid
        return data


class Generate_lstm():
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.config = parse_args()
        self.Nt, self.Nlat, self.Nlon, self.Nfeature = inputs.shape

    def gen_time_series(self, inputs):
        dates, jd, fourier = Time(begin_date='2015-03-31',
                                  end_date='2019-01-24')()
        period = np.concatenate(
            (dates.day[:, np.newaxis]), axis=-1)[:, np.newaxis, np.newaxis, :]
        period = fourier[:, 0][:, np.newaxis, np.newaxis, np.newaxis]
        period = np.tile(period, (1, inputs.shape[1], inputs.shape[2], 1))
        print(period.shape)
        print(inputs.shape)
        inputs = np.concatenate((inputs, period), axis=-1)
        return inputs

    def train_valid_split(self, inputs, outputs,
                          train_valid_ratio, num_samples,
                          len_input, window_size):
        """Split train & valid dataset

        .. Notes:: We include the last time of train set into valid set, to
                   ensure that the model can predict the first timestep in
                   valid set.
        """
        # get train & valid index
        index_train = round(train_valid_ratio * num_samples)
        index_valid = index_train-(len_input+window_size)
        # generate train and valid
        inputs_train = inputs[:index_train, :, :, :]
        inputs_valid = inputs[index_valid:, :, :, :]
        outputs_train = outputs[:index_train, :, :, :]
        outputs_valid = outputs[index_valid:, :, :, :]
        return inputs_train, inputs_valid, outputs_train, outputs_valid

    def generate(self, inputs, outputs,
                 len_input, len_output, window_size):
        """Generate inputs and outputs for SMNET."""
        # caculate the last time point to generate batch
        end_idx = inputs.shape[0] - len_input - len_output - window_size
        # generate index of batch start point in order
        batch_start_idx = range(end_idx)
        # get batch_size
        batch_size = len(batch_start_idx)
        # generate inputs
        input_batch_idx = [
            (range(i, i + len_input)) for i in batch_start_idx]
        inputs = np.take(inputs, input_batch_idx, axis=0). \
            reshape(batch_size, len_input,
                    inputs.shape[1], inputs.shape[2], inputs.shape[3])
        # generate outputs
        output_batch_idx = [
            (range(i + len_input + window_size, i + len_input + window_size +
                   len_output)) for i in batch_start_idx]
        outputs = np.take(outputs, output_batch_idx, axis=0). \
            reshape(batch_size,  len_output,
                    outputs.shape[1], outputs.shape[2], 1)
        return inputs, outputs

    def __call__(self):
        # add time
        inputs = self.gen_time_series(self.inputs)

        # split data
        inputs_train, inputs_valid, outputs_train, outputs_valid = \
            self.train_valid_split(inputs=inputs,
                                   outputs=self.outputs,
                                   train_valid_ratio=0.8,
                                   num_samples=self.Nt,
                                   len_input=self.config.len_inputs,
                                   window_size=6)
        # generate
        x_train, y_train = self.generate(inputs=inputs_train,
                                         outputs=outputs_train,
                                         len_input=self.config.len_inputs,
                                         len_output=1,
                                         window_size=6)
        x_valid, y_valid = self.generate(inputs=inputs_valid,
                                         outputs=outputs_valid,
                                         len_input=self.config.len_inputs,
                                         len_output=1,
                                         window_size=6)
        print(x_train.shape)
        print(y_valid.shape)
        # package into a dict
        data = {}
        data['x_train'] = x_train
        data['x_valid'] = x_valid
        data['y_train'] = y_train
        data['y_valid'] = y_valid
        return data


class Generate_patch():
    """a class for generating inputs & outputs for SMNET"""

    def __init__(self, inputs, outputs):
        """inputs and outputs

        Parameters
        ----------
        inputs, outputs: float, ndarray
            array-like of shape = [samples, lat,lon, feature]
        config: dict, parser
            config contain all hyperparmaters for SMNET, includes:
            0. train_valid_ratio: float
                split ratio of train and valid dataset.
            1. num_height: int
                the number of inputs height
            2. num_width: int
                the number of inputs width
            3. len_input: int
                the number of input len
            4. len_output: int
                the number of output len
            5. window_size: int
                the number of window length between input and output

        .. notes::
            For soil moisture prediction task, inputs must be 4 dimensions
            and array-like of shape = [time, lat, lon, feature]

        Attributes
        ----------
        data: dict.

        'x_train' & 'x_valid': float.
            train dataset of shape = [samples, in-step, height, wigth, feature]
        'y_train' & 'y_valid':
            train dataset of shape = [samples, out-step, height, wigth, 1]
        """
        self.inputs = inputs
        self.outputs = outputs
        self.config = parse_args()
        self.Nt, self.Nlat, self.Nlon, self.Nfeature = inputs.shape

    def train_valid_split(self, inputs, outputs,
                          train_valid_ratio, num_samples,
                          len_input, window_size):
        """Split train & valid dataset

        .. Notes:: We include the last time of train set into valid set, to
                   ensure that the model can predict the first timestep in
                   valid set.
        """
        # get train & valid index
        index_train = round(train_valid_ratio * num_samples)
        index_valid = index_train-(len_input+window_size)
        # generate train and valid
        inputs_train = inputs[:index_train, :, :, :]
        inputs_valid = inputs[index_valid:, :, :, :]
        outputs_train = outputs[:index_train, :, :, :]
        outputs_valid = outputs[index_valid:, :, :, :]
        return inputs_train, inputs_valid, outputs_train, outputs_valid

    def generate(self, inputs, outputs,
                 len_input, len_output, window_size):
        """Generate inputs and outputs for SMNET."""
        # caculate the last time point to generate batch
        end_idx = inputs.shape[0] - len_input - len_output - window_size
        # generate index of batch start point in order
        batch_start_idx = range(end_idx)
        # get batch_size
        batch_size = len(batch_start_idx)
        # generate inputs
        input_batch_idx = [
            (range(i, i + len_input)) for i in batch_start_idx]
        inputs = np.take(inputs, input_batch_idx, axis=0). \
            reshape(batch_size, len_input,
                    inputs.shape[1], inputs.shape[2], inputs.shape[3])
        # generate outputs
        output_batch_idx = [
            (range(i + len_input + window_size, i + len_input + window_size +
                   len_output)) for i in batch_start_idx]
        outputs = np.take(outputs, output_batch_idx, axis=0). \
            reshape(batch_size,  len_output,
                    outputs.shape[1], outputs.shape[2], 1)
        return inputs, outputs

    def __call__(self):
        """Exec generate inputs and labels for SMNET

        .. rubic:: process loop
                   0. split data
                   1. generate specific dataset for SMNET
                   2. cropping dataset
                   3. package into a dict
        """
        # split data
        inputs_train, inputs_valid, outputs_train, outputs_valid = \
            self.train_valid_split(inputs=self.inputs,
                                   outputs=self.outputs,
                                   train_valid_ratio=0.8,
                                   num_samples=self.Nt,
                                   len_input=self.config.len_inputs,
                                   window_size=self.config.window_size)
        # generate
        x_train, y_train = self.generate(inputs=inputs_train,
                                         outputs=outputs_train,
                                         len_input=self.config.len_inputs,
                                         len_output=self.config.len_outputs,
                                         window_size=self.config.window_size)
        x_valid, y_valid = self.generate(inputs=inputs_valid,
                                         outputs=outputs_valid,
                                         len_input=self.config.len_inputs,
                                         len_output=self.config.len_outputs,
                                         window_size=self.config.window_size)
        # crop
        y_train = y_train[:, :,
                          int(self.config.height_inputs / 2) - 4:
                          int(self.config.height_inputs / 2) + 4,
                          int(self.config.width_inputs / 2) - 4:
                          int(self.config.width_inputs / 2) + 4,
                          :]
        y_valid = y_valid[:, :,
                          int(self.config.height_inputs / 2) - 4:
                          int(self.config.height_inputs / 2) + 4,
                          int(self.config.width_inputs / 2) - 4:
                          int(self.config.width_inputs / 2) + 4,
                          :]
        # package into a dict
        data = {}
        data['x_train'] = x_train
        data['x_valid'] = x_valid
        data['y_train'] = y_train
        data['y_valid'] = y_valid
        return data


def tasks(num_lat, num_lon, interval):
    """Generate begin index of lat & lon"""
    # config
    config = parse_args()
    lat = np.arange(0, num_lat-1, interval)
    lon = np.arange(0, num_lon-1, interval)

    region = []
    for i in range(len(lat)):
        for j in range(len(lon)):
            if lat[i] + config.height_inputs <= num_lat \
                    and (lon[j] + config.width_inputs <= num_lon):
                region.append([int(lat[i]), int(lon[j])])
    return region


def inverse(pred, true):
    """Inverse prediction array with true array"""
    # scaler
    scaler = MinMaxScaler()
    # inverse
    for i in range(true.shape[1]):
        for j in range(true.shape[2]):
            if (np.isnan(true[:, i, j, :]).any()) or (np.isnan(pred[:, i, j, :]).any()):
                print('This pixel ')
            else:
                scaler.fit_transform(true[:, i, j, :])
                pred[:, i, j, :] = scaler.inverse_transform(pred[:, i, j, :])
    return pred


def run_grid():
    """Exec grid data generating process

    .. rubic:: process
               0. load raw inputs
               1. cropping
    """
    def tasks_grid(num_lat, num_lon, interval):
        """Generate begin index of lat & lon"""
        # config
        config = parse_args()
        lat = np.arange(0, num_lat-1, interval)
        lon = np.arange(0, num_lon-1, interval)
        # generate region
        region = []
        for i in range(len(lat)):
            for j in range(len(lon)):
                region.append([int(lat[i]), int(lon[j])])
        return region
    # config
    config = parse_args()
    # load USA data
    sm = np.load(config.path_rawinputs+'/SM_US.npy')[:, 12:140, 12:292, :]
    p = np.load(config.path_rawinputs+'/P_US.npy')[:, 12:140, 12:292, :]
    st = np.load(config.path_rawinputs + '/ST_US.npy')[:, 12:140, 12:292, :]
    # set dims
    timesteps, lat, lon, _ = np.shape(sm)
    # concat input
    inputs_ = np.concatenate((sm, p, st), axis=-1)
    # region
    region = tasks_grid(np.shape(inputs_)[1], np.shape(inputs_)[2], 8)
    # loop
    for num_jobs in range(len(region)):
        # crop
        df = inputs_[:,
                     region[num_jobs][0]: region[num_jobs][0] + 8,
                     region[num_jobs][1]: region[num_jobs][1] + 8,
                     :]
        # preprocessing
        print('[ML][DATA] Preprocessing')
        inputs, outputs = Preprocessing(df)()
        # generate inputs
        print('[ML][DATA] Generating')
        data = Generate_grid(inputs, outputs)()
        # save
        print('[ML][DATA] Saving')
        save2pickle(data, config.path_inputs, str(num_jobs) + '.pickle')


def run_lstm():

    def tasks_grid(num_lat, num_lon, interval):
        """Generate begin index of lat & lon"""
        # config
        config = parse_args()
        lat = np.arange(0, num_lat-1, interval)
        lon = np.arange(0, num_lon-1, interval)
        # generate region
        region = []
        for i in range(len(lat)):
            for j in range(len(lon)):
                region.append([int(lat[i]), int(lon[j])])
        return region
    # config
    config = parse_args()
    # load USA data
    sm = np.load(config.path_rawinputs+'/SM_US.npy')[:, 12:140, 12:292, :]
    p = np.load(config.path_rawinputs+'/P_US.npy')[:, 12:140, 12:292, :]
    st = np.load(config.path_rawinputs + '/ST_US.npy')[:, 12:140, 12:292, :]
    # set dims
    timesteps, lat, lon, _ = np.shape(sm)
    # concat input
    inputs_ = np.concatenate((sm, p, st), axis=-1)
    # region
    region = tasks_grid(np.shape(inputs_)[1], np.shape(inputs_)[2], 8)
    # loop
    for num_jobs in range(len(region)):
            # crop
        df = inputs_[:,
                     region[num_jobs][0]: region[num_jobs][0] + 8,
                     region[num_jobs][1]: region[num_jobs][1] + 8,
                     :]
        print(df.shape)
        # preprocessing
        print('[ML][DATA] Preprocessing')
        inputs, outputs = Preprocessing(df)()
        print(inputs.shape)
        # generate inputs
        print('[ML][DATA] Generating')
        data = Generate_lstm(inputs, outputs)()
        # save
        print('[ML][DATA] Saving')
        save2pickle(data, config.path_inputs, str(num_jobs) + '.pickle')


def run_patch():
    """Exec patch data generating process"""
    # config
    config = parse_args()
    # load data
    print('[SMNET][DATA] Loading')
    raw = load()
    # generate tasks lat/lon
    region = tasks(np.shape(raw)[1], np.shape(raw)[2], 8)
    print(len(region))

    if config.gen_patch_data == 0:
        data_range = range(len(region) // 4 * 3)
        # data_range = range(100, 101)

    elif config.gen_patch_data == 1:
        data_range = range(len(region)//4*3, len(region))

    for num_jobs in data_range:
        # cropping
        df = raw[:,
                 region[num_jobs][0]: region[num_jobs][0] + config.height_inputs,
                 region[num_jobs][1]: region[num_jobs][1] + config.width_inputs,
                 :]
        # preprocessing
        print('[SMNET][DATA] Preprocessing')
        inputs, outputs = Preprocessing(df)()
        # generate inputs
        print('[SMNET][DATA] Generating')
        data = Generate_patch(inputs, outputs)()
        print(data['y_valid'].shape)
        print('[SMNET][DATA] Saving')
        save2pickle(data, config.path_inputs, str(num_jobs) + '.pickle')


if __name__ == "__main__":
    # run_patch()
    # run_grid()
    run_lstm()
    # dates, jd, fourier = Time(begin_date='2015-03-31',
    #                          end_date='2019-01-24')()
    # load()

"""
@author: Lu Li
@mail: <lilu35@mail2.sysu.edu.cn>
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import Preprocessing, inverse
from utils import KG, autocorr, parse_args


"""This part implement processing result get from parrell experiments from TH-2.
and generate corresponding figure file, which is used to plot final edition
figures.
    This file contain two main part, first is process the pickle file and
generate output.npz which contain prediction & target array. Second is generate
figure files, e.g., figure5.npz ... figure15.npz.
"""


class Process_grid():

    def __init__(self):
        self.config = parse_args()
        handle = open(self.config.path_outputs+str(0)+'.pickle', 'rb')
        outputs = pickle.load(handle)
        self.N, self.Nlat, self.Nlon, _ = outputs['pred_valid'].shape
        self.load(560)

    def load(self, num_jobs):
        # ----------------------------------------------------------------------
        #                  load & transform output files
        # ----------------------------------------------------------------------
        # initial
        pred = np.full((self.N, 16 * 8, 35 * 8), np.nan)
        target = np.full((self.N, 16 * 8, 35 * 8), np.nan)
        r2 = np.full((16 * 8, 35 * 8), np.nan)
        # init
        j = 0
        # concat all pixels
        for i in range(num_jobs):
            # read pickle file and give empty file 0
            try:
                # read pickle
                handle = open(self.config.path_outputs +
                              str(i)+'.pickle', 'rb')
                outputs = pickle.load(handle)
            except:
                outputs = 0
            # reshape
            if outputs == 0:
                pred_valid = np.full((self.N, 8, 8, 3), np.nan)
                y_valid = np.full((self.N, 8, 8, 3), np.nan)
            else:
                # read pred, y [None,8,8,8]
                pred_valid = np.reshape(
                    outputs['pred_valid'], [-1, 8, 8, 3])
                y_valid = np.reshape(outputs['y_valid'], [-1, 8, 8, 3])
            # concat all figure in lat dimension
            pred[:, j * 8:j * 8 + 8, (i % 35) * 8:(i % 35) *
                 8 + 8] = np.squeeze(pred_valid[:, :, :, 2])
            target[:, j * 8:j * 8 + 8, (i % 35) * 8:(i % 35)
                   * 8 + 8] = np.squeeze(y_valid[:, :, :, 2])
            # exclude pixels with no prediction ability.
            r2_ = np.full((8, 8), np.nan)
            for m in range(8):
                for n in range(8):
                    if np.isnan(y_valid[:, m, n, 2]).all():
                        pass
                    else:
                        r2_[m, n] = r2_score(
                            y_valid[:,  m, n, 2], pred_valid[:, m, n, 2])
            # generate r2
            r2[j * 8:j * 8 + 8, (i % 35) * 8:(i % 35)
               * 8 + 8] = r2_
            # reset parameters
            if (i + 1) % 35 == 0 and i != 0:
                j += 1
        # exclude
        for i in range(target.shape[1]):
            for j in range(target.shape[2]):
                if r2[i, j] < 0:
                    pred[:, i, j] = np.full((self.N,), np.nan)

        np.savez(self.config.path_figures+'output',
                 pred, target)
        # inverse
        # load sm
        s = np.load(self.config.path_rawinputs +
                    'SM_US.npy')[:, 12:140, 12:292, :]
        s[s == -9999] = np.nan

        inverse_pred = np.squeeze(inverse(pred[:, :, :, np.newaxis], s))
        inverse_target = np.squeeze(s[-2234:, :, :, :])
        print(inverse_pred.shape)
        print(inverse_target.shape)
        np.savez(self.config.path_figures+'inverse_output',
                 inverse_pred, inverse_target)


class Process_lstm():

    def __init__(self):
        self.config = parse_args()
        handle = open(self.config.path_outputs+str(0)+'.pickle', 'rb')
        outputs = pickle.load(handle)
        self.N, _, self.Nlat, self.Nlon, _ = outputs['pred_valid'].shape
        self.load(560)

    def load(self, num_jobs):
        # ----------------------------------------------------------------------
        #                  load & transform output files
        # ----------------------------------------------------------------------
        # initial
        pred = np.full((self.N, 16 * 8, 35 * 8), np.nan)
        target = np.full((self.N, 16 * 8, 35 * 8), np.nan)
        r2 = np.full((16 * 8, 35 * 8), np.nan)

        # init
        j = 0
        # concat all pixels
        for i in range(num_jobs):
            # read pickle file and give empty file 0
            try:
                # read pickle
                handle = open(self.config.path_outputs +
                              str(i)+'.pickle', 'rb')
                outputs = pickle.load(handle)
            except:
                outputs = 0
            # reshape
            if outputs == 0:
                pred_valid = np.full((self.N, 1, 8, 8, 1), np.nan)
                y_valid = np.full((self.N, 1, 8, 8, 1), np.nan)
            else:
                # read pred, y [None,8,8,8]
                pred_valid = np.reshape(
                    outputs['pred_valid'], [-1, 1, 8, 8, 1])
                y_valid = np.reshape(outputs['y_valid'], [-1, 1, 8, 8, 1])
            # concat all figure in lat dimension
            pred[:, j * 8:j * 8 + 8, (i % 35) * 8:(i % 35) *
                 8 + 8] = np.squeeze(pred_valid[:, 0, :, :, 0])
            target[:, j * 8:j * 8 + 8, (i % 35) * 8:(i % 35)
                   * 8 + 8] = np.squeeze(y_valid[:, 0, :, :, 0])

            r2_ = np.full((8, 8), np.nan)
            for m in range(8):
                for n in range(8):

                    if np.isnan(y_valid[:, 0, m, n, 0]).all():
                        pass
                    else:
                        r2_[m, n] = r2_score(
                            y_valid[:, 0, m, n, 0], pred_valid[:, 0, m, n, 0])

            r2[j * 8:j * 8 + 8, (i % 35) * 8:(i % 35)
               * 8 + 8] = r2_
            # reset parameters
            if (i + 1) % 35 == 0 and i != 0:
                j += 1

        for i in range(target.shape[1]):
            for j in range(target.shape[2]):
                if r2[i, j] < 0:
                    pred[:, i, j] = np.full((self.N, ), np.nan)

        np.savez(self.config.path_figures+'output',
                 pred, target)

        s = np.load(self.config.path_rawinputs +
                    'SM_US.npy')[:, 12:140, 12:292, :]
        s[s == -9999] = np.nan

        inverse_pred = np.squeeze(inverse(pred[:, :, :, np.newaxis], s))
        inverse_target = np.squeeze(s[-2233:, :, :, :])
        print(inverse_pred.shape)
        print(inverse_target.shape)

        # save
        np.savez(self.config.path_figures+'inverse_output',
                 inverse_pred, inverse_target)


class Process_patch():
    """Process for output pickle files from SMNET."""

    def __init__(self):
        """Describe matrix & initial parameters
        """
        self.config = parse_args()
        handle = open(self.config.path_outputs+str(0)+'.pickle', 'rb')
        outputs = pickle.load(handle)
        self.N, _, self.Nlat, self.Nlon, _ = outputs['pred_valid'].shape
        self.load(self.config.num_jobs)

    def load(self, num_jobs):
        """Loading all output files & generate prediction & target matrix
        """
        # ----------------------------------------------------------------------
        #                  load & transform output files
        # ----------------------------------------------------------------------
        # initial
        pred = np.full((self.N, 8, 16 * 8, 35 * 8), np.nan)
        target = np.full((self.N, 8, 16 * 8, 35 * 8), np.nan)
        r2 = np.full((16 * 8, 35 * 8), np.nan)

        # init
        j = 0
        # concat all pixels
        for i in range(num_jobs):
            # read pickle file and give empty file 0
            try:
                # read pickle
                handle = open(self.config.path_outputs +
                              str(i)+'.pickle', 'rb')
                outputs = pickle.load(handle)
            except:
                outputs = 0
            # reshape
            if outputs == 0:
                pred_valid = np.full((self.N, 8, 8, 8, 1), np.nan)
                y_valid = np.full((self.N, 8, 8, 8, 1), np.nan)
            else:
                # read pred, y [None,8,8,8]
                pred_valid = np.reshape(
                    outputs['pred_valid'], [-1, 8, 8, 8, 1])
                y_valid = np.reshape(outputs['y_valid'], [-1, 8, 8, 8, 1])
            # concat all figure in lat dimension
            pred[:, :, j * 8:j * 8 + 8, (i % 35) * 8:(i % 35) *
                 8 + 8] = np.squeeze(pred_valid[:, :, :, :, 0])
            target[:, :, j * 8:j * 8 + 8, (i % 35) * 8:(i % 35)
                   * 8 + 8] = np.squeeze(y_valid[:, :, :, :, 0])

            r2_ = np.full((8, 8), np.nan)
            for m in range(8):
                for n in range(8):
                    if np.isnan(y_valid[:, 0, m, n, 0]).all():
                        pass
                    else:
                        r2_[m, n] = r2_score(
                            y_valid[:, 0, m, n, 0], pred_valid[:, 0, m, n, 0])

            r2[j * 8:j * 8 + 8, (i % 35) * 8:(i % 35)
               * 8 + 8] = r2_

            # reset parameters
            if (i + 1) % 35 == 0 and i != 0:
                j += 1
        # mean the timestep
        pred = np.squeeze(np.nanmean(pred, axis=1))
        target = np.squeeze(np.nanmean(target, axis=1))
        """
        for i in range(target.shape[1]):
            for j in range(target.shape[2]):
                if r2[i, j] < 0:
                    pred[:, i, j] = np.full((self.N, ), np.nan)
        """
        np.savez(self.config.path_figures+'output', pred, target)

        # ----------------------------------------------------------------------
        #                        inverse output files
        # ----------------------------------------------------------------------
        # load sm
        if config.case == 0:
            s = np.load(self.config.path_rawinputs +
                        'SM_US.npy')[:, 12:140, 12:292, :]
            s[s == -9999] = np.nan
        elif config.case == 1:
            s = np.load(self.config.path_rawinputs +
                        'SM_US_6HH.npy')[:, 12:140, 12:292, :]
            s[s == -9999] = np.nan
        elif config.case == 2:
            s = np.load(self.config.path_rawinputs +
                        'SM_US_12HH.npy')[:, 12:140, 12:292, :]
            s[s == -9999] = np.nan

        # inverse pred
        inverse_pred = np.squeeze(
            inverse(pred[:, :, :, np.newaxis], s))
        # init and generate inverse target
        inverse_target = np.full((self.N, 8, s.shape[1], s.shape[2]), np.nan)
        for i in range(8):
            inverse_target[:, 0, :, :] = np.squeeze(
                s[-(self.N+8):-8, :, :, :])
            inverse_target[:, 1, :, :] = np.squeeze(
                s[-(self.N+7):-7, :, :, :])
            inverse_target[:, 2, :, :] = np.squeeze(
                s[-(self.N+6):-6, :, :, :])
            inverse_target[:, 3, :, :] = np.squeeze(
                s[-(self.N+5):-5, :, :, :])
            inverse_target[:, 4, :, :] = np.squeeze(
                s[-(self.N+4):-4, :, :, :])
            inverse_target[:, 5, :, :] = np.squeeze(
                s[-(self.N + 3):-3, :, :, :])
            inverse_target[:, 6, :, :] = np.squeeze(
                s[-(self.N + 2):-2, :, :, :])
            inverse_target[:, 7, :, :] = np.squeeze(
                s[-(self.N + 1):-1, :, :, :])
        # mean
        inverse_target = np.squeeze(np.mean(inverse_target, axis=1))
        # ----------------------------------------------------------------------
        #                       exclude bad pixels
        # ----------------------------------------------------------------------
        #
        index = np.isnan(np.mean(inverse_target, axis=0))

        for i in range(inverse_target.shape[1]):
            for j in range(inverse_target.shape[2]):
                if index[i, j]:
                    inverse_pred[:, i, j] = np.full((self.N,), np.nan)

        rmse_valid = np.full((s.shape[1], s.shape[2]), np.nan)

        for i in range(inverse_target.shape[1]):
            for j in range(inverse_target.shape[2]):
                if np.isnan(inverse_pred[:, i, j]).any():
                    pass
                else:
                    rmse_valid[i, j] = np.sqrt(mean_squared_error(
                        inverse_target[:, i, j], inverse_pred[:, i, j]))

        for i in range(target.shape[1]):
            for j in range(target.shape[2]):
                if rmse_valid[i, j] > 0.05:
                    inverse_pred[:, i, j] = np.full((self.N, ), np.nan)

        # clean board
        index = np.isnan(np.mean(inverse_target, axis=0))

        for i in range(inverse_target.shape[1]):
            for j in range(inverse_target.shape[2]):
                if index[i, j]:
                    inverse_pred[:, i, j] = np.full((self.N, ), np.nan)

        # ensure path figures
        if not os.path.exists(self.config.path_figures):
            os.mkdir(self.config.path_figures)

        # save
        np.savez(self.config.path_figures+'inverse_output',
                 inverse_pred, inverse_target)


class figures():
    def __init__(self, pred, target, inverse_pred, inverse_target):
        self.figureC(pred, target, inverse_pred, inverse_target)

        #self.figure5(pred, target, inverse_pred, inverse_target)
        #self.figure6(pred, target)
        #self.figure8(inverse_pred, inverse_target)
        #self.figure10(inverse_pred, inverse_target)
    def smooth_avg(self, inputs, WSZ=160, mode='valid'):
        out = np.convolve(inputs, np.ones(WSZ, dtype=int), mode=mode)/WSZ
        r = np.arange(1, WSZ-1, 2)
        start = np.cumsum(inputs[:WSZ-1])[::2]/r
        stop = (np.cumsum(inputs[:-WSZ:-1])[::2]/r)[::-1]
        return np.concatenate((start, out, stop))
    

    def msess(self, target, pred):

        ref = self.smooth_avg(target, WSZ=641) # must be odd number
        forecast_mse = mean_squared_error(target, pred)
        ref_mse = mean_squared_error(target,ref)
        msess = 1.0-(forecast_mse/ref_mse)
        print(msess)
        print(ref.shape)
        print(forecast_mse)
        print(ref_mse)

        return msess, forecast_mse, ref_mse 

    def figureC(self, pred, target, inverse_pred, inverse_output):

        config = parse_args()
        msess_ = np.full(
            (inverse_target.shape[1], inverse_target.shape[2]), np.nan)
        forecast_mse = np.full(
            (inverse_target.shape[1], inverse_target.shape[2]), np.nan)
        ref_mse = np.full(
            (inverse_target.shape[1], inverse_target.shape[2]), np.nan)
        
        for i in range(inverse_target.shape[1]):
            for j in range(inverse_target.shape[2]):
                if (np.isnan(inverse_pred[:, i, j]).any()) \
                        or (np.isnan(inverse_target[:, i, j]).any()):
                    pass
                else:
                    msess_[i, j], forecast_mse[i,j], ref_mse[i,j] = self.msess(
                        inverse_target[:, i, j], inverse_pred[:, i, j])
        # save
        np.savez(config.path_figures+'figureC', msess_, forecast_mse, ref_mse)

    def figure5(self, pred, target, inverse_pred, inverse_target):
        """Contain 4 figures shows performance of SMNET from several aspects.
           (a)target time mean
           (b)target-pred time mean
           (c)RMSE
           (d)R2
        """
        config = parse_args()
        # (a)
        target_spatial_mean = np.mean(inverse_target, axis=0)
        pred_spatial_mean = np.mean(inverse_pred, axis=0)
        # (b)
        d = target_spatial_mean - pred_spatial_mean
        # (c) & (d)
        rmse = np.full(
            (inverse_target.shape[1], inverse_target.shape[2]), np.nan)
        r2 = np.full(
            (inverse_target.shape[1], inverse_target.shape[2]), np.nan)
        for i in range(inverse_target.shape[1]):
            for j in range(inverse_target.shape[2]):
                if (np.isnan(inverse_pred[:, i, j]).any()) \
                        or (np.isnan(inverse_target[:, i, j]).any()):
                    pass
                else:
                    rmse[i, j] = np.sqrt(mean_squared_error(
                        inverse_target[:, i, j], inverse_pred[:, i, j]))

        for i in range(inverse_target.shape[1]):
            for j in range(inverse_target.shape[2]):
                if (np.isnan(pred[:, i, j]).any()) \
                        or (np.isnan(target[:, i, j]).any()):
                    pass
                else:
                    r2[i, j] = r2_score(
                        target[:, i, j], pred[:, i, j])
        # save
        np.savez(config.path_figures+'figure5',
                 target_spatial_mean, d, rmse, r2)

    def figure6(self, pred, target):
        """Contain 2 figures shows SMNET advantages.
           (a)multi models prediction spatial mean
           (b)boxplot of multi models
        """

        config = parse_args()
        s = np.load(config.path_rawinputs + 'SM_US.npy')
        s[s == -9999] = np.nan

        # clean board
        index = np.isnan(np.mean(s[:, :, :, 0], axis=0))
        for i in range(pred.shape[1]):
            for j in range(pred.shape[2]):
                if index[i, j]:
                    pred[:, i, j] = np.squeeze(
                        np.full((pred.shape[0], 1), np.nan))
                    target[:, i, j] = np.squeeze(
                        np.full((pred.shape[0], 1), np.nan))
        # scaler time series
        scaler_mean_pred = np.nanmean(pred, axis=(-1, -2))
        scaler_mean_target = np.nanmean(target, axis=(-1, -2))
        # save
        np.savez(config.path_figures+'figure6',
                 scaler_mean_pred, scaler_mean_target)

    def figure8(self, pred, target):
        """Scatter density plot of of different climate regions.
           Only choose 20 pixels to represent all pixels.
        """
        # generate climate regions and corresponding index
        data, index_KG = KG()
        # init
        pred_grids = np.full((pred.shape[0], 20, len(index_KG)), np.nan)
        target_grids = np.full((pred.shape[0], 20, len(index_KG)), np.nan)
        # loop for all climate regions
        for k in range(len(index_KG)):
            print(k)
            # if index KG have more than 20 pixels
            if len(index_KG[str(k)][0]) > 20:
                print('1')
                # selected time series of 20 points
                a = pred[:, index_KG[str(k)][0], index_KG[str(k)][1]]
                b = target[:, index_KG[str(k)][0], index_KG[str(k)][1]]
                #
                index = np.random.randint(0, a.shape[-1], 20)
                pred_grids[:, :, k] = a[:, index]
                target_grids[:, :, k] = b[:, index]
        # save
        np.savez(config.path_figures+'figure8',
                 pred_grids, target_grids)

    def figure10(self, pred, target):
        """Time correlation and spatial correlation.
           (a)time correlation of SMAP
           (b)different of time correlation of SMAP-SMNET
        """
        # (a) & (b)
        # pred = pred[230:3139, :, :]
        # target = target[230:3139, :, :]
        # init
        acf_target = np.full((pred.shape[1], pred.shape[2]), np.nan)
        acf_pred = np.full((pred.shape[1], pred.shape[2]), np.nan)
        # loop month
        for i in range(pred.shape[1]):
            for j in range(pred.shape[2]):
                if np.isnan(pred[:, i, j]).any() \
                        or np.isnan(target[:, i, j]).any():
                    pass
                else:
                    acf_target[i, j] = autocorr(target[:, i, j], t=8)[0, 1]
                    acf_pred[i, j] = autocorr(pred[:, i, j], t=8)[0, 1]
        # save
        np.savez(config.path_figures+'figure10',
                 acf_target, acf_pred)


if __name__ == "__main__":
    config = parse_args()
    # generate prediction & target
    # Process_grid()
    # Process_lstm()

    #Process_patch()
    # load prediction & target
    data = np.load(config.path_figures+'output.npz')
    pred = data['arr_0']
    target = data['arr_1']
    data = np.load(config.path_figures+'inverse_output.npz')
    # data = np.load(config.path_figures+'inverse_output.npz')
    inverse_pred = data['arr_0']
    inverse_target = data['arr_1']

    print(pred.shape)
    print(target.shape)
    print(inverse_pred.shape)
    print(inverse_target.shape)

    print(np.nansum(inverse_pred - pred))
    print(np.nansum(inverse_target-target))

    # generate figures files
    figures(pred, target, inverse_pred, inverse_target)


"""divide metrics by months and climate regions.


# load data
data = np.load('/WORK/sysu_yjdai_6/lilu/output.npz')
pred = data['arr_0']
target = data['arr_1']
# get size
timestep = pred.shape[0]
# generate month index
index = month(timestep)
# generate kG index
index_KG = KG()

# crop
pred = pred[230:3139, :, :]
target = target[230:3139, :, :]

# init
r2 = np.full((12, len(index_KG), pred.shape[1] * pred.shape[2]), np.nan)
acf_target = np.full(
    (12, len(index_KG), pred.shape[1] * pred.shape[2]), np.nan)
acf_pred = np.full(
    (12, len(index_KG), pred.shape[1]*pred.shape[2]), np.nan)
mse = np.full((12, len(index_KG), pred.shape[1]*pred.shape[2]), np.nan)
mae = np.full((12, len(index_KG), pred.shape[1] * pred.shape[2]), np.nan)
rmse = np.full((12, len(index_KG), pred.shape[1] * pred.shape[2]), np.nan)

# loop month
for i in range(12):
    # loop KG
    for k in range(len(index_KG)):
        # judge
        if len(index[str(i+1)]) != 0 and len(index_KG[str(k)]) != 0:
            # generate series of each month of each KG regions
            a = pred[index[str(i + 1)][0]:index[str(i + 1)][-1],
                        index_KG[str(k)][0], index_KG[str(k)][1]]
            b = target[index[str(i + 1)][0]:index[str(i + 1)][-1],
                        index_KG[str(k)][0], index_KG[str(k)][1]]
            print('have {} points'.format(a.shape[1]))
            print(np.sum(np.isnan(a)))
            print(np.sum(np.isnan(b)))
            # loop all points for each KG regions
            for l in range(a.shape[1]):
                # crop
                c = a[:, l]
                d = b[:, l]
                # judge
                if np.isnan(c).any() or np.isnan(d).any() or len(c) == 0:
                    pass
                else:
                    r2[i, k, l] = r2_score(d, c)

                    acf_target[i, k, l] = autocorr(d, t=8)[0, 1]
                    acf_pred[i, k, l] = autocorr(c, t=8)[0, 1]

                    mse[i, k, l] = mean_squared_error(d, c)
                    mae[i, k, l] = mean_absolute_error(d, c)
                    rmse[i, k, l] = np.sqrt(mean_absolute_error(d, c))

np.savez('figure13', r2, mse, mae, rmse, acf_target, acf_pred)
"""


"""precipitation & soil temperature divided by month and climate regions.
data = np.load('/WORK/sysu_yjdai_6/lilu/output.npz')
pred = data['arr_0']
target = data['arr_1']

index = month(pred.shape[0])
index_KG = KG()
p = np.full((12, len(index_KG)), np.nan)
st = np.full((12, len(index_KG)), np.nan)

p_ = np.squeeze(np.load(
    '/WORK/sysu_yjdai_6/lilu/inputs/raw/P_US.npy')[-pred.shape[0]:, :, :, 0])
st_ = np.squeeze(np.load(
    '/WORK/sysu_yjdai_6/lilu/inputs/raw/ST_US.npy')[-pred.shape[0]:, :, :, 0])
p_[p_ == -9999] = np.nan
st_[st_ == -9999] = np.nan
print(p_.shape)

for i in range(12):
    for k in range(len(index_KG)):
        print('Month {} have {} timepoints'.format(
            i, len(index[str(i + 1)])))
        print('Region {} have {} spatialpoints'.format(
            k, len(index_KG[str(k)][0])))
        print('----------')

        if len(index[str(i+1)]) != 0 and len(index_KG[str(k)]) != 0:

            a = p_[index[str(i+1)][0]:index[str(i+1)][-1],
                    index_KG[str(k)][0], index_KG[str(k)][1]]
            b = st_[index[str(i + 1)][0]:index[str(i + 1)]
                    [-1], index_KG[str(k)][0], index_KG[str(k)][1]]
            print('Selected shape is')
            print(a.shape)
            print('---------')

            p[i, k] = np.nanmean(a, axis=(0, 1))
            st[i, k] = np.nanmean(b, axis=(0, 1))

np.savez('figure12', p, st)
"""

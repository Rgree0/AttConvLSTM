# coding: utf-8
# pylint:
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn

import argparse
import colorsys as cs
import os
import pickle
import time

import h5py
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from fancyimpute import KNN
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fancyimpute import KNN

#from data import Time


def fillna(inputs):
    return pd.DataFrame(
        KNN(k=6).fit_transform(pd.DataFrame(inputs))).to_numpy()


def gen_region_index(series, target_range):
    return np.argwhere(
        [(series > target_range[0]) & (series < target_range[1])])[:, 1]


def print_log():
    """Basic info"""
    print('welcome to deep learning world \n')
    print('            _____     __  __     __     _     _______     _______  ')
    print('           / ____|   |  \/  |   | \ \  | |   |  _____|   |___ ___| ')
    print('           \  \      | \  / |   |  \ \ | |   | |_____       | |    ')
    print('            \  \     | |\/| |   | | \ \| |   |  _____|      | |    ')
    print('           __\  \    | |  | |   | |  \ \ |   | |_____       | |    ')
    print('          |_____/    |_|  |_|   |_|   \__|   |_______|      |_|    ')
    print('\n[SMNET][INFO] @author: Lu Li')
    print('[SMNET][INFO] @mail: lilu35@mail2.sysu.edu.cn \n')


def tictoc(func):
    """Time caculate"""
    def wrapper(*args, **kwargs):
        begin = time.time()
        log = func(*args, **kwargs)
        end = time.time()
        print('cost {} second'.format(end-begin))
        return log
    return wrapper


def save2pickle(data, out_path, out_file):
    """Save to pickle file"""
    # if not have output path, mkdir it
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # save to pickle
    handle = open(out_path + out_file, 'wb')
    pickle.dump(data, handle, protocol=4)
    handle.close()


def parse_args():
    """Hyperparameters

    Parameters
    ----------
    PATH:
    0. path_rawinputs: str
        path to load raw inputs
    1. path_inputs: str
        path to load inputs or save inputs
    2. path_outputs: str
        path to save outputs
    3. path_log: str
        path to save log, including best model etc.
    BLOCKS:
    0. downsampling: bool, optional, (default true)
        control downsampling module
    1. channel_attention: bool, optional, (default true)
        control channel attention module
    2. spatial_attention: bool, optional, (default true)
        control spatial attention module
    3. convlstm: bool, optional, (default False)
        control convlstm module:
        if true, exec covlstm module
        if false, exec encoder-decoder convlstm
    4. self_attention: bool, optional, (default true)
        control self attention module
    HYPERPARAMETERS:
    0. len_inputs: int, (default 10)
    1. height_inputs: int, (default 32)
    2. width_inputs: int, (default 32)
    3. channel_inputs: int, (default 3)
    4. len_outputs: int, (default 8)
    5. height_outputs: int, (default 8)
    6. width_outputs: int, (default 8)
    7. window_size: int, (default 1)
    8. fillvalue: float, (default -9999)
    9. train_test_ratio: float, (default 0.8)
    10. nums_input_attention: int, (default 1)
    11. nums_self_attention: int, (default 1)
    12. channel_dense_ratio: int, (default 1)
    13. spatial_kernel_size: int, (default 3)
    MODEL PARAMETERS:
    0. epoch, int, (default 1)
    1. batch_size, int, (default 100)
    2. loss, str, (default 'mse')
    3. learning_rate, float, (default 0.01)
    4. metrics, list, (default ['mae','mse'])
    5. split_ratio, float, (default 0.2)
    """
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--path_rawinputs', type=str, default='')
    parser.add_argument('--path_inputs', type=str, default='')
    parser.add_argument('--path_outputs', type=str, default='')
    parser.add_argument('--path_log', type=str,
                        default='/work/lilu/log/')
    parser.add_argument('--path_figures', type=str, default='')
    # data
    parser.add_argument('--gen_patch_data', type=int, default=0)
    parser.add_argument('--case', type=int, default=0)

    # blocks
    parser.add_argument('--downsampling', type=bool, default=True)
    parser.add_argument('--channel_attention', type=bool, default=True)
    parser.add_argument('--spatial_attention', type=bool, default=True)
    parser.add_argument('--convlstm', type=bool, default=False)
    parser.add_argument('--self_attention', type=bool, default=True)
    # hyperparameters
    parser.add_argument('--len_inputs', type=int, default=10)
    parser.add_argument('--height_inputs', type=int, default=32)
    parser.add_argument('--width_inputs', type=int, default=32)
    parser.add_argument('--channel_inputs', type=int, default=3)
    parser.add_argument('--len_outputs', type=int, default=8)
    parser.add_argument('--height_outputs', type=int, default=8)
    parser.add_argument('--width_outputs', type=int, default=8)
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--fillvalue', type=float, default=-9999)
    parser.add_argument('--train_test_ratio', type=float, default=0.2)
    parser.add_argument('--nums_input_attention', type=int, default=1)
    parser.add_argument('--nums_self_attention', type=int, default=1)
    parser.add_argument('--channel_dense_ratio', type=int, default=1)
    parser.add_argument('--spatial_kernel_size', type=int, default=3)
    # model paramters
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--metrics', type=list, default=['mae', 'mse'])
    parser.add_argument('--split_ratio', type=float, default=0.3)
    # parfor paramters
    parser.add_argument('--num_jobs', type=int, default=1)
    return parser.parse_args()


def KG():
    """Generate Koppen-Geiger index used to climate classification

    classes = {...
        1,1,'Af','Tropical, rainforest',[0 0 255];...
        2,1,'Am','Tropical, monsoon',[0 120 255];...
        3,1,'Aw','Tropical, savannah',[70 170 250];...
        4,2,'BWh','Arid, desert, hot',[255 0 0];...
        5,2,'BWk','Arid, desert, cold',[255 150 150];...
        6,2,'BSh','Arid, steppe, hot',[245 165 0];...
        7,2,'BSk','Arid, steppe, cold',[255 220 100];...
        8,3,'Csa','Temperate, dry summer, hot summer',[255 255 0];...
        9,3,'Csb','Temperate, dry summer, warm summer',[200 200 0];...
        10,3,'Csc','Temperate, dry summer, cold summer',[150 150 0];...
        11,3,'Cwa','Temperate, dry winter, hot summer',[150 255 150];...
        12,3,'Cwb','Temperate, dry winter, warm summer',[100 200 100];...
        13,3,'Cwc','Temperate, dry winter, cold summer',[50 150 50];...
        14,3,'Cfa','Temperate, no dry season, hot summer',[200 255 80];...
        15,3,'Cfb','Temperate, no dry season, warm summer',[100 255 80];...
        16,3,'Cfc','Temperate, no dry season, cold summer',[50 200 0];...
        17,4,'Dsa','Cold, dry summer, hot summer',[255 0 255];...
        18,4,'Dsb','Cold, dry summer, warm summer',[200 0 200];...
        19,4,'Dsc','Cold, dry summer, cold summer',[150 50 150];...
        20,4,'Dsd','Cold, dry summer, very cold winter',[150 100 150];...
        21,4,'Dwa','Cold, dry winter, hot summer',[170 175 255];...
        22,4,'Dwb','Cold, dry winter, warm summer',[90 120 220];...
        23,4,'Dwc','Cold, dry winter, cold summer',[75 80 180];...
        24,4,'Dwd','Cold, dry winter, very cold winter',[50 0 135];...
        25,4,'Dfa','Cold, no dry season, hot summer',[0 255 255];...
        26,4,'Dfb','Cold, no dry season, warm summer',[55 200 255];...
        27,4,'Dfc','Cold, no dry season, cold summer',[0 125 125];...
        28,4,'Dfd','Cold, no dry season, very cold winter',[0 70 95];...
        29,5,'ET','Polar, tundra',[178 178 178];...
        30,5,'EF','Polar, frost',[102 102 102];...
        };
    """
    # load index from mat file
    data = h5py.File(
        #  '/WORK/sysu_yjdai_6/lilu/inputs/index_KG.mat')
        '/Users/lewlee/Desktop/AttConvLSTM/outputs/index_KG.mat')
    data = data['index_KG']
    print(data.shape)
    data = np.transpose(data)

    # generate lat lon of SMAP
    f = h5py.File(
        #    '/WORK/sysu_yjdai_6/lilu/inputs/SMAP_L4.h5')
        '/Users/lewlee/Desktop/AttConvLSTM/outputs/SMAP_L4.h5')

    # all lat, lon world
    SMAP_LAT_RANGE = np.array(f['cell_lat'][:, 1])
    SMAP_LON_RANGE = np.array(f['cell_lon'][1, :])
    # index of us
    lat_range = gen_region_index(SMAP_LAT_RANGE, [25, 53])
    lon_range = gen_region_index(SMAP_LON_RANGE, [-125, -67])
    data = data[lat_range[0]:lat_range[-1]:2, lon_range[0]:lon_range[-1]:2]
    data = data[12:140, 12:292]
    # exclude Tropical area cuz only few points
    data[data == 0] = np.nan
    # set more rough spatial
    for i in range(1, 4):
        data[data == i] = 0
    for i in range(4, 6):
        data[data == i] = 1
    for i in range(6, 8):
        data[data == i] = 2
    for i in range(8, 11):
        data[data == i] = 3
    for i in range(11, 14):
        data[data == i] = 4
    for i in range(14, 17):
        data[data == i] = 5
    for i in range(17, 21):
        data[data == i] = 6
    for i in range(21, 25):
        data[data == i] = 7
    for i in range(25, 29):
        data[data == i] = 8
    for i in range(29, 31):
        data[data == i] = 9
    # get all climate regions
    a = np.unique(data)
    # get corresponding index
    index = {}
    for i in range(len(a[:10])):
        index[str(i)] = np.where(data == a[i])

    return data, index


def KG_all():
    """Generate Koppen-Geiger index used to climate classification

    classes = {...
        1,1,'Af','Tropical, rainforest',[0 0 255];...
        2,1,'Am','Tropical, monsoon',[0 120 255];...
        3,1,'Aw','Tropical, savannah',[70 170 250];...
        4,2,'BWh','Arid, desert, hot',[255 0 0];...
        5,2,'BWk','Arid, desert, cold',[255 150 150];...
        6,2,'BSh','Arid, steppe, hot',[245 165 0];...
        7,2,'BSk','Arid, steppe, cold',[255 220 100];...
        8,3,'Csa','Temperate, dry summer, hot summer',[255 255 0];...
        9,3,'Csb','Temperate, dry summer, warm summer',[200 200 0];...
        10,3,'Csc','Temperate, dry summer, cold summer',[150 150 0];...
        11,3,'Cwa','Temperate, dry winter, hot summer',[150 255 150];...
        12,3,'Cwb','Temperate, dry winter, warm summer',[100 200 100];...
        13,3,'Cwc','Temperate, dry winter, cold summer',[50 150 50];...
        14,3,'Cfa','Temperate, no dry season, hot summer',[200 255 80];...
        15,3,'Cfb','Temperate, no dry season, warm summer',[100 255 80];...
        16,3,'Cfc','Temperate, no dry season, cold summer',[50 200 0];...
        17,4,'Dsa','Cold, dry summer, hot summer',[255 0 255];...
        18,4,'Dsb','Cold, dry summer, warm summer',[200 0 200];...
        19,4,'Dsc','Cold, dry summer, cold summer',[150 50 150];...
        20,4,'Dsd','Cold, dry summer, very cold winter',[150 100 150];...
        21,4,'Dwa','Cold, dry winter, hot summer',[170 175 255];...
        22,4,'Dwb','Cold, dry winter, warm summer',[90 120 220];...
        23,4,'Dwc','Cold, dry winter, cold summer',[75 80 180];...
        24,4,'Dwd','Cold, dry winter, very cold winter',[50 0 135];...
        25,4,'Dfa','Cold, no dry season, hot summer',[0 255 255];...
        26,4,'Dfb','Cold, no dry season, warm summer',[55 200 255];...
        27,4,'Dfc','Cold, no dry season, cold summer',[0 125 125];...
        28,4,'Dfd','Cold, no dry season, very cold winter',[0 70 95];...
        29,5,'ET','Polar, tundra',[178 178 178];...
        30,5,'EF','Polar, frost',[102 102 102];...
        };
    """
    # load index from mat file
    data = h5py.File(
        #  '/WORK/sysu_yjdai_6/lilu/inputs/index_KG.mat')
        '/Users/lewlee/Documents/Github/SMNet/output/index_KG.mat')
    data = data['index_KG']
    print(data.shape)
    data = np.transpose(data)

    # generate lat lon of SMAP
    f = h5py.File(
        #    '/WORK/sysu_yjdai_6/lilu/inputs/SMAP_L4.h5')
        '/Users/lewlee/Documents/Github/SMNet/output/SMAP_L4.h5')

    # all lat, lon world
    SMAP_LAT_RANGE = np.array(f['cell_lat'][:, 1])
    SMAP_LON_RANGE = np.array(f['cell_lon'][1, :])
    # index of us
    lat_range = gen_region_index(SMAP_LAT_RANGE, [25, 53])
    lon_range = gen_region_index(SMAP_LON_RANGE, [-125, -67])
    data = data[lat_range[0]:lat_range[-1]:2, lon_range[0]:lon_range[-1]:2]
    data = data[12:140, 12:292]
    # exclude Tropical area cuz only few points
    data[data == 0] = np.nan
    # set more rough spatial
    for i in range(4, 8):
        data[data == i] = 0
    for i in range(8, 17):
        data[data == i] = 1
    for i in range(17, 29):
        data[data == i] = 2
    # get all climate regions
    a = np.unique(data)
    print(a)
    # get corresponding index
    index = {}
    for i in range(len(a[:3])):
        index[str(i)] = np.where(data == a[i])

    return data, index


def autocorr(x, t=1):
    return np.corrcoef(np.array(x[0:len(x) - t]), np.array(x[t:len(x)]))


def month(N):

    months = np.squeeze(Time('2015-03-31', '2019-01-24')())[-N:, 0] * 12
    months = months[230:3139]
    print(months.shape)
    index = {}
    for i in range(12):
        index[str(i+1)] = []
        for j in range(len(months)):
            if months[j] == i+1:
                index[str(i+1)].append(j)
    return index


def bwr_cmap():
    """Adds the 'stoplight' and 'RdGn' colormaps to matplotlib's database
    """
    # bias colormap
    val = 0.8
    per = 0.2 / 2
    Rd = cs.rgb_to_hsv(1, 0, 0)
    Rd = cs.hsv_to_rgb(Rd[0], Rd[1], val)
    Bl = cs.rgb_to_hsv(0, 0, 1)
    Bl = cs.hsv_to_rgb(Bl[0], Bl[1], val)
    RdBl = {'red':   ((0.0, 0.0,   Bl[0]),
                      (0.5-per, 1.0, 1.0),
                      (0.5+per, 1.0, 1.0),
                      (1.0, Rd[0], 0.0)),
            'green': ((0.0, 0.0,   Bl[1]),
                      (0.5-per, 1.0, 1.0),
                      (0.5+per, 1.0, 1.0),
                      (1.0, Rd[1], 0.0)),
            'blue':  ((0.0, 0.0,   Bl[2]),
                      (0.5-per, 1.0, 1.0),
                      (0.5+per, 1.0, 1.0),
                      (1.0, Rd[2], 0.0))}
    plt.register_cmap(name='bias', data=RdBl)


def gen_meshgrid():
    """Generate meshgrid of pcolormesh"""
    # generate lat lon of SMAP
    f = h5py.File(
        '/hard/lilu/1_AttConvLSTM/outputs/SMAP_L4.h5', 'r')
    # all lat, lon world
    SMAP_LAT_RANGE = np.array(f['cell_lat'][:, 1])
    SMAP_LON_RANGE = np.array(f['cell_lon'][1, :])
    # index of us
    lat_range = gen_region_index(SMAP_LAT_RANGE, [25, 53])
    lon_range = gen_region_index(SMAP_LON_RANGE, [-125, -67])
    us_lat = SMAP_LAT_RANGE[lat_range][:-1]
    us_lon = SMAP_LON_RANGE[lon_range][:-1]
    # get lat & lon by intervel
    us_1_lat = us_lat[0::2][12:140]
    us_1_lon = us_lon[0::2][12:292]
    # lat = us_1_lat  # [12:140]
    # lon = us_1_lon  # [12:292]
    # lat = us_1_lat[0:128]
    # lon = us_1_lon[0:288]
    lon, lat = np.meshgrid(us_1_lon, us_1_lat)
    return lon, lat


def set_lonlat(_m, lon_list, lat_list, lon_labels, lat_labels, lonlat_size):

    lon_dict = _m.drawmeridians(
        lon_list, labels=lon_labels, color='none', fontsize=lonlat_size)
    lat_dict = _m.drawparallels(
        lat_list, labels=lat_labels, color='none', fontsize=lonlat_size)
    lon_list = []
    lat_list = []
    for lon_key in lon_dict.keys():
        try:
            lon_list.append(lon_dict[lon_key][1][0].get_position()[0])
        except:
            continue

    for lat_key in lat_dict.keys():
        try:
            lat_list.append(lat_dict[lat_key][1][0].get_position()[1])
        except:
            continue
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.set_yticks(lat_list)
    ax.set_xticks(lon_list)
    ax.tick_params(labelcolor='none')


def test_pickle(path):
    """print R2 of each output pickle file"""
    # read
    handle = open(path, 'rb')
    outputs = pickle.load(handle)
    y_valid = outputs['y_valid'][:, :, :, 1][:, np.newaxis, :, :, np.newaxis]
    pred_valid = outputs['pred_valid'][:, :,
                                       :, 1][:, np.newaxis, :, :, np.newaxis]
    print('The shape of valid data is')
    print(y_valid.shape)

    # print spatial average
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(np.nanmean(y_valid, axis=(1, 2, 3, -1)))
    plt.plot(np.nanmean(pred_valid, axis=(1, 2, 3, -1)))
    plt.legend(['1', '2'])

    # print spatial R2
    plt.subplot(2, 2, 2)
    r = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if np.isnan(y_valid[:, 0, i, j, 1]).all():
                pass
            else:
                r[i, j] = r2_score(y_valid[:, 0, i, j, 1],
                                   pred_valid[:, 0, i, j, 1])
    #r[r < 0.6] = np.nan
    print(np.nanmean(r))

    plt.imshow(r, vmin=0.6, vmax=1)
    plt.colorbar()

    plt.show()


def folder_list(folder_path, datasets=1):

    # return list of files in folder
    l = glob.glob(folder_path, recursive=True)

    # sorted using the file name (for ESACCI, SMAP, GLDAS et al)
    num_in_string = []
    # print(l[0].split("_"))
    for i in range(len(l)):
        if datasets == 0:  # flx
            # l.sort()
            return l
        if datasets == 1:
            num_in_string.append(
                int(re.sub("\D", "", l[i].split("-")[6])))  # ESACCI
        elif datasets == 2:
            num_in_string.append(
                int(re.sub("\D", "", l[i].split("_")[6])))  # SMAP
        elif datasets == 3:
            num_in_string.append(
                int(re.sub("\D", "", l[i].split(".")[1])))  # GLDAS
    # print(num_in_string)

    # sorted index
    sorted_index = sorted(range(len(num_in_string)),
                          key=lambda i: num_in_string[i])

    # sorted list by index
    sorted_l = []
    for i in range(len(sorted_index)):
        sorted_l.append(l[sorted_index[i]])
    return sorted_l  # list


def smooth(data):

    # get shape
    Nlat, Nlon = data.shape

    try:
        for i in range(Nlat // 8):
            for j in range(Nlon):
                data[i, j] = (data[i - 1, j] + data[i + 1, j]) * 0.5

        for i in range(Nlat):
            for j in range(Nlon // 8):
                data[i, j] = (data[i, j - 1] + data[i, j + 1]) * 0.5
    except:
        print('lalala')
    return data


def gen_metric(path):

    # load figure5.npz
    data = np.load(path)

    # generate metric
    spatial_mean_target = data['arr_0']
    rmse = data['arr_2']
    r2 = data['arr_3']

    # turn nan
    rmse[rmse < 0] = np.nan
    r2[r2 < 0] = np.nan

    # interplote nan
    rmse = fillna(rmse)
    r2 = fillna(r2)

    # exclude points outside polygon
    index = np.where(np.isnan(spatial_mean_target))
    rmse[index[0], index[1]] = np.nan
    r2[index[0], index[1]] = np.nan

    return spatial_mean_target, rmse, r2


def gen_tac_sac():

    # generate sac
    target, _, _ = gen_metric(
                '/hard/lilu/1_AttConvLSTM/outputs/figures_DAC/figure5.npz')

    # init
    sac = np.full((target.shape[0], target.shape[1]), np.nan)
    Z = np.full((target.shape[0], target.shape[1]), np.nan)
    WZ = np.full((target.shape[0], target.shape[1]), np.nan)
    Z2 = np.full((target.shape[0], target.shape[1]), np.nan)

    # Caculate local moran's I
    x_mean = np.nanmean(target)
    for i in range(1, target.shape[0]-1):
        for j in range(1, target.shape[1]-1):
            Z[i, j] = (target[i, j]-x_mean)
            WZ[i, j] = ((np.nansum(
                [target[i-1, j]-x_mean, target[i+1, j]-x_mean,
                 target[i, j+1]-x_mean, target[i, j-1]-x_mean])))
            sac[i, j] = Z[i, j]*WZ[i, j]
            Z2[i, j] = Z[i, j] * Z[i, j]

    # generate tac
    data = np.load(
        '/hard/lilu/1_AttConvLSTM/outputs/figures_DAC/figure10.npz')
    tac = data['arr_0']

    # fill nan
    tac = fillna(tac)
    index = np.where(np.isnan(target))
    tac[index[0], index[1]] = np.nan

    # turn nan
    """
    tac[np.where((tac < 0.8))] = 0.8
    tac[np.where((tac > 1))] = np.nan

    #tac[np.where((tac > 1) | (tac < 0.6))] = np.nan
    #sac[np.where((sac > 0.2) | (sac < 0))] = np.nan
    sac[np.where((sac < 0))] = np.nan
    sac[np.where((sac > 0.2))] = 0.2

    #tac[np.isnan(sac)] = np.nan
    sac[np.isnan(tac)] = np.nan
    """
    return tac, sac


if __name__ == "__main__":
    index = KG()

# coding: utf-8
# pylint:
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn

import pickle

import numpy as np

from model import ML_Models, Train, trainL
from utils import parse_args, print_log, save2pickle, tictoc


"""MAIN MODULE

.. notes:: The main module all caculate future several days instead only 
           only predict one step.
"""
@tictoc
def mainD():
    """Exec SMNET, i.e., deep learning model."""
    # print info
    print_log()
    # config
    config = parse_args()
    # read pickle file
    handle = open(config.path_inputs+str(config.num_jobs)+".pickle", 'rb')
    inputs = pickle.load(handle)

    if len(inputs) == 0:
        log = 0
        save2pickle(log,
                    out_path=config.path_outputs,
                    out_file=str(config.num_jobs)+".pickle")
    else:
        Train(inputs)
        #logs = trainL(inputs)
        # save
        #save2pickle(logs,
        #        out_path=config.path_outputs,
        #        out_file=str(config.num_jobs) + ".pickle")


@tictoc
def mainL():
    """Exec LSTM"""
    # config
    config = parse_args()
    # read pickle file
    handle = open(config.path_inputs+str(config.num_jobs)+".pickle", 'rb')
    inputs = pickle.load(handle)
    # get shape
    N, _, Nlat, Nlon, Nfeature = inputs['y_valid'].shape
   
     
    print(inputs['x_valid'].shape)
    # init
    y_valid = np.full((N, 1, Nlat, Nlon, 1), np.nan)
    pred_valid = np.full((N, 1, Nlat, Nlon, 1), np.nan)
    # loop
    for i in range(Nlat):
        for j in range(Nlon):
            print(i)
            print(j)
            # crop pixel
            _inputs = {}
            _inputs['x_train'] = inputs['x_train'][:, :, i, j, :]
            _inputs['x_valid'] = inputs['x_valid'][:, :, i, j, :]
            _inputs['y_train'] = inputs['y_train'][:, :, i, j, :]
            _inputs['y_valid'] = inputs['y_valid'][:, :, i, j, :]
            print(inputs['x_train'].shape)
            print(inputs['y_valid'].shape)

            # run
            if len(_inputs) == 0:
                log = 0
                save2pickle(log,
                            out_path=config.path_outputs,
                            out_file=str(config.num_jobs) + ".pickle")
            else:
                log = trainL(_inputs)
                # save
                y_valid[:, 0, i, j, 0] = np.squeeze(log['y_valid'])
                pred_valid[:, 0, i, j, 0] = np.squeeze(log['pred_valid'])
    logs = {}
    logs['y_valid'] = y_valid
    logs['pred_valid'] = pred_valid
    # save
    save2pickle(logs,
                out_path=config.path_outputs,
                out_file=str(config.num_jobs) + ".pickle")


@tictoc
def mainML():
    """Exec machine learning models"""
    # config
    config = parse_args()
    # read pickle file
    handle = open(config.path_inputs+str(config.num_jobs)+".pickle", 'rb')
    inputs = pickle.load(handle)
    # get shape
    N, Nlat, Nlon, Nfeature = inputs['y_valid'].shape
    # init
    y_valid = np.full((N, Nlat, Nlon, 3), np.nan)
    pred_valid = np.full((N, Nlat, Nlon, 3), np.nan)
    # loop
    for i in range(Nlat):
        for j in range(Nlon):
            print(i)
            print(j)
            # crop pixel
            _inputs = {}
            _inputs['x_train'] = inputs['x_train'][:, i, j, :]
            _inputs['x_valid'] = inputs['x_valid'][:, i, j, :]
            _inputs['y_train'] = inputs['y_train'][:, i, j, :]
            _inputs['y_valid'] = inputs['y_valid'][:, i, j, :]
            # run
            if (np.isnan(_inputs['x_train']).any()) and (np.isnan(_inputs['y_train']).any()):
                print('all nan')
            else:
                log = ML_Models(_inputs)()
                # save
                y_valid[:, i, j, 0] = np.squeeze(log['ridge']['y_valid'])
                y_valid[:, i, j, 1] = np.squeeze(log['SVR']['y_valid'])
                y_valid[:, i, j, 2] = np.squeeze(log['RF']['y_valid'])
                pred_valid[:, i, j, 0] = np.squeeze(log['ridge']['pred_valid'])
                pred_valid[:, i, j, 1] = np.squeeze(log['SVR']['pred_valid'])
                pred_valid[:, i, j, 2] = np.squeeze(log['RF']['pred_valid'])
    # generate logs
    logs = {}
    logs['y_valid'] = y_valid
    logs['pred_valid'] = pred_valid
    # save 2 pickle
    save2pickle(logs,
                out_path=config.path_outputs,
                out_file=str(config.num_jobs) + ".pickle")


if __name__ == "__main__":
    #mainD()
    mainL()
    # mainML()

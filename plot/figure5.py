import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.basemap import Basemap
plt.rc('font', family='Times New Roman')

from utils import bwr_cmap, fillna, gen_meshgrid



def figure5(path_file):

    # load figure5.npz
    data = np.load(path_file)

    # load seperate variables
    target_spatial_mean = data['arr_0']  # mean of predict
    d = data['arr_1']  # delta of label and predict
    rmse_valid = data['arr_2']  # rmse
    r2 = data['arr_3']  # r2

    # interplote nan
    rmse_valid[rmse_valid < 0] = np.nan
    rmse_valid = fillna(rmse_valid)
    r2[r2 < 0] = np.nan
    r2 = fillna(r2)

    # exclude points on sea
    index = np.where(np.isnan(target_spatial_mean))
    rmse_valid[index[0], index[1]] = np.nan
    r2[index[0], index[1]] = np.nan

    # generate meshgrid
    lon, lat = gen_meshgrid()

    # --------------------------------------------------------------------------
    # figure 5
    # (a) temporal average
    # (b) delta SMAP-pred
    # (c) RMSE
    # (d) R2
    # --------------------------------------------------------------------------
    plt.figure(figsize=(12, 6.7))

    # ----------------------------- Figure 5 (a) -------------------------------
    plt.subplot2grid((2, 2), (0, 0))

    # draw projection
    # m = Basemap(width=5000000, height=3000000,
    #             projection='lcc',
    #             lat_0=39, lon_0= -96.)  # lambert projection
    m = Basemap(projection='mill',
                llcrnrlat=27, urcrnrlat=50,
                llcrnrlon=-122.9, urcrnrlon=-70.2,)  # mill projection

    # draw edge lines
    m.drawcoastlines()
    # m.drawcountries()

    # give meshgrid
    x, y = m(lon, lat)

    # coutourf
    sc = m.pcolormesh(x, y,
                      target_spatial_mean,
                      cmap=plt.cm.get_cmap('jet_r'),
                      vmin=0, vmax=0.4,
                      shading='flat')

    # set edge color
    sc.set_edgecolor('face')

    # colorbar
    m.colorbar(sc, location='bottom')

    # set text
    x, y = m(-123, 50.5)
    plt.text(x, y, '(a) MEAN', fontweight='bold', fontsize=14)

    # ----------------------------- Figure 5 (b) -------------------------------
    plt.subplot2grid((2, 2), (0, 1))

    # define own colormaps
    bwr_cmap()
    cmap = plt.get_cmap('bias')

    # draw projection
    m = Basemap(projection='mill',
                llcrnrlat=27, urcrnrlat=50,
                llcrnrlon=-122.9, urcrnrlon=-70.2,)

    # lines
    m.drawcoastlines()

    # meshgrid
    x, y = m(lon, lat)

    # contourf
    sc = m.pcolor(x, y,
                  d,
                  cmap=cmap,
                  vmin=-0.05, vmax=0.05)

    # edge
    sc.set_edgecolor('face')

    # colorbar
    m.colorbar(sc, location='bottom')

    # text
    x, y = m(-123, 50.5)
    plt.text(x, y, '(b) DELTA', fontweight='bold', fontsize=14)

    # ----------------------------- Figure 5 (c) -------------------------------
    plt.subplot2grid((2, 2), (1, 0))

    # draw projection
    m = Basemap(projection='mill',
                llcrnrlat=27, urcrnrlat=50,
                llcrnrlon=-122.9, urcrnrlon=-70.2,)

    # draw lines
    m.drawcoastlines()

    # meshgrid
    x, y = m(lon, lat)

    # countourf
    sc = m.pcolormesh(x, y,
                      rmse_valid,
                      cmap=plt.cm.get_cmap('jet_r'),
                      vmin=0, vmax=0.04)

    # edge
    sc.set_edgecolor('face')

    # colorbar
    cb = m.colorbar(sc, location='bottom')
    tick_locator = ticker.MaxNLocator(nbins=8)  # give tick of bar
    cb.locator = tick_locator
    cb.update_ticks()

    # text
    x, y = m(-123, 50.5)
    plt.text(x, y, '(c) RMSE', fontweight='bold', fontsize=14)

    # ----------------------------- Figure 5 (d) -------------------------------
    plt.subplot2grid((2, 2), (1, 1))

    # projection
    m = Basemap(projection='mill',
                llcrnrlat=27, urcrnrlat=50,
                llcrnrlon=-122.9, urcrnrlon=-70.2,)

    # lines
    m.drawcoastlines()

    # meshgrid
    x, y = m(lon, lat)

    # countourf
    sc = m.pcolormesh(x, y,
                      r2,
                      cmap=plt.cm.get_cmap('jet'),
                      vmin=0.6, vmax=1)

    # edge
    sc.set_edgecolor('face')

    # colorbar
    m.colorbar(sc, location='bottom')

    # text
    x, y = m(-123, 50.5)
    plt.text(x, y, '(d) $R^{2}$', fontweight='bold', fontsize=14)

    plt.subplots_adjust(hspace=0.2, wspace=0.01)

    # --------------------------------------------------------------------------
    # save
    plt.savefig('/Users/lewlee/Desktop/figure5.pdf')


if __name__ == "__main__":

    MAIN_PATH = '/Users/lewlee/Documents/Github/SMNet/output/'
    FILE_PATH = 'figures_DAC/figure5.npz'
    figure5(MAIN_PATH + FILE_PATH)

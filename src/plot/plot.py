# coding: utf-8
# pylint:
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn

from utils import KG, RegisterCustomColormaps, gen_meshgrid, gen_region_index, smooth
from data import Time
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


plt.rc('font', family='Times New Roman')



def gen_r2():

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')
    target_spatial_mean = data['arr_0']
    r2 = data['arr_3']
    r2[r2 < 0.4] = np.nan
    # interplote nan
    r2 = fillna(r2)
    index = np.where(np.isnan(target_spatial_mean))
    r2[index[0], index[1]] = np.nan

    return r2


def fillna(inputs):
    return pd.DataFrame(
        KNN(k=6).fit_transform(pd.DataFrame(inputs))).to_numpy()


def figure7():
    """Plot figure7, i.e., Koppen-Geiger rough index map.
    """
    # index
    data, index_KG = KG()
    # plot
    plt.figure(figsize=(10, 10))

    # load figure5.npz
    data1 = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')
    target_spatial_mean = data1['arr_0']
    index = np.where(np.isnan(target_spatial_mean))
    print(index)
    data[index[0], index[1]] = np.nan
    # ----------------------------- Figure 7 ----------------------------------
    colors = ('#FF1493', '#FFC0CB',
              '#20B2AA', '#32CD32', '#7CFC00',
              '#00FFFF',  '#00BFFF', '#0000FF')
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.5,)
    m.drawcoastlines()
    x, y = m(lon, lat)
    levels = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    sc = m.contourf(x, y,
                    data,
                    colors=colors,
                    levels=levels, vmin=0, vmax=7)
    x, y = m(-123, 50.8)
    plt.text(x, y, 'Köppen-Geiger climate index',
             fontweight='bold', fontsize=14)
    # create proxy artists to make legend
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
             for pc in sc.collections]
    title = ['Arid, desert', 'Arid, steppe',
             'Temperate, dry summer', 'Temperate, dry winter',
             'Temperate, no dry season', 'Cold, dry summer',
             'Cold, dry winter', 'Cold, no dry season']
    plt.legend(proxy, title, loc='upper right',
               bbox_to_anchor=(1, 1.44))
    plt.savefig('/Users/lewlee/Desktop/figure7.pdf')


def figure8():
    """Plot figure8, i.e.,
       Scatter density plot of of different climate regions

       Notes: the same with figure 7, only generate csv.
    """
    # load
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure8.npz')
    pred_grids = data['arr_0']
    target_grids = data['arr_1']
    # flatten 20 grids
    pred_grids = pred_grids.reshape(-1, pred_grids.shape[-1])
    target_grids = target_grids.reshape(-1, target_grids.shape[-1])
    # plt.plot(pred_grids)
    # plt.show()
    # concat
    inputs = np.concatenate([pred_grids, target_grids], axis=-1)
    np.savetxt('/Users/lewlee/Documents/MATLAB/figure8.csv',
               inputs, delimiter=',')


def figure9():
    """Plot figure9, i.e., density curves of R2 in different climate regions.
       & save csv file of R2, RMSE of spatial mean of each regions.
    """
    title = ['Arid, desert', 'Arid, steppe',
             'Temperate, dry summer', 'Temperate, dry winter',
             'Temperate, no dry season', 'Cold, dry summer',
             'Cold, dry winter', 'Cold, no dry season', ]
    # -------------------------kdeplot------------------------------------------
    # load figure5.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')
    rmse_valid = data['arr_2']
    r2 = data['arr_3']
    r2[r2 < 0] = np.nan
    # index
    _, index_KG = KG()
    # loop for climate regions
    for k in range(0, len(index_KG)):
        # kdeplot
        if (k <= 1) and (k >= 0):
            sns.kdeplot(
                r2[index_KG[str(k)][0], index_KG[str(k)][1]],
                label=title[k], linestyle='--', lw=2)
        if (k <= 4) and (k >= 2):
            sns.kdeplot(
                r2[index_KG[str(k)][0], index_KG[str(k)][1]],
                label=title[k], linestyle=':', lw=2)
        if (k <= 8) and (k >= 5):
            sns.kdeplot(
                r2[index_KG[str(k)][0], index_KG[str(k)][1]],
                label=title[k], linestyle='-', lw=2)
        plt.xlim(0, 1)
    plt.xlabel('determination coefficient $R^{2}$')
    plt.ylabel('estimation density')
    # save
    plt.savefig('/Users/lewlee/Desktop/figure9.pdf', dpi=600)
    # --------------------------------------------------------------------------
    # caculate avg r2, rmse
    avg_r2 = np.full((len(index_KG), 1), np.nan)
    avg_rmse = np.full((len(index_KG), 1), np.nan)
    # loop for climate region
    for i in range(len(index_KG)):
        avg_r2[i] = np.nanmean(
            r2[index_KG[str(i)][0], index_KG[str(i)][1]])
        avg_rmse[i] = np.nanmean(
            rmse_valid[index_KG[str(i)][0], index_KG[str(i)][1]])
    # concat
    inputs = np.concatenate([avg_r2, avg_rmse], axis=-1)
    # save
    np.savetxt('/Users/lewlee/Documents/MATLAB/figure8_metrics.csv',
               inputs, delimiter=',')


def figure11():
    # load figure5.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')
    target = data['arr_0']
    rmse_valid = data['arr_2']
    r2 = data['arr_3']
    r2[r2 < 0] = np.nan
    rmse_valid[rmse_valid < 0] = np.nan
    # load figure10.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure10.npz')
    acf_target = data['arr_0']

    # init
    I = np.full((target.shape[0], target.shape[1]), np.nan)
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
            I[i, j] = Z[i, j]*WZ[i, j]
            Z2[i, j] = Z[i, j] * Z[i, j]

    plt.figure(figsize=(14, 5))
    # ----------------------------- Figure 11 (a) -------------------------------
    ax = plt.subplot(1, 3, 1)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', np.nanmean(I)))
    ax.spines['left'].set_position(('data', np.nanmean(acf_target)))
    plt.scatter(acf_target.reshape(-1, 1), I.reshape(-1, 1),
                c=r2.reshape(-1, 1), linewidths=0.01, s=0.7,
                cmap=plt.cm.get_cmap('jet'), vmin=0.5, vmax=1)
    plt.xlim(0.6, 1.01)
    plt.ylim(-0.05, 0.25)
    cb = plt.colorbar(orientation='vertical', shrink=0.3, pad=0.1)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    plt.text(0.6, 0.24, '(a) $R^{2}$', fontweight='bold', fontsize=14)
    plt.text(0.6, 0.045, 'TAC', fontweight='bold', fontsize=10, color='r')
    plt.text(0.92, 0.24, "SC",
             fontweight='bold', fontsize=10, color='r')
    # ----------------------------- Figure 11 (b) -------------------------------
    ax = plt.subplot(1, 3, 2)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', np.nanmean(I)))
    ax.spines['left'].set_position(('data', np.nanmean(acf_target)))
    plt.scatter(acf_target.reshape(-1, 1), I.reshape(-1, 1),
                c=rmse_valid.reshape(-1, 1), linewidths=0.01, s=0.7,
                cmap=plt.cm.get_cmap('jet_r'), vmin=0.00, vmax=0.04)
    cb = plt.colorbar(orientation='vertical',
                      shrink=0.3, pad=0.1)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    plt.xlim(0.6, 1.01)
    plt.ylim(-0.05, 0.25)
    plt.text(0.6, 0.24, '(b) RMSE', fontweight='bold', fontsize=14)
    plt.text(0.6, 0.045, 'TAC', fontweight='bold', fontsize=10, color='r')
    plt.text(0.92, 0.24, "SC",
             fontweight='bold', fontsize=10, color='r')
    # ----------------------------- Figure 11 (c) -------------------------------
    data, index_KG = KG()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] != 7 and data[i, j] != 6 and data[i, j] != 1 \
                    and data[i, j] != 3:
                data[i, j] = np.nan
                acf_target[i, j] = np.nan
                I[i, j] = np.nan
    acf_target = acf_target.reshape(-1, 1)
    acf_target_1 = acf_target[~np.isnan(acf_target)]
    print(acf_target.shape)

    I = I.reshape(-1, 1)
    I = I[~np.isnan(acf_target)]
    print(I.shape)

    data = data.reshape(-1, 1)
    data = data[~np.isnan(acf_target)]
    print(data.shape)

    ax = plt.subplot(1, 3, 3)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', np.nanmean(I)))
    ax.spines['left'].set_position(('data', np.nanmean(acf_target)))
    idx1 = np.where(data == 1)
    plt.scatter(acf_target_1[idx1], I[idx1],
                c='r', s=0.7, label='Arid, desert')
    idx1 = np.where(data == 3)
    plt.scatter(acf_target_1[idx1], I[idx1],
                c='grey', s=0.7, label='Temperate,dry summer')
    idx1 = np.where(data == 6)
    plt.scatter(acf_target_1[idx1], I[idx1], c='g',
                s=0.7, label='Cold, dry summer')
    idx1 = np.where(data == 7)
    plt.scatter(acf_target_1[idx1], I[idx1],
                c='blue', s=0.7, label='Cold, dry winter')

    plt.legend(loc='best', bbox_to_anchor=(0.1, 0.45, 0.5, 0.5),
               title='Köppen-Geiger climate index')

    plt.xlim(0.6, 1.01)
    plt.ylim(-0.05, 0.25)
    plt.text(0.6, 0.24, '(c) Climate regions', fontweight='bold', fontsize=14)
    plt.text(0.6, 0.05, 'TAC', fontweight='bold', fontsize=10, color='r')
    plt.text(0.92, 0.24, "SC",
             fontweight='bold', fontsize=10, color='r')
    plt.savefig('/Users/lewlee/Desktop/figure11.pdf')


def figure12():
    """Scatter autocorrelation of DAC, DC, RF, LSTM ... models."""
    # load figure10.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure10.npz')
    acf_target_DAC = data['arr_0'].reshape(-1, 1)
    acf_pred_DAC = data['arr_1'].reshape(-1, 1)
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DC/figure10.npz')
    acf_target_DC = data['arr_0'].reshape(-1, 1)
    acf_pred_DC = data['arr_1'].reshape(-1, 1)
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_rf/figure10.npz')
    acf_target_rf = data['arr_0'].reshape(-1, 1)
    acf_pred_rf = data['arr_1'].reshape(-1, 1)
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_lstm/figure10.npz')
    acf_target_lstm = data['arr_0'].reshape(-1, 1)
    acf_pred_lstm = data['arr_1'].reshape(-1, 1)

    acf_pred_DC[np.where(np.isnan(acf_pred_DAC))] = 0
    acf_pred_rf[np.where(np.isnan(acf_pred_DAC))] = 0
    acf_pred_lstm[np.where(np.isnan(acf_pred_DAC))] = 0

    def gen_est_err(x, y):

        y[np.where(np.isnan(x))] = 0
        x[np.where(np.isnan(x))] = 0
        x[np.where(np.isnan(y))] = 0
        y[np.where(np.isnan(y))] = 0
        print(x.shape)
        print(y.shape)

        x_ = x[(np.where(x != 0)) and (np.where(y != 0))]
        y_ = y[(np.where(x != 0)) and (np.where(y != 0))]
        x = x_
        y = y_
        print(x_.shape)
        print(y_.shape)
        a, b = np.polyfit(x, y, deg=1)
        print(a)
        print(b)
        y_est = a*x+b
        y_err = x.std()*np.sqrt(1/len(x) +
                                (x - x.mean())**2 / np.sum((x - x.mean())**2))
        return x, y, y_est, y_err, a, b

    plt.figure(figsize=(12, 12))
    # (a) DAC
    x = acf_target_DAC.reshape(-1, )
    y = acf_pred_DAC.reshape(-1, )
    x, y, y_est, y_err, a, b = gen_est_err(x, y)

    ax = plt.subplot(2, 2, 1)
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    ax.scatter(x, y, c=z, s=0.5, cmap='jet')

    ax.set_xlim(0.2, 1)
    ax.set_ylim(0.2, 1)
    ax.plot(x, y_est, '-', color='r', lw=3)
    plt.xlabel('target ACF', fontweight='bold', fontsize=15)
    plt.ylabel('prediction ACF', fontweight='bold', fontsize=15)
    plt.text(0.25, 0.90, 'Y = '+str(round(a, 3))+'X - ' +
             str(round(-b, 3)), fontweight='bold', fontsize=15)
    plt.text(0.2, 1.01, '(a) DAC', fontweight='bold', fontsize=15)
    # (b) DC
    ax = plt.subplot(2, 2, 2)
    x = acf_target_DAC.reshape(-1, )
    y = acf_pred_DC.reshape(-1, )
    x, y, y_est, y_err, a, b = gen_est_err(x, y)
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    ax.scatter(x, y, c=z, s=0.5, cmap='jet')
    ax.set_xlim(0.2, 1)
    ax.set_ylim(0.2, 1)
    ax.plot(x, y_est, '-', color='r', lw=3)
    plt.xlabel('target ACF', fontweight='bold', fontsize=15)
    plt.ylabel('prediction ACF', fontweight='bold', fontsize=15)
    plt.text(0.25, 0.90, 'Y = '+str(round(a, 3))+'X + ' +
             str(round(b, 3)), fontweight='bold', fontsize=15)
    plt.text(0.2, 1.01, '(b) DC', fontweight='bold', fontsize=15)
    # (c) RF
    ax = plt.subplot(2, 2, 3)
    x = acf_target_DAC.reshape(-1, )
    y = acf_pred_rf.reshape(-1, )
    x, y, y_est, y_err, a, b = gen_est_err(x, y)

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    ax.scatter(x, y, c=z, s=0.5, cmap='jet')
    ax.set_xlim(0.2, 1)
    ax.set_ylim(0.2, 1)
    ax.plot(x, y_est, '-', color='r', lw=3)
    plt.xlabel('target ACF', fontweight='bold', fontsize=15)
    plt.ylabel('prediction ACF', fontweight='bold', fontsize=15)
    plt.text(0.25, 0.90, 'Y = '+str(round(a, 3))+'X + ' +
             str(round(b, 3)), fontweight='bold', fontsize=15)
    plt.text(0.2, 1.01, '(c) RF', fontweight='bold', fontsize=15)

    # (d) LSTM
    ax = plt.subplot(2, 2, 4)

    x = acf_target_DAC.reshape(-1, )
    y = acf_pred_lstm.reshape(-1, )
    x, y, y_est, y_err, a, b = gen_est_err(x, y)

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    ax.scatter(x, y, c=z, s=0.5, cmap='jet')
    ax.set_xlim(0.2, 1)
    ax.set_ylim(0.2, 1)
    ax.plot(x, y_est, '-', color='r', lw=3)
    plt.xlabel('target ACF', fontweight='bold', fontsize=15)
    plt.ylabel('prediction ACF', fontweight='bold', fontsize=15)
    plt.text(0.25, 0.90, 'Y = '+str(round(a, 3))+'X + ' +
             str(round(b, 3)), fontweight='bold', fontsize=15)
    plt.text(0.2, 1.01, '(d) LSTM', fontweight='bold', fontsize=15)
    plt.savefig('/Users/lewlee/Desktop/figure12.pdf')


def figure13():
    """Plot difference of R2, RMSE and autocorrelation between SMNET and RF.
    """

    # load figure5.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')
    target_spatial_mean = data['arr_0']
    d = data['arr_1']
    rmse_valid = data['arr_2']
    rmse_valid[rmse_valid < 0] = np.nan
    r2 = data['arr_3']
    r2[r2 < 0] = np.nan
    # interplote nan
    rmse_valid = fillna(rmse_valid)
    r2 = fillna(r2)
    index = np.where(np.isnan(target_spatial_mean))
    rmse_valid[index[0], index[1]] = np.nan
    r2[index[0], index[1]] = np.nan

    rmse_DAC = rmse_valid
    r2_DAC = r2
    print(r2_DAC.shape)

    # load figure5.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DC/figure5.npz')
    target_spatial_mean = data['arr_0']
    d = data['arr_1']
    rmse_valid = data['arr_2']
    rmse_valid[rmse_valid < 0] = np.nan
    r2 = data['arr_3']
    r2[r2 < 0] = np.nan
    # interplote nan
    rmse_valid = fillna(rmse_valid)
    r2 = fillna(r2)
    index = np.where(np.isnan(target_spatial_mean))
    rmse_valid[index[0], index[1]] = np.nan
    r2[index[0], index[1]] = np.nan

    rmse_DC = rmse_valid
    r2_DC = r2
    print(r2_DC.shape)

    # load figure5.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_rf/figure5.npz')
    target_spatial_mean = data['arr_0']
    d = data['arr_1']
    rmse_valid = data['arr_2']
    rmse_valid[rmse_valid < 0] = np.nan
    r2 = data['arr_3']
    r2[r2 < 0] = np.nan
    # interplote nan
    rmse_valid = fillna(rmse_valid)
    r2 = fillna(r2)
    index = np.where(np.isnan(target_spatial_mean))
    rmse_valid[index[0], index[1]] = np.nan
    r2[index[0], index[1]] = np.nan

    rmse_rf = rmse_valid
    r2_rf = r2
    print(r2_rf.shape)

    # load figure5.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_lstm/figure5.npz')
    target_spatial_mean = data['arr_0']
    d = data['arr_1']
    rmse_valid = data['arr_2']
    rmse_valid[rmse_valid < 0] = np.nan
    r2 = data['arr_3']
    r2[r2 < 0.3] = np.nan
    # interplote nan
    rmse_valid = fillna(rmse_valid)
    r2 = fillna(r2)
    index = np.where(np.isnan(target_spatial_mean))
    rmse_valid[index[0], index[1]] = np.nan
    r2[index[0], index[1]] = np.nan

    rmse_lstm = rmse_valid
    r2_lstm = r2
    print(r2_lstm.shape)

    colors1 = ('#FFFFFF', '#FFFFFF', '#FF7F50', '#FF7F50', '#FF0000')
    colors = ('white', 'white', 'white', 'palegreen',
              'yellow', 'gold', 'orange', 'red', 'darkred')
    clrmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)

    # --------------------------------------------------------------------------
    plt.figure(figsize=(12, 6.7))
    """
    # ----------------------------- Figure 5 (a) -------------------------------
    plt.subplot2grid((2, 1), (0, 0))
    # m = Basemap(width=5000000, height=3000000, projection='lcc',
    #            lat_0=39, lon_0= -96.)
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)
    m.drawcoastlines()
    # m.drawcountries()
    x, y = m(lon, lat)
    a = r2_DAC-r2_DC
    # a[a < 0] = np.nan
    sc = m.pcolormesh(x, y,
                      a,
                      cmap=clrmap,  # 'jet',
                      vmin=0, vmax=0.3)
    sc.set_edgecolor('face')
    m.colorbar(sc, location='bottom')
    x, y = m(-123, 50.5)
    plt.text(x, y, '(a) DAC-DC',
             fontweight='bold', fontsize=14)
    """
    # ----------------------------- Figure 5 (b) -------------------------------
    plt.subplot2grid((1, 2), (0, 0))
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)
    m.drawcoastlines()
    # m.drawcountries()
    x, y = m(lon, lat)
    a = r2_DAC-r2_rf
    # a[a < 0] = np.nan

    sc = m.pcolormesh(x, y,
                      a,
                      # cmap=plt.cm.get_cmap('Reds'),
                      cmap=clrmap,  # 'jet',
                      vmin=0, vmax=0.3)
    sc.set_edgecolor('face')
    m.colorbar(sc, location='bottom')
    x, y = m(-123, 50.5)
    plt.text(x, y, '(a) AttConvLSTM-RF',
             fontweight='bold', fontsize=14)
    # ----------------------------- Figure 5 (c) -------------------------------
    plt.subplot2grid((1, 2), (0, 1))
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)
    m.drawcoastlines()
    # m.drawcountries()
    x, y = m(lon, lat)
    a = r2_DAC-r2_lstm
    # a[a < 0] = np.nan
    sc = m.pcolormesh(x, y,
                      a,

                      cmap=clrmap,  # 'jet',
                      vmin=0, vmax=0.3)
    sc.set_edgecolor('face')
    m.colorbar(sc, location='bottom')

    data, index_KG = KG()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] != 6:
                data[i, j] = np.nan

    # m.contourf(x, y, data, hatches=['-'], cmap='gray', alpha=0.5)
    x, y = m(-123, 50.5)
    plt.text(x, y, '(b) AttConvLSTM-LSTM',
             fontweight='bold', fontsize=14)

    # --------------------------------------------------------------------------
    plt.savefig('/Users/lewlee/Desktop/figure13.pdf')


def figure14():

    colors = ["white", "white", "oldlace", "mistyrose",
              "lightpink", "hotpink", "deeppink"]
    clrmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')
    target = data['arr_0']
    index = np.where(np.isnan(target))
    r2 = data['arr_3']
    r2[r2 < 0] = np.nan
    r2 = fillna(r2)
    r2[index[0], index[1]] = np.nan
    r2_3HH = r2
    print(np.nanmean(r2_3HH))
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_6HH/figure5.npz')
    r2 = data['arr_3']
    r2[r2 < 0] = np.nan
    r2 = fillna(r2)
    r2[index[0], index[1]] = np.nan

    r2_6HH = r2
    print(np.nanmean(r2_6HH))

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_12HH/figure5.npz')
    r2 = data['arr_3']
    r2[r2 < 0] = np.nan
    r2 = fillna(r2)
    r2[index[0], index[1]] = np.nan

    r2_12HH = r2
    print(np.nanmean(r2_12HH))
    """
    plt.figure(figsize=(12, 12))
    MM = r2_3HH.shape[0]*r2_3HH.shape[1]
    a = np.full((MM, 8), np.nan)
    print(a.shape)
    data, index = KG()
    print(data.shape)
    # boxplot
    for i in range(1, 9):
        print(i)
        N = len(index[str(i)][0])
        print(N)
        a[0:N, i-1] = r2_3HH[index[str(i)][0], index[str(i)][1]].reshape(-1, )

    df = pd.DataFrame(a, columns=['1', '2', '3', '4', '5', '6', '7', '8'])

    plt.subplot(1, 3, 1)
    plt.boxplot(df.dropna().values, )

    for i in range(1, 9):
        print(i)
        N = len(index[str(i)][0])
        print(N)
        a[0:N, i - 1] = r2_6HH[index[str(i)][0], index[str(i)][1]].reshape(-1,)
    df = pd.DataFrame(a, columns=['1', '2', '3', '4', '5', '6', '7', '8'])

    plt.subplot(1, 3, 2)
    plt.boxplot(df.dropna().values)

    for i in range(1, 9):
        print(i)
        N = len(index[str(i)][0])
        print(N)
        a[0:N, i - 1] = r2_12HH[index[str(i)]
                                [0], index[str(i)][1]].reshape(-1,)
    df = pd.DataFrame(a, columns=['1', '2', '3', '4', '5', '6', '7', '8'])

    plt.subplot(1, 3, 3)
    plt.boxplot(df.dropna().values)

    plt.xticks(range(1, 9),  ['Arid, desert', 'Arid, steppe',
                              'Temperate, dry summer', 'Temperate, dry winter',
                              'Temperate, no dry season', 'Colde, dry summer',
                              'Cold, dry winter', 'Cold, no dry season', ])

    # , positions=[0],
    # widths=0.4, whis=0.5, patch_artist=True, showfliers=False,
    # boxprops=dict(facecolor='cyan', color='cyan'),
    # meanprops=dict(color='dodgerblue'))

    plt.savefig('/Users/lewlee/Desktop/figure14.pdf')
    
    # --------------------------------------------------------------------------
    plt.figure(figsize=(12, 6.7))
    # ----------------------------- Figure 5 (b) -------------------------------
    plt.subplot2grid((2, 2), (0, 0))
    RegisterCustomColormaps()
    cmap = plt.get_cmap('bias')
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)
    m.drawcoastlines()
    # m.drawcountries()
    x, y = m(lon, lat)
    sc = m.pcolormesh(x, y,
                      r2_3HH-r2_6HH,
                      cmap='jet',  vmin=0, vmax=0.6, shading='flat')
    sc.set_edgecolor('face')
    m.colorbar(sc, location='bottom')
    x, y = m(-123, 50.5)
    plt.text(x, y, '(a) d(3h-6h)',
             fontweight='bold', fontsize=14)
    # ----------------------------- Figure 5 (b) -------------------------------
    plt.subplot2grid((2, 2), (0, 1))
    RegisterCustomColormaps()
    cmap = plt.get_cmap('bias')
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)
    m.drawcoastlines()
    # m.drawcountries()
    x, y = m(lon, lat)
    sc = m.pcolormesh(x, y,
                      r2_3HH-r2_12HH,
                      cmap='jet', vmin=0, vmax=0.6, shading='flat')
    sc.set_edgecolor('face')
    m.colorbar(sc, location='bottom')
    x, y = m(-123, 50.5)
    plt.text(x, y, '(b) d(3h-12h)',
             fontweight='bold', fontsize=14)

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure10.npz')

    acf_target_DAC_3 = data['arr_0']
    acf_pred_DAC_3 = data['arr_1']

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_6HH/figure10.npz')

    acf_target_DAC_6 = data['arr_0']
    acf_pred_DAC_6 = data['arr_1']

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_12HH/figure10.npz')

    acf_target_DAC_12 = data['arr_0']
    acf_pred_DAC_12 = data['arr_1']

    colors = ["white", "white", "lavenderblush", "darksalmon",
              "salmon", "tomato", "red"]
    clrmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)
    plt.subplot2grid((2, 2), (1, 0))
    RegisterCustomColormaps()
    cmap = plt.get_cmap('bias')
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)
    m.drawcoastlines()
    # m.drawcountries()
    x, y = m(lon, lat)
    sc = m.pcolormesh(x, y,
                      acf_target_DAC_3-acf_target_DAC_6,
                      cmap='jet', vmin=0, vmax=0.3)
    sc.set_edgecolor('face')
    m.colorbar(sc, location='bottom')
    x, y = m(-123, 50.5)
    plt.text(x, y, '(c) ACF(3h-6h)',
             fontweight='bold', fontsize=14,)
    # ----------------------------- Figure 5 (b) -------------------------------
    plt.subplot2grid((2, 2), (1, 1))
    RegisterCustomColormaps()
    cmap = plt.get_cmap('bias')
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)
    m.drawcoastlines()
    # m.drawcountries()
    x, y = m(lon, lat)
    sc = m.pcolormesh(x, y,
                      acf_target_DAC_3-acf_target_DAC_12,
                      cmap='jet', vmin=0, vmax=0.3)
    sc.set_edgecolor('face')
    m.colorbar(sc, location='bottom')
    x, y = m(-123, 50.5)
    plt.text(x, y, '(d) ACF(3h-12h)',
             fontweight='bold', fontsize=14)
    plt.savefig('/Users/lewlee/Desktop/figure15.pdf')
    """
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure6.npz')

    acf_target_DAC_3 = data['arr_0']
    acf_pred_DAC_3 = data['arr_1']

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_6HH/figure6.npz')

    acf_target_DAC_6 = data['arr_0']
    acf_pred_DAC_6 = data['arr_1']

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_12HH/figure6.npz')

    acf_target_DAC_12 = data['arr_0']
    acf_pred_DAC_12 = data['arr_1']

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(acf_target_DAC_3)
    plt.plot(acf_pred_DAC_3)
    plt.legend(['3h target', '3h predict', ], loc=3)
    r2_score(acf_target_DAC_3, acf_pred_DAC_3)
    plt.text(20, 0.55, '$R^{2}$ = ' +
             str(round(r2_score(acf_target_DAC_3, acf_pred_DAC_3), 3)))
    plt.text(len(acf_target_DAC_3)-80, 0.15,
             '(a)', fontweight='bold', fontsize=14)
    plt.xlim(0, len(acf_target_DAC_3))
    plt.ylim(0.1, 0.6)

    plt.subplot(3, 1, 2)
    plt.plot(acf_target_DAC_6)
    plt.plot(acf_pred_DAC_6)
    plt.legend(['6h target', '6h predict', ], loc=3)

    plt.text(10, 0.55, '$R^{2}$ = ' +
             str(round(r2_score(acf_target_DAC_6, acf_pred_DAC_6), 3)))
    plt.ylabel('Scaler soil moisture')
    plt.xlim(0, len(acf_target_DAC_6))
    plt.text(len(acf_target_DAC_6)-40, 0.15,
             '(b)', fontweight='bold', fontsize=14)
    plt.ylim(0.1, 0.6)

    plt.subplot(3, 1, 3)
    plt.plot(acf_target_DAC_12)
    plt.plot(acf_pred_DAC_12)
    plt.legend(['12h target', '12h predict', ], loc=3)

    plt.text(5, 0.55, '$R^{2}$ = ' +
             str(round(r2_score(acf_target_DAC_12, acf_pred_DAC_12), 3)))
    plt.xlim(0, len(acf_target_DAC_12))
    plt.text(len(acf_target_DAC_12)-20, 0.15,
             '(c)', fontweight='bold', fontsize=14)
    plt.ylim(0.1, 0.6)

    plt.xlabel('Time')
    plt.savefig('/Users/lewlee/Desktop/figure16.pdf')


def figure15():

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_lstm/figure10.npz')
    acf_target_lstm = data['arr_0']
    acf_pred_lstm = data['arr_1']

    plt.figure()
    # ----------------------------- Figure 5 (a) -------------------------------
    plt.subplot2grid((1, 2), (0, 0))
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)
    m.drawcoastlines()
    x, y = m(lon, lat)
    sc = m.pcolor(x, y,
                  acf_target_lstm, vmin=0.8, vmax=1)
    sc.set_edgecolor('face')
    m.colorbar(sc, location='bottom')
    x, y = m(-123, 50.5)
    plt.text(x, y, '(b) AC', fontweight='bold', fontsize=14)
    plt.savefig('/Users/lewlee/Desktop/figure17.pdf')


def table1():
    """Caculate value for table 1
    1. average soil moisture
    2. bias
    3. R2
    4. RMSE
    5. ACF
    6. Moran
    7. R2(R2, ACF)
    """
    # load
    # load figure5.npz
    data = np.load(
        # figure5.npz')
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')
    target = data['arr_0']
    d = data['arr_1']
    rmse = data['arr_2']
    # rmse[rmse < 0] = np.nan
    r2 = data['arr_3']
    # r2[r2 < 0] = np.nan
    # load figure10.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure10.npz')
    acf = data['arr_0']
    # init
    I = np.full((target.shape[0], target.shape[1]), np.nan)
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
            I[i, j] = Z[i, j]*WZ[i, j]
            Z2[i, j] = Z[i, j] * Z[i, j]
    # index
    _, index_KG = KG()
    for k in range(1, len(index_KG)):
        T = target[index_KG[str(k)][0], index_KG[str(k)][1]]
        D = d[index_KG[str(k)][0], index_KG[str(k)][1]]
        R = r2[index_KG[str(k)][0], index_KG[str(k)][1]]
        RM = rmse[index_KG[str(k)][0], index_KG[str(k)][1]]
        A = acf[index_KG[str(k)][0], index_KG[str(k)][1]]
        M = I[index_KG[str(k)][0], index_KG[str(k)][1]]
        print("region {}: \naverage SM is {} \naverage bias is {}"
              .format(k, np.nanmean(T), np.nanmean(D)))
        print("average R2 is {} \naverage RMSE is {}"
              .format(np.nanmean(R), np.nanmean(RM)))
        print("average AC is {} \naverage Moran is {}"
              .format(np.nanmean(A), np.nanmean(M)))

        R[np.where(np.isnan(A))] = 0
        A[np.where(np.isnan(A))] = 0

        A[np.where(np.isnan(R))] = 0
        R[np.where(np.isnan(R))] = 0

        print("average R2(R2,AC) is {}"
              .format(np.corrcoef(A, R)[0, 1]))

        A[np.where(np.isnan(M))] = 0
        R[np.where(np.isnan(M))] = 0
        M[np.where(np.isnan(M))] = 0
        # A = np.concatenate((A[].reshape(-1, 1), M.reshape(-1, 1)), axis=-1)
        print(A.shape)
        print("average R2(R2,MI) is {}"
              .format(np.corrcoef(M, R)[0, 1]))
        print('---------------------------------\n')

        print('---------------------------------\n')


if __name__ == "__main__":
    # figure5()
    # figure6()
    # figure7()
    # figure8()
    # figure9()
    # figure10()
    # figure11()
    # figure12()
    # figure13()
    # figure14()
    # figure15()
    table1()

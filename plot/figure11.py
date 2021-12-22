
import sys

sys.path.append('../')

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

plt.rc('font', family='Times New Roman')
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.basemap import Basemap
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import fillna, gen_meshgrid, gen_metric, gen_tac_sac

import sys

sys.path.append('../')

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

plt.rc('font', family='Times New Roman')

from src.utils import fillna, gen_meshgrid, gen_metric, gen_tac_sac


def figure11():

    # generate DAC
    _, rmse_DAC, r2_DAC = gen_metric(
        '/hard/lilu/1_AttConvLSTM/outputs/figures_DAC/figure5.npz')

    # generate rf
    _, rmse_rf, r2_rf = gen_metric(
        '/hard/lilu/1_AttConvLSTM/outputs/figures_rf/figure5.npz')

    # generate lstm
    _, rmse_lstm, r2_lstm = gen_metric(
        '/hard/lilu/1_AttConvLSTM/outputs/figures_lstm/figure5.npz')

    # generate d()
    a = r2_DAC - r2_rf
    b = r2_DAC - r2_lstm

    print(a > 0)
    print(~np.isnan(a))
    print(sum(sum(a > 0)) / sum(sum(~np.isnan(a))))
    print(sum(sum(b>0))/sum(sum(~np.isnan(b))))


    a[a < 0] = 0
    t1 = str(round(np.nanmean(a[:]), 3))
    a[a > 0.3] = 0.3


    b[b < 0] = 0
    t2 = str(round(np.nanmean(b[:]), 3))
    b[b > 0.3] = 0.3

    # load sac, tac
    tac, sac = gen_tac_sac()

    quartertac, mediumtac, quarter3tac = np.nanpercentile(tac.reshape(-1,), (25,50,75))
    quartertac, mediumsac, quarter3sac = np.nanpercentile(sac.reshape(-1,), (25,50,65))

    high_tac = np.full((tac.shape[0], tac.shape[1]), np.nan)
    high_tac[np.where(tac > quarter3tac)] = 1

    high_sac = np.full((tac.shape[0], tac.shape[1]), np.nan)
    high_sac[np.where(sac > quarter3sac)] = 1
    # plot
    plt.figure(figsize=(20, 10))
    
    # generate colormap
    colors = ('white', 'lightcyan', 'cyan', 
              'deepskyblue', 'dodgerblue', 'lightgreen', 'yellow')
    colors1 = ('white', 'white')
    colors1 = ('red','red')

    clrmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)
    whitemap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors1)

    # -----------------------------figure 11(a)---------------------------------
    # subplot
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)

    # projection
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)

    # draw lines
    m.drawcoastlines()
    # m.drawcountries()

    # meshgrid
    lon, lat = gen_meshgrid()
    x, y = m(lon, lat)

    # pcolor
    sc = m.contourf(x, y,
                    a,
                    cmap=clrmap,
                    #color=colors,
                    #levels=levels,
                    vmin=0, vmax=0.3)
    m.contourf(x, y, high_tac, hatches=['\\\\'],cmap=whitemap, alpha=0)
    #m.contourf(x, y, high_sac, hatches=['*'],cmap=whitemap, alpha=0)

    # set edge
    #sc.set_edgecolor('face')

    # text
    x, y = m(-123, 50.5)
    plt.text(x, y, '(a) difference between $R^{2}$ of AttConvLSTM & RF ', fontweight='bold', fontsize=14)
    
    #x, y = m(-122.6, 27.3)
    #plt.text(x, y, '$\overline{\Delta}$ =' + str(t1), fontsize=10, fontweight='bold', c='red')

    #x, y = m(-122.6, 29.0)
    #plt.text(x, y, "\\\ > 75% TAC", fontsize=10, c='red')
    # inset colorbar
    axin1 = ax1.inset_axes([0.899, 0.024, 0.02, 0.3])
    plt.colorbar(sc, cax=axin1,)

    

    # -----------------------------figure 11(b)---------------------------------
    # subplot
    ax2 = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)

    # projection
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)

    # draw lines
    m.drawcoastlines()
    # m.drawcountries()

    # meshgrid
    lon, lat = gen_meshgrid()
    x, y = m(lon, lat)

    # pcolor
    sc = m.contourf(x, y,
                      b,
                      cmap=clrmap, 
                      vmin=0, vmax=0.3)

    # set edge
    #sc.set_edgecolor('face')
    #m.contourf(x, y, high_tac, hatches=['\\\\'],cmap=whitemap, alpha=0)
    m.contourf(x, y, high_sac, hatches=['..'],cmap=whitemap, alpha=0)

    # text
    x, y = m(-123., 50.5)
    plt.text(x, y, '(b) difference between $R^{2}$ of AttConvLSTM & LSTM ',
 fontweight='bold', fontsize=14)

    #x, y = m(-122.6, 27.3)
    #plt.text(x, y, '$\overline{\delta}$ =' + str(t2),fontsize=10, fontweight='bold', c='red')
    #x, y = m(-122.6, 29.0)
    #plt.text(x, y, ".. > 75% SAC",fontsize=10, c='red')
    # inset colorbar
    axin2 = ax2.inset_axes([0.899, 0.024, 0.02, 0.3])
    plt.colorbar(sc, cax=axin2,drawedges=False)

    # -------------------------figure 11(c)-------------------------------------
    # subplot
    ax2 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=2)

    # projection
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)

    # draw lines
    m.drawcoastlines()
    # m.drawcountries()

    # meshgrid
    lon, lat = gen_meshgrid()
    x, y = m(lon, lat)

    # pcolor
    sc = m.contourf(x, y,
                      a-b,
                      cmap=clrmap, 
                      vmin=0, vmax=0.3)

    # set edge
    #sc.set_edgecolor('face')
    #m.contourf(x, y, high_tac, hatches=['\\\\'],cmap=whitemap, alpha=0)




    # text
    x, y = m(-123., 50.5)
    plt.text(x, y, '(c) difference between $R^{2}$ of LSTM & RF ',
 fontweight='bold', fontsize=14)

    # inset colorbar
    axin2 = ax2.inset_axes([0.899, 0.024, 0.02, 0.3])
    plt.colorbar(sc, cax=axin2,drawedges=False)

    plt.subplots_adjust(wspace=0.05, hspace=0.18)
    # --------------------------------------------------------------------------
    
    # save
    plt.savefig('figure11.pdf')
    
def _figure11():

    # generate DAC
    _, rmse_DAC, r2_DAC = gen_metric(
        '/hard/lilu/1_AttConvLSTM/outputs/figures_DAC/figure5.npz')

    # generate rf
    _, rmse_rf, r2_rf = gen_metric(
        '/hard/lilu/1_AttConvLSTM/outputs/figures_rf/figure5.npz')

    # generate lstm
    _, rmse_lstm, r2_lstm = gen_metric(
        '/hard/lilu/1_AttConvLSTM/outputs/figures_lstm/figure5.npz')

    # generate d()
    a = r2_DAC - r2_rf
    b = r2_DAC - r2_lstm

    print(a > 0)
    print(~np.isnan(a))
    print(sum(sum(a > 0)) / sum(sum(~np.isnan(a))))
    print(sum(sum(b>0))/sum(sum(~np.isnan(b))))


    a[a < 0] = 0
    t1 = str(round(np.nanmean(a[:]), 3))
    a[a > 0.3] = 0.3


    b[b < 0] = 0
    t2 = str(round(np.nanmean(b[:]), 3))
    b[b > 0.3] = 0.3

    # load sac, tac
    tac, sac = gen_tac_sac()

    quartertac, mediumtac, quarter3tac = np.nanpercentile(tac.reshape(-1,), (25,50,75))
    quartertac, mediumsac, quarter3sac = np.nanpercentile(sac.reshape(-1,), (25,50,65))

    high_tac = np.full((tac.shape[0], tac.shape[1]), np.nan)
    high_tac[np.where(tac > quarter3tac)] = 1

    high_sac = np.full((tac.shape[0], tac.shape[1]), np.nan)
    high_sac[np.where(sac > quarter3sac)] = 1
    # plot
    plt.figure(figsize=(10, 7))
    
    # generate colormap
    colors = ('white', 'lightcyan', 'cyan', 
              'deepskyblue', 'dodgerblue', 'lightgreen', 'yellow')
    colors1 = ('white', 'white')
    colors1 = ('red','red')

    clrmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)
    whitemap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors1)

    # -----------------------------figure 11(a)---------------------------------
    # subplot
    ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=2)

    # projection
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)

    # draw lines
    m.drawcoastlines()
    # m.drawcountries()

    # meshgrid
    lon, lat = gen_meshgrid()
    x, y = m(lon, lat)

    # pcolor
    sc = m.contourf(x, y,
                    a,
                    cmap=clrmap,
                    #color=colors,
                    #levels=levels,
                    vmin=0, vmax=0.3)
    m.contourf(x, y, high_tac, hatches=['\\\\'],cmap=whitemap, alpha=0)
    #m.contourf(x, y, high_sac, hatches=['*'],cmap=whitemap, alpha=0)

    # set edge
    #sc.set_edgecolor('face')

    # text
    x, y = m(-123, 50.5)
    plt.text(x, y, '(a) difference between $R^{2}$ of AttConvLSTM & RF ($\Delta$)', fontweight='bold', fontsize=14)
    
    x, y = m(-122.6, 27.3)
    plt.text(x, y, '$\overline{\Delta}$ =' + str(t1),
fontsize=10, fontweight='bold', c='red')

    x, y = m(-122.6, 29.0)
    plt.text(x, y, "\\\ > 75% TAC",
fontsize=10, c='red')
    # inset colorbar
    axin1 = ax1.inset_axes([0.899, 0.024, 0.02, 0.3])
    plt.colorbar(sc, cax=axin1,)



    # -----------------------------figure 11(b)---------------------------------
    # subplot
    ax2 = plt.subplot2grid((4, 3), (2, 0), rowspan=2, colspan=2)

    # projection
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)

    # draw lines
    m.drawcoastlines()
    # m.drawcountries()

    # meshgrid
    lon, lat = gen_meshgrid()
    x, y = m(lon, lat)

    # pcolor
    sc = m.contourf(x, y,
                      b,
                      cmap=clrmap, 
                      vmin=0, vmax=0.3)

    # set edge
    #sc.set_edgecolor('face')
    #m.contourf(x, y, high_tac, hatches=['\\\\'],cmap=whitemap, alpha=0)
    m.contourf(x, y, high_sac, hatches=['..'],cmap=whitemap, alpha=0)

    # text
    x, y = m(-123., 50.5)
    plt.text(x, y, '(b) difference between $R^{2}$ of AttConvLSTM & LSTM ($\delta$)',
 fontweight='bold', fontsize=14)

    x, y = m(-122.6, 27.3)
    plt.text(x, y, '$\overline{\delta}$ =' + str(t2),
fontsize=10, fontweight='bold', c='red')
    x, y = m(-122.6, 29.0)
    plt.text(x, y, ".. > 75% SAC",
fontsize=10, c='red')
    # inset colorbar
    axin2 = ax2.inset_axes([0.899, 0.024, 0.02, 0.3])
    plt.colorbar(sc, cax=axin2,drawedges=False)

    # -------------------------figure 11(c)-------------------------------------

    # subplot
    ax2 = plt.subplot2grid((4, 3), (2, 0), rowspan=2, colspan=2)

    # projection
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)

    # draw lines
    m.drawcoastlines()
    # m.drawcountries()

    # meshgrid
    lon, lat = gen_meshgrid()
    x, y = m(lon, lat)

    # pcolor
    sc = m.contourf(x, y,
                      a-b,
                      cmap=clrmap, 
                      vmin=0, vmax=0.3)

    # set edge
    #sc.set_edgecolor('face')
    #m.contourf(x, y, high_tac, hatches=['\\\\'],cmap=whitemap, alpha=0)




    # text
    x, y = m(-123., 50.5)
    plt.text(x, y, '(c) difference between $R^{2}$ of AttConvLSTM & LSTM ($\delta$)',
 fontweight='bold', fontsize=14)

    x, y = m(-122.6, 27.3)
    plt.text(x, y, '$\overline{\delta}$ =' + str(t2),
fontsize=10, fontweight='bold', c='red')
    x, y = m(-122.6, 29.0)
    plt.text(x, y, ".. > 75% SAC",
fontsize=10, c='red')
    # inset colorbar
    axin2 = ax2.inset_axes([0.899, 0.024, 0.02, 0.3])
    plt.colorbar(sc, cax=axin2,drawedges=False)
    """
    # 
    tac = tac.reshape(-1,)
    sac = sac.reshape(-1,)

    # set parameters
    NLEVEL = 100

    # generate min/max of tac
    mintac, maxtac = np.nanmin(tac), np.nanmax(tac)
    #mediumtac = np.nanmean(tac)
    quartertac, mediumtac, quarter3tac = np.nanpercentile(tac, (25,50,75))
    seq_tac = np.arange(mintac, maxtac, (maxtac-mintac)/NLEVEL)

    # generate min/max, medium of sac
    minsac, maxsac = np.nanmin(sac), np.nanmax(sac)
    #mediumsac = np.nanmean(sac)
    quartersac, mediumsac, quarter3sac = np.nanpercentile(sac, (25,50,75))

    seq_sac = np.arange(minsac, maxsac, (maxsac-minsac)/NLEVEL)

    # init
    mean_a_tac = np.zeros_like(seq_tac) # d(DAC-RF)
    mean_a_sac = np.zeros_like(seq_sac)
    mean_b_tac = np.zeros_like(seq_tac) # d(DAC-LSTM)
    mean_b_sac = np.zeros_like(seq_sac)
    std_a_tac = np.zeros_like(seq_tac) # d(DAC-RF)
    std_a_sac = np.zeros_like(seq_sac)
    std_b_tac = np.zeros_like(seq_tac) # d(DAC-LSTM)
    std_b_sac = np.zeros_like(seq_sac)
    
    # 
    a = a.reshape(-1,)
    b = b.reshape(-1,)

    # generate mean r2 according to sac, tac
    for i in range(seq_tac.shape[0] - 1):

        #
        a_tac = a[np.where((tac > seq_tac[i]) & (tac < seq_tac[i + 1]))]
        a_sac = a[np.where((sac > seq_sac[i]) & (sac < seq_sac[i + 1]))]
        b_tac = b[np.where((tac > seq_tac[i]) & (tac < seq_tac[i + 1]))]
        b_sac = b[np.where((sac > seq_sac[i]) & (sac < seq_sac[i + 1]))]

        # generate mean value
        mean_a_tac[i] = np.nanmean(a_tac[:])
        mean_a_sac[i] = np.nanmean(a_sac[:])
        mean_b_tac[i] = np.nanmean(b_tac[:])
        mean_b_sac[i] = np.nanmean(b_sac[:])

        # generate std value
        std_b_tac[i] = np.nanstd(b_tac[:])
        std_b_sac[i] = np.nanstd(b_sac[:])
        std_a_tac[i] = np.nanstd(a_tac[:])
        std_a_sac[i] = np.nanstd(a_sac[:])

    # plot
    

    ax = plt.subplot2grid((4, 3), (0, 2), rowspan=4, colspan=1)
    #ax.errorbar(seq_tac, mean_a_tac, yerr=std_a_tac, linestyle='dotted')
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)



    ax.plot(mean_a_tac, seq_tac, lw=3, color='blue')
    ax.plot(mean_b_tac, seq_tac, lw=1.5, color='cyan')
    ax.set_xlim(xmin=0, xmax=0.2)
    ax.set_ylim(0.8,1)
    ax.set_ylabel('temporal autocorrelation (TAC)', fontsize=12, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    print(mediumtac)
    #ax.axhline(y=mediumtac, c="blue", lw=0.5, linestyle='--')
    ax.axhline(y=mediumtac, c="blue", lw=0.5, linestyle='--')
    ax.text(0.12, mediumtac+0.003, 'median TAC', color='blue')

    ax.scatter(-0.02, mediumtac, s=2, c='blue', marker='<')
    ax.set_xlabel('average $R^{2}$ difference' ,fontsize=12)
    ax.text(0,1.005,'(c)',fontweight='bold', fontsize=14)
    
    ax.plot([0.125, 0.155], [0.97, 0.97], lw=3, color='blue')
    ax.plot([0.125, 0.155], [0.965, 0.965], lw=3, color='red')
    ax.text(0.17, 0.965, '$\Delta$', fontweight='bold', fontsize=14)

    ax.plot([0.125, 0.155], [0.94, 0.94], lw=1.5, color='cyan')
    ax.plot([0.125, 0.155], [0.935, 0.935], lw=1.5, color='fuchsia')
    ax.text(0.17, 0.935, '$\delta$',fontweight='bold', fontsize=14)

    axtwin = ax.twinx()
    mean_a_sac[np.isnan(mean_a_sac)] = 0.05
    mean_b_sac[np.isnan(mean_b_sac)] = 0.05

    axtwin.plot(mean_a_sac, seq_sac, lw=3, color='red')
    axtwin.plot(mean_b_sac, seq_sac, lw=1.5, color='fuchsia')
    axtwin.set_ylabel('spatial autocorrelation (SAC)', fontsize=12, color='hotpink')
    axtwin.tick_params(axis='y', labelcolor='hotpink')
    #axtwin.set_ylim(-0.05+ mediumsac, 0.165+ mediumsac)
    axtwin.set_ylim(0, 0.1)
    axtwin.set_xlim(xmin=0, xmax=0.2)


    axtwin.axhline(y=mediumsac, c="hotpink", lw=0.5, linestyle='--')
    axtwin.text(0.12, mediumsac+0.001, 'median SAC', color='hotpink')
    """


    plt.subplots_adjust(wspace=0.35, hspace=0.27)
    # --------------------------------------------------------------------------
    
    # save
    plt.savefig('figure11.pdf')

if __name__ == "__main__":
    figure11()


    """
    data, index_KG = KG()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] != 6:
                data[i, j] = np.nan
    # m.contourf(x, y, data, hatches=['-'], cmap='gray', alpha=0.5)
    """

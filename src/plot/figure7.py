import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors

plt.rc('font', family='Times New Roman')

from utils import fillna, gen_meshgrid, gen_metric, gen_tac_sac



def figure7():

    tac, sac = gen_tac_sac()

    # fig
    plt.figure(figsize=(10, 7))

    # generate meshgrid
    lon, lat = gen_meshgrid()

    colors = ('white', 'lightcyan', 'cyan', 'darkturquoise',
              'deepskyblue', 'dodgerblue', 'lightgreen','gold','yellow')
    clrmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)

    # ----------------------------- Figure 7 (a) -------------------------------
    ax1 = plt.subplot2grid((1, 2), (0, 0))

    # projection
    m = Basemap(projection='mill',
                llcrnrlat=27, urcrnrlat=50,
                llcrnrlon=-122.9, urcrnrlon=-70.5,)

    # lines
    m.drawcoastlines()

    # generate meshgrid
    x, y = m(lon, lat)

    # countourf
    sc = m.contourf(x, y,
                      sac,
                      cmap=clrmap,
                      vmin=0, vmax=0.2)
    m.colorbar(sc, location='bottom', extend='both', fraction=0.7,
shrink=0.8, pad=0.1, label='Local Moran Index')
    
    # text
    x, y = m(-123, 50.5)
    plt.text(x, y, "(a) spatial autocorrelation (SAC)", fontweight='bold', fontsize=14)
    # inset colorbar
    #axin1 = ax1.inset_axes([0.899, 0.024, 0.02, 0.35])
    #plt.colorbar(sc, cax=axin1,)

    # ----------------------------- Figure 7 (b) -------------------------------
    plt.subplot2grid((1, 2), (0, 1))

    # projection
    m = Basemap(projection='mill',
                llcrnrlat=27, urcrnrlat=50,
                llcrnrlon=-122.9, urcrnrlon=-70.5,)

    # lines
    m.drawcoastlines()

    # meshgrid
    x, y = m(lon, lat)

    # countourf
    sc = m.contourf(x, y,
                      tac,
                      cmap='jet',
                      vmin=0.8, vmax=1)
    m.colorbar(sc, location='bottom',pad=0.1, label='Time Autocorrelation Index')

    # text
    x, y = m(-123, 50.5)
    plt.text(x, y, '(b) temporal autocorrelation (TAC)', fontweight='bold', fontsize=14)

    plt.subplots_adjust(wspace=0.1)
    # --------------------------------------------------------------------------

    # save
    plt.savefig('/Users/lewlee/Desktop/figure7.pdf', dpi=600)


if __name__ == "__main__":
    figure7()

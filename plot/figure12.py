import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
plt.rc('font', family='Times New Roman')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.colors as mcolors

from figure11 import gen_metric
from utils import gen_meshgrid


def figure12():

    # load 3H 
    target_3H, rmse_3H, r2_3H = gen_metric(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')

    # load 6H 
    target_6H, rmse_6H, r2_6H = gen_metric(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_6HH/figure5.npz')

    # load 12H 
    target_12H, rmse_12H, r2_12H = gen_metric(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_12HH/figure5.npz')

    # load spatial mean 6H
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_6HH/figure6.npz')

    target_mean_6 = data['arr_0']
    pred_mean_6 = data['arr_1']

    # load spatial mean 12H
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_12HH/figure6.npz')

    target_mean_12 = data['arr_0']
    pred_mean_12 = data['arr_1']

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure10.npz')

    acf_target_3 = data['arr_0']
    acf_pred_3 = data['arr_1']

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_6HH/figure10.npz')

    acf_target_6 = data['arr_0']
    acf_pred_6 = data['arr_1']

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC_12HH/figure10.npz')

    acf_target_12 = data['arr_0']
    acf_pred_12 = data['arr_1']

    
    # plot
    fig = plt.figure(figsize=(9, 7))

    # colorbar
    #colors = ('white', 'lightcyan', 'cyan', 
    #          'deepskyblue', 'dodgerblue', 'lightgreen',
    #          'yellow', 'darkorange', 'fuchsia', 'hotpink')
    colors = ('white', 'lightcyan', 'cyan','darkturquoise',
              'deepskyblue', 'dodgerblue', 'royalblue',
              'blue')
    clrmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)

    # ----------------------------- Figure 12 (b) ------------------------------
    ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=2)

    # project
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)
    
    # line
    m.drawcoastlines()
    # m.drawcountries()
    
    # meshgrid
    lon, lat = gen_meshgrid()
    x, y = m(lon, lat)

    # turn nan
    a = r2_3H - r2_6H
    a[a<0] = np.nan
    a[a>0.5] = 0.5

    # contourf
    sc = m.contourf(x, y, a,
                    cmap=clrmap,
                    vmin=0, vmax=0.5)

    d3 = acf_target_3 - acf_target_6
    mm = np.nanpercentile(d3, 75)

    kk = np.full((d3.shape[0], d3.shape[1]), np.nan)
    kk[d3 > mm] = 1

    m.contourf(x, y,
               kk,
               hatches=['..'],cmap='gray', alpha=0)
    #sc.set_edgecolor('face')
    
    # test
    x, y = m(-123, 50.5)
    plt.text(x, y, '(a) difference between $R^{2}$ of 3HH & 6HH case',
             fontweight='bold', fontsize=14)

    # inset colorbar
    axin1 = ax1.inset_axes([0.899, 0.024, 0.02, 0.3])
    plt.colorbar(sc, cax=axin1,drawedges=False)

    # ----------------------------- Figure 5 (b) -------------------------------
    ax2 = plt.subplot2grid((4, 3), (2, 0), colspan=2, rowspan=2)

    # project
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.2,)
    
    # line
    m.drawcoastlines()
    # m.drawcountries()
    
    # meshgrid
    lon, lat = gen_meshgrid()
    x, y = m(lon, lat)

    # 

    # turn nan
    b = r2_3H - r2_12H
    b[b<0] = np.nan
    b[b>0.5] = 0.5

    # contourf
    sc = m.contourf(x, y, b,
                    cmap=clrmap,
                    vmin=0, vmax=0.5)
    #sc.set_edgecolor('face')
    d3 = acf_target_3 - acf_target_12
    mm = np.nanpercentile(d3, 75)

    kk = np.full((d3.shape[0], d3.shape[1]), np.nan)
    kk[d3 > mm] = 1

    m.contourf(x, y,
               kk,
               hatches=['..'],cmap='gray', alpha=0)
    # text
    x, y = m(-123, 50.5)
    plt.text(x, y, '(b) difference between $R^{2}$ of 3HH & 12HH case',
             fontweight='bold', fontsize=14)

    # inset colorbar
    axin2 = ax2.inset_axes([0.899, 0.024, 0.02, 0.3])
    plt.colorbar(sc, cax=axin2,drawedges=False)


    # add axes
    axx1 = fig.add_axes([0.629, 0.5324, 0.15,0.329])
    axx1.set_yticks([])
    axx1.spines['right'].set_linewidth(2)
    axx1.spines['top'].set_visible(False)
    axx1.spines['left'].set_visible(False)
    axx1.spines['bottom'].set_linewidth(2)
    axx1.set_xlabel('mean SM')

    axxtwin1 = axx1.twinx()
    axxtwin1.plot(target_mean_6, range(len(target_mean_6)), lw=2, c='black')
    axxtwin1.plot(pred_mean_6, range(len(target_mean_6)), lw=2, c='hotpink')
    axxtwin1.set_ylim(0, len(target_mean_6))
    axxtwin1.spines['right'].set_linewidth(2)
    axxtwin1.spines['top'].set_visible(False)
    axxtwin1.spines['left'].set_visible(False)
    axxtwin1.spines['bottom'].set_linewidth(2)
    axxtwin1.text(0.28, 1000, '$R^{2}$ = ' +
             str(round(r2_score(target_mean_6, pred_mean_6), 3)))
    axxtwin1.text(0.28, 940, 'RMSE = ' +
             str(round(np.sqrt(mean_squared_error(target_mean_6, pred_mean_6)), 3)))

    #axxtwin1.set_ylabel('time')
    axxtwin1.plot([0.35, 0.38], [500, 500], lw=2, c='black')
    axxtwin1.plot([0.35, 0.38], [450, 450], lw=2, c='hotpink')
    axxtwin1.text(0.39, 490, 'SMAP(12HH)', fontsize=6)
    axxtwin1.text(0.39, 440, 'AttConvLSTM(6H)', fontsize=6, c='hotpink')
    axxtwin1.set_yticks([32, 156, 280, 404, 528, 652, 776,900, 1024, 1148,])
    axxtwin1.set_yticklabels(['2018-04', '2018-05', '2018-06',
'2018-07','2018-08','2018-09','2018-10', '2018-11','2018-12','2019-01'])
    axxtwin1.scatter(0.53, 640, s=15, marker='<', c='red')
    axxtwin1.scatter(0.53, 100, s=15, marker='<', c='red')

    axxtwin1.scatter(0.365, 400, s=15, marker='<', c='red')
    axxtwin1.text(0.39, 390, 'catastrophe point', fontsize=6, c='red')

    #axxtwin1.set_xlabel('mean soil moisture')

    axx2 = fig.add_axes([0.629, 0.1285, 0.15,0.329])
    axx2.set_yticks([])
    axx2.spines['right'].set_linewidth(2)
    axx2.spines['top'].set_visible(False)
    axx2.spines['left'].set_visible(False)
    axx2.spines['bottom'].set_linewidth(2)
    
    axxtwin2 = axx2.twinx()
    axxtwin2.plot(target_mean_12, range(len(target_mean_12)),lw=2, c='black')
    axxtwin2.plot(pred_mean_12, range(len(target_mean_12)), lw=2, c='hotpink')
    print(target_mean_6.shape)
    #axxtwin2.set_ylabel('time')
    #axxtwin2.set_xlabel('mean soil moisture')
    axxtwin2.set_ylim(0, len(target_mean_12))
    axxtwin2.spines['right'].set_linewidth(2)
    axxtwin2.spines['top'].set_visible(False)
    axxtwin2.spines['left'].set_visible(False)
    axxtwin2.spines['bottom'].set_linewidth(2)
    axxtwin2.text(0.3, 500, '$R^{2}$ = ' +
             str(round(r2_score(target_mean_12, pred_mean_12), 3)))
    axxtwin2.text(0.3, 470, 'RMSE = ' +
             str(round(np.sqrt(mean_squared_error(target_mean_12, pred_mean_12)), 3)))
    axxtwin2.plot([0.36, 0.39], [250, 250], lw=2, c='black')
    axxtwin2.plot([0.36, 0.39], [225, 225], lw=2, c='hotpink')
    axxtwin2.text(0.40, 245, 'SMAP(12HH)',fontsize=6)
    axxtwin2.text(0.40, 220, 'AttConvLSTM(12HH)', fontsize=6, c='hotpink')
    axxtwin2.set_yticks([16, 78, 140, 202, 264,326, 388,450, 512, 574,])
    axxtwin2.set_yticklabels(['2018-04', '2018-05', '2018-06',
'2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12', '2019-01'])
    axxtwin2.scatter(0.53, 320, s=15, marker='<', c='red')
    axxtwin2.scatter(0.53, 50, s=15, marker='<', c='red')

    axxtwin2.scatter(0.375, 200, s=15, marker='<', c='red')
    axxtwin2.text(0.4, 195, 'catastrophe point', fontsize=6, c='red')

    plt.subplots_adjust(hspace=0.2)
    
    plt.savefig('/Users/lewlee/Desktop/figure12.pdf')

if __name__ == "__main__":
    figure12()


 
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

from figure7 import figure7

plt.rc('font', family='Times New Roman')


def figure8():

    # load sac, tac
    tac, sac = figure7()

    # reshape
    sac = sac.reshape(-1,)
    tac = tac.reshape(-1,)

    # turn nan
    tac[np.where((tac > 1) | (tac < 0.6))] = np.nan
    sac[np.where((sac > 0.2) | (sac < -0.03))] = np.nan
    tac[np.isnan(sac)] = np.nan
    sac[np.isnan(tac)] = np.nan

    # load figure5.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')

    target = data['arr_0']
    rmse = data['arr_2'].reshape(-1,)
    r2 = data['arr_3'].reshape(-1,)
    print(rmse.shape)
    print(r2.shape)
    # turn nan
    r2[r2 < 0] = np.nan
    rmse[rmse < 0] = np.nan

    # remove nan
    r2[np.isnan(tac)] = np.nan
    rmse[np.isnan(tac)] = np.nan
    r2[np.isnan(rmse)] = np.nan

    sac = sac[~np.isnan(r2)]
    tac = tac[~np.isnan(r2)]
    rmse = rmse[~np.isnan(r2)]
    r2 = r2[~np.isnan(r2)]

    # plot
    # ----------------------------- Figure 8(a) --------------------------------
    #colors = ('cyan', 'aqua', 'c', 'mediumseagreen', 'limegreen',
    #            'yellow','lightsalmon','fuchsia','magenta','m')

    # set parameters
    NLEVEL = 100

    # generate min/max of tac
    mintac, maxtac = min(tac), max(tac)
    mediumtac = np.mean(tac)#np.percentile(tac, 50,)
    seq_tac = np.arange(mintac, maxtac, (maxtac-mintac)/NLEVEL)

    # generate min/max, medium of sac
    minsac, maxsac = min(sac), max(sac)
    mediumsac = np.mean(sac)#np.percentile(sac,50,)
    seq_sac = np.arange(minsac, maxsac, (maxsac-minsac)/NLEVEL)

    # init
    mean_r2_tac = np.zeros_like(seq_tac)
    mean_r2_sac = np.zeros_like(seq_sac)
    std_r2_tac = np.zeros_like(seq_tac)
    std_r2_sac = np.zeros_like(seq_sac)

    mean_rmse_tac = np.zeros_like(seq_tac)
    mean_rmse_sac = np.zeros_like(seq_sac)

    # generate mean r2 according to sac, tac
    for i in range(seq_tac.shape[0] - 1):

        # select r2
        r2_tac = r2[np.where((tac > seq_tac[i]) & (tac < seq_tac[i + 1]))]
        r2_sac = r2[np.where((sac > seq_sac[i]) & (sac < seq_sac[i + 1]))]

        # generate mean value
        mean_r2_tac[i] = np.nanmean(r2_tac[:])
        mean_r2_sac[i] = np.nanmean(r2_sac[:])

        # generate std value
        std_r2_tac = np.std(r2_tac[:])
        std_r2_sac = np.std(r2_sac[:])

        # select rmse
        rmse_tac = rmse[np.where((tac > seq_tac[i]) & (tac < seq_tac[i + 1]))]
        rmse_sac = rmse[np.where((sac > seq_sac[i]) & (sac < seq_sac[i + 1]))]

        # generate mean value
        mean_rmse_tac[i] = np.nanmean(rmse_tac[:])
        mean_rmse_sac[i] = np.nanmean(rmse_sac[:])


    # plot
    fig = plt.figure(figsize=(10.5,6.2))
    
    # subplot 1
    ax1 = plt.subplot(1, 2, 1)

    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)

    # plot medium lines 
    ax1.axhline(y=0, c="black", lw=0.1)
    ax1.axvline(x=0, c="black", lw=0.1)

    # scatter
    plt.scatter(tac - mediumtac, sac - mediumsac,
                c=r2, s=6, marker='^', cmap='jet')
    plt.xlim(-0.45,0.23)
    plt.ylim(-0.1, 0.25)
    plt.xlabel('deviation of temporal autocorrelation (TAC)')
    plt.ylabel('deviation of spatial autocorrelation (SAC)')
    plt.text(-0.44, 0.258, '(a)', fontweight='bold', fontsize=12)

    plt.colorbar(orientation='horizontal', extend='both', pad=0.12, label='determinate coefficient ($R^{2}$)')
    # text 
    plt.text(-0.44, -0.085, 'low TAC', c='blue', fontsize=8)
    plt.text(-0.44, -0.095, 'low SAC', c='blue', fontsize=8)

    plt.text(0.13, -0.085, 'high TAC', c='red',fontsize=8)
    plt.text(0.13, -0.095, 'low SAC', c='blue',fontsize=8)

    plt.text(0.13, 0.015, 'high TAC', c='red',fontsize=8)
    plt.text(0.13, 0.005, 'high SAC', c='red',fontsize=8)

    plt.text(-0.44, 0.015, 'low TAC', c='blue',fontsize=8)
    plt.text(-0.44, 0.005, 'high SAC', c='red',fontsize=8)
    
    plt.plot(0.10, 0.23, marker='*', color='red')
    plt.text(0.115, 0.228, 'mean value', fontsize=8)
    # create nest subplots
    axin1 = ax1.inset_axes([0.075, 0.75, 0.4, 0.2])
    axin1.set_facecolor('whitesmoke')

    # plot
    axin1.plot(seq_tac, mean_r2_tac, color='dodgerblue')
    axin1.scatter(mediumtac, 0.34, marker='*', color='red')

    # exclude ticks of x,y
    axin1.set_xticks([])
    axin1.set_yticks([])

    axin1.spines['top'].set_color('fuchsia')
    axin1.spines['bottom'].set_color('dodgerblue')

    # set range of x,y
    axin1.set_xlim(min(seq_tac)+0.01,max(seq_tac)-0.02)
    axin1.set_ylim(0.3, max(mean_r2_tac))

    # give label of x,y
    axin1.set_xlabel('TAC', color='dodgerblue', fontsize=8)
    axin1.set_ylabel('$R^{2}$', color='black', fontsize=8)

    # twit axis
    axin2 = axin1.twiny()

    # plot
    axin2.plot(seq_sac, mean_r2_sac, color='fuchsia')
    axin2.scatter(mediumsac, max(mean_r2_sac)-0.04, marker='*', color='red')

    # exclude ticks of x,y
    axin2.set_xticks([])
    axin2.set_yticks([])

    # set range of x,y 
    axin2.set_xlim(min(seq_sac)+0.01, max(seq_sac)-0.01)
    axin2.set_ylim(0.3, max(mean_r2_sac))

    # give label of x,y
    axin2.set_xlabel('SAC', color='fuchsia', fontsize=8)


    # subplot 1
    ax2 = plt.subplot(1, 2, 2)

    ax2.spines['top'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)

    # plot medium lines 
    ax2.axhline(y=0, c="black", lw=0.1)
    ax2.axvline(x=0, c="black", lw=0.1)

    # scatter
    plt.scatter(tac - mediumtac, sac - mediumsac,
                c=rmse, s=6, marker='^', cmap='jet')
    plt.plot(0.10, 0.23, marker='*', color='red')
    plt.text(0.115, 0.228, 'mean value', fontsize=8)
    plt.xlim(-0.45,0.23)
    plt.ylim(-0.1, 0.25)

    plt.xlabel('deviation of temporal autocorrelation (TAC)')
    plt.ylabel('deviation of spatial autocorrelation (SAC)')
    plt.text(-0.44, 0.258, '(b)', fontweight='bold', fontsize=12)

    plt.colorbar(orientation='horizontal', extend='both', pad=0.12,
 label='root mean squared error (RMSE)')
    # text 
    plt.text(-0.44, -0.085, 'low TAC', c='blue', fontsize=8)
    plt.text(-0.44, -0.095, 'low SAC', c='blue', fontsize=8)

    plt.text(0.13, -0.085, 'high TAC', c='red',fontsize=8)
    plt.text(0.13, -0.095, 'low SAC', c='blue',fontsize=8)

    plt.text(0.13, 0.015, 'high TAC', c='red',fontsize=8)
    plt.text(0.13, 0.005, 'high SAC', c='red',fontsize=8)

    plt.text(-0.44, 0.015, 'low TAC', c='blue',fontsize=8)
    plt.text(-0.44, 0.005, 'high SAC', c='red',fontsize=8)
    
    # create nest subplots
    axin1 = ax2.inset_axes([0.075, 0.75, 0.4, 0.2])
    axin1.set_facecolor('whitesmoke')

    # plot
    axin1.plot(seq_tac, mean_rmse_tac, color='dodgerblue')
    axin1.scatter(mediumtac, min(mean_rmse_sac)+0.003, marker='*', color='red')

    # exclude ticks of x,y
    axin1.set_xticks([])
    axin1.set_yticks([])

    axin1.spines['top'].set_color('fuchsia')
    axin1.spines['bottom'].set_color('dodgerblue')

    # set range of x,y
    axin1.set_xlim(min(seq_tac)+0.01,max(seq_tac)-0.02)
    axin1.set_ylim(min(mean_rmse_sac), max(mean_rmse_sac))

    # give label of x,y
    axin1.set_xlabel('TAC', color='dodgerblue', fontsize=8)
    axin1.set_ylabel('RMSE', color='black', fontsize=8)

    # twit axis
    axin2 = axin1.twiny()

    # plot
    axin2.plot(seq_sac, mean_rmse_sac, color='fuchsia')
    axin2.scatter(mediumsac, max(mean_rmse_sac)-0.003, marker='*', color='red')

    # exclude ticks of x,y
    axin2.set_xticks([])
    axin2.set_yticks([])

    # set range of x,y 
    axin2.set_xlim(min(seq_sac)+0.01, max(seq_sac)-0.01)
    axin2.set_ylim(min(mean_rmse_sac), max(mean_rmse_sac))

    # give label of x,y
    axin2.set_xlabel('SAC', color='fuchsia', fontsize=8)
    

    """
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
    """
    
    plt.savefig('/Users/lewlee/Desktop/figure8.pdf')


if __name__ == "__main__":
    figure8()

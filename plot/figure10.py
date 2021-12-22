import seaborn as sns
import numpy as np
from figure7 import figure7
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils import KG

plt.rc('font', family='Times New Roman')


def figure11():

    # load sac, tac
    sac, tac = figure7()
    # plot
    fig = plt.figure(figsize=(6,6))
    ax1 = plt.gca()
    # subplot 1

    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)

    # plot medium lines 
    ax1.axhline(y=0, c="black", lw=0.1)
    ax1.axvline(x=0, c="black", lw=0.1)



    # turn nan
    tac[np.where((tac > 1) | (tac < 0.6))] = np.nan
    sac[np.where((sac > 0.2) | (sac < -0.03))] = np.nan
    tac[np.isnan(sac)] = np.nan
    sac[np.isnan(tac)] = np.nan

    # generate min/max of tac
    mintac, maxtac = np.nanmin(tac[:]), np.nanmax(tac[:])
    mediumtac = np.nanmean(tac[:])#np.percentile(tac, 50,)

    # generate min/max, medium of sac
    minsac, maxsac = np.nanmin(sac[:]), np.nanmax(sac[:])
    mediumsac = np.nanmean(sac[:])#np.percentile(sac,50,)

    # ----------------------------- Figure 11 (c) -------------------------------
    data, index_KG = KG()


    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] != 7 and data[i, j] != 6 and data[i, j] != 1 \
                    and data[i, j] != 3:
                data[i, j] = np.nan
                tac[i, j] = np.nan
                sac[i, j] = np.nan

    tac = tac.reshape(-1, 1)
    sac = sac.reshape(-1, 1)
    data = data.reshape(-1, 1)

    sac = sac[~np.isnan(tac)]
    data = data[~np.isnan(tac)]
    tac = tac[~np.isnan(tac)]


    idx1 = np.where(data == 1)

    print(len(idx1))
    ax1.scatter(tac[idx1]-mediumtac, sac[idx1]-mediumsac,
                c='r', marker='^' , s=6, label='Arid, desert')
    idx1 = np.where(data == 3)
    ax1.scatter(tac[idx1]-mediumtac, sac[idx1]-mediumsac,
                c='grey', marker='*', s=6, label='Temperate,dry summer')
    idx1 = np.where(data == 6)
    ax1.scatter(tac[idx1]-mediumtac, sac[idx1]-mediumsac, c='g',marker='v',
                s=6, label='Cold, dry summer')
    idx1 = np.where(data == 7)
    ax1.scatter(tac[idx1]-mediumtac, sac[idx1]-mediumsac, marker='o',
                c='blue', s=6, label='Cold, dry winter')


    # set label
    plt.xlim(-0.3,0.2)
    plt.ylim(-0.05, 0.15)
    plt.xlabel('deviation of temporal autocorrelation (TAC)')
    plt.ylabel('deviation of spatial autocorrelation (SAC)')
    #plt.text(-0.44, 0.255, 'Distribution of typical climate zones', fontweight='bold', fontsize=12)

    # text 
    plt.text(-0.28, -0.04, 'low TAC', c='blue', fontsize=8)
    plt.text(-0.28, -0.045, 'low SAC', c='blue', fontsize=8)

    plt.text(0.13, -0.04, 'high TAC', c='red',fontsize=8)
    plt.text(0.13, -0.045, 'low SAC', c='blue',fontsize=8)

    plt.text(0.13, 0.01, 'high TAC', c='red',fontsize=8)
    plt.text(0.13, 0.005, 'high SAC', c='red',fontsize=8)

    plt.text(-0.28, 0.01, 'low TAC', c='blue',fontsize=8)
    plt.text(-0.28, 0.005, 'high SAC', c='red',fontsize=8)

    plt.legend(loc='best', bbox_to_anchor=(0.05, 0.6, 0.4, 0.4),
                title='Köppen-Geiger climate index')

    axin1 = ax1.inset_axes([0.75, 0.70, 0.2, 0.1])
    axin1.bar(range(4), [0.692, 0.802, 0.536, 0.763], color=['r', 'grey','g','blue'])
    axin2 = ax1.inset_axes([0.75, 0.85, 0.2, 0.1])
    axin2.bar(range(4), [0.097, 0.21,-0.078, -0.341], color=['r', 'grey','g','blue'])

    axin1.spines['right'].set_visible(False)
    axin1.spines['top'].set_visible(False)
    axin1.set_xticks([])

    axin2.axhline(y=0, c="black", lw=1)
    axin2.spines['bottom'].set_visible(False)
    axin2.spines['right'].set_visible(False)
    axin2.spines['top'].set_visible(False)
    ax1.text(0.08, 0.14, 'R($R^{2}$, SAC)', fontsize=8)
    ax1.text(0.08, 0.11, 'R($R^{2}$, TAC)', fontsize=8)

    axin2.set_xticks([])
#['Arid, desert', 'temperate, dry summer', 'Cold, dry summer', 'Cold, dry winter'])

    """
    ax2 = plt.subplot2grid((4, 8), (0, 4), rowspan=1, colspan=4)
    axin2 = ax2.twinx()
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('yellow')
    ax2.set_yticks([])
    ax2.set_xticks([])
    mean_sm = [0.01, 0.152, 0.159, 0.157, 0.261, 0.169, 0.188, 0.245]
    mean_r2 = [0.793, 0.795, 0.918, 0.845, 0.824, 0.882, 0.629, 0.712]
    mean_rmse = [0.017, 0.022, 0.017, 0.023, 0.025, 0.019, 0.023, 0.02]
    mean_tac = [0.897, 0.906, 0.965, 0.921, 0.927, 0.970, 0.828, 0.870]
    mean_sac = [0.047, 0.018, 0.022, 0.011, 0.024, 0.010, 0.006, 0.025]
    mean_r = [0.692, 0.661, 0.802, 0.723, 0.839, 0.536, 0.763, 0.814]

    axin2.spines['left'].set_visible(False)
    axin2.spines['top'].set_visible(False)
    axin2.set_xticks([])
    axin2.set_ylabel('mean value', color='cyan')
    x =   ['Arid, desert', 'Arid, steppe',
             'Temperate, dry summer', 'Temperate, dry winter',
             'Temperate, no dry season', 'Cold, dry summer',
             'Cold, dry winter', 'Cold, no dry season']
    sns.barplot(x=x, y=mean_r2, palette='vlag', ax=axin2)

    ax3 = plt.subplot2grid((4, 8), (1, 4), rowspan=1, colspan=4)
    axin3 = ax3.twinx()
    ax3.spines['left'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_color('yellow')
    ax3.set_yticks([])
    ax3.set_xticks([])
    axin3.spines['left'].set_visible(False)
    axin3.spines['top'].set_visible(False)
    axin3.set_xticks([])
    axin3.set_ylabel('mean value', color='cyan')
    sns.barplot(x=x, y=mean_rmse, palette='vlag', ax=axin3)

    ax3 = plt.subplot2grid((4, 8), (2, 4), rowspan=1, colspan=4)
    axin3 = ax3.twinx()
    ax3.spines['left'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_color('yellow')
    ax3.set_yticks([])
    ax3.set_xticks([])
    axin3.spines['left'].set_visible(False)
    axin3.spines['top'].set_visible(False)
    axin3.set_xticks([])
    axin3.set_ylabel('', color='cyan')
    sns.barplot(x=x, y=mean_tac, palette='vlag', ax=axin3)

    ax3 = plt.subplot2grid((4, 8), (3, 4), rowspan=1, colspan=4)
    axin3 = ax3.twinx()
    ax3.spines['left'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_color('yellow')
    ax3.set_yticks([])
    ax3.set_xticks([])
    axin3.spines['left'].set_visible(False)
    axin3.spines['top'].set_visible(False)
    axin3.set_xticks([])
    axin3.set_ylabel('sac', color='cyan')
    sns.barplot(x=x, y=mean_sac, palette='vlag', ax=axin3)
    """

    #plt.scatter(range(8), [0.01, 0.152, 0.159, 0.157, 0.261, 0.169, 0.188, 0.245])

    plt.savefig('/Users/lewlee/Desktop/figure11.pdf')

if __name__ == "__main__":
    figure11()

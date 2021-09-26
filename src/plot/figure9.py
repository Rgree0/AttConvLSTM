import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import KG, KG_all, gen_meshgrid
from mpl_toolkits.basemap import Basemap

plt.rc('font', family='Times New Roman')


def figure10():

    # index
    data, index_KG = KG()

    # plot
    fig = plt.figure(figsize=(6,10))
    ax = plt.subplot(2,1,1)
    # load figure5.npz
    data1 = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')
    target_spatial_mean = data1['arr_0']
    index = np.where(np.isnan(target_spatial_mean))
    print(index)
    data[index[0], index[1]] = np.nan
    # ----------------------------- Figure 7 ----------------------------------
    #colors = ('deeppink', 'fuchsia',
    #          'limegreen','darkgreen','forestgreen',
    #          'cyan', 'royalblue', 'deepskyblue')
    colors = ('#FF1493', '#FFC0CB',
              '#20B2AA', '#32CD32', '#7CFC00',
              '#00FFFF',  '#00BFFF', '#0000FF')
    lon, lat = gen_meshgrid()
    m = Basemap(projection='mill', llcrnrlat=27,
                urcrnrlat=50, llcrnrlon=-122.9, urcrnrlon=-70.5,)
    m.drawcoastlines()
    x, y = m(lon, lat)
    levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,8.5]
    sc = ax.contourf(x, y,
                    data,
                    colors=colors,
                    levels=levels, vmin=0, vmax=9)
    data_all, index_KG_all = KG_all()
    levels = [-0.5, 0.5, 1.5, 2.5,]

    #ax.contourf(x, y, data_all, levels=levels, hatches=['-', '/', '\\'], cmap='gray', vmin=0, vmax=2)

    x, y = m(-123, 50.8)
    plt.text(x, y, '(a) Köppen-Geiger climate index',
             fontweight='bold', fontsize=12)
    # create proxy artists to make legend
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
             for pc in sc.collections]
    title = ['Arid, desert', 'Arid, steppe',
             'Temperate, dry summer', 'Temperate, dry winter',
             'Temperate, no dry season', 'Cold, dry summer',
             'Cold, dry winter', 'Cold, no dry season']
    plt.legend(proxy, title, loc='upper right',
               bbox_to_anchor=(1, 1.72))








    title = ['Arid, desert', 'Arid, steppe',
             'Temperate, dry summer', 'Temperate, dry winter',
             'Temperate, no dry season', 'Cold, dry summer',
             'Cold, dry winter', 'Cold, no dry season',]

    # -------------------------kdeplot------------------------------------------
    # load figure5.npz
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')
    rmse = data['arr_2']
    r2 = data['arr_3']
    r2[r2 < 0] = np.nan
    rmse[rmse < 0] = np.nan

    # get medium
    quarterr2, mediumr2, quarter3r2 = np.nanpercentile(r2.reshape(-1,), (25, 50, 75))

    # index
    _, index_KG = KG()



    ax1 = plt.subplot(2, 1, 2)

    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)

    # loop for climate regions
    for k in range(0, len(index_KG)):

        # kdeplot
        if (k <= 2) and (k >= 1):
            color=['#FF1493', '#FFC0CB']#['deeppink', 'fuchsia']
            sns.kdeplot(
                r2[index_KG[str(k)][0], index_KG[str(k)][1]],
                label=title[k-1], linestyle='dashdot', lw=2, color=color[k-1]) 
        if (k <= 5) and (k >= 3):
            color=['#20B2AA', '#32CD32', '#7CFC00']#['limegreen','darkgreen','forestgreen']
            sns.kdeplot(
                r2[index_KG[str(k)][0], index_KG[str(k)][1]],
                label=title[k-1], linestyle='dashed',lw=2, color=color[k-3]) #, linestyle=':'
        if (k <= 8) and (k >= 6):
            color=['#00FFFF',  '#00BFFF', '#0000FF']#['cyan','royalblue','deepskyblue']
            sns.kdeplot(
                r2[index_KG[str(k)][0], index_KG[str(k)][1]],
                label=title[k - 1], linestyle='-', lw=2, color=color[k-6])

    # 
    axin1 = ax1.inset_axes([0.10, 0.2, 0.4, 0.3])
    #axin1.set_facecolor('whitesmoke')
    axin1.spines['top'].set_visible(False)
    axin1.spines['right'].set_visible(False)

    _, index_KG_all = KG_all()

    # loop for climate regions
    for k in range(0, len(index_KG_all)):

        # kdeplot
        if k == 0:
            sns.kdeplot(
                r2[index_KG_all[str(k)][0], index_KG_all[str(k)][1]],
               shade=True,linestyle='dashdot', lw=2.5, color='red', ax=axin1) 
        if k == 1:
            sns.kdeplot(
                r2[index_KG_all[str(k)][0], index_KG_all[str(k)][1]],
               shade=True,linestyle='dashed',lw=2.5, color='green', ax=axin1) #, linestyle=':'
        if k == 2:
            sns.kdeplot(
                r2[index_KG_all[str(k)][0], index_KG_all[str(k)][1]],
                shade=True,linestyle='-', lw=2.5, color='blue', ax=axin1)

    axin1.text(0.2, 4.8, 'arid')
    axin1.text(0.2, 4.0, 'temperate')
    axin1.text(0.2, 3.2, 'cold')
    axin1.plot([0.01,0.15],[3.3,3.3], c='blue', linestyle='-')
    axin1.plot([0.01,0.15],[4.1,4.1], c='green', linestyle='dashed')
    axin1.plot([0.01,0.15],[4.9,4.9], c='red', linestyle='dashdot')


    axin1.set_xticks([])
    axin1.set_yticks([])

    # give label of x,y
    axin1.set_xlabel('$R^{2}$', color='black', fontsize=8)
    axin1.set_ylabel('ED', color='black', fontsize=8)
    plt.ylim(ymax=11.5)
    # 
    plt.scatter(mediumr2, 11.3, marker='^', color='red', s=40)
    plt.scatter(quarterr2, 11.3, marker='^', color='blue', s=40)
    plt.scatter(quarter3r2, 11.3, marker='^', color='pink', s=40)

    plt.text(0, 11.7, '(b) Estimate density of $R^{2}$ in different climate zones', fontweight='bold', fontsize=12)

    plt.scatter(0.52, 10.7, marker='^', color='blue', s=40)
    plt.text(0.55, 10.5, '25 percentile of $R^{2}$')

    plt.scatter(0.52, 10.3, marker='^', color='red', s=40)
    plt.text(0.55, 10.1, '50 percentile of $R^{2}$')
    plt.scatter(0.52, 9.9, marker='^', color='pink', s=40)
    plt.text(0.55, 9.7, '75 percentile of $R^{2}$')

    plt.xlim(0, 1)
    plt.ylim(ymin=0)
    plt.xlabel('determination coefficient ($R^{2}$)')
    plt.ylabel('estimation density (ED)')

    plt.subplots_adjust(hspace=-0.1)

    # save
    plt.savefig('/Users/lewlee/Desktop/figure10.pdf')



    """
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

if __name__ == "__main__":
    figure10()
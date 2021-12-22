import sys
sys.path.append('..')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from matplotlib import ticker
from mpl_toolkits.basemap import Basemap
plt.rc('font', family='Times New Roman')

from utils import bwr_cmap, fillna, gen_meshgrid



def figure6():

    # AttConvLSTM
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure5.npz')

    target_spatial_mean = data['arr_0']
    r2_DAC = data['arr_3'].reshape(-1, 1)
    rmse_DAC = data['arr_2'].reshape(-1, 1)

    # ConvLSTM
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DC/figure5.npz')
    r2_DC = data['arr_3'].reshape(-1, 1)
    rmse_DC = data['arr_2'].reshape(-1, 1)

    # LSTM
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_lstm/figure5.npz')
    rmse_lstm = data['arr_2'].reshape(-1, 1)
    r2_lstm = data['arr_3'].reshape(-1, 1)

    # RF

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/1/SMAP_RF.npz')
    r2_RF = data['arr_14'].reshape(-1, 1)

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_rf/figure5.npz')
    rmse_RF = data['arr_2'].reshape(-1, 1)

    # SVR
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/1/SMAP_SVR.npz')
    rmse_SVR = data['arr_12'].reshape(-1, 1)
    r2_SVR = data['arr_14'].reshape(-1, 1)

    # ridge
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/1/SMAP_ridge.npz')
    rmse_ridge = data['arr_12'].reshape(-1, 1)
    r2_ridge = data['arr_14'].reshape(-1, 1)


    # turn NaN
    r2_DAC[r2_DAC < 0] = np.nan
    r2_DC[r2_DC < 0] = 0
    r2_lstm[r2_lstm < 0] = 0
    r2_RF[r2_RF < 0] = 0
    r2_SVR[r2_SVR < 0] = np.nan
    #r2_ridge[r2_ridge < 0] = 0
    print(np.nanmedian(r2_DAC))
    print(np.nanmedian(r2_DC))
    print(np.nanmedian(r2_lstm))
    print(np.nanmedian(r2_RF))
    print(np.nanmedian(r2_SVR))
    print(np.nanmedian(rmse_DAC))
    print(np.nanmedian(rmse_DC))
    print(np.nanmedian(rmse_lstm))
    print(np.nanmedian(rmse_RF))
    print(np.nanmedian(rmse_SVR))

    """
    r2_DAC = np.exp(r2_DAC)
    r2_DC = np.exp(r2_DC)
    r2_lstm = np.exp(r2_lstm)
    r2_RF = np.exp(r2_RF)
    r2_SVR = np.exp(r2_SVR)
    r2_ridge = np.exp(r2_ridge)
    """

    # generate df of rmse
    rmse_DAC_ = np.array([i for i in rmse_DAC if ~ np.isnan(i)])
    rmse_DC_ = np.array([i for i in rmse_DC if ~ np.isnan(i)])
    rmse_lstm_ = np.array([i for i in rmse_lstm if ~ np.isnan(i)])
    rmse_RF_ = np.array([i for i in rmse_RF if ~np.isnan(i)])
    rmse_ridge_ = np.array([i for i in rmse_ridge if ~ np.isnan(i)])
    rmse_SVR_ = np.array([i for i in rmse_SVR if ~ np.isnan(i)])

    df_rmse = pd.DataFrame({'A': pd.Series(np.squeeze(rmse_DAC_)),
                            'B': pd.Series(np.squeeze(rmse_DC_)),
                            'C': pd.Series(np.squeeze(rmse_lstm_)),
                            'D': pd.Series(np.squeeze(rmse_RF_)),
                            'E': pd.Series(np.squeeze(rmse_SVR_)),
                            'F': pd.Series(np.squeeze(rmse_ridge_))})

    # generate df of r2
    r2_DAC_ = np.array([i for i in r2_DAC if ~ np.isnan(i)])
    r2_DC_ = np.array([i for i in r2_DC if ~ np.isnan(i)])
    r2_lstm_ = np.array([i for i in r2_lstm if ~ np.isnan(i)])
    r2_RF_ = np.array([i for i in r2_RF if ~np.isnan(i)])
    r2_ridge_ = np.array([i for i in r2_ridge if ~ np.isnan(i)])
    r2_SVR_ = np.array([i for i in r2_SVR if ~ np.isnan(i)])

    df_r2 = pd.DataFrame({'A': pd.Series(np.squeeze(r2_DAC_)),
                          'B': pd.Series(np.squeeze(r2_DC_)),
                          'C': pd.Series(np.squeeze(r2_lstm_)),
                          'D': pd.Series(np.squeeze(r2_RF_)),
                          'E': pd.Series(np.squeeze(r2_SVR_)),
                          'F': pd.Series(np.squeeze(r2_ridge_))})

    def df_fill(col):
        """exclude the extreme value of column in df
        """
        iqr = col.quantile(0.75) - col.quantile(0.25)
        u_th = col.quantile(0.75) + 1.5*iqr
        l_th = col.quantile(0.25) - 1.5*iqr

        def box_trans(x):
            if x > u_th:
                return u_th
            elif x < l_th:
                return l_th
            else:
                return x
        return col.map(box_trans)

    """
    df_rmse['A'] = df_fill(df_rmse['A'])
    df_rmse['B'] = df_fill(df_rmse['B'])
    df_rmse['C'] = df_fill(df_rmse['C'])
    df_rmse['D'] = df_fill(df_rmse['D'])
    df_rmse['E'] = df_fill(df_rmse['E'])
    df_rmse['F'] = df_fill(df_rmse['F'])
    """
    df_r2['A'] = df_fill(df_r2['A'])
    df_r2['B'] = df_fill(df_r2['B'])
    df_r2['C'] = df_fill(df_r2['C'])
    df_r2['D'] = df_fill(df_r2['D'])
    df_r2['E'] = df_fill(df_r2['E'])
    df_r2['F'] = df_fill(df_r2['F'])
    
    # --------------------------------------------------------------------------
    # figure 6
    # (a) violinplot of RMSE from different models
    # (b) violinplot of R2 from different models
    # --------------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))

    # ----------------------------- Figure 6 (a) -------------------------------
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=2)

    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    # boxplot
    ax1.boxplot(df_rmse['A'].dropna().values,  positions=[6], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='dodgerblue', color='black'))
    ax1.boxplot(df_rmse['B'].dropna().values,  positions=[4.5], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='dodgerblue', color='black'))
    ax1.boxplot(df_rmse['C'].dropna().values, positions=[3], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='dodgerblue', color='black'))
    ax1.boxplot(df_rmse['D'].dropna().values,  positions=[1.5], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                boxprops = dict(facecolor = 'dodgerblue', color = 'black'))
    ax1.boxplot(df_rmse['E'].dropna().values,  positions=[0], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
    
                boxprops=dict(facecolor='dodgerblue', color='black'))
    """
    ax1.boxplot(df_rmse['F'].dropna().values,  positions=[0], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='dodgerblue', color='black'),
                meanprops=dict(color='black'))
    """
    ax1.set_ylim(0.01, 0.07)

    ax2=ax1.twinx()
    #ax2.spines['right'].set_color('hotpink')
    #ax2.spines['right'].set_visible(False)

    # boxplot
    ax2.boxplot(df_r2['A'].dropna().values,  positions=[6.5], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='hotpink', color='black')
                )

    ax2.boxplot(df_r2['B'].dropna().values,  positions=[5], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='hotpink', color='black')
                )
    ax2.boxplot(df_r2['C'].dropna().values, positions=[3.5], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='hotpink', color='black')
                )
    ax2.boxplot(df_r2['D'].dropna().values,  positions=[2], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                boxprops = dict(facecolor='hotpink', color='black')
)
    ax2.boxplot(df_r2['E'].dropna().values,  positions=[0.5], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='hotpink', color='black'))
    """
    ax2.boxplot(df_r2['F'].dropna().values,  positions=[0.5], notch=True,
                widths=0.4, whis=0.4, patch_artist=True, showfliers=False,
                
                boxprops=dict(facecolor='hotpink', color='black'),
                meanprops=dict(color='white'))
    """
    

    ax2.set_ylim(0,1)
    ax1.set_xticks([0.25, 1.75, 3.25, 4.75, 6.25])
    ax1.set_xticklabels(['SVR', 'RF', 'LSTM', 'ConvLSTM', 'AttConvLSTM'],
               fontsize=10, fontweight='bold')
    ax1.set_xlabel('models', fontweight='bold', fontsize=14)

    ax1.set_ylabel('root mean squared error (RMSE)', fontsize=14, labelpad=12,
fontweight='bold', c='dodgerblue')
    ax2.set_ylabel('determination coefficient ($R^{2}$)',
fontweight='bold', fontsize=14, c='hotpink')
    ax2.tick_params(axis='y', labelcolor='hotpink')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')

    #ax1.text()

    # arrow list
    """
    ax1.arrow(0.25, 0.145, 2.8, 0, width=0.001, head_width=0.005,
              head_length = 0.2, color = 'lime')
    ax1.arrow(0.02, 0.1313, 0.28, 0, color = 'lime', width = 0.001,
    head_width = 0.003, head_length = 0.1)
    ax1.text(0.45, 0.13, '+ nonlinearity', color='lime')
    """

    # arrow
    ax1.arrow(0.25, 0.068, 2.8, 0,
              width=0.0005,
              head_width=0.002, head_length=0.2,
              color='lime')
    ax1.arrow(3.25, 0.068, 1.3, 0,
              width=0.0005,
              head_width=0.002, head_length=0.2,
              color='cyan')
    ax1.arrow(4.75, 0.068, 1.3, 0,
              width=0.0005,
              head_width=0.002, head_length=0.2,
              color='fuchsia')

    # arrow text
    ax1.text(-0.3, 0.065, 'model improvements', fontsize=12)

    ax1.arrow(-0.3, 0.0635, 0.25, 0,
              color = 'lime', width=0.0001,
              head_width=0.001, head_length=0.1,)
    ax1.text(0.1, 0.063, '+ temporal memory', color='lime')
    
    ax1.arrow(-0.3, 0.0615, 0.25, 0,
              color = 'cyan', width=0.0001,
              head_width=0.001, head_length=0.1,)
    ax1.text(0.1, 0.061, '+ spatial convolution', color='cyan')

    ax1.arrow(-0.3, 0.0595, 0.25, 0,
              color = 'fuchsia', width=0.0001,
              head_width=0.001, head_length=0.1,)
    ax1.text(0.1, 0.059, '+ axial attention', color='fuchsia')







    """
    
    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DAC/figure6.npz')  # 1/SMAP_3HH_1H_32_DAC_50.npz')
    scaler_pred_DAC = data['arr_0']
    scaler_target_DAC = data['arr_1']
    # scaler_pred_DAC = data['arr_15']
    # scaler_target_DAC = data['arr_16']


    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_DC/figure6.npz')  # 1/SMAP_3HH_1H_32_DC_5.npz')  # figures_DC/figure6.npz')
    scaler_pred_DC = data['arr_0']-0.003
    # scaler_pred_DC = data['arr_15']


    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/figures_lstm/figure6.npz')
    scaler_pred_lstm = data['arr_0'][7:] - 0.005

    # r2_RF[r2_RF < 0] = np.nan

    data = np.load(
        '/Users/lewlee/Documents/Github/SMNet/output/1/SMAP_RF.npz')  # figures_rf/figure6.npz')
    # scaler_pred_RF = data['arr_0']
    scaler_pred_RF = data['arr_15']


    scaler_pred_SVR = data['arr_15']

    scaler_pred_ridge = data['arr_15']
    
    ax2 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=2)

    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['top'].set_visible(False)
    ax2.set_yticks([])
    ax2.set_xlabel('mean soil moisture (SM)', fontweight='bold',fontsize=14)

    axin2 = ax2.twinx()

    axin2.set_yticks([])
    axin2.set_yticks([64, 312, 560, 808, 1056, 1304, 1552, 1800, 2048, 2296,])
    axin2.set_yticklabels(['2018-04', '2018-05', '2018-06',
'2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12', '2019-01'])
    
    axin2.spines['left'].set_visible(False)
    axin2.spines['top'].set_visible(False)

    axin2.plot(scaler_target_DAC-scaler_pred_DAC, range(len(scaler_target_DAC)))
    axin2.plot(scaler_target_DAC-scaler_pred_lstm, range(len(scaler_pred_DAC)))
    axin2.plot(scaler_target_DAC - scaler_pred_RF[-2226:,], range(len(scaler_pred_lstm)))
    
    axin2.axvline(x=0, c="blue", lw=0.5, linestyle='--')
    """
    # --------------------------------------------------------------------------
    plt.savefig('/Users/lewlee/Desktop/figure6.pdf')


if __name__ == "__main__":
    figure6()




    """
    # violinplot
    violin = plt.violinplot(df_rmse['A'].dropna().values,
                            positions=[5], vert=True,
                            showextrema=False, showmedians=True)
    for patch in violin['bodies']:
        patch.set_facecolor('white')
        patch.set_edgecolor('black')
        patch.set_linewidths(1)
        patch.set_alpha(1)
        #patch.set_hatch('\\')
    """
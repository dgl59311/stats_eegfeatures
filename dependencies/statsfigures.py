import os
import pickle
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def cm2inch(val):
    """since plt works with inch
    need to convert"""
    return val/2.54


def figures_subplot(font_size, width, height, ratio_2):

    plt.style.use('seaborn-white')
    plt.rcParams.update({'font.size': font_size,
                         'xtick.labelsize': font_size,
                         'ytick.labelsize': font_size})

    fig, ax = plt.subplots(1, 2, figsize=(cm2inch(width), cm2inch(height)), gridspec_kw={'width_ratios': [1, ratio_2]})

    return fig, ax


def figures_plot(font_size, width, height):

    plt.style.use('seaborn-white')
    plt.rcParams.update({'font.size': font_size,
                         'xtick.labelsize': font_size,
                         'ytick.labelsize': font_size})

    fig, ax = plt.subplots(1, 1, figsize=(cm2inch(width), cm2inch(height)))

    return fig, ax


def figure_1_ax_0(ax, sp_data, dc_data, font_text, color_sp, color_dc):

    n_vars = len(sp_data['N significant'])
    sp_data = sp_data.iloc[::-1]
    dc_data = dc_data.iloc[::-1]

    ax[0].barh(np.arange(n_vars) + 0.35, sp_data['N significant'], 0.3, label=r'Spearman |$\rho$|', color=color_sp)
    ax[0].barh(np.arange(n_vars), dc_data['N significant'], 0.3, label="Distance correlation $\mathscr{R}_\mathscr{n}$", color=color_dc)
    ax[0].legend(frameon=True, loc='lower right', ncol=1, bbox_to_anchor=(-0.05, -0.1), fontsize=font_text, framealpha=1, shadow=1)
    ax[0].set(yticks=np.arange(n_vars)+0.15, yticklabels=[s.replace("_", " ") for s in list(sp_data.index)])
    ax[0].spines['left'].set_alpha(0.5)
    ax[0].spines['bottom'].set_alpha(0.5)

    ax[0].spines['left'].set_bounds(-0.5, 12)
    ax[0].set_xlim(left=-1, right=51)
    ax[0].set_ylim(bottom=-1, top=12)
    ax[0].tick_params(axis='y', which='major', length=3, width=1.5)
    ax[0].tick_params(axis='x', which='major', length=3, width=1.5)

    ax[0].set_xticks(np.arange(0, 60, 10))
    ax[0].set_xticks(np.arange(0, 50, 10), minor=True)
    ax[0].set_xlabel('# significant EEG features')

    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    for i in range(n_vars):

        x_i = sp_data['N significant'][i]
        num_f = str(int(x_i))
        ax[0].text(x_i+0.4, i+0.2, num_f, fontsize=font_text, color='dimgray')
        min_r = sp_data['min corr'][i]
        max_r = sp_data['max corr'][i]

        if x_i > 1:
            ax[0].text(43.5, i+0.2, format(round(min_r, 2), '.2f') + ' - ' + format(round(max_r, 2), '.2f'), fontsize=font_text, color='dimgray', style='italic')

        elif x_i == 1:
            ax[0].text(43.5, i+0.2, format(round(max_r, 2), '.2f') + ' - ' + format(round(max_r, 2), '.2f'), fontsize=font_text, color='dimgray', style='italic')

        elif x_i == 0:
            print('0')

        x_i_dc = dc_data['N significant'][i]
        num_f = str(int(x_i_dc))
        ax[0].text(x_i_dc+0.4, i-0.2, num_f, fontsize=font_text, color='k')

        min_r = dc_data['min corr'][i]
        max_r = dc_data['max corr'][i]

        if x_i_dc > 1:
            ax[0].text(43.5, i-0.2, format(round(min_r, 2), '.2f') + ' - ' + format(round(max_r, 2), '.2f'), fontsize=font_text, color='k', style='italic')

        elif x_i_dc == 1:
            ax[0].text(43.5, i-0.2, format(round(max_r, 2), '.2f') + ' - ' + format(round(max_r, 2), '.2f'), fontsize=font_text, color='k', style='italic')

        elif x_i_dc == 0:
            print('0')

    ax[0].text(43.5, 12.5, 'range (min-max)\nof significant \ncorrelations', fontsize=font_text,
               verticalalignment='center', style='oblique')

    ax[0].text(0, 12, 'A', fontsize=12)

    return ax


def figure_1_ax_1(ax, sp_data, dc_data, font_text, color_sp, color_dc):

    n_vars = len(sp_data['N significant'])
    sp_data = sp_data.iloc[::-1]
    dc_data = dc_data.iloc[::-1]

    ax[1].barh(np.arange(n_vars) + 0.35, sp_data['within eeg 50'], 0.3,
               xerr=[sp_data['within eeg 25'], sp_data['within eeg 75']-sp_data['within eeg 50']],
               error_kw=dict(lw=1, capsize=2, capthick=1, alpha=0.5), label='Spearman r', color=color_sp)

    ax[1].barh(np.arange(n_vars), dc_data['within eeg 50'], 0.3,
               xerr=[dc_data['within eeg 25'], dc_data['within eeg 75']-dc_data['within eeg 50']],
               error_kw=dict(lw=1, capsize=2, capthick=1, alpha=0.5), label='Distance correlation', color=color_dc)

    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_alpha(0.5)
    ax[1].spines['top'].set_visible(False)

    ax[1].set(yticks=np.arange(12)+0.15, yticklabels=[])
    ax[1].set_xticks(np.arange(0, 1.2, 0.2))
    ax[1].set_xticks(np.arange(0, 1, 0.2), minor=True)
    ax[1].set_xlabel('Correlation between EEG features')

    ax[1].tick_params(axis='y', which='major', length=3, width=1.5)
    ax[1].tick_params(axis='x', which='major', length=3, width=1.5)

    ax[1].set_xlim(left=-0.03, right=1.03)
    ax[1].set_ylim(bottom=-1, top=12)
    ax[1].spines['left'].set_bounds(-0.5, 12)
    ax[1].spines['bottom'].set_alpha(0.5)

    for i in range(n_vars):

        in_25 = sp_data['multivar dc eeg 25'][i]
        in_50 = sp_data['multivar dc eeg 50'][i]
        in_75 = sp_data['multivar dc eeg 75'][i]

        if in_25 != 0 and in_50 != 0 and in_75 != 0:
            ax[1].text(1.1, i+0.2,
                       format(round(in_25, 2), '.2f') + ' - ' + format(round(in_50, 2), '.2f') + ' - ' + format(round(in_75, 2), '.2f'),
                       fontsize=font_text, color='dimgray', style='italic')

        in_25 = dc_data['multivar dc eeg 25'][i]
        in_50 = dc_data['multivar dc eeg 50'][i]
        in_75 = dc_data['multivar dc eeg 75'][i]

        if in_25 != 0 and in_50 != 0 and in_75 !=0:
            ax[1].text(1.1, i-0.2,
                       format(round(in_25, 2), '.2f') + ' - ' + format(round(in_50, 2), '.2f') + ' - ' + format(round(in_75, 2), '.2f'),
                       fontsize=font_text, color='k', style='italic')

    ax[1].text(1.1, 12.5, "multivariate $\sqrt{|\mathscr{R}_\mathscr{n}^*|}$\n(25, 50, 75 percentiles)", fontsize=font_text,
               verticalalignment='center', style='oblique')
    ax[1].text(0, 12, 'B', fontsize=12)
    return ax


def load_sp_dc_data(idgroup, sp_data, task, results_1_dir, results_5_dir):

    # load data for figures 2 and 3
    files_1 = os.listdir(results_1_dir)
    multivardc_mat = pd.read_csv(os.path.join(results_5_dir, '5_dc_fx_' + idgroup + '.csv'), index_col=0)
    str_corr_sp = list(filter(lambda x: '1_correlations_eeg_' + task + '_' + sp_data in x, files_1))
    data_results = pd.read_csv(os.path.join(results_1_dir, str_corr_sp[0]), index_col=0)
    features = list(data_results)
    results_multivardc = multivardc_mat.loc[features][features].T

    return data_results, results_multivardc


def figure_2_3_ax_0(fig, ax, data_results, title, method):

    fig_correlation = np.array(data_results)
    ax[0].set_xticks(np.arange(0, fig_correlation.shape[1], 1))
    ax[0].set_yticks(np.arange(0, fig_correlation.shape[1], 1))
    ax[0].set_xticklabels(np.arange(1, fig_correlation.shape[1]+1, 1))
    ax[0].set_yticklabels(np.arange(1, fig_correlation.shape[1]+1, 1))
    # Minor ticks
    ax[0].set_xticks(np.arange(-0.5, fig_correlation.shape[1], 1), minor=True)
    ax[0].set_yticks(np.arange(-0.5, fig_correlation.shape[1], 1), minor=True)

    if method == 'spearman':
        task_1 = ax[0].imshow(fig_correlation, interpolation="none", cmap='bwr')
        task_1.set_clim(-1, 1)
        ax[0].text(0, -1, title, fontsize=13, weight='bold')
    else:
        task_1 = ax[0].imshow(fig_correlation, interpolation="none", cmap='PuBu')
        task_1.set_clim(0, 1)
        ax[0].text(0, -1.5, title, fontsize=13, weight='bold')

    ax[0].grid(which='minor', color='silver', linestyle='-', linewidth=0.7)
    ax[0].set(xticks=range(fig_correlation.shape[1]), xticklabels=list(data_results))
    ax[0].set_xticks(np.arange(0, fig_correlation.shape[1], 1))

    ax[0].set(yticks=range(fig_correlation.shape[1]), yticklabels=list(data_results))
    ax[0].set_yticks(np.arange(0, fig_correlation.shape[1], 1))
    ax[0].tick_params(axis='x', rotation=90)

    ax[0].set_xticks(np.arange(-0.5, fig_correlation.shape[1], 1), minor=True)
    ax[0].set_yticks(np.arange(-0.5, fig_correlation.shape[1], 1), minor=True)

    ax[0].tick_params(axis='both', which='major', length=1.8)

    plt.tight_layout()

    cbaxes = fig.add_axes([0.14, 0.06, 0.015, 0.175])
    cb = fig.colorbar(task_1, cax=cbaxes)
    cb.ax.tick_params(labelsize=9, labelleft=1, labelright=0)

    if method == 'spearman':
        cb.set_label(label=r"$\rho$", weight='bold', rotation=0, fontsize=10, style='italic', labelpad=10)
    else:
        cb.set_label(label=r"$\mathscr{R}_\mathscr{n}$", rotation=0, fontsize=10, style='italic', labelpad=10)

    return fig, ax


def figure_2_3_ax_1(fig, ax, results_multivardc):

    # get the sqrt of multivariate dc since it approximates the population squared distance correlation     
    fig_multivardc = np.sqrt(np.abs(results_multivardc))

    ax[1].set_xticks(np.arange(0, fig_multivardc.shape[1], 1))
    ax[1].set_yticks(np.arange(0, fig_multivardc.shape[1], 1))
    ax[1].set_xticklabels(np.arange(1, fig_multivardc.shape[1]+1, 1))
    ax[1].set_yticklabels(np.arange(1, fig_multivardc.shape[1]+1, 1))
    # Minor ticks
    ax[1].set_xticks(np.arange(-0.5, fig_multivardc.shape[1], 1), minor=True)
    ax[1].set_yticks(np.arange(-0.5, fig_multivardc.shape[1], 1), minor=True)
    task_2 = ax[1].imshow((fig_multivardc + fig_multivardc.T)-2*np.eye(len(fig_multivardc)), interpolation="none", cmap='gray')
    task_2.set_clim(0, 1)

    divider = make_axes_locatable(ax[1])
    cax = divider.new_vertical(size="7%", pad=0.1, pack_start=True)
    fig.add_axes(cax)
    cb_2 = fig.colorbar(task_2, cax=cax, orientation="horizontal")
    cb_2.ax.tick_params(labelsize=9)
    cb_2.set_label(label=r"$\sqrt{|\mathscr{R}_\mathscr{n}^*|}$", rotation=0, fontsize=10, style='italic', labelpad=10)

    ax[1].set(yticklabels=[])
    ax[1].set(xticklabels=[])

    return fig, ax


def fig_4_load_data(results_1_dir, results_2_dir, method):
    # Young data
    with open(os.path.join(results_2_dir, '2_regression_y.pkl'), 'rb') as f:
        y_r2 = pickle.load(f)
    # Older data
    with open(os.path.join(results_2_dir, '2_regression_o.pkl'), 'rb') as f:
        o_r2 = pickle.load(f)

    if method == 'ridge':
        Y_r2 = np.median(y_r2['ridge_r2_test'], 0)
        O_r2 = np.median(o_r2['ridge_r2_test'], 0)
    else:
        Y_r2 = np.median(y_r2['rf_r2_test'], 0)
        O_r2 = np.median(o_r2['rf_r2_test'], 0)

    # load mask of significant correlations _ young
    Y_MaskFile_sp = os.path.join(results_1_dir, '1_mask_spearman_y.csv')
    Y_dataMF = pd.read_csv(Y_MaskFile_sp, index_col=0)
    Y_dataMF.where(Y_dataMF == 'NS', 1, inplace=True)
    Y_data_sp = Y_dataMF.replace(['NS'], 0)
    Y_MaskFile_dc = os.path.join(results_1_dir, '1_mask_distcorr_y.csv')
    Y_dataMF = pd.read_csv(Y_MaskFile_dc, index_col=0)
    Y_dataMF.where(Y_dataMF == 'NS', 1, inplace=True)
    Y_data_dc = Y_dataMF.replace(['NS'], 0)

    Y_mask = Y_data_sp + 2*Y_data_dc

    # load mask of significant correlations _ old
    O_MaskFile_sp = os.path.join(results_1_dir, '1_mask_spearman_o.csv')
    O_dataMF = pd.read_csv(O_MaskFile_sp, index_col=0)
    O_dataMF.where(O_dataMF == 'NS', 1, inplace=True)
    O_data_sp = O_dataMF.replace(['NS'], 0)

    O_MaskFile_dc = os.path.join(results_1_dir, '1_mask_distcorr_o.csv')
    O_dataMF = pd.read_csv(O_MaskFile_dc, index_col=0)
    O_dataMF.where(O_dataMF == 'NS', 1, inplace=True)
    O_data_dc = O_dataMF.replace(['NS'], 0)

    O_mask = O_data_sp + 2*O_data_dc

    return Y_r2, Y_mask, O_r2, O_mask


def figure_4_l(fig, ax, o_r2, o_mask, y_r2, y_mask, upper_r2, side, colormap, group):

    if group == 'Older':
        ax_id = 0
    else:
        ax_id = 1

    end_n = 87

    if side == 'left':
        n_feats = 87

        if group == 'Older':
            cmfig = ax[ax_id].imshow(o_r2[0:end_n, :], cmap=colormap)
            ax[ax_id].set(yticks=np.arange(n_feats), yticklabels=o_mask.index[0:87])
            full_null = o_mask.values[0:n_feats, :]
        else:
            cmfig = ax[ax_id].imshow(y_r2[0:end_n, :], cmap=colormap)
            ax[ax_id].set(yticks=np.arange(n_feats), yticklabels=[])
            full_null = y_mask.values[0:n_feats, :]

    else:
        n_feats = 88
        if group == 'Older':
            cmfig = ax[ax_id].imshow(o_r2[end_n:, :], cmap=colormap)
            ax[ax_id].set(yticks=np.arange(n_feats), yticklabels=o_mask.index[end_n:])
            full_null = o_mask.values[end_n:, :]
        else:
            cmfig = ax[ax_id].imshow(y_r2[end_n:, :], cmap=colormap)
            ax[ax_id].set(yticks=np.arange(n_feats), yticklabels=[])
            full_null = y_mask.values[end_n:, :]

    cmfig.set_clim(0, upper_r2)
    ax[ax_id].text(5.5, -2, group, fontsize=13, weight='bold', ha='center', va='center')

    ax[ax_id].set_yticks(np.arange(0, n_feats, 1))
    ax[ax_id].set_yticks(np.arange(-0.5, n_feats, 1), minor=True)

    ax[ax_id].set_xticks(np.arange(0, 12, 1))
    ax[ax_id].set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax[ax_id].grid(which='minor', color='silver', linestyle='-', linewidth=0.3)
    ax[ax_id].set(xticks=np.arange(12), xticklabels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'])
    ax[ax_id].set_xlabel('Cognitive scores', labelpad=0.4)
    ax[ax_id].tick_params(axis='y', which='major', length=2)
    ax[ax_id].tick_params(axis='x', which='major', length=2)

    # full mask
    full_null_mask = np.where(full_null > 0)

    for i in range(len(full_null_mask[0])):
        pos_task = full_null_mask[0][i]
        pos_eeg = full_null_mask[1][i]
        sp_dc = full_null[pos_task, pos_eeg]

        if sp_dc == 1:
            ax[ax_id].add_patch(Rectangle((pos_eeg-.5, pos_task-.5), 1, 1, fc='none', ec='limegreen', lw=0.4))
        if sp_dc == 2:
            ax[ax_id].add_patch(Rectangle((pos_eeg-.5, pos_task-.5), 1, 1, fc='none', ec='darkorange', lw=0.4))
        if sp_dc == 3:
            ax[ax_id].add_patch(Rectangle((pos_eeg-.5, pos_task-.5), 1, 1, fc='none', ec='slateblue', lw=0.4))

    if group == 'Younger':

        cax = fig.add_axes([0.03, 0.04, 0.3, 0.007])
        cb = fig.colorbar(cmfig, cax=cax, orientation='horizontal')
        cb.set_label(label="$R^2$", rotation=0, fontsize=10, style='italic', labelpad=0)

    return fig, ax


def figure_4_b(fig, ax):
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set(yticklabels=[])
    ax.set(xticklabels=[])
    ax.scatter(0, 1, s=28, marker='s', edgecolor='limegreen', linewidth=1, c='white',
               label='Significant using Spearman correlation')
    ax.scatter(0, 1, s=28, marker='s', edgecolor='darkorange', linewidth=1, c='white',
               label='Significant using distance correlation')
    ax.scatter(0, 1, s=28, marker='s', edgecolor='slateblue', linewidth=1, c='white',
               label='Significant using Spearman and distance correlation')
    ax.legend(frameon=True, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1), fontsize=7, framealpha=1,
              columnspacing=0.1, labelspacing=1)
    return fig, ax


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2, crop_=0):
    # Crop fig with legends
    if crop_ != 0:
        im2 = im2.crop((0, crop_, im2.width - 1, im2.height))
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def figure_5(width, height, results_3_dir):

    df_BigDiff = pd.read_csv(os.path.join(results_3_dir, '3_mwu_r.csv'), index_col=0)
    df_BigDiff = df_BigDiff.iloc[::-1]
    df_BigDiff.index = [k.replace(".csv", "") for k in list(df_BigDiff.index)]

    sign_fx = np.sign(df_BigDiff['z_stat'].values)
    # to depict negative effect sizes as reduced values for older adults
    sign_fx = sign_fx * -1 
    fx_values = df_BigDiff['r_stat']*sign_fx
    pvals = df_BigDiff['pvalues']

    # see figure size requirements
    fig, ax = plt.subplots(1, figsize=(cm2inch(width), cm2inch(height)))
    plt.style.use('seaborn-white')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-2, len(fx_values)+1])
    # reference lines for cohen's D
    ref_d = [0.1, 0.3, 0.5]
    ref_effect = ['Small', 'Medium', 'Large']
    linestyles = [':', ':', ':']

    for d, e, l in zip(ref_d, ref_effect, linestyles):
        ax.axvline(d,  label='{} effect'.format(e), linewidth=1.3,
                   c='gray', linestyle=l, zorder=1, alpha=0.5)

    ref_d = [-0.1, -0.3, -0.5]
    for d, l in zip(ref_d, linestyles):
        ax.axvline(d, c='gray', linewidth=1.3, linestyle=l, zorder=1, alpha=0.5)

    locs_sig = np.where(pvals < 0.05)[0]
    locs_cd = fx_values

    for i in range(len(locs_sig)):
        ioi = locs_sig[i]
        cdoi = locs_cd[ioi]

        if (ioi % 2) == 0:
            plt.plot([cdoi, 10], [ioi, ioi], c='k', linewidth=0.5, linestyle='--', alpha=0.5)
        else:
            plt.plot([-10, cdoi], [ioi, ioi], c='k', linewidth=0.5, linestyle='--', alpha=0.5)

        ax.tick_params(axis='both', which='both', length=1.5)

    color_alphas = ['darkslateblue' if val < 0.05 else 'lavender'
                    for val in pvals]

    ax.barh(range(df_BigDiff.shape[0]), fx_values,
            xerr=[fx_values-sign_fx*df_BigDiff['rcil_stat'], sign_fx*df_BigDiff['rcih_stat']-fx_values],
            error_kw=dict(lw=0.5, capsize=1, capthick=0.5), color=color_alphas, alpha=0.8, edgecolor='black', linewidth=0.5, height=0.9, zorder=10)

    ax.set_xticks([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])
    plt.xticks(fontsize=8)
    ax.set_xlabel("Effect size " + r"($r$)", fontsize=8)

    plt.yticks(np.arange(1, len(locs_cd), 2), df_BigDiff.index[np.arange(1, len(locs_cd), 2)], fontsize=7)
    ax2 = ax.twinx()
    ax2.set_ylim([-2, len(fx_values)+1])
    ax2.set_yticks(np.arange(0, len(fx_values), 1))
    ax2.tick_params(axis='both', which='both', length=1.5)

    plt.yticks(np.arange(0, len(locs_cd), 2), df_BigDiff.index[np.arange(0, len(locs_cd), 2)], fontsize=7)

    ax2.scatter(-3, 4, s=28, marker='s', edgecolor='black', linewidth=1, c='darkslateblue', label='significant group difference')
    ax2.scatter(-3, 4, s=28, marker='s', edgecolor='black', linewidth=1, c='lavender', label='non-significant group difference')
    ax2.legend(frameon=True, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035), fontsize=7)

    fig.tight_layout()

    return fig, ax


def figure_6(results_4_dir, size):

    Y_correlation = pd.read_csv(os.path.join(results_4_dir, '4_correlation_eeg_y.csv'), index_col=0)
    O_correlation = pd.read_csv(os.path.join(results_4_dir, '4_correlation_eeg_o.csv'), index_col=0)

    fig_correlation = np.triu(Y_correlation.values, 1)+np.tril(O_correlation.values)
    fig_correlation = fig_correlation - np.eye(len(fig_correlation))

    plt.style.use('seaborn-white')
    plt.rcParams.update({'font.size': 5})

    fig, ax = plt.subplots(1, figsize=(cm2inch(size), cm2inch(size)))

    ax.set_xticks(np.arange(0, fig_correlation.shape[1], 1))
    ax.set_yticks(np.arange(0, fig_correlation.shape[1], 1))

    ax.set_xticklabels(np.arange(1, fig_correlation.shape[1]+1, 1))
    ax.set_yticklabels(np.arange(1, fig_correlation.shape[1]+1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, fig_correlation.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, fig_correlation.shape[1], 1), minor=True)

    plt.imshow(fig_correlation, interpolation="none", cmap='bwr')
    plt.clim(-1, 1)
    plt.text(54.5, -3, 'Younger', fontsize=13, weight='bold', ha='center', va='center')
    plt.text(-3, 54.5, 'Older', fontsize=13, rotation='vertical', weight='bold', ha='center', va='center')

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='silver', linestyle='-', linewidth=0.7)
    plt.xticks(range(fig_correlation.shape[1]), list(Y_correlation), rotation=90)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.yticks(range(fig_correlation.shape[1]), list(Y_correlation))
    ax.tick_params(axis='both', which='major', length=1.8)

    cbaxes = fig.add_axes([0.875, 0.012, 0.01, 0.11])
    cb = plt.colorbar(cax=cbaxes)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(label=r"$\rho$", weight='bold', rotation=0, fontsize=8, style='italic')

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, fig_correlation.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, fig_correlation.shape[1], 1), minor=True)

    for i in range(fig_correlation.shape[1]):
        ax.add_patch(Rectangle((i-.5, i-.5), 1, 1, fc='silver', ec='silver', lw=0.1))

    plt.tight_layout()

    return fig, ax


def figure_s2(results_4_dir, results_5_dir, size):

    # load correlation data to get same features
    Y_correlation = pd.read_csv(os.path.join(results_4_dir, '4_correlation_eeg_y.csv'), index_col=0)
    sel_features = list(Y_correlation.index)
    Y_mvdc = pd.read_csv(os.path.join(results_5_dir, '5_dc_fx_y.csv'), index_col=0)
    Y_mvdc = Y_mvdc.loc[sel_features][sel_features]
    O_mvdc = pd.read_csv(os.path.join(results_5_dir, '5_dc_fx_o.csv'), index_col=0)
    O_mvdc = O_mvdc.loc[sel_features][sel_features]

    fig_mvdc = np.abs(np.triu(Y_mvdc.values, 1)+np.triu(O_mvdc.values).T)

    # get the sqrt of multivariate dc since it approximates the population squared distance correlation
    fig_mvdc = np.sqrt(fig_mvdc)
    fig_mvdc = fig_mvdc - np.eye(len(fig_mvdc))

    plt.style.use('seaborn-white')
    plt.rcParams.update({'font.size': 5})

    fig, ax = plt.subplots(1, figsize=(cm2inch(size), cm2inch(size)))

    ax.set_xticks(np.arange(0, fig_mvdc.shape[1], 1))
    ax.set_yticks(np.arange(0, fig_mvdc.shape[1], 1))

    ax.set_xticklabels(np.arange(1, fig_mvdc.shape[1]+1, 1))
    ax.set_yticklabels(np.arange(1, fig_mvdc.shape[1]+1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, fig_mvdc.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, fig_mvdc.shape[1], 1), minor=True)

    plt.imshow(fig_mvdc, interpolation="none", cmap='PuBu')
    plt.clim(0, 1)
    plt.text(54.5, -3, 'Younger', fontsize=13, weight='bold', ha='center', va='center')
    plt.text(-3, 54.5, 'Older', fontsize=13, rotation='vertical', weight='bold', ha='center', va='center')

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='silver', linestyle='-', linewidth=0.7)
    plt.xticks(range(fig_mvdc.shape[1]), list(Y_mvdc), rotation=90)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.yticks(range(fig_mvdc.shape[1]), list(Y_mvdc))
    ax.tick_params(axis='both', which='major', length=1.8)

    cbaxes = fig.add_axes([0.875, 0.012, 0.01, 0.11])
    cb = plt.colorbar(cax=cbaxes)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(label=r"$\sqrt{|\mathscr{R}_\mathscr{n}^*|}$", rotation=0, labelpad=15, fontsize=8, style='italic')

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, fig_mvdc.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, fig_mvdc.shape[1], 1), minor=True)

    for i in range(fig_mvdc.shape[1]):
        ax.add_patch(Rectangle((i-.5, i-.5), 1, 1, fc='silver', ec='silver', lw=0.1))

    plt.tight_layout()

    return fig, ax


def figure_s3_4(results_5_dir, results_6_dir, size, group='y'):

    # load average/csd and zero - referenced data
    if group == 'y':
        multivardc_ = np.sqrt(pd.read_csv(os.path.join(results_5_dir, '5_dc_fx_y.csv'), index_col=0).abs())
        multivardc_z = np.sqrt(pd.read_csv(os.path.join(results_6_dir, '6_dc_fx_y.csv'), index_col=0).abs())
        np.fill_diagonal(multivardc_.values, 0)
    else:
        multivardc_ = np.sqrt(pd.read_csv(os.path.join(results_5_dir, '5_dc_fx_o.csv'), index_col=0).abs())
        multivardc_z = np.sqrt(pd.read_csv(os.path.join(results_6_dir, '6_dc_fx_o.csv'), index_col=0).abs())
        np.fill_diagonal(multivardc_.values, 0)

    # List of zero ref features
    zero_ref_feats = list(multivardc_z)
    feat_index = [feature.replace(' zero', '') for feature in zero_ref_feats]
    fig_mvdc = multivardc_z.T.values + multivardc_.loc[feat_index][feat_index].values

    plt.style.use('seaborn-white')
    plt.rcParams.update({'font.size': 4})

    fig, ax = plt.subplots(1, figsize=(cm2inch(size), cm2inch(size)))

    ax.set_xticks(np.arange(0, fig_mvdc.shape[1], 1))
    ax.set_yticks(np.arange(0, fig_mvdc.shape[1], 1))

    ax.set_xticklabels(np.arange(1, fig_mvdc.shape[1]+1, 1))
    ax.set_yticklabels(np.arange(1, fig_mvdc.shape[1]+1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, fig_mvdc.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, fig_mvdc.shape[1], 1), minor=True)

    plt.imshow(fig_mvdc, interpolation="none", cmap='PuBu')
    plt.clim(0, 1)
    plt.text(fig_mvdc.shape[1]/2, -3, 'AVG/CSD Reference', fontsize=10, weight='bold', ha='center', va='center')
    plt.text(-3, fig_mvdc.shape[1]/2, 'Zero Reference', fontsize=10, rotation='vertical', weight='bold', ha='center', va='center')

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='silver', linestyle='-', linewidth=0.7)
    plt.xticks(range(fig_mvdc.shape[1]), feat_index, rotation=90)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.yticks(range(fig_mvdc.shape[1]), feat_index)
    ax.tick_params(axis='both', which='major', length=1.8)
    cbaxes = fig.add_axes([0.9, 0.006, 0.01, 0.091])
    cb = plt.colorbar(cax=cbaxes)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(label=r"$\sqrt{|\mathscr{R}_\mathscr{n}^*|}$", rotation=0, labelpad=15, fontsize=8, style='italic')

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, fig_mvdc.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, fig_mvdc.shape[1], 1), minor=True)

    plt.tight_layout()

    return fig, ax


def figure_pc_s(ax, tasks, results_dir, method='sp', group='y'):

    for k in range(2):

        task = tasks[k]
        fig_task = task.replace("_", " ")
        # For spearman rho
        file_n = '8_' + task + '_' + group + '_pca_results_' + method
        files_summ = os.listdir(results_dir)
        str_corr = list(filter(lambda x: file_n in x, files_summ))

        if len(str_corr) > 0:
            data_results = pd.read_csv(os.path.join(results_dir, str_corr[0]), index_col=0)
            evar = np.round(data_results['explained variance'].values*100, 2)
            # plot the first three principal components
            ax[k].plot(np.abs(data_results.values[0:3, :-1]).T, np.arange(data_results.shape[0]), linewidth=0.8)
            ax[k].set(yticks=np.arange(data_results.shape[0]), yticklabels=data_results.columns[:-1])
            ax[k].set_xlim(left=0, right=1)
            ax[k].set_xlabel('|PC loadings|\n' + fig_task)
            # For legend
            l_1 = 'PC1: ' + str(evar[0]) + '%'
            l_2 = 'PC2: ' + str(evar[1]) + '%'
            l_3 = 'PC3: ' + str(evar[2]) + '%'

            ax[k].legend([l_1, l_2, l_3], frameon=True, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2), fontsize=7)
            ax[k].tick_params(axis='y', which='major', length=3, width=1)
            plt.tight_layout()

    return ax


def figure_s11(width, height, results_8_dir, group="o"):

    fig, ax = plt.subplots(1, figsize=(cm2inch(width), cm2inch(height)))
    plt.style.use('seaborn-white')
    data_results = pd.read_csv(os.path.join(results_8_dir, '8_group_difference_' + group + '_pca_results_sp.csv'), index_col=0)
    evar = np.round(data_results['explained variance'].values * 100, 2)
    # plot the first three principal components
    ax.plot(np.abs(data_results.values[0:3, :-1]).T, np.arange(data_results.shape[1]-1), linewidth=1)
    ax.set(yticks=np.arange(data_results.shape[1]-1))
    ax.set_yticklabels(labels=data_results.columns[:-1], fontsize=5)
    ax.set_xlim(left=0, right=0.3)
    ax.set_ylim(bottom=-1, top=data_results.shape[1])
    ax.set(xticks=np.arange(0, 0.4, 0.1))
    ax.set_xticklabels(labels=['0.0', '0.1', '0.2', '0.3'], fontsize=8)
    ax.set_xlabel('|PC loadings|', fontsize=10)

    # For legend
    l_1 = 'PC1: ' + str(evar[0]) + '%'
    l_2 = 'PC2: ' + str(evar[1]) + '%'
    l_3 = 'PC3: ' + str(evar[2]) + '%'

    ax.legend([l_1, l_2, l_3], frameon=True, loc='upper right', ncol=1, fontsize=7)
    ax.tick_params(axis='y', which='major', length=3, width=1)
    ax.grid(which='major', color='silver', linestyle='--', linewidth=0.3)
    plt.tight_layout()

    return fig, ax

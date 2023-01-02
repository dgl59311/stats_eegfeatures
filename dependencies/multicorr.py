import os
import tqdm
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.base import clone


def match_et_merge(data_in_beh, data_in_eeg):
    """
    Checks nan and index in data_in_1, matches, 
    and returns one dataframe

    data_in_beh = data frame with id = sub-xxxxx and 
    one cognitive variable
    data_in_eeg = data frame with id = sub-xxxxx
    
    """
    col_beh = list(data_in_beh)[0]
    na_beh = data_in_beh[data_in_beh[col_beh].isna()]        
    na_drop = na_beh.index
    dropped_out = len(na_drop)
    
    if dropped_out > 0:
        # remove nan
        data_in_beh = data_in_beh.drop(na_drop)
        # print('found and removed',len(na_drop),'nan')
        # print('number of subjects = ', len(data_in_beh))
    # match eeg et merge on id
    data_in_eeg = data_in_eeg.loc[data_in_beh.index]
    data_out = pd.merge(left=data_in_beh, right=data_in_eeg, on='id')
    
    return data_out, dropped_out


def id_group_match(ix):
    
    """ 
    return 1 and 2 for y or o, respectively
    to match csv files
    """
    if ix == 'y':
        idgroup = 1
    elif ix == 'o':
        idgroup = 2
    
    return idgroup


def correlate_eeg_beh(data_eeg, data_beh, behvar, metric, group):  
    """ 
    Computes spearman/distance correlations & correct p-values
    Args
    data_eeg = dataframe imported using pd.read_csv
    data_beh = dataframe imported using pd.read_csv
    metric = string either 'spearman' or 'distcorr'
    """
    
    id_group = id_group_match(group)
        
    # match beh et eeg and remove nan
    group_data_beh = data_beh.loc[data_beh['Group'] == id_group, [behvar]]
    # find nan, drop, and match subject id
    merged_data, nan_dropped = match_et_merge(group_data_beh, data_eeg)   
    # get data for analysis
    beh = merged_data[behvar]
    # drop unused data
    eeg = merged_data.drop(['Group', 'Gender', 'Age', behvar], axis=1)
    drop_no_var = eeg.columns[(eeg == 0).all()]
    no_var_n = len(drop_no_var)
    eeg = eeg.drop(columns=drop_no_var)
    # eeg varables n
    vars_eeg = eeg.shape[1] 

    # correlate EEG variables, i.e., electrodes, brain regions or microstate parameters
    if metric == 'spearman':
        corr_variables = [spearmanr(eeg.values[:, k], beh) for k in range(vars_eeg)]
    elif metric == 'distcorr':
        corr_variables = [pg.distance_corr(eeg.values[:, k], beh, n_boot=1000, seed=234) for k in range(vars_eeg)]
    
    corr_variables = np.array(corr_variables)      
    # append zero corr for no variance column
    if no_var_n > 0:
        app_novar = np.zeros((no_var_n, 2))
        app_novar[:, 1] = np.ones(no_var_n)
        corr_variables = np.vstack((corr_variables, app_novar))
        
    # Correct for multiple comparisons
    corr_variables = np.array(corr_variables)
    r_values = corr_variables[:, 0]
    p_values = corr_variables[:, 1]
    _, pvals_corr = pg.multicomp(p_values, method='fdr_bh', alpha=0.05)
    n_sig = len(np.where(pvals_corr < 0.05)[0])
    where_max = np.argmax(abs(r_values))
    max_correlation = r_values[where_max]
    number_significant = n_sig
        
    if n_sig > 0:
        max_correlation_EEGvar_MASK = list(eeg.columns)[where_max]
    else:
        max_correlation_EEGvar_MASK = 'NS'

    return max_correlation, max_correlation_EEGvar_MASK, number_significant, no_var_n
           
           
def distcorr_loop(matrix):
    """
    computes pairwise distance correlations
    returns magnitude and pvalues
    """
    rows, _ = matrix.shape[0], matrix.shape[1]
    r = np.ones(shape=(rows, rows))
    p = np.ones(shape=(rows, rows))
    for i in range(rows):
        for j in range(i+1, rows):
            r_, p_ = pg.distance_corr(matrix[i], matrix[j], n_boot=1000, seed=234)
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
            
    return r, p
        

def task_eeg_variables(path_results, path_eeg_data, data_beh, behvar, group, metric):
    
    """
    Generates a dataframe containing the variables of the EEG features
    that correlated significantly with a cognitive variable
    
    """    
    id_group = id_group_match(group)
    
    file_name = '1_mask_' + metric + '_' + group + '.csv' 
    mask_file = pd.read_csv(os.path.join(path_results, file_name), index_col=0)
    eeg_var = mask_file[behvar].loc[mask_file[behvar] != 'NS']
    significant_analysis = len(eeg_var) 
    
    if significant_analysis > 1:
        
        for k in range(significant_analysis):
            
            name_feat = eeg_var.index[k]
            eeg_feat = pd.read_csv(os.path.join(path_eeg_data, name_feat + '.csv'), index_col=0)
            maxcorrvar_eegfeat = eeg_feat[eeg_var[k]] 
            
            # match beh et eeg and remove nan
            group_data_beh = data_beh.loc[data_beh['Group'] == id_group, [behvar]]
            # find nan, drop, and match subject id
            merged_data, nan_dropped = match_et_merge(group_data_beh, maxcorrvar_eegfeat)   
            
            if k == 0:
                local_concat = merged_data.copy(deep=True)
            else:
                merged_data = merged_data.drop([behvar], axis=1)
                local_concat = pd.concat([local_concat, merged_data], axis=1)
        
        # correlate features  
        if metric == 'spearman':            
            eff_val, p_val = spearmanr(local_concat)       
            
        elif metric == 'distcorr':
            to_correlate = local_concat.values
            eff_val, p_val = distcorr_loop(to_correlate.T)
        
        corr_to_beh = eff_val[0, 1:]
        eff_val = np.delete(eff_val, 0, 0)
        eff_val = np.delete(eff_val, 0, 1)
        # pvalue
        p_val = np.delete(p_val, 0, 0)
        p_val = np.delete(p_val, 0, 1)
        # fill diagonal with max correlation to cognitive variable
        np.fill_diagonal(eff_val, corr_to_beh)
        # index with info on electrode, brain region, or microstate parameter
        to_save = eeg_var.index + '_' + eeg_var.values
        # create correlation magnitude dataframe
        eff_val = pd.DataFrame(data=eff_val, index=to_save, columns=eeg_var.index)
        # create pvalues magnitude dataframe
        p_val = pd.DataFrame(data=p_val, index=to_save, columns=eeg_var.index)
    else:
        local_concat = []
        eff_val = []
        p_val = []
                                                                                        
    return local_concat, eff_val, p_val, metric






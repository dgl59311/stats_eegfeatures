% Scripts to calculate network measures in electrode space

% Directory management
clc; 
clear; 
close all;

% open eeglab
% we used eeglab version 2021.1
eeglab
% we used fieldtrip-20200327
addpath('...\fieldtrip-20200327')
ft_defaults

% get current directory
% use .../amplitude_features
CurrDir = pwd;                                    

% Paths to the data
SubjectsDir = uigetdir([],'Path to the raw data');

% Subjects pre-processed data files
Subject_data = dir(fullfile(SubjectsDir,'*.set'));

% The pool of subjects
Subject_pool = {Subject_data(:).name}';

% load chanlocs
load(fullfile(fileparts(CurrDir),'chanlocs.mat'))

% Path to save .mat results
ResultsDir = uigetdir([], 'Save .mat files with results');

%% 

% use zero ref
zero_ref = 1;

epoch = 4;  % in seconds

% Loop across subjects

for iSubject = 1:size(Subject_pool,1)
    
    % Load participant data
    EEGFile = fullfile(SubjectsDir,Subject_pool{iSubject});
    EEG = pop_loadset(EEGFile);
    
    % Interpolate and resample data
    EEG = interpol(EEG,chanlocs);
    EEG = pop_reref(EEG, []);
    EEG = pop_resample(EEG, 125);
    EEG = eeg_checkset(EEG);
    
    if zero_ref == 1
        % Rereference using REST algorithm
        EEG = zero_ref_f(EEG);
    end
    
    eeglab redraw

    % To analyze with fieldtrip
    FT_eeg = eeglab2fieldtrip(EEG,'preprocessing','none');
    
    if zero_ref ~= 1
        % get Current source density signals using fieldtrip
        cfg_csd = [];
        cfd_csd.method = 'spline';
        FT_eeg = ft_scalpcurrentdensity(cfg_csd, FT_eeg);
    end
    
    % define epochs of 4 seconds in ft.
    cfg_ep = [];
    cfg_ep.length  = epoch; % seconds
    FT_eeg_4    = ft_redefinetrial(cfg_ep, FT_eeg);
    
    %%% frequency analysis for COH and PLV
    
    cfg_freq         = [];
    cfg_freq.method  = 'mtmfft';
    cfg_freq.output  = 'fourier';
    cfg_freq.taper   = 'hanning';
    cfg_freq.foi  = 1:0.5:45;
    cfg_freq.channel = 'all';
    freq   = ft_freqanalysis(cfg_freq, FT_eeg_4);
    
	%%% frequency analysis for WPLI (uses cross-spectrum and all trials)
   
    cfg_freq_2         = [];
    cfg_freq_2.method  = 'mtmfft';
    cfg_freq_2.output  = 'powandcsd';
    cfg_freq_2.taper   = 'hanning';
    cfg_freq.foi  = 1:0.5:45;
    cfg_freq_2.channel = 'all';
    cfg_freq_2.keeptrials  = 'yes';
    freq_2   = ft_freqanalysis(cfg_freq_2, FT_eeg_4);
    
    %%% connectivity parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% more can be found from the following link:
    % http://www.fieldtriptoolbox.org/reference/ft_connectivityanalysis/
    
    % using COH
    cfg_conn           = [];
    cfg_conn.method    = 'coh';
    cfg_conn.complex = 'absimag';
    freq_conn_coh = ft_connectivityanalysis(cfg_conn, freq);
    % connectivity matrix
    nx_coh=freq_conn_coh.cohspctrm;
    
    % using PLV
	cfg_conn           = [];
    cfg_conn.method    = 'plv';
    freq_conn_plv = ft_connectivityanalysis(cfg_conn, freq);
    % connectivity matrix
    nx_plv = freq_conn_plv.plvspctrm;
    
    % using WPLI 
	cfg_conn           = [];
    cfg_conn.method    = 'wpli_debiased';
    freq_conn_wpli = ft_connectivityanalysis(cfg_conn, freq_2);
    % connectivity matrix
    nx_wpli = freq_conn_wpli.wpli_debiasedspctrm;
    
    % get connectivity measures for 5 frequency bands
    
    for band = 1:5
        
        [locutoff,hicutoff] = FiltLims(band);
        
        % coherence
        f_vec_coh = freq_conn_coh.freq;
        [~,ix_low] = min(abs(f_vec_coh-locutoff));
        [~,ix_hi] = min(abs(f_vec_coh-hicutoff));
        coh_m = mean(nx_coh(:,:,ix_low:ix_hi),3);
        
        % nx
        cc_coh = clustering_coef_wu(coh_m)';
        stre_coh = strengths_und(coh_m)';
        betw_centr_coh = betweenness_wei(coh_m)';
        
        % phase locking value
        f_vec_plv = freq_conn_plv.freq;
        [~,ix_low] = min(abs(f_vec_plv-locutoff));
        [~,ix_hi] = min(abs(f_vec_plv-hicutoff));
        plv_m = mean(nx_plv(:,:,ix_low:ix_hi),3);
        
        % nx
        cc_plv = clustering_coef_wu(plv_m)';
        stre_plv = strengths_und(plv_m)';
        betw_centr_plv = betweenness_wei(plv_m)';
        
        % wpli
        f_vec_wpli = freq_conn_wpli.freq;
        [~,ix_low] = min(abs(f_vec_wpli-locutoff));
        [~,ix_hi] = min(abs(f_vec_wpli-hicutoff));
        wpli_values = nx_wpli(:,ix_low:ix_hi);
        wpli_values(wpli_values<0) = 0;
        wpli_m = mean(wpli_values,2);
        wpli_mat = tril(ones(61),-1);
        wpli_mat(wpli_mat==1) = wpli_m;
        wpli_mat = wpli_mat+wpli_mat';
        
        % nx
        cc_wpli = clustering_coef_wu(wpli_mat)';
        stre_wpli = strengths_und(wpli_mat)';
        betw_centr_wpli = betweenness_wei(wpli_mat)';
        
        if band == 1
            
            ICOHdelta_clusteringcoefficient(iSubject,:) = cc_coh;
            ICOHdelta_nodestrength(iSubject,:) = stre_coh;
            ICOHdelta_betweennesscentrality(iSubject,:) = betw_centr_coh;
    
            PLVdelta_clusteringcoefficient(iSubject,:) = cc_plv;
            PLVdelta_nodestrength(iSubject,:) = stre_plv;
            PLVdelta_betweennesscentrality(iSubject,:) = betw_centr_plv;
    
            WPLIdelta_clusteringcoefficient(iSubject,:) = cc_wpli;
            WPLIdelta_nodestrength(iSubject,:) = stre_wpli;
            WPLIdelta_betweennesscentrality(iSubject,:) = betw_centr_wpli;
        
        elseif band == 2
            
            ICOHtheta_clusteringcoefficient(iSubject,:)=cc_coh;
            ICOHtheta_nodestrength(iSubject,:)=stre_coh;
            ICOHtheta_betweennesscentrality(iSubject,:)=betw_centr_coh;
    
            PLVtheta_clusteringcoefficient(iSubject,:)=cc_plv;
            PLVtheta_nodestrength(iSubject,:)=stre_plv;
            PLVtheta_betweennesscentrality(iSubject,:)=betw_centr_plv;
    
            WPLItheta_clusteringcoefficient(iSubject,:)=cc_wpli;
            WPLItheta_nodestrength(iSubject,:)=stre_wpli;
            WPLItheta_betweennesscentrality(iSubject,:)=betw_centr_wpli; 
    
        elseif band == 3
            
            ICOHalpha_clusteringcoefficient(iSubject,:) = cc_coh;
            ICOHalpha_nodestrength(iSubject,:) = stre_coh;
            ICOHalpha_betweennesscentrality(iSubject,:) = betw_centr_coh;
    
            PLValpha_clusteringcoefficient(iSubject,:) = cc_plv;
            PLValpha_nodestrength(iSubject,:) = stre_plv;
            PLValpha_betweennesscentrality(iSubject,:) = betw_centr_plv;
    
            WPLIalpha_clusteringcoefficient(iSubject,:) = cc_wpli;
            WPLIalpha_nodestrength(iSubject,:) = stre_wpli;
            WPLIalpha_betweennesscentrality(iSubject,:) = betw_centr_wpli;
    
        elseif band == 4
            
            ICOHbeta_clusteringcoefficient(iSubject,:) = cc_coh;
            ICOHbeta_nodestrength(iSubject,:) = stre_coh;
            ICOHbeta_betweennesscentrality(iSubject,:) = betw_centr_coh;
    
            PLVbeta_clusteringcoefficient(iSubject,:) = cc_plv;
            PLVbeta_nodestrength(iSubject,:) = stre_plv;
            PLVbeta_betweennesscentrality(iSubject,:) = betw_centr_plv;
    
            WPLIbeta_clusteringcoefficient(iSubject,:) = cc_wpli;
            WPLIbeta_nodestrength(iSubject,:) = stre_wpli;
            WPLIbeta_betweennesscentrality(iSubject,:) = betw_centr_wpli;
    
        elseif band == 5
            
            ICOHgamma_clusteringcoefficient(iSubject,:) = cc_coh;
            ICOHgamma_nodestrength(iSubject,:) = stre_coh;
            ICOHgamma_betweennesscentrality(iSubject,:) = betw_centr_coh;
    
            PLVgamma_clusteringcoefficient(iSubject,:) = cc_plv;
            PLVgamma_nodestrength(iSubject,:) = stre_plv;
            PLVgamma_betweennesscentrality(iSubject,:) = betw_centr_plv;
    
            WPLIgamma_clusteringcoefficient(iSubject,:) = cc_wpli;
            WPLIgamma_nodestrength(iSubject,:) = stre_wpli;
            WPLIgamma_betweennesscentrality(iSubject,:) = betw_centr_wpli;
            
        end
    end
    clear FT_eeg

end

%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1
    
    save('e_ICOH_nx_ZERO.mat','ICOHdelta_clusteringcoefficient','ICOHdelta_nodestrength','ICOHdelta_betweennesscentrality',...
        'ICOHtheta_clusteringcoefficient','ICOHtheta_nodestrength','ICOHtheta_betweennesscentrality',...
        'ICOHalpha_clusteringcoefficient','ICOHalpha_nodestrength','ICOHalpha_betweennesscentrality',...
        'ICOHbeta_clusteringcoefficient','ICOHbeta_nodestrength','ICOHbeta_betweennesscentrality',...
        'ICOHgamma_clusteringcoefficient','ICOHgamma_nodestrength','ICOHgamma_betweennesscentrality');

    save('e_PLV_nx_ZERO.mat','PLVdelta_clusteringcoefficient','PLVdelta_nodestrength','PLVdelta_betweennesscentrality',...
        'PLVtheta_clusteringcoefficient','PLVtheta_nodestrength','PLVtheta_betweennesscentrality',...
        'PLValpha_clusteringcoefficient','PLValpha_nodestrength','PLValpha_betweennesscentrality',...
        'PLVbeta_clusteringcoefficient','PLVbeta_nodestrength','PLVbeta_betweennesscentrality',...
        'PLVgamma_clusteringcoefficient','PLVgamma_nodestrength','PLVgamma_betweennesscentrality');

    save('e_WPLI_nx_ZERO.mat','WPLIdelta_clusteringcoefficient','WPLIdelta_nodestrength','WPLIdelta_betweennesscentrality',...
        'WPLItheta_clusteringcoefficient','WPLItheta_nodestrength','WPLItheta_betweennesscentrality',...
        'WPLIalpha_clusteringcoefficient','WPLIalpha_nodestrength','WPLIalpha_betweennesscentrality',...
        'WPLIbeta_clusteringcoefficient','WPLIbeta_nodestrength','WPLIbeta_betweennesscentrality',...
        'WPLIgamma_clusteringcoefficient','WPLIgamma_nodestrength','WPLIgamma_betweennesscentrality');

else
    
    save('e_ICOH_nx.mat','ICOHdelta_clusteringcoefficient','ICOHdelta_nodestrength','ICOHdelta_betweennesscentrality',...
        'ICOHtheta_clusteringcoefficient','ICOHtheta_nodestrength','ICOHtheta_betweennesscentrality',...
        'ICOHalpha_clusteringcoefficient','ICOHalpha_nodestrength','ICOHalpha_betweennesscentrality',...
        'ICOHbeta_clusteringcoefficient','ICOHbeta_nodestrength','ICOHbeta_betweennesscentrality',...
        'ICOHgamma_clusteringcoefficient','ICOHgamma_nodestrength','ICOHgamma_betweennesscentrality');

    save('e_PLV_nx.mat','PLVdelta_clusteringcoefficient','PLVdelta_nodestrength','PLVdelta_betweennesscentrality',...
        'PLVtheta_clusteringcoefficient','PLVtheta_nodestrength','PLVtheta_betweennesscentrality',...
        'PLValpha_clusteringcoefficient','PLValpha_nodestrength','PLValpha_betweennesscentrality',...
        'PLVbeta_clusteringcoefficient','PLVbeta_nodestrength','PLVbeta_betweennesscentrality',...
        'PLVgamma_clusteringcoefficient','PLVgamma_nodestrength','PLVgamma_betweennesscentrality');

    save('e_WPLI_nx.mat','WPLIdelta_clusteringcoefficient','WPLIdelta_nodestrength','WPLIdelta_betweennesscentrality',...
        'WPLItheta_clusteringcoefficient','WPLItheta_nodestrength','WPLItheta_betweennesscentrality',...
        'WPLIalpha_clusteringcoefficient','WPLIalpha_nodestrength','WPLIalpha_betweennesscentrality',...
        'WPLIbeta_clusteringcoefficient','WPLIbeta_nodestrength','WPLIbeta_betweennesscentrality',...
        'WPLIgamma_clusteringcoefficient','WPLIgamma_nodestrength','WPLIgamma_betweennesscentrality');
end



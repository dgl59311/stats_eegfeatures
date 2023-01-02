% RQA using CPR toolbox and Pred. Maintenance Toolbox
% clc;clear
% Scripts to calculate nonlinear EEG features using crp toolbox and
% the predictive maintenance toolbox

% Directory management
clc; 
clear; 
close all;

% open eeglab
% we used eeglab version 2021.1
eeglab

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

epoch=4; % epoch length in seconds

nchan = 61; % number of channels

% use zero ref
zero_ref = 1;

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

	% Epoch data in x seconds epochs
    EEG = eeg_regepochs( EEG, 'limits', [0 epoch], 'rmbase', NaN, 'recurrence', epoch);
    EEG = eeg_checkset( EEG );
    
    % Dataset info
    n_eps = size(EEG.data,3);
    n_ch = size(EEG.data,1);
    sampl = EEG.srate;
	RQA_C = zeros(n_ch,13,n_eps);
    PRED_M = zeros(n_ch,2,n_eps);
    EPS = zeros(2,n_ch,n_eps);
    time_series = EEG.data;
    
    parfor (ne = 1:n_eps, 6)
        
        signals = squeeze(time_series(:,:,ne));
        
        for c = 1:61
            
            x = double(signals(c,:));
            [~, lag, dim] = phaseSpaceReconstruction(x);
            recrqa = crqa(x',dim,lag,0.1,'rr','nogui');
            co_di = correlationDimension(x,lag,dim);
            ly_exp = lyapunovExponent(x,sampl,lag,dim);
            pred_m = [co_di;ly_exp];
            RQA_C(c,:,ne) = recrqa;
            PRED_M(c,:,ne) = pred_m;
            EPS(:,c,ne) = [lag;dim];
            
        end
    end
    
    RQA_data{iSubject} = RQA_C;
    PRED_data{iSubject} = PRED_M;
    EPS_data{iSubject} = EPS;
    
    fprintf(' %d', iSubject); 

end

%%

for i = 1:size(Subject_pool,1)
    
    Subj_data_1=RQA_data{i};
    Subj_data_2=PRED_data{i};
    
    for nchan = 1:61
        
        % Nonlinear measures
        correlation_dimension(i,nchan)=myBiweight(squeeze(Subj_data_2(nchan,1,:))');
        lyapunov_exponent(i,nchan)=myBiweight(squeeze(Subj_data_2(nchan,2,:))');
        
        % RQA metrics
        RQA_determinism(i,nchan)=myBiweight(squeeze(Subj_data_1(nchan,2,:))');
        RQA_mean_diag(i,nchan)=myBiweight(squeeze(Subj_data_1(nchan,3,:))');
        RQA_entropy(i,nchan)=myBiweight(squeeze(Subj_data_1(nchan,5,:))');
        RQA_laminarity(i,nchan)=myBiweight(squeeze(Subj_data_1(nchan,6,:))');
        RQA_trappingtime(i,nchan)=myBiweight(squeeze(Subj_data_1(nchan,7,:))');
        RQA_rte(i,nchan)=myBiweight(squeeze(Subj_data_1(nchan,11,:))');
        RQA_max_vert(i,nchan)=myBiweight(squeeze(Subj_data_1(nchan,8,:))');     
        
    end
end

%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1
    
    save('rqa_pred_ZERO.mat',...
        'correlation_dimension',...
        'lyapunov_exponent','RQA_determinism',...
        'RQA_mean_diag','RQA_max_diag','RQA_entropy',...
        'RQA_laminarity','RQA_trappingtime','RQA_max_vert','RQA_rte')

else
    
    save('rqa_pred.mat',...
        'correlation_dimension',...
        'lyapunov_exponent','RQA_determinism',...
        'RQA_mean_diag','RQA_max_diag','RQA_entropy',...
        'RQA_laminarity','RQA_trappingtime','RQA_max_vert','RQA_rte')
    
end
%%

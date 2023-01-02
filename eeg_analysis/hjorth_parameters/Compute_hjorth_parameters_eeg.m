% Scripts to calculate Hjorth parameters

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
% Analysis settings

epoch = 4; % epoch length in seconds

% number of channels
nchan = 61;

% use zero ref
zero_ref = 0;

%% 

% To store data

hjorth_activity = zeros(length(Subject_pool),nchan);
hjorth_mobility = zeros(length(Subject_pool),nchan);
hjorth_complexity = zeros(length(Subject_pool),nchan);

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
    
    % Number of epochs
    n_eps = size(EEG.data,3);

    % Features to obtain
    features_s = zeros(n_eps,nchan,3);
    
    for n_epochs = 1:n_eps
        
        for chan = 1:nchan
            
            % Obtain time-series
            x_c=squeeze(EEG.data(chan,:,n_epochs));
            
            % Hjorth parameters
            
            % Activity
            activity = var(x_c);
   
            % Mobility
            mobility = std(diff(x_c))./std(x_c);

            % Complexity
            complexity = std(diff(diff(x_c)))./std(diff(x_c))./mobility;
 
            features_s(n_epochs,chan,1) = activity;
            features_s(n_epochs,chan,2) = mobility;
            features_s(n_epochs,chan,3) = complexity;
                
        end        
    end
    
    hjorth_activity(iSubject,:)=myBiweight(squeeze(features_s(:,:,1))');
    hjorth_mobility(iSubject,:)=myBiweight(squeeze(features_s(:,:,2))');
    hjorth_complexity(iSubject,:)=myBiweight(squeeze(features_s(:,:,3))');
    
end


%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1
    
    save('hjorth_parameters_ZERO.mat',...
        'hjorth_activity','hjorth_mobility','hjorth_complexity');
     
else
    
    save('hjorth_parameters.mat',...
        'hjorth_activity','hjorth_mobility','hjorth_complexity');

end


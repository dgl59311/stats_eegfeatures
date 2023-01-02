% Scripts to calculate entropies

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

% Number of EEG channels
nchan = 61;

% use zero ref
zero_ref = 0;

%%

% To store data

sample_entropy = zeros(length(Subject_pool),nchan);
permutation_entropy = zeros(length(Subject_pool),nchan);
approximate_entropy = zeros(length(Subject_pool),nchan);

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
    
    % Number of available epochs
    n_epochs = size(EEG.data,3);
    
    % Store entropy features
    features_s = zeros(n_epochs,nchan,3);
    
    for n_ep = 1:n_epochs
        
        signals = squeeze(EEG.data(:,:,n_ep));

        parfor (chan = 1:nchan,5)
        %for chan = 1:nchan
            
            % Get time series
            x_c = signals(chan,:);
            
            % Embedding dimension = 3
            
            get_samp_en = sampen(x_c,3,0.2,'euclidean');
            get_perm_en = PE(x_c,1,3,length(x_c)-3);
            get_approx_entr = approximateEntropy(x_c,1,3);
            
            % Save entropy features
            ENS = [get_samp_en;get_perm_en;get_approx_entr];
            features_s(n_ep,chan,:) = ENS;
                
        end    
    end
    
    % Store data
    
    sample_entropy(iSubject,:) = myBiweight(squeeze(features_s(:,:,1))');
    permutation_entropy(iSubject,:) = myBiweight(squeeze(features_s(:,:,2))');
    approximate_entropy(iSubject,:) = myBiweight(squeeze(features_s(:,:,3))');
    
end

%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1
    
    save('entropies_fullband_ZERO.mat',...
         'permutation_entropy','sample_entropy','approximate_entropy');
     
else
    
    save('entropies_fullband.mat',...
         'permutation_entropy','sample_entropy','approximate_entropy');
end

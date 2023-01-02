% Scripts to calculate range EEG measures
% Janir Ramos da Cruz @IST and @EPFL, 30/06/2020
% adapted for statseeg, Dario 2022

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

% use zero ref
zero_ref = 0;

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

    % filter the data for different frequency bands
    Delta = pop_eegfiltnew(EEG, 1, 4, [], 0, [], 0);
    Theta = pop_eegfiltnew(EEG, 4, 8, [], 0, [], 0);
    Alpha = pop_eegfiltnew(EEG, 8, 13, [], 0, [], 0);
    Beta = pop_eegfiltnew(EEG, 13, 30, [], 0, [], 0);
    Gamma = pop_eegfiltnew(EEG, 30, 45, [], 0, [], 0);
    
    % For the first run of subjects, claim some memory to store individual
    % subjects values for the coefficient of variation (rEEG_CV)
    % and the EEG assymetry (rEEG_asymmetry)
    % subject x channels
    if iSubject == 1
        % coefficient of variation
       rEEG_CV_Delta = zeros(size(Subject_pool,1),size(EEG.data,1));
       rEEG_CV_Theta = zeros(size(Subject_pool,1),size(EEG.data,1)); 
       rEEG_CV_Alpha = zeros(size(Subject_pool,1),size(EEG.data,1)); 
       rEEG_CV_Beta = zeros(size(Subject_pool,1),size(EEG.data,1)); 
       rEEG_CV_Gamma = zeros(size(Subject_pool,1),size(EEG.data,1));
       
       % EEG assymetry
       rEEG_asymmetry_Delta = zeros(size(Subject_pool,1),size(EEG.data,1));
       rEEG_asymmetry_Theta = zeros(size(Subject_pool,1),size(EEG.data,1));
       rEEG_asymmetry_Alpha = zeros(size(Subject_pool,1),size(EEG.data,1));
       rEEG_asymmetry_Beta = zeros(size(Subject_pool,1),size(EEG.data,1));
       rEEG_asymmetry_Gamma = zeros(size(Subject_pool,1),size(EEG.data,1));
    end
    
    
    % Loop for each channel
    for ichan = 1:size(EEG.data,1)
        
        % Calculate the rEEG_CV and symmetry for each channel
        [rEEG_CV_Delta(iSubject, ichan),rEEG_asymmetry_Delta(iSubject, ichan)] = rangeEEG(Delta.data(ichan,:),Delta.srate);
        [rEEG_CV_Theta(iSubject, ichan),rEEG_asymmetry_Theta(iSubject, ichan)] = rangeEEG(Theta.data(ichan,:),Theta.srate);
        [rEEG_CV_Alpha(iSubject, ichan),rEEG_asymmetry_Alpha(iSubject, ichan)] = rangeEEG(Alpha.data(ichan,:),Alpha.srate);
        [rEEG_CV_Beta(iSubject, ichan),rEEG_asymmetry_Beta(iSubject, ichan)] = rangeEEG(Beta.data(ichan,:),Beta.srate);
        [rEEG_CV_Gamma(iSubject, ichan),rEEG_asymmetry_Gamma(iSubject, ichan)] = rangeEEG(Gamma.data(ichan,:),Gamma.srate);
            
    end
end

%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1
    
    save('range_eeg_ZERO.mat',...
        'rEEG_asymmetry_Delta','rEEG_asymmetry_Theta','rEEG_asymmetry_Alpha',...
        'rEEG_asymmetry_Beta','rEEG_asymmetry_Gamma','rEEG_CV_Delta',...
        'rEEG_CV_Theta','rEEG_CV_Alpha','rEEG_CV_Beta','rEEG_CV_Gamma');

else
    
    save('range_eeg.mat',...
        'rEEG_asymmetry_Delta','rEEG_asymmetry_Theta','rEEG_asymmetry_Alpha',...
        'rEEG_asymmetry_Beta','rEEG_asymmetry_Gamma','rEEG_CV_Delta',...
        'rEEG_CV_Theta','rEEG_CV_Alpha','rEEG_CV_Beta','rEEG_CV_Gamma');
end


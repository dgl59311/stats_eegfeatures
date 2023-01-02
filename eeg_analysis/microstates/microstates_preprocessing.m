% export data for CARTOOL

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

% Path to save .eph data for Cartool
ResultsDir = uigetdir([], 'Save .eph files with results');

%%

% Use zero ref
zero_ref = 1;

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
    
    % Save ime series
    id_s = Subject_pool{iSubject};
    
    if zero_ref ==1
        out_file = [ResultsDir '\' id_s(1:end-4) '_zero.eph'];
    else
        out_file = [ResultsDir '\' id_s(1:end-4) '.eph'];
    end
    
    vc_mat2eph(EEG.data, EEG.srate, out_file)
    
end

%%

            
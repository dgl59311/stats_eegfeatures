% Scripts to calculate the Lempel-Ziv Complexity (LZC)
%   Janir Ramos da Cruz @IST and @EPFL, 03/02/2020

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

	% Epoch data in x seconds epochs
    EEG = eeg_regepochs( EEG, 'limits', [0 epoch], 'rmbase', NaN, 'recurrence', epoch);
    EEG = eeg_checkset( EEG );
     
    % For the first run of subjects, claim some memory to store individual
    % subjects values for the Lempel-Ziv Complexity (LZC)
    % subject x channels
    if iSubject == 1
       LZC_exhaustive = zeros(size(Subject_pool,1),size(EEG.data,1)); 
       LZC_primitive = zeros(size(Subject_pool,1),size(EEG.data,1)); 
    end
    
    % Claim some memory for the LZC calculation
    % channels x epochs
    tmpLZC_exhaustive = zeros(size(EEG.data,1),size(EEG.data,3));
    tmpLZC_primitive = zeros(size(EEG.data,1),size(EEG.data,3));
    
    % Loop for each channel
    for ichan = 1:size(EEG.data,1)
        % Loop for each segment to have a better estimate
        for iepoch = 1:size(EEG.data,3)
            
            % Calculate the LZC for each channel and epoch
            m_data = median(EEG.data(ichan,:,iepoch));
            logical_data = double(EEG.data(ichan,:,iepoch)>m_data);
            
            [C, ~, ~] = calc_lz_complexity(logical_data, 'exhaustive', true);
            tmpLZC_exhaustive(ichan,iepoch) = C;
            
            [C_2, ~, ~] = calc_lz_complexity(logical_data, 'primitive', true);
            tmpLZC_primitive(ichan,iepoch) = C_2;
            
        end
    end
    
    % Assign the biweight estimate of the mean to avoid influence of
    % outliers
    [LZC_exhaustive(iSubject,:),~] = myBiweight(tmpLZC_exhaustive);
    [LZC_primitive(iSubject,:),~] = myBiweight(tmpLZC_primitive);
    
end


%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1
    
    save('LZC_ZERO.mat',...
        'LZC_exhaustive','LZC_primitive');
     
else
    
    save('LZC.mat',...
        'LZC_exhaustive','LZC_primitive');

end

%%





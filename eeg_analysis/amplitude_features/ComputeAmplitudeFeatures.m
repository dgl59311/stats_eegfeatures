% Scripts to calculate time-frequency features
% Original data: 1-45 Hz

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

% epoch length in seconds
epoch = 4; 

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
    
	% Epoch data in x seconds epochs
    EEG = eeg_regepochs(EEG, 'limits', [0 epoch], 'rmbase', NaN, 'recurrence', epoch);
    EEG = eeg_checkset(EEG);
    
    % Filter the data for different frequency bands
    Delta = pop_eegfiltnew(EEG, 1, 4, [], 0, [], 0);
    Theta = pop_eegfiltnew(EEG, 4, 8, [], 0, [], 0);
    Alpha = pop_eegfiltnew(EEG, 8, 13, [], 0, [], 0);
    Beta = pop_eegfiltnew(EEG, 13, 30, [], 0, [], 0);
    Gamma = pop_eegfiltnew(EEG, 30, 45, [], 0, [], 0);
    
    % amp_total_pow - amplitude total power
    % amp_env_mean - mean of the envelope
    % amp_env_std - standard deviation of the envelope
    % amp_skew - skewness of the signal amplitude
    % amp_kurt - kurtosis of the signal amplitude

    % For the first run of subjects, claim some memory to store individual
    % subjects values for different bands
    % subject x channels
    
    if iSubject == 1
       % for amplitude total power
       amp_total_powDelta = zeros(size(Subject_pool,1),size(EEG.data,1)); 
       amp_total_powTheta = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_total_powAlpha = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_total_powBeta = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_total_powGamma = zeros(size(Subject_pool,1),size(EEG.data,1));
       
       % for mean of the envelope
       amp_env_meanDelta = zeros(size(Subject_pool,1),size(EEG.data,1)); 
       amp_env_meanTheta = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_env_meanAlpha = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_env_meanBeta = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_env_meanGamma = zeros(size(Subject_pool,1),size(EEG.data,1));
       
       % for standard deviation of the envelope
       amp_env_stdDelta = zeros(size(Subject_pool,1),size(EEG.data,1)); 
       amp_env_stdTheta = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_env_stdAlpha = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_env_stdBeta = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_env_stdGamma = zeros(size(Subject_pool,1),size(EEG.data,1));
       
       % for skewness of the signal amplitude
       amp_skewDelta = zeros(size(Subject_pool,1),size(EEG.data,1)); 
       amp_skewTheta = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_skewAlpha = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_skewBeta = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_skewGamma = zeros(size(Subject_pool,1),size(EEG.data,1));
       
       % for kurtosis of the signal amplitude
       amp_kurtDelta = zeros(size(Subject_pool,1),size(EEG.data,1)); 
       amp_kurtTheta = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_kurtAlpha = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_kurtBeta = zeros(size(Subject_pool,1),size(EEG.data,1));
       amp_kurtGamma = zeros(size(Subject_pool,1),size(EEG.data,1));
    end
    
    % Claim some memory for the calculations for different bands
    % channels x epochs
    % for amplitude total power
    tmp_amp_total_powDelta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_total_powTheta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_total_powAlpha = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_total_powBeta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_total_powGamma = zeros(size(EEG.data,1),size(EEG.data,3));
    
    % for mean of the envelope
    tmp_amp_env_meanDelta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_env_meanTheta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_env_meanAlpha = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_env_meanBeta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_env_meanGamma = zeros(size(EEG.data,1),size(EEG.data,3));
    
    % for standard deviation of the envelope
    tmp_amp_env_stdDelta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_env_stdTheta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_env_stdAlpha = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_env_stdBeta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_env_stdGamma = zeros(size(EEG.data,1),size(EEG.data,3));
    
    % for skewness of the signal amplitude
    tmp_amp_skewDelta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_skewTheta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_skewAlpha = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_skewBeta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_skewGamma = zeros(size(EEG.data,1),size(EEG.data,3));
    
    % for kurtosis of the signal amplitude
    tmp_amp_kurtDelta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_kurtTheta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_kurtAlpha = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_kurtBeta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmp_amp_kurtGamma = zeros(size(EEG.data,1),size(EEG.data,3));
    
    % Loop for each channel
    for ichan = 1:size(EEG.data,1)
        % Loop for each segment to have a better estimate
        for iepoch = 1:size(EEG.data,3)
            % for delta - from 1 to 4 Hz
            [tmp_amp_total_powDelta(ichan,iepoch), tmp_amp_env_meanDelta(ichan,iepoch),...
                tmp_amp_env_stdDelta(ichan,iepoch), tmp_amp_skewDelta(ichan,iepoch),...
                tmp_amp_kurtDelta(ichan,iepoch)] = amplitudeFeatures(Delta.data(ichan,:,iepoch));
            
            % for theta - from 4 to 8 Hz
            [tmp_amp_total_powTheta(ichan,iepoch), tmp_amp_env_meanTheta(ichan,iepoch),...
                tmp_amp_env_stdTheta(ichan,iepoch), tmp_amp_skewTheta(ichan,iepoch),...
                tmp_amp_kurtTheta(ichan,iepoch)] = amplitudeFeatures(Theta.data(ichan,:,iepoch));
            
            % for alpha - from 8 to 13 Hz
            [tmp_amp_total_powAlpha(ichan,iepoch), tmp_amp_env_meanAlpha(ichan,iepoch),...
                tmp_amp_env_stdAlpha(ichan,iepoch), tmp_amp_skewAlpha(ichan,iepoch),...
                tmp_amp_kurtAlpha(ichan,iepoch)] = amplitudeFeatures(Alpha.data(ichan,:,iepoch));
            
            % for beta - from 13 to 30 Hz
            [tmp_amp_total_powBeta(ichan,iepoch), tmp_amp_env_meanBeta(ichan,iepoch),...
                tmp_amp_env_stdBeta(ichan,iepoch), tmp_amp_skewBeta(ichan,iepoch),...
                tmp_amp_kurtBeta(ichan,iepoch)] = amplitudeFeatures(Beta.data(ichan,:,iepoch));
            
            % for gamma - from 30 to 70 Hz
            [tmp_amp_total_powGamma(ichan,iepoch), tmp_amp_env_meanGamma(ichan,iepoch),...
                tmp_amp_env_stdGamma(ichan,iepoch), tmp_amp_skewGamma(ichan,iepoch),...
                tmp_amp_kurtGamma(ichan,iepoch)] = amplitudeFeatures(Gamma.data(ichan,:,iepoch));
        end
    end
    
    % Assign the biweight estimate of the mean to avoid influence of
    % outliers
    % for amplitude total power
    [amp_total_powDelta(iSubject,:), ~] = myBiweight(tmp_amp_total_powDelta);
    [amp_total_powTheta(iSubject,:), ~] = myBiweight(tmp_amp_total_powTheta);
    [amp_total_powAlpha(iSubject,:), ~] = myBiweight(tmp_amp_total_powAlpha);
    [amp_total_powBeta(iSubject,:), ~] = myBiweight(tmp_amp_total_powBeta);
    [amp_total_powGamma(iSubject,:), ~] = myBiweight(tmp_amp_total_powGamma);
    
    % for mean of the envelope
    [amp_env_meanDelta(iSubject,:), ~] = myBiweight(tmp_amp_env_meanDelta);
    [amp_env_meanTheta(iSubject,:), ~] = myBiweight(tmp_amp_env_meanTheta);
    [amp_env_meanAlpha(iSubject,:), ~] = myBiweight(tmp_amp_env_meanAlpha);
    [amp_env_meanBeta(iSubject,:), ~] = myBiweight(tmp_amp_env_meanBeta);
    [amp_env_meanGamma(iSubject,:), ~] = myBiweight(tmp_amp_env_meanGamma);
    
    % for standard deviation of the envelope
    [amp_env_stdDelta(iSubject,:), ~] = myBiweight(tmp_amp_env_stdDelta);
    [amp_env_stdTheta(iSubject,:), ~] = myBiweight(tmp_amp_env_stdTheta);
    [amp_env_stdAlpha(iSubject,:), ~] = myBiweight(tmp_amp_env_stdAlpha);
    [amp_env_stdBeta(iSubject,:), ~] = myBiweight(tmp_amp_env_stdBeta);
    [amp_env_stdGamma(iSubject,:), ~] = myBiweight(tmp_amp_env_stdGamma);
    
    % for skewness of the signal amplitude
    [amp_skewDelta(iSubject,:), ~] = myBiweight(tmp_amp_skewDelta);
    [amp_skewTheta(iSubject,:), ~] = myBiweight(tmp_amp_skewTheta);
    [amp_skewAlpha(iSubject,:), ~] = myBiweight(tmp_amp_skewAlpha);
    [amp_skewBeta(iSubject,:), ~] = myBiweight(tmp_amp_skewBeta);
    [amp_skewGamma(iSubject,:), ~] = myBiweight(tmp_amp_skewGamma);
    
    % for kurtosis of the signal amplitude
    [amp_kurtDelta(iSubject,:), ~] = myBiweight(tmp_amp_kurtDelta);
    [amp_kurtTheta(iSubject,:), ~] = myBiweight(tmp_amp_kurtTheta);
    [amp_kurtAlpha(iSubject,:), ~] = myBiweight(tmp_amp_kurtAlpha);
    [amp_kurtBeta(iSubject,:), ~] = myBiweight(tmp_amp_kurtBeta);
    [amp_kurtGamma(iSubject,:), ~] = myBiweight(tmp_amp_kurtGamma);
    
    eeglab redraw

end

%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1
    
    save('amplitude_features_ZERO.mat',...
        'amp_total_powDelta','amp_total_powTheta','amp_total_powAlpha',...
        'amp_total_powBeta','amp_total_powGamma',...
        'amp_env_meanDelta','amp_env_meanTheta','amp_env_meanAlpha',...
        'amp_env_meanBeta','amp_env_meanGamma',...
        'amp_env_stdDelta','amp_env_stdTheta','amp_env_stdAlpha',...
        'amp_env_stdBeta','amp_env_stdGamma',...
        'amp_skewDelta','amp_skewTheta','amp_skewAlpha',...
        'amp_skewBeta','amp_skewGamma',...
        'amp_kurtDelta','amp_kurtTheta','amp_kurtAlpha',...
        'amp_kurtBeta','amp_kurtGamma')

else
    
    save('amplitude_features.mat',...
        'amp_total_powDelta','amp_total_powTheta','amp_total_powAlpha',...
        'amp_total_powBeta','amp_total_powGamma',...
        'amp_env_meanDelta','amp_env_meanTheta','amp_env_meanAlpha',...
        'amp_env_meanBeta','amp_env_meanGamma',...
        'amp_env_stdDelta','amp_env_stdTheta','amp_env_stdAlpha',...
        'amp_env_stdBeta','amp_env_stdGamma',...
        'amp_skewDelta','amp_skewTheta','amp_skewAlpha',...
        'amp_skewBeta','amp_skewGamma',...
        'amp_kurtDelta','amp_kurtTheta','amp_kurtAlpha',...
        'amp_kurtBeta','amp_kurtGamma')
    
end

%%
   
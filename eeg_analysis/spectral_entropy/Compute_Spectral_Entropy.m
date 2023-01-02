% Scripts to calculate Spectral Entropies
% Original data was filtered 1-45 Hz

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
    
    % For the first run of subjects, claim some memory to store individual
    % subjects values for different bands
    % subject x channels
    if iSubject == 1
       % for mean
       SE_Delta = zeros(size(Subject_pool,1),size(EEG.data,1)); 
       SE_Theta = zeros(size(Subject_pool,1),size(EEG.data,1));
       SE_Alpha = zeros(size(Subject_pool,1),size(EEG.data,1));
       SE_Beta = zeros(size(Subject_pool,1),size(EEG.data,1));
       SE_Gamma = zeros(size(Subject_pool,1),size(EEG.data,1));
    end
    
    % Claim some memory for the calculations for different bands
    % channels x epochs
    tmpSE_Delta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmpSE_Theta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmpSE_Alpha = zeros(size(EEG.data,1),size(EEG.data,3));
    tmpSE_Beta = zeros(size(EEG.data,1),size(EEG.data,3));
    tmpSE_Gamma = zeros(size(EEG.data,1),size(EEG.data,3));
    
    % Loop for each channel
    for ichan = 1:size(EEG.data,1)
        % Loop for each segment to have a better estimate
        for iepoch = 1:size(EEG.data,3)
            
            inSignal = EEG.data(ichan,:,iepoch);
            L = length(inSignal);                           % length of the signal of interest
            NFFT = 2^nextpow2(length(inSignal));            % number of points for FFT computation
            f = EEG.srate/2*linspace(0,1,NFFT/2+1);         % the frequencies axis
            Y_tmp = fft(inSignal,NFFT)/L;                   % fft transformation (double-sided)
            Y = 2*abs(Y_tmp(1:NFFT/2+1)).^2;                  
            NY = Y/sum(Y);                                  % normalized spectrum

            % for delta - from 1 to 4 Hz
            % Delta SE
            [~,f_in_delta] = min(abs((f-1)));               % for the initial freq of interest
            [~,f_fin_delta] = min(abs((f-4)));              % for the end freq of interest
            delta_values = NY(f_in_delta:f_fin_delta);
            SEdelta = -sum(delta_values.*log(delta_values+eps))/log(length(delta_values));
            tmpSE_Delta(ichan,iepoch) = SEdelta;
            
            % for theta - from 4 to 8 Hz
            % Theta SE
            [~,f_in_theta] = min(abs((f-4)));               % for the initial freq of interest
            [~,f_fin_theta] = min(abs((f-8)));              % for the end freq of interest
            theta_values = NY(f_in_theta:f_fin_theta);
            SEtheta = -sum(theta_values.*log(theta_values+eps))/log(length(theta_values));
            tmpSE_Theta(ichan,iepoch) = SEtheta;
            
            % for alpha - from 8 to 13 Hz
            % Alpha SE
            [~,f_in_alpha] = min(abs((f-8)));                % for the initial freq of interest
            [~,f_fin_alpha] = min(abs((f-13)));              % for the end freq of interest
            alpha_values = NY(f_in_alpha:f_fin_alpha);
            SEalpha = -sum(alpha_values.*log(alpha_values+eps))/log(length(alpha_values));
            tmpSE_Alpha(ichan,iepoch) = SEalpha;
            
            % for beta - from 13 to 30 Hz
            % Beta SE
            [~,f_in_beta] = min(abs((f-13)));               % for the initial freq of interest
            [~,f_fin_beta] = min(abs((f-30)));              % for the end freq of interest
            beta_values = NY(f_in_beta:f_fin_beta);
            SEbeta = -sum(beta_values.*log(beta_values+eps))/log(length(beta_values));
            tmpSE_Beta(ichan,iepoch) = SEbeta;
            
            % for gamma - from 30 to 70 Hz            
            % Gamma SE
            [~,f_in_gamma] = min(abs((f-30)));               % for the initial freq of interest
            [~,f_fin_gamma] = min(abs((f-45)));              % for the end freq of interest
            gamma_values = NY(f_in_gamma:f_fin_gamma);
            SEgamma = -sum(gamma_values.*log(gamma_values+eps))/log(length(gamma_values));
            tmpSE_Gamma(ichan,iepoch) =  SEgamma;
        end
    end
    
    % Assign the biweight estimate of the mean to avoid influence of
    % outliers
    [SE_Delta(iSubject,:),~] = myBiweight(tmpSE_Delta);
    [SE_Theta(iSubject,:),~] = myBiweight(tmpSE_Theta);
    [SE_Alpha(iSubject,:),~] = myBiweight(tmpSE_Alpha);
    [SE_Beta(iSubject,:),~] = myBiweight(tmpSE_Beta);
    [SE_Gamma(iSubject,:),~] = myBiweight(tmpSE_Gamma);
    
end

%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1

    save('spectral_entropy_ZERO.mat',...
        'SE_Delta','SE_Theta',...
        'SE_Alpha','SE_Beta','SE_Gamma')

else
    
    save('spectral_entropy.mat',...
        'SE_Delta','SE_Theta',...
        'SE_Alpha','SE_Beta','SE_Gamma')   
end

%%

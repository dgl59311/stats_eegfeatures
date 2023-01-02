% Detrended fluctuation analysis exponent

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

% Number of channels
nchan = 61;

% Number of equally spaced windows
nwindows = 30;

% Min and max window length
minwindowl = 3;
maxwindowl = 50;

% Windows for analysis 
lgspace = logspace(log10(minwindowl), log10(maxwindowl), nwindows);

%%

% use zero ref
zero_ref = 0;

% To store fluctuation functions for each window
FLUC_FUNCTIONS = zeros(5, length(Subject_pool), nchan, nwindows);

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
    
    % Sampling rate
    srate_eeg = EEG.srate;
    % Window sizes in time frames
    wsizes = round(lgspace*srate_eeg);
    % Time series
	signals = EEG.data;
    % Total data length
    largeset = length(signals);   

    parfor (band = 1:5,5)
        
        % Set bandpass filter
        [locutoff,hicutoff] = FiltLims(band);
        filtorder = 2*fix(srate_eeg/locutoff);
        b = fir1(filtorder, [locutoff, hicutoff]./(srate_eeg/2),'bandpass');
        
        for es = 1:nchan
            
            % Filter time series and obtain cumulative sum of amplitude
            % envelopes
            x = signals(es,:)-mean(signals(es,:));
            xf = filter(b, 1, double(x));
            A = abs(hilbert(xf));
            A = A-mean(A)
            Y = cumsum(A);
            
            for nw = 1:nwindows
                
                % Obtain fluctuation function for each window
                test_w = wsizes(nw);
                start_p = 1:test_w/2:largeset-test_w; %50 overlap
                avw = length(start_p);
                % Save how many windows are available
                navailable(band,iSubject,es,nw) = avw;
                varavw = zeros(1,avw);
                
                for wt=1:avw
                    
                    % Detrend signal and get the variance
                    initp = round(start_p(wt));
                    fluc_signal = Y(initp:initp+test_w-1);
                    p = polyfit(initp:initp+test_w-1,fluc_signal,1);
                    lsreg = polyval(p,initp:initp+test_w-1);
                    det_signal = fluc_signal-lsreg;
                    varavw(wt) = var(det_signal);
                    
                end
                
            % Save fluctuation functions
            FLUC_FUNCTIONS(band,iSubject,es,nw) = sqrt(mean(varavw));
            
            end
        end
    end
end
%% 

% Obtain DFA exponents
dfa_exponents = zeros(5,length(Subject_pool),nchan);

for band = 1:5
    
    for iSubject = 1:length(Subject_pool)
        
        for ch = 1:nchan
            
            pf = polyfit(log10(lgspace'),...
                log10(squeeze(FLUC_FUNCTIONS(band,iSubject,ch,:))),1);
            dfa_exponents(band,iSubject,ch) = pf(1);
            
        end
    end
end

dfaexponent_delta=squeeze(dfa_exponents(1,:,:));
dfaexponent_theta=squeeze(dfa_exponents(2,:,:));
dfaexponent_alpha=squeeze(dfa_exponents(3,:,:));
dfaexponent_beta=squeeze(dfa_exponents(4,:,:));
dfaexponent_gamma=squeeze(dfa_exponents(5,:,:));

%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1
    
    save('dfa_exponents_30_3_50_ZERO.mat',...
        'dfaexponent_delta','dfaexponent_theta',...
        'dfaexponent_alpha','dfaexponent_beta','dfaexponent_gamma')

else
    
    save('dfa_exponents_30_3_50.mat',...
        'dfaexponent_delta','dfaexponent_theta',...
        'dfaexponent_alpha','dfaexponent_beta','dfaexponent_gamma')
    
end
%%
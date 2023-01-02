% Scripts to calculate Higuchi and Katz FD       

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
zero_ref = 1;

%%

% To store data

fractal_dimension_katz=zeros(length(Subject_pool),nchan);
fractal_dimension_higuchi=zeros(length(Subject_pool),nchan);

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
    
    n_eps = size(EEG.data,3);
    
    features_s = zeros(n_eps,nchan,2);
    
    for n_epochs = 1:n_eps
        
        % Get time series
        signals = squeeze(EEG.data(:,:,n_epochs));

        parfor (chan = 1:nchan,5)
            
            x_c = signals(chan,:);
            
            % Calculate fractal dimensions for each time series
            
            % Katz fractal dimension
            get_katz_fd = Katz_FD(x_c);
            % Higuchi fractal dimension
            get_higuchi_fd = Higuchi_FD(x_c,25);
            
            FDS = [get_katz_fd;get_higuchi_fd];
            features_s(n_epochs,chan,:) = FDS;
                
        end
    end
    
    fractal_dimension_katz(iSubject,:) = myBiweight(squeeze(features_s(:,:,1))');
    fractal_dimension_higuchi(iSubject,:) = myBiweight(squeeze(features_s(:,:,2))');

end

%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1
    
    save('fractal_dimensions_ZERO.mat',...
        'fractal_dimension_katz','fractal_dimension_higuchi');
     
else
    
    save('fractal_dimensions.mat',...
        'fractal_dimension_katz','fractal_dimension_higuchi');

end



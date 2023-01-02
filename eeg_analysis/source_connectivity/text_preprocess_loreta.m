% Create folders & text data for LORETA
% .asc instead of .txt

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
% Save .asc files locally (~4gb)
LocalDir = uigetdir([], 'Save .asc files locally');

%% 

epoch = 4; % epoch length in seconds

n_ep = []; % save number of epochs

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
    
    % Get participant id
    id_s = Subject_pool{iSubject};
    SubjectID = id_s(1:end-4);
    
    eeglab redraw
    
    % Epoch data in x seconds epochs
    EEG = eeg_regepochs(EEG, 'limits', [0 epoch], 'rmbase', NaN, 'recurrence', epoch);
    EEG = eeg_checkset(EEG);
    
    % Create new folder
    folder_name = [LocalDir '\' SubjectID];
	mkdir(folder_name)
    
    % Save number of epochs
    n_ep = [n_ep size(EEG.data,3)];
    
    % Same number of epochs for all participants
    n_epochs = 90;
    
    for ini = 1:n_epochs
        
        To_write = squeeze(EEG.data(:,:,ini));
        n_decimal = 1;
        new_T = varfun(@(x) num2str(x,...
                 ['%' sprintf('.%df', n_decimal)]),...
                 array2table(To_write));
        C1 = table2cell(new_T);
        fileID = fopen([folder_name '\' SubjectID '_' num2str(ini,'%03.f') '.asc'],'w');
        fprintf(fileID,['%6s\t' repmat('%6s\t',1,59) '%6s\n'],C1{:});
        fclose(fileID);   
        
    end
end

%% 





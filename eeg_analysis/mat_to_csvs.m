% Create csv files

clc;
clear;
close all

% Paths to the data
SubjectsDir = uigetdir([],'Path to the raw data');

% Subjects pre-processed data files
Subject_data = dir(fullfile(SubjectsDir,'*.set'));

% The pool of subjects
Subject_pool = {Subject_data(:).name}';
Subject_pool = regexprep(Subject_pool,{'_EC.set'},{''});

% Path to read .mat EEG data
DirData = uigetdir([], '.mat EEG features');
measure_matFiles = dir(fullfile(DirData, '*.mat'));

% Path to save .csv results
ResultsDir = uigetdir([], 'Save .csv files with EEG features');

% load demog data
demog = readtable("age_bins_values.csv");

% load EEG features names
names_eeg = readtable('eeg_features_names.csv');

% load name of channels and brain regions
load('chanlocs.mat')
load('sourcelocs.mat')

% Check that data of participants are there
if isempty(find(strcmpi(Subject_pool,table2cell(demog(:,1)))==0))
    disp('Subject_pool matches demog data')
end

%%
cd(ResultsDir)

for i = 1:length(measure_matFiles)
    
    measureID = measure_matFiles(i).name;
    X = load(fullfile(DirData, measureID));
    f_names = fieldnames(X);
    
    for j = 1:length(f_names)
        
        name_mat = f_names{j};
        data_eeg = X.(name_mat);
        s_2 = size(data_eeg,2);
        table_eeg = array2table(data_eeg);
        
        if s_2==61
            table_eeg.Properties.VariableNames = {chanlocs(:).labels};
        
        elseif s_2==80
            table_eeg.Properties.VariableNames = sourcelocs';
        
        end
        
        full_data = [demog table_eeg];
        ix_name = find(ismember(names_eeg.mat,name_mat));
        
        
        if contains(measureID, 'ZERO')
            writetable(full_data,[char(names_eeg.csv(ix_name)) ' zero.csv']);
        else
            writetable(full_data,[char(names_eeg.csv(ix_name)) '.csv']);
        end

    end
end


%%

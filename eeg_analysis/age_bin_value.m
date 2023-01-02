clc;
clear;

% Paths to the data
SubjectsDir = uigetdir([],'Path to the raw data');

% Subjects pre-processed data files
SubjectData = dir(fullfile(SubjectsDir,'*.set'));

% read original demographic file
demogs = readtable("META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv");

% Get data 
ids = table2array(demogs(:,1));
gender = table2array(demogs(:,2));
age = table2array(demogs(:,3));

%%

% Identify age group
% 1 == younger 
group_age = zeros(1,length(age));
bin_age = zeros(1,length(age));

% Replace binned values with average
for i = 1:length(age)
    ap = age{i};
    if contains(ap,'20-25')
        group_age(i) = 1;
        bin_age(i) = 22.5;
    elseif contains(ap,'25-30')
        group_age(i) = 1;  
        bin_age(i) = 27.5;
    elseif contains(ap,'30-35')
        group_age(i) = 1;  
        bin_age(i) = 32.5;
	elseif contains(ap,'35-40')
        group_age(i) = 1;  
        bin_age(i) = 37.5;
    elseif contains(ap,'55-60')
        group_age(i) = 2;  
        bin_age(i) = 57.5;
	elseif contains(ap,'60-65')
        group_age(i) = 2;  
        bin_age(i) = 62.5;
	elseif contains(ap,'65-70')
        group_age(i) = 2;  
        bin_age(i) = 67.5;
	elseif contains(ap,'70-75')
        group_age(i) = 2;  
        bin_age(i) = 72.5;
	elseif contains(ap,'75-80')
        group_age(i) = 2;  
        bin_age(i) = 77.5;
    end
end

%%

% Find participants with EEG recordings and write new demog file

young1_old2 = zeros(length(SubjectData),1);
female1_male2 = zeros(length(SubjectData),1);
age_bins = zeros(length(SubjectData),1);
id_subx = [];

for i = 1:length(young1_old2)
    name_i = SubjectData(i).name(1:end-7);
    in_demog = find(contains(ids,name_i));
    young1_old2(i) = group_age(in_demog);
    female1_male2(i) = gender(in_demog);
    age_bins(i) = bin_age(in_demog);
    id_subx = [id_subx;name_i];
end

% Write file

output_matrix=[{'id'} {'Gender'} {'Group'} {'Age'};...
    [cellstr(id_subx) num2cell([female1_male2 young1_old2 age_bins])]];

writecell(output_matrix,'age_bins_values.csv');
%%

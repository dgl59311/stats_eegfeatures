% build csv tables

clc;
clear;
close all;

% load demog data
demog = readtable("age_bins_values.csv");

% load table with microstates data
mss_table = uigetdir([],'Path to microstates results');
mss_table = fullfile(mss_table, 'Fit GC microstates_preprocessing.m.05.(05).Group1LinesAsSamples.csv');

% for non-zero ref
%mss_table = fullfile(mss_table, 'Fit GC RSWhole sub.05.(05).Group1LinesAsSamples.csv');


ms = readtable(mss_table);
ms(:,1) = [];

% Path to save .csv results
ResultsDir = uigetdir([], 'Save .csv files with results');

%%
% gev, duration, time cov, seg density
ms{:,[1 5 9 13 17]}=ms{:,[1 5 9 13 17]}*100;
ms{:,[2 6 10 14 18]}=ms{:,[2 6 10 14 18]}*1000/125;
ms{:,[3 7 11 15 19]}=ms{:,[3 7 11 15 19]}*100;
ms{:,[4 8 12 16 20]}=ms{:,[4 8 12 16 20]}*100;

%% With zero reference

writetable([demog ms(:, 1:4)], fullfile(ResultsDir,'microstate A zero.csv'))
writetable([demog ms(:, 5:8)], fullfile(ResultsDir,'microstate B zero.csv'))
writetable([demog ms(:, 9:12)], fullfile(ResultsDir,'microstate C zero.csv'))
writetable([demog ms(:, 13:16)], fullfile(ResultsDir,'microstate D zero.csv'))
writetable([demog ms(:, 17:20)], fullfile(ResultsDir,'microstate E zero.csv'))

%% Average reference

writetable([demog ms(:, 1:4)], fullfile(ResultsDir,'microstate A.csv'))
writetable([demog ms(:, 5:8)], fullfile(ResultsDir,'microstate B.csv'))
writetable([demog ms(:, 9:12)], fullfile(ResultsDir,'microstate C.csv'))
writetable([demog ms(:, 13:16)], fullfile(ResultsDir,'microstate D.csv'))
writetable([demog ms(:, 17:20)], fullfile(ResultsDir,'microstate E.csv'))

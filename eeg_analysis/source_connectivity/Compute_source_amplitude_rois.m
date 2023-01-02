% eLORETA power

clc;
clear;
close all;

% Brain regions
n_ROIS = 80;

% Paths to the crss files
Dir_mats = uigetdir([],'Path to the .txt and crss files');

% Path to save .mat results
ResultsDir = uigetdir([], 'Save .mat files with results');

%% build data tensors 

filePattern = fullfile(Dir_mats, '*.txt');
matr_c = dir(filePattern);
    
for nsub = 1:length(matr_c)
   
    fid = fopen(fullfile(matr_c(nsub).folder, matr_c(nsub).name));
    tline = fgetl(fid);
    tlines = cell(0,1);
    
    while ischar(tline)
        
        tlines{end+1,1} = tline;
        tline = fgetl(fid);
        
    end
    
    fclose(fid);
    
    for fb = 1:5
        
        ix_s = [find(diff(isspace(tlines{fb,:}))==-1) length(tlines{fb,:})];
        
        for nroi = 1:(length(ix_s)-1)
            
            C_ALL(nsub,fb,nroi) = str2num(tlines{fb,:}(ix_s(nroi):ix_s(nroi+1)));
            
        end
    end
end

%% 

slor_delta = squeeze(C_ALL(:,1,:));
slor_theta = squeeze(C_ALL(:,2,:));
slor_alpha = squeeze(C_ALL(:,3,:));
slor_beta = squeeze(C_ALL(:,4,:));
slor_gamma = squeeze(C_ALL(:,5,:));

%% 
% select where to save data

cd(ResultsDir)

save('source_amplitudes.mat',...
    'slor_delta','slor_theta','slor_alpha',...
    'slor_beta','slor_gamma');


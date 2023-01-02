% Scripts to calculate phase amplitude coupling
% Original data is filtered 1-45 Hz

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

% Delta phase-theta amplitude
% Delta phase-alpha amplitude
% Delta phase-beta amplitude
% Delta phase-gamma amplitude 
% Theta phase-alpha amplitude 
% Theta phase-beta amplitude
% Theta phase-gamma amplitude 
% Alpha phase-beta amplitude
% Alpha phase-gamma amplitude
% Beta phase-gamma amplitude

fil_op=[1 2;1 3;1 4;1 5;2 3;2 4;2 5;3 4;3 5;4 5];

% refs
% "Measuring Phase-Amplitude Coupling Between Neuronal 
% Oscillations of Different Frequencies", Tort et al (2010)

%% 

% Analysis parameters

nbins = 18; % number of bins for KL calculation

epoch = 4; % in seconds

nchan = 61; % number of channels

% Define bin centers and edges
BinEdges = linspace(-pi,pi,nbins+1);

BinCenters = BinEdges(1:end-1)-diff(BinEdges)/2;

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
    
    for combination = 1:10
        
        % Filter data
        
        % Phase data
        [locutoff_p,hicutoff_p] = FiltLims(fil_op(combination,1));
        Phase_F = pop_eegfiltnew(EEG, locutoff_p,hicutoff_p, [], 0, [], 0);
        
        % Amplitude data
        [locutoff_a,hicutoff_a]=FiltLims(fil_op(combination,2));
        Amp_F = pop_eegfiltnew(EEG, locutoff_a,hicutoff_a, [], 0, [], 0);
        
        for n_epochs=1:size(EEG.data,3)
            
            for chan=1:size(EEG.data,1)
                
                % Get phase and amplitude time series
                x_phase = squeeze(Phase_F.data(chan,:,n_epochs));
                x_amp = squeeze(Amp_F.data(chan,:,n_epochs));
                Band_Phases = angle(hilbert(x_phase));
                Band_Amp = abs(hilbert(x_amp));
                [~,BinIndex] = histc(Band_Phases,BinEdges);
                Amp_Bins = zeros(1,nbins);
                
                % Put amplitudes in the corresponding bins 
                for i = 1:nbins
                    Amp_Bins(i) = mean(Band_Amp(find(BinIndex==i)));
                end
                
                % Average amplitude values
                Amp_Mod = Amp_Bins/sum(Amp_Bins);
                Uniform = ones(1,nbins)/nbins;
                
                if any(Amp_Mod==0)
                   
                    Amp_Mod(Amp_Mod==0)=eps;
                
                end
                % KL divergence
                Mod_Index = sum(Amp_Mod.*log(Amp_Mod./Uniform))/(log(nbins));
                M_Indexes(combination,chan,n_epochs) = Mod_Index;
                
            end
        end
    end
    
    MI_S{iSubject} = M_Indexes;

end
%%
% Delta phase-theta amplitude
% Delta phase-alpha amplitude
% Delta phase-beta amplitude
% Delta phase-gamma amplitude 
% Theta phase-alpha amplitude 
% Theta phase-beta amplitude
% Theta phase-gamma amplitude 
% Alpha phase-beta amplitude
% Alpha phase-gamma amplitude
% Beta phase-gamma amplitude

X = MI_S;

for i=1:length(X)
    
    d_x=MI_S{i};
    
    for c = 1:nchan
        % Average across trials
        modindex_delta_theta(i,c) = myBiweight(squeeze(d_x(1,c,:))');
        modindex_delta_alpha(i,c) = myBiweight(squeeze(d_x(2,c,:))');
        modindex_delta_beta(i,c) = myBiweight(squeeze(d_x(3,c,:))');
        modindex_delta_gamma(i,c) = myBiweight(squeeze(d_x(4,c,:))');
        modindex_theta_alpha(i,c) = myBiweight(squeeze(d_x(5,c,:))');
        modindex_theta_beta(i,c) = myBiweight(squeeze(d_x(6,c,:))');
        modindex_theta_gamma(i,c) = myBiweight(squeeze(d_x(7,c,:))');
        modindex_alpha_beta(i,c) = myBiweight(squeeze(d_x(8,c,:))');
        modindex_alpha_gamma(i,c) = myBiweight(squeeze(d_x(9,c,:))');
        modindex_beta_gamma(i,c) = myBiweight(squeeze(d_x(10,c,:))');
    end
end


%%
% select where to save data
cd(ResultsDir)

% save features 
if zero_ref == 1
    
    save('modulation_index_ZERO.mat',...
        'modindex_delta_theta','modindex_delta_alpha','modindex_delta_beta','modindex_delta_gamma',...
        'modindex_theta_alpha','modindex_theta_beta','modindex_theta_gamma',...
        'modindex_alpha_beta','modindex_alpha_gamma','modindex_beta_gamma');

else
    
    save('modulation_index.mat',...
        'modindex_delta_theta','modindex_delta_alpha','modindex_delta_beta','modindex_delta_gamma',...
        'modindex_theta_alpha','modindex_theta_beta','modindex_theta_gamma',...
        'modindex_alpha_beta','modindex_alpha_gamma','modindex_beta_gamma'); 
end





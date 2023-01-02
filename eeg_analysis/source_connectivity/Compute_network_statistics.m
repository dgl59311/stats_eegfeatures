% Read Connectivity Matrices & estimate NX statistics using BCT

clc;
clear;
close all;

% Brain regions
n_ROIS = 80;

% Paths to the connectivity files
Dir_mats = uigetdir([],'Path to the connectivity results');

% Path to save .mat results
ResultsDir = uigetdir([], 'Save .mat files with results');

%% Read data

lcoh_files = read_connectivity(dir(fullfile(Dir_mats, '*ROIlaggedCoh-crss.txt')));
lps_files = read_connectivity(dir(fullfile(Dir_mats, '*ROIlaggedPhase-crss.txt')));

%% Estimate connectivity measures 

for nsub = 1:size(lcoh_files,1)
    % number of frequency bands used before
	for nb = 1:5
        
        X = squeeze(lcoh_files(nsub,nb,:,:));
        cc_ = clustering_coef_wu(X)';
        stre_ = strengths_und(X)';
        betw_centr = betweenness_wei(X)';

        if nb == 1
            
            LCOHdelta_clusteringcoefficient(nsub,:) = cc_;
            LCOHdelta_nodestrength(nsub,:) = stre_;
            LCOHdelta_betwcentr(nsub,:) = betw_centr;
            
        elseif nb == 2
            
            LCOHtheta_clusteringcoefficient(nsub,:) = cc_;
            LCOHtheta_nodestrength(nsub,:) = stre_;
            LCOHtheta_betwcentr(nsub,:) = betw_centr;
        
        elseif nb == 3
        
            LCOHalpha_clusteringcoefficient(nsub,:) = cc_;
            LCOHalpha_nodestrength(nsub,:) = stre_;
            LCOHalpha_betwcentr(nsub,:) = betw_centr;
        
        elseif nb == 4
            
            LCOHbeta_clusteringcoefficient(nsub,:) = cc_;
            LCOHbeta_nodestrength(nsub,:) = stre_;
            LCOHbeta_betwcentr(nsub,:) = betw_centr;
        
        elseif nb == 5
            
            LCOHgamma_clusteringcoefficient(nsub,:) = cc_;
            LCOHgamma_nodestrength(nsub,:) = stre_;
            LCOHgamma_betwcentr(nsub,:) = betw_centr;
        
        end
	end
end
%% 

% select where to save data
cd(ResultsDir)

save('s_LCOH_nx.mat',...
    'LCOHdelta_clusteringcoefficient','LCOHdelta_nodestrength','LCOHdelta_betwcentr',...
	'LCOHtheta_clusteringcoefficient','LCOHtheta_nodestrength','LCOHtheta_betwcentr',...
	'LCOHalpha_clusteringcoefficient','LCOHalpha_nodestrength','LCOHalpha_betwcentr',...
	'LCOHbeta_clusteringcoefficient','LCOHbeta_nodestrength','LCOHbeta_betwcentr',...
	'LCOHgamma_clusteringcoefficient','LCOHgamma_nodestrength','LCOHgamma_betwcentr');


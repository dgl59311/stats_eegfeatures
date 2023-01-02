# stats_eegfeatures

Gordillo, da Cruz, Moreno, Garobbio, Herzog

In this repository we present six main analyses:
- Correlation of EEG features with cognitive variables ('1_correlations_eeg_beh.ipynb')
- Prediction of cognitive variables with EEG features ('2_predict_beh.ipynb')
- Group comparisons between older and younger participants using each EEG feature ('3_mwu_r.ipynb')
- Correlation of EEG features showing significant group differences ('4_correlations_groups.ipynb' and '5_multivariate_distance_correlation.ipynb')
- Comparison of EEG features across reference choices ('6_multivariate_distance_correlation_references.ipynb' and '7_correlations_EEG_refs.ipynb' )
- Dimensionality reduction of EEG features and multiple regression using latent variables ('8_dimensionality_reduction_pc_regression.ipynb')

Before running the above pipelines we need to analyze the EEG data and obtained the EEG features.

For EEG analysis:

- The folders in 'eeg_analysis' contains MATLAB code to analyze EEG data.
- For analysis methods using third-party software, we include the verbose files in the respective folders. 
- Before running the code add 'utils_eeg' to the MATLAB path
- All code runs using the .set/.fdt files provided by the LEMON study
- Each code outputs a .mat file with the analysis results in a folder specified by the user (we used '/eeg_analysis/mat_results)
- Before statistical analyses, we generate .csv files from the .mat files using the script 'mat_to_csv'


For statistics:

- We used Python and R for statistical analyses with the environments specified in stats_eeg.yml and rstats_eeg.yml
- The results of each statistical analysis (effect sizes, p-values) are stored in .csv files
- For example, '1_correlations_eeg_beh.ipynb' produces .csv files stored in 'stats_eegfeatures/results/1_correlations_eeg_beh_results/'
- Each notebook contains all the necessary information about where and how results are stored
- '9_summarize_results.ipynb' is meant to provide a general overview of the results of the analysis pipeline
- All the figures of the manuscript are stored in 'stats_eegfeatures/results/figures_results/' and are generated using the codes presented in '10_figures.ipynb' 

if you have any comments or questions please send them to: dario.gordillolopez@epfl.ch

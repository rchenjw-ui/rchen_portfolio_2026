NOTE: 

The research code is shared with permission from my mentor at Children's Hospital Los Angeles (CHLA). Two unpublished original research data files are not publicly available due to data-sharing restrictions 

To demonstrate the computational workflow, this repository provides synthetic data that can be used to run the code. These demonstration inputs are not the original data, and their outputs should not be interpreted as reproducing or validating the study findings

This is the first README.md I wrote for this repo, and as such is not as clean as the others.

BEFORE YOU USE:
- Make sure you are running in Python! This is a Python program
- eval_given_saved.py evaluates the logistic regression method on a combination of existing and synthetic data (the UM samples are synthetic, as are some evaulations for tumor fraction)
- model_and_scaler_50.json, top_50_probes_probe_beta_pairs, and pilot_8samples_merged-s.csv (S for synthetic) are necessary for eval_given_saved.py to run out of the box

Problem:
- The problem is that retinoblastoma cannot easily be biopsied without spreading
- Using cfDNA to estimate tumor fraction (to track clinical progress, for example) can be used
- However, using ICHOR-CNA and other CNV methods has a few limitations, namely CNVs are not uniform
- Methylation based methods can sidestep this

Features:
- Uses logistic regression to avoid anomalous results outside of 0-1 range
- Trains a logistic regression given known tumor and retina samples
- Uses logistic regression to estimate tumor fraction 
- Estimates feature importance and graphs feature coefficients vs beta values 
- Evaluates on a few known clinical samples
- Also synthetically generates samples with known tumor fraction for evaluation

data:
- embryo_data: retina data from embryonic sample (public)
- methyl_data: three cleared patient samples (released by lab)
- tumor_data: data from known retinoblastoma tumors (public)
- model_and_scaler_50.json: the file containing the ML model for calculating tumor fraction
- pilot_8samples_merged-s.csv: Contains two synthetic uveal melanoma plus 6 released patient samples (released by lab/synthetic)
- top_50_probes_probe_beta_pairs.csv: a list of probes and reference values for retinoblastoma (just the ones ML model uses)

code:
- data_extraction.py: file handling utils
- test_probe_gen.py: generates the probe sets
- matrices_for_use: due to some heavy NumPy data processing, had to separately generate .npy files
- logistic_regression_test.py: trains and tests models on existing data
- eval_given_saved.py: runs a model on an evaluation suite and synthetic data 
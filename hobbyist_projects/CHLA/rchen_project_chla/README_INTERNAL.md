NOTE: 

The research code is shared with permission from my mentor at Children's Hospital Los Angeles. Two unpublished original research data files are not publicly available due to data-sharing restrictions 

To demonstrate the computational workflow, this repository provides synthetic data that can be used to run the code. These dmonestration inputs are not the original data, and their outputs should not be interpreted as reproducing or validating the study findings


BEFORE YOU USE:
- add the files "GSE57362_Embryo_retinas_meth_signal.txt" and "GSE57362_Embryo_retinas_Unmeth_signal.txt" to directory "embryo_data"
- add the files "GSE58783_Methylation_processed_data.RData" to directory "tumor_data"
- Make sure you are running in Python! This is a Python program

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

Future maneuvers:
- Better visualizations
- More feature importance estimates
- More data, plus clinical evaluations
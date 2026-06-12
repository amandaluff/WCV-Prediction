**Predicting Infant Nonattendance at the Next Recommended Well-Child Visit** [![DOI](https://zenodo.org/badge/1092086249.svg)](https://doi.org/10.5281/zenodo.20648820)

**Cite As**: Luff A. Predicting Infant Nonattendance at the Next Recommended Well-Child Visit. Analytic Code. Zenodo. https://doi.org/10.5281/zenodo.20648821

OVERVIEW

This project evaluates prediction of a child attending their next well-child visit, given that they attended their current visit, using data from two clinics within a single health system. Practice A (Park Ridge area) is used for model training and internal validation. Practice B (Oak Lawn area) is used for external validation. Three models are compared: regularized logistic regression, random forest, and XGBoost.

DATA AVAILABILITY

The source data contain protected health information and are not included in this repository. De-identified data may be available from the corresponding author upon reasonable request and in accordance with institutional policies.

SOURCE DATA

Five files were extracted from the electronic health record via SlicerDicer.

1. Well-child visits: Completed visit records including appointment details, demographics, insurance, and provider information
2. Birth records: Birth weight, gestational age, and NICU admission status, linked by baby MRN
3. ED encounters: Emergency department visits with MRN and departure date
4. Immunization refusal diagnoses: Diagnosis events for documented immunization refusal with MRN and event date
5. All visits: Full visit file used to identify prior no-show appointments

DATA PROCESSING

The cleaning script (data_cleaning.py) links the source files and produces four output parquet files. The unit of analysis is the visit: one row per patient per AAP-recommended well-child visit timepoint (0, 1, 2, 4, 6, 9, and 12 months), deduplicated to the earliest completed visit date per patient-timepoint pair. Each patient contributes 1 to 7 visits.

Output files:
- wcv: Full analytic dataset, both clinics, all variables
- wcv_demo: Demographics only (MRN, clinic_group, race, ethnicity, language)
- wcv_train: Practice A visits only, demographics and clinic indicator removed, all columns cast to float64
- wcv_test: Practice B visits only, demographics and clinic indicator removed, all columns cast to float64

OUTCOME

missed: Binary (1/0). Indicates whether the patient did not complete a visit at the next expected AAP timepoint. For example, if no visit occurs between 6 and 9 months, the 6-month visit row is coded as missed = 1.

PREDICTORS IN TRAINING AND TEST DATA

All variables below are present in wcv_train and wcv_test as float64. MRN is a patient identifier and is removed before model fitting. missed is the outcome.

- MRN: Patient identifier (not used in modeling) 
- timepoint: Expected AAP visit month (0, 1, 2, 4, 6, 9, 12) 
- visit_delay: Months between expected timepoint and actual visit age 
- sch_lead_days: Days between scheduling date and visit date 
- new_to_dpt: First visit at this clinic (binary) 
- portal_active: Patient portal active at time of scheduling (binary) 
- male: Patient sex is male (binary) 
- medicaid: Medicaid insurance (binary) 
- attending: Treating provider is an attending physician (binary)
- physician: Treating provider is a physician (attending, resident, or fellow) vs. nurse practitioner (binary) 
- prior_ed_visit: Any in-system ED visit before the current visit date (binary) 
- imm_refusal: Documented immunization refusal on or before the current visit date (binary) 
- prior_noshow: Any no-show appointment before the current visit date (binary) 
- any_prior_missed: Any prior missed well-child visit within the dataset (binary) 
- has_birth: Birth record exists in the health system (binary) 
- nicu: NICU admission; 0 if no birth record (binary) 
- lbw: Low birth weight, less than 2500g; 0 if no birth record (binary) 
- preterm: Preterm birth, less than 37 weeks gestation; 0 if no birth record (binary) 

DEMOGRAPHIC VARIABLES

The following variables are stored in wcv_demo and were not used in model training. They are used for cohort description and post hoc fairness assessment.

race: White, Black, Asian, Another Race, Not Reported
ethnicity: Hispanic or Latino, Not Hispanic or Latino, Not Reported
language: English, Spanish, Another Language

CODE FILES

1. data_cleaning.py
   Merging, data cleaning, and feature creation from files exported from SlicerDicer. Outputs train, test, demographic, and full analytic datasets to parquet files.

2. wcv_lr.py
   Model training, hyperparameter tuning, and interpretability for regularized logistic regression (LASSO).

3. wcv_rf.py
   Model training, hyperparameter tuning, and interpretability for random forest.

4. wcv_xgb.py
   Model training, hyperparameter tuning, and interpretability for XGBoost.

5. wcv_diagnostic_plots.py
   Combined diagnostic plots and performance metric tables across all three models.

6. wcv_fairness.py
   Subgroup performance assessment by race/ethnicity, preferred language, and insurance type using the external validation set.

MODELING WORKFLOW

Each modeling file (wcv_lr.py, wcv_rf.py, wcv_xgb.py) contains the following steps:
- Read training and test parquet files
- Create pipeline (SMOTE for tree-based models; class_weight='balanced' for logistic regression)
- Set up group k-fold cross-validation grouped by patient (MRN)
- Randomized hyperparameter search scored on average precision
- Fit best estimator
- Save fitted model to pickle
- Generate and save out-of-fold predictions to parquet
- Calculate feature importance (coefficients for logistic regression; Gini importance for RF and XGBoost) and permutation importance
- Calculate and plot SHAP beeswarm values for out-of-fold and test sets

The diagnostic plot and evaluation file (wcv_diagnostic_plots.py) contains:
- Combined ROC curves across all three models for internal and external validation
- Combined precision-recall curves across all three models for internal and external validation
- Performance metric tables (AUC, average precision, accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1) with 95% bootstrap confidence intervals (2,000 replicates)
- Metrics reported at the F1-maximizing threshold (primary) and at a 0.5 threshold (secondary)

FAIRNESS ASSESSMENT

The fairness assessment (wcv_fairness.py) computes AUC stratified by race/ethnicity, preferred language, and insurance type for all three models in the external validation set, with 95% bootstrap confidence intervals (2,000 replicates). Demographic variables are merged from wcv_demo onto the scored test set predictions.

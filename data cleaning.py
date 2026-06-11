# Data processing
import pandas as pd

#Completed visits
visits=pd.read_excel('WELL Visits_2025-11-24T13_11_22.xlsx')
visits = visits[visits['Appointment Status'] == 'Completed'] 

##-----BIRTH RECORD VARIABLES----###
birth=pd.read_excel('Births_2025-04-29T14_27_06.xlsx')

#NICU admission
birth['nicu'] = birth['Baby Was Admitted to ICU?'].notna().astype(bool)

#low birth weight
birth['lbw']=(birth['Birth Weight (g) (g)'] < 2500).astype(bool)

#preterm birth
birth['preterm']=(birth['Gestational Age (Days)'] < 259).astype(bool)

#drop unneeded cols
birth.drop(columns=['Birth Date', 'Birth Weight (g) (g)', 'Baby Was Admitted to ICU?', 'Baby Discharge Date', 'Mother Discharge Date', 'Gestational Age (Days)'], inplace=True) 

birth.rename(columns={'Baby MRN': 'MRN'}, inplace=True)


##-----MERGE BIRTH VARS WITH VISITS----###
wcv = pd.merge(visits, birth, on='MRN', how='left')

#create indicator for having birth record
wcv['has_birth'] = wcv['nicu'].notna().astype(bool)

#fill birth vars with false if missing
bool_vars = ['nicu', 'preterm', 'lbw'] 
for col in bool_vars:
    wcv[col] = wcv[col].where(wcv[col].notna(), False).astype(bool)


##-----DROP IRRELEVANT VISITS----###
# List of visit types to exclude
exclude_types = [
    'MY WORKERS COMP', 'IL US HIPS BILATERAL INFANT WITH MANIPULATION', 'LAB VISIT', 'LAB WALK IN',
    'AUDIOGRAM', 'LAB ON MY WAY', 'IL XR ANKLE LT MIN 3V', 'IL XR FOOT LT MIN 3V', 'WALK-IN VISIT',
    'IL XR BONE SURVEY INFANT', 'AUDIO BRAINSTEM RESPONSE', 'NURSE VISIT'
]
# Drop rows where Visit Type is in the exclude list
wcv = wcv[~wcv['Visit Type'].isin(exclude_types)]

#Identify clinics in sample
wcv['sample'] = wcv['Department'].isin([
#[redacted]
]).astype(int)

# Define clinic lists
grp_clinics = [
#[redacted]
]

# Create a new column for clinic group: 
wcv['clinic_group'] = wcv['Department'].apply(lambda x: 1 if x in grp_clinics else 0)

##-----WCV TIMEPOINT MAPPING----###
# Define WCV timepoint mapping function
def map_wcv(age):
    if 0 <= age < 1:
        return 0
    elif 1 <= age < 2:
        return 1
    elif 2 <= age < 4:
        return 2
    elif 4 <= age < 6:
        return 4
    elif 6 <= age < 9:
        return 6
    elif 9 <= age < 12:
        return 9
    elif 12 <= age < 15:
        return 12
    elif 15 <= age < 24:
        return 15
    else:
        return None  # For negative ages or missing data

wcv.rename(columns={'Age in Months (At Visit) (months)': 'age_months'}, inplace=True)
wcv['timepoint'] = wcv['age_months'].apply(map_wcv)

# Sort by MRN and Visit Date
wcv = wcv.sort_values(by=['MRN', 'Visit Date'])

# Define the standard timepoints in order
timepoints_order = [0, 1, 2, 4, 6, 9, 12, 15, 24]

# Create a mapping for next expected timepoint
next_tp_map = {tp: timepoints_order[i+1] if i+1 < len(timepoints_order) else None
               for i, tp in enumerate(timepoints_order)}

# Compute next timepoint for each row
wcv['next_expected_wcv'] = wcv['timepoint'].map(next_tp_map)

# Check if patient has that next timepoint anywhere
wcv['has_next_wcv'] = wcv.apply(
    lambda row: row['next_expected_wcv'] in wcv.loc[wcv['MRN'] == row['MRN'], 'timepoint'].values,
    axis=1
)

#visit delay (age minus timepoint)
wcv['visit_delay']=wcv['age_months'] - wcv['timepoint']

# Drop rows where timepoint > 12 and outside included clinic
wcv = wcv[wcv['timepoint'] <= 12]
wcv = wcv[wcv['sample'] == 1]

# Sort by MRN, timepoint, and Visit Date so earliest dates come first
wcv = wcv.sort_values(by=['MRN', 'timepoint', 'Visit Date'], ascending=[True, True, True])

# Drop duplicates, keeping the first (earliest Visit Date)
wcv = wcv.drop_duplicates(subset=['MRN', 'timepoint'], keep='first')

##-----CLEAN AND RECODE COLS----###
#recode insurance to medicaid, commercial, self-pay
wcv.rename(columns={'Primary Payer Financial Class': 'insurance'}, inplace=True)
wcv['insurance'] = wcv['insurance'].where(
    wcv['insurance'].isin(['Medicaid', 'Medicaid Other', 'Self-Pay']),
    'Commercial'
    ).replace({
    'Medicaid Other': 'Medicaid'
    }).astype('category')

#binary insurance
wcv['medicaid']=(wcv['insurance'] == 'Medicaid').astype(bool)

#recode language
wcv.rename(columns={'Language': 'language'}, inplace=True)
wcv['language'] = wcv['language'].where(
    wcv['language'].isin(['English', 'Spanish']),
    'Another Language'
    ).astype('category')

#recode race
wcv.rename(columns={'OMB Race': 'race'}, inplace=True)
wcv['race'] = wcv['race'].where(
    wcv['race'].isin(['White', 'Black or African American', 'Asian', 'Asked but No Answer', 'Unknown']),
    'Another Race'
    ).replace({
    'Asked but No Answer': 'Not Reported',  'Unknown': 'Not Reported', 'Black or African American': 'Black'
    }).astype('category')

#recode ethnicity
wcv.rename(columns={'OMB Ethnicity': 'ethnicity'}, inplace=True)
wcv['ethnicity'] = wcv['ethnicity'].where(
    wcv['ethnicity'].isin(['Not Hispanic or Latino', 'Hispanic or Latino']),
    'Not Reported'
    ).astype('category')

#recode provider type
wcv.rename(columns={'Primary Provider Type': 'provider_type'}, inplace=True)
wcv['provider_type'] = wcv['provider_type'].where(
    wcv['provider_type'].isin(['Physician', 'Nurse Practitioner']),
    'Resident/Fellow'
    ).astype('category')

wcv['attending']=(wcv['provider_type'] == 'Physician').astype(bool)
wcv['physician']=(wcv['provider_type'] != 'Nurse Practitioner').astype(bool)

# Convert to desired types
wcv = wcv.astype({
    'Confirmed?': 'bool',
    'Portal Active at Scheduling?': 'bool',
    'New to Department?': 'bool',
})
wcv['Lead Time (days)'] = wcv['Lead Time (days)'].astype('Int64')

wcv.rename(columns={'Portal Active at Scheduling?': 'portal_active'}, inplace=True)
wcv.rename(columns={'Lead Time (days)': 'sch_lead_days'}, inplace=True)
wcv.rename(columns={'New to Department?': 'new_to_dpt'}, inplace=True)
wcv['male']=(wcv['Patient Sex'] == 'Male').astype(bool)


##-----ED VISITS----###
#Check for prior ed visit
ed=pd.read_excel('ED Encounters_2025-11-05T19_30_55.xlsx', usecols=['MRN', 'Departure Date'])

# Sort MRN and Departure Date
ed = ed.sort_values(['MRN', 'Departure Date'])
wcv = wcv.sort_values(['MRN', 'Visit Date'])

# Function to check prior ED visit
def had_prior_ed(mrn, visit_date):
    return (ed.loc[ed['MRN'] == mrn, 'Departure Date'] < visit_date).any()

# Apply to WCV
wcv['prior_ed_visit'] = wcv.apply(lambda row: had_prior_ed(row['MRN'], row['Visit Date']), axis=1)


##-----IMUNIZATION REFUSALS----###
# Check for immunization refusal
imm_ref = pd.read_excel('Diagnosis Events_2025-05-12T10_09_17.xlsx', usecols=['MRN', 'Diagnosis Event Start Date'])

# Sort MRN and Date
imm_ref = imm_ref.sort_values(['MRN', 'Diagnosis Event Start Date'])
wcv = wcv.sort_values(['MRN', 'Visit Date'])

# Function to check immunization refusal before or on WCV date
def had_prior_imm_refusal(mrn, visit_date):
    return (imm_ref.loc[imm_ref['MRN'] == mrn, 'Diagnosis Event Start Date'] <= visit_date).any()

# Apply to WCV
wcv['imm_refusal'] = wcv.apply(lambda row: had_prior_imm_refusal(row['MRN'], row['Visit Date']), axis=1)


##-----NO SHOWS----###
# Check for no show
noshow = pd.read_excel('Visits_2025-11-05T19_19_03.xlsx', usecols=['MRN', 'Visit Date', 'Appointment Status'])
noshow = noshow[noshow['Appointment Status'] == 'No Show']

# Sort MRN and Date
noshow = noshow.sort_values(['MRN', 'Visit Date'])
wcv = wcv.sort_values(['MRN', 'Visit Date'])

# Function to check prior noshow
def prior_noshow(mrn, visit_date):
    return (noshow.loc[noshow['MRN'] == mrn, 'Visit Date'] <= visit_date).any()

# Apply to WCV
wcv['prior_noshow'] = wcv.apply(lambda row: prior_noshow(row['MRN'], row['Visit Date']), axis=1)


##-----HAS PRIOR MISSED WCV----###
wcv['missed'] = ~wcv['has_next_wcv']
#flag for prior missed WCV
wcv['any_prior_missed'] = (
    wcv.groupby('MRN')['missed']
       .transform(lambda x: x.shift().cumsum() > 0)
)

##-----FINAL DROP----###
#drop unneeded columns
wcv.drop(['MyChart Status at Instant', 'Cancellation Reason', 'Appointment Scheduling Mode', 'Product Type Current Benefit Plan', 'Patient Birth Date', 'Weight (lb)', 'has_next_wcv', 'Department', 'Visit Type', 'Visit Date', 'Appointment Status', 'Confirmed?', 'Late Cancel?', 'Appointment Confirmation Status', 'eCheck-In Status', 'BMI (kg/m^2)', 'Primary Diagnosis Code ICD-10-CM', 'Department Specialty', 'sample', 'next_expected_wcv', 'insurance', 'provider_type', 'Patient Sex', 'Day of Week', 'age_months', 'Interpreter Needed?'], axis=1, inplace=True)

##-----SAVE----###
wcv.to_parquet("wcv")
wcv[['MRN', 'clinic_group', 'race', 'ethnicity', 'language']].to_parquet("wcv_demo")
wcv[wcv['clinic_group'] == 0].drop(columns=['clinic_group', 'race', 'ethnicity', 'language']).astype('float64').to_parquet("wcv_train")
wcv[wcv['clinic_group'] == 1].drop(columns=['clinic_group', 'race', 'ethnicity', 'language']).astype('float64').to_parquet("wcv_test")


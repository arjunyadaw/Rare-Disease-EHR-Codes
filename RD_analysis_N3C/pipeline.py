

@transform_pandas(
    Output(rid="ri.vector.main.execute.6b9ebc59-5588-48ec-95ce-2de1d7ad0eeb"),
    rd_chisqr_age_gender=Input(rid="ri.foundry.main.dataset.79e2e404-6e61-470d-aa1b-2f56aff3e3a5")
)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def chisqr_female_death(rd_chisqr_age_gender):
    df = rd_chisqr_age_gender
    # Filter data where gender_MALE == 1
    df_male = df[df['Gender'] == 'FEMALE']

    # Initialize a dictionary to store results
    results = {}

    # Loop through each unique Age_bin
    for age_bin in df_male['Age_bin'].unique():
        # Filter data for the current Age_bin
        df_age_bin = df_male[df_male['Age_bin'] == age_bin]
        
        # Create a contingency table for rare_disease (1 vs 0) vs death (1 vs 0)
        contingency_table = pd.crosstab(df_age_bin['rare_disease'], df_age_bin['death'])
        
        # Perform the Chi-Square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Store the results for the current Age_bin
        results[age_bin] = {
            'Chi-Square Statistic': chi2_stat,
            'P-value': p_value,
            'Degrees of Freedom': dof,
            'Expected Frequencies': expected
        }

    # Print the results
    for age_bin, result in results.items():
        print(f"Age Bin: {age_bin}")
        print(f"Chi-Square Statistic: {result['Chi-Square Statistic']}")
        print(f"P-value: {result['P-value']}")
        print(f"Degrees of Freedom: {result['Degrees of Freedom']}")
        print("Expected Frequencies:")
        print(result['Expected Frequencies'])
        print("\n")

@transform_pandas(
    Output(rid="ri.vector.main.execute.1c948907-cb0e-4820-b5a3-810c8853d20d"),
    rd_chisqr_age_gender=Input(rid="ri.foundry.main.dataset.79e2e404-6e61-470d-aa1b-2f56aff3e3a5")
)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def chisqr_female_hosp(rd_chisqr_age_gender):
    df = rd_chisqr_age_gender
    # Filter data where gender_MALE == 1
    df_male = df[df['Gender'] == 'FEMALE']

    # Initialize a dictionary to store results
    results = {}

    # Loop through each unique Age_bin
    for age_bin in df_male['Age_bin'].unique():
        # Filter data for the current Age_bin
        df_age_bin = df_male[df_male['Age_bin'] == age_bin]
        
        # Create a contingency table for rare_disease (1 vs 0) vs death (1 vs 0)
        contingency_table = pd.crosstab(df_age_bin['rare_disease'], df_age_bin['hospitalized'])
        
        # Perform the Chi-Square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Store the results for the current Age_bin
        results[age_bin] = {
            'Chi-Square Statistic': chi2_stat,
            'P-value': p_value,
            'Degrees of Freedom': dof,
            'Expected Frequencies': expected
        }

    # Print the results
    for age_bin, result in results.items():
        print(f"Age Bin: {age_bin}")
        print(f"Chi-Square Statistic: {result['Chi-Square Statistic']}")
        print(f"P-value: {result['P-value']}")
        print(f"Degrees of Freedom: {result['Degrees of Freedom']}")
        print("Expected Frequencies:")
        print(result['Expected Frequencies'])
        print("\n")

@transform_pandas(
    Output(rid="ri.vector.main.execute.bb1f2e67-29c5-414b-8251-164c248b2802"),
    rd_chisqr_age_gender=Input(rid="ri.foundry.main.dataset.79e2e404-6e61-470d-aa1b-2f56aff3e3a5")
)
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def chisqr_male_death(rd_chisqr_age_gender):
    df = rd_chisqr_age_gender
    # Filter data where gender_MALE == 1
    df_male = df[df['Gender'] == 'MALE']

    # Initialize a dictionary to store results
    results = {}

    # Loop through each unique Age_bin
    for age_bin in df_male['Age_bin'].unique():
        # Filter data for the current Age_bin
        df_age_bin = df_male[df_male['Age_bin'] == age_bin]
        
        # Create a contingency table for rare_disease (1 vs 0) vs death (1 vs 0)
        contingency_table = pd.crosstab(df_age_bin['rare_disease'], df_age_bin['death'])
        
        # Perform the Chi-Square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Store the results for the current Age_bin
        results[age_bin] = {
            'Chi-Square Statistic': chi2_stat,
            'P-value': p_value,
            'Degrees of Freedom': dof,
            'Expected Frequencies': expected
        }

    # Print the results
    for age_bin, result in results.items():
        print(f"Age Bin: {age_bin}")
        print(f"Chi-Square Statistic: {result['Chi-Square Statistic']}")
        print(f"P-value: {result['P-value']}")
        print(f"Degrees of Freedom: {result['Degrees of Freedom']}")
        print("Expected Frequencies:")
        print(result['Expected Frequencies'])
        print("\n")

@transform_pandas(
    Output(rid="ri.vector.main.execute.725853b8-de92-4a16-8bbd-d123e2aa5619"),
    rd_chisqr_age_gender=Input(rid="ri.foundry.main.dataset.79e2e404-6e61-470d-aa1b-2f56aff3e3a5")
)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def chisqr_male_hosp(rd_chisqr_age_gender):
    df = rd_chisqr_age_gender
    # Filter data where gender = MALE
    df_male = df[df['Gender'] == 'MALE']

    # Initialize a dictionary to store results
    results = {}

    # Loop through each unique Age_bin
    for age_bin in df_male['Age_bin'].unique():
        # Filter data for the current Age_bin
        df_age_bin = df_male[df_male['Age_bin'] == age_bin]
        
        # Create a contingency table for rare_disease (1 vs 0) vs death (1 vs 0)
        contingency_table = pd.crosstab(df_age_bin['rare_disease'], df_age_bin['hospitalized'])
        
        # Perform the Chi-Square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Store the results for the current Age_bin
        results[age_bin] = {
            'Chi-Square Statistic': chi2_stat,
            'P-value': p_value,
            'Degrees of Freedom': dof,
            'Expected Frequencies': expected
        }

    # Print the results
    for age_bin, result in results.items():
        print(f"Age Bin: {age_bin}")
        print(f"Chi-Square Statistic: {result['Chi-Square Statistic']}")
        print(f"P-value: {result['P-value']}")
        print(f"Degrees of Freedom: {result['Degrees of Freedom']}")
        print("Expected Frequencies:")
        print(result['Expected Frequencies'])
        print("\n")

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.045b70e3-bf68-4747-8bec-2f2d33ba1776"),
    covid_rd_vac_longc_antiv=Input(rid="ri.foundry.main.dataset.2fef5ac1-8965-432c-b9ea-90b85346e843")
)
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan, when, count, col, lit, regexp_replace,mean,desc,row_number, array, array_contains

def cohort_data(covid_rd_vac_longc_antiv):
    df = covid_rd_vac_longc_antiv.select("hospitalized","death","age_1_20","age_20_40","age_40_65","age_65_above","Gender","Race","Ethnicity","smoking_status","rare_disease","rare_bone_diseases","rare_cardiac_diseases", "rare_circulatory_system_disease","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_maxillo_facial_surgical_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease","rare_systemic_or_rheumatologic_disease","rare_urogenital_disease")

    # Encoding object type features into binary features
    gender_categories = df.select('Gender').distinct().rdd.flatMap(lambda x: x).collect()
    ethnicity_categories = df.select('Ethnicity').distinct().rdd.flatMap(lambda x: x).collect()
    race_categories = df.select('Race').distinct().rdd.flatMap(lambda x: x).collect()
    smoking_status_categories = df.select('smoking_status').distinct().rdd.flatMap(lambda x: x).collect()

    gender_exprs = [F.when(F.col('Gender') == cat, 1).otherwise(0).alias(f"gender_{cat}") for cat in gender_categories]
    ethnicity_exprs = [F.when(F.col('Ethnicity') == cat, 1).otherwise(0).alias(f"ethnicity_{cat}") for cat in ethnicity_categories]
    race_exprs = [F.when(F.col('Race') == cat, 1).otherwise(0).alias(f"race_{cat}") for cat in race_categories]
    smoking_status_exprs = [F.when(F.col('smoking_status') == cat, 1).otherwise(0).alias(f"smoking_status_{cat}") for cat in smoking_status_categories]

    df = df.select(gender_exprs + race_exprs + ethnicity_exprs + smoking_status_exprs + df.columns)
    cols_to_drop = ["Gender", "Race", "Ethnicity","smoking_status"]
    df2 = df.drop(*cols_to_drop)

    return df2

"""
"rare_odontologic_disease",
"rare_abdominal_surgical_diseases",
"Tuberculosis",
"Rheumatologic_disease",
"Dementia_before",
"Congestive_heart_failure",
"Kidney_disease",
"Cerebro_vascular_disease",
"Peripheral_vascular_disease",
"Heart_failure",
"Hemiplegia_or_paraplegia",
"Psychosis",
"OBESITY_before_covid_indicator",
"Coronary_artery_disease",
"Systemic_corticosteroids",
"Depression",
"HIV_infection",
"Chronic_lung_disease",
"PEPTICULCER_before_covid_indicator",
"Myocardial_infraction",
"Cardiomyophaties",
"HTN",
"Tobacco_smoker",
"liver_disease",
"Cancer",
"Diabetes"
"""

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3eca198a-b21a-4336-8cdb-227fbf966dff"),
    cohort_data=Input(rid="ri.foundry.main.dataset.045b70e3-bf68-4747-8bec-2f2d33ba1776")
)
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

def cohort_death(cohort_data):
    df = cohort_data.select("hospitalized", "death", "gender_MALE", "gender_FEMALE", "race_Black_or_African_American",  
                        "race_Unknown", "race_White", "race_Asian", "ethnicity_Not_Hispanic_or_Latino",
                        "ethnicity_Hispanic_or_Latino", "ethnicity_Unknown", "smoking_status_Non_smoker",
                        "smoking_status_Current_or_Former","age_1_20", "age_20_40", "age_40_65", "age_65_above",
                        "rare_disease", "rare_bone_diseases","rare_cardiac_diseases",
                        "rare_circulatory_system_disease","rare_developmental_defect_during_embryogenesis",
                        "rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease",
                        "rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease",
                        "rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_maxillo_facial_surgical_disease",
                        "rare_neoplastic_disease","rare_neurologic_disease",
                        "rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease",
                        "rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease",
                        "rare_systemic_or_rheumatologic_disease","rare_urogenital_disease")

    feature_columns = ["gender_MALE", "gender_FEMALE", "race_Black_or_African_American", "race_Unknown",
                       "race_White", "race_Asian", "ethnicity_Not_Hispanic_or_Latino",
                       "ethnicity_Hispanic_or_Latino", "ethnicity_Unknown", "smoking_status_Non_smoker",
                       "smoking_status_Current_or_Former", "age_1_20", "age_20_40", "age_40_65",
                       "age_65_above", "rare_disease", "rare_bone_diseases",
                       "rare_cardiac_diseases", "rare_circulatory_system_disease", 
                       "rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease",
                       "rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease",
                       "rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease",
                       "rare_maxillo_facial_surgical_disease","rare_neoplastic_disease","rare_neurologic_disease",
                       "rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease",
                       "rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease",
                       "rare_systemic_or_rheumatologic_disease","rare_urogenital_disease"]

    feature_list = []
    total_count = []
    number_incident = []
    number_hosp = []
    death_percentage = []
    hosp_percentage = []

    for feature in feature_columns:
        # Count number of patients for each feature
        num_patients = df.filter(col(feature) == 1).count()
        # Count number of deaths for each feature
        num_deaths = df.filter((col(feature) == 1) & (col("death") == 1)).count()
        # Count number of hospitalizations for each feature
        num_hosp = df.filter((col(feature) == 1) & (col("hospitalized") == 1)).count()
        
        # Calculate percentages
        if num_patients > 0:
            death_perc = (num_deaths / num_patients) * 100
            hosp_perc = (num_hosp / num_patients) * 100
        else:
            death_perc = 0.0
            hosp_perc = 0.0
        
        # Append results to lists
        feature_list.append(feature)
        total_count.append(num_patients)
        number_incident.append(num_deaths)
        number_hosp.append(num_hosp)
        death_percentage.append(death_perc)
        hosp_percentage.append(hosp_perc)
    
    # Create DataFrame with new columns
    result_df = spark.createDataFrame(zip(feature_list, total_count, number_incident, number_hosp,
                                          death_percentage, hosp_percentage),
                                      schema=["features", "total_patients", "death_count", "hosp_count",
                                              "death_percentage", "hosp_percentage"])

    return result_df

    """ 
def cohort_death(cohort_data):
    df = cohort_data.select("hospitalized","death","gender_MALE","gender_FEMALE","race_Black_or_African_American","race_Unknown","race_White","race_Asian","ethnicity_Not_Hispanic_or_Latino","ethnicity_Hispanic_or_Latino","ethnicity_Unknown","smoking_status_Non_smoker","smoking_status_Current_or_Former","age_1_20","age_20_40","age_40_65","age_65_above","rare_disease","rare_bone_diseases","rare_cardiac_diseases","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_systemic_or_rheumatologic_disease","rare_transplantation_disease")

    feature_columns = ["gender_MALE","gender_FEMALE","race_Black_or_African_American","race_Unknown","race_White","race_Asian","ethnicity_Not_Hispanic_or_Latino","ethnicity_Hispanic_or_Latino","ethnicity_Unknown","smoking_status_Non_smoker","smoking_status_Current_or_Former","age_1_20","age_20_40","age_40_65","age_65_above","rare_disease","rare_bone_diseases","rare_cardiac_diseases","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_systemic_or_rheumatologic_disease","rare_transplantation_disease"]

    feature_list = []
    total_count = []
    number_incident = []
    number_hosp = []

    for feature in feature_columns:
        # count number of patients for each features
        num_patients = df.filter(col(feature) == 1).count()
        # count number of deaths for each features
        num_deaths = df.filter((col(feature) == 1) & (col("death") ==1 )).count()
        # count number of hospitalized for each features
        num_hosp = df.filter((col(feature) == 1) & (col("hospitalized") ==1 )).count()        
        # append results to list 
        feature_list.append(feature)
        total_count.append(num_patients)
        number_incident.append(num_deaths)
        number_hosp.append(num_hosp)
    #create dataframe
    result_df = spark.createDataFrame(zip(feature_list,total_count,number_incident, number_hosp), schema = ["features","total_patients","death_count","hosp_count"])

    return result_df   
    """

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8759c318-64c1-447e-ac27-6d21a2c93ccd"),
    covid_rd_vac_longc_antiv=Input(rid="ri.foundry.main.dataset.2fef5ac1-8965-432c-b9ea-90b85346e843")
)
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan, when, count, col, lit, regexp_replace,mean,desc,row_number, array, array_contains
def covid_rd_cohort_data(covid_rd_vac_longc_antiv):
    df = covid_rd_vac_longc_antiv

    # Vaccination status defined if person has taken atleast 1 dose of vaccine
    df = df.withColumn('vaccination_status', when(df.Covid_vaccine_dose >= 1,1).otherwise(0)) 
   
    df = df.withColumn("life_thretening",
                     when(col("Severity_Type")=='Death', 1)
                    .when(col("Severity_Type")=='Severe', 1)
                    .otherwise(0))

    df = df.withColumn("antiviral_treatment",
                     when(col("Paxlovid")== 1, 1)
                    .when(col("Molnupiravir")==1, 1)
                    .when(col("Bebtelovimab")==1, 1)
                    .otherwise(0))

    # Add a new column 'Age_bin' based on the binning conditions
    df = df.withColumn('Age_bin', 
                    when((df.Age > 1) & (df.Age <= 20), '1-20')
                    .when((df.Age > 20) & (df.Age <= 40), '20-40')
                    .when((df.Age > 40) & (df.Age <= 65), '40-65')
                    .otherwise('65+'))

#    binary_cols = ["rare_odontologic_disease", 
#                "rare_disorder_due_to_toxic_effects",
#                "rare_abdominal_surgical_diseases",
#                "rare_transplantation_disease"]

    # Create a new column that is an array of the binary columns
#    df = df.withColumn("binary_array", array(*binary_cols))

    # Create a new column that is 1 if the array contains 1, and 0 otherwise
#    df = df.withColumn("other_rare_diseases", array_contains("binary_array", 1).cast("int"))

    # Drop the intermediate column
#    df = df.drop("binary_array",
#                "rare_odontologic_disease",
#                "rare_disorder_due_to_toxic_effects",
#                "rare_abdominal_surgical_diseases",
#                "rare_transplantation_disease")

    # Encoding object type features into binary features
    bmi_categories = df.select('BMI_category').distinct().rdd.flatMap(lambda x: x).collect()
    gender_categories = df.select('Gender').distinct().rdd.flatMap(lambda x: x).collect()
    ethnicity_categories = df.select('Ethnicity').distinct().rdd.flatMap(lambda x: x).collect()
    race_categories = df.select('Race').distinct().rdd.flatMap(lambda x: x).collect()
    smoking_status_categories = df.select('smoking_status').distinct().rdd.flatMap(lambda x: x).collect()

    bmi_exprs = [F.when(F.col('BMI_category') == cat, 1).otherwise(0).alias(f"{cat}") for cat in bmi_categories]
    gender_exprs = [F.when(F.col('Gender') == cat, 1).otherwise(0).alias(f"gender_{cat}") for cat in gender_categories]
    ethnicity_exprs = [F.when(F.col('Ethnicity') == cat, 1).otherwise(0).alias(f"ethnicity_{cat}") for cat in ethnicity_categories]
    race_exprs = [F.when(F.col('Race') == cat, 1).otherwise(0).alias(f"race_{cat}") for cat in race_categories]
    smoking_status_exprs = [F.when(F.col('smoking_status') == cat, 1).otherwise(0).alias(f"smoking_status_{cat}") for cat in smoking_status_categories]

    df = df.select(bmi_exprs + gender_exprs + race_exprs + ethnicity_exprs + smoking_status_exprs + df.columns)
#    cols_to_drop = ["BMI_category","Gender", "Race", "Ethnicity","smoking_status"]
#    df2 = df.drop(*cols_to_drop)
    # Age binning
    df = df.withColumn("age_1_20", when((col('Age') >1) & (col('Age') <=20), 1).otherwise(0))\
            .withColumn("age_20_40", when((col('Age') >20) & (col('Age') <=40), 1).otherwise(0))\
            .withColumn("age_40_65", when((col('Age') >40) & (col('Age') <=65), 1).otherwise(0))\
            .withColumn("age_65_above", when(col('Age')>65, 1).otherwise(0))

    return df.select("death","hospitalized","Age","Age_bin","BMI_category","Gender","Race","Ethnicity","smoking_status","age_1_20","age_20_40","age_40_65","age_65_above","BMI_obese","BMI_over_weight","BMI_normal","BMI_under_weight","gender_MALE","gender_FEMALE","race_Black_or_African_American","race_Unknown","race_White","race_Asian","ethnicity_Not_Hispanic_or_Latino","ethnicity_Hispanic_or_Latino","ethnicity_Unknown","smoking_status_Non_smoker","smoking_status_Current_or_Former","vaccination_status","antiviral_treatment","COVID_reinfection","Cancer","Cardiomyophaties","Cerebro_vascular_disease","Chronic_lung_disease","Coronary_artery_disease","Dementia_before","Depression","Diabetes","Heart_failure","HIV_infection","HTN","Kidney_disease","liver_disease","Myocardial_infraction","Peripheral_vascular_disease","Rheumatologic_disease","Systemic_corticosteroids","rare_disease","rare_bone_diseases","rare_cardiac_diseases", "rare_circulatory_system_disease","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_maxillo_facial_surgical_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease","rare_systemic_or_rheumatologic_disease","rare_urogenital_disease")   # "long_covid", ,"other_rare_diseases"

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2fef5ac1-8965-432c-b9ea-90b85346e843"),
    Covid_drug_treatment=Input(rid="ri.foundry.main.dataset.9b49d62c-05a8-4bfc-8e62-cca5c09b6291"),
    Hormonised_data_rds=Input(rid="ri.foundry.main.dataset.8f8570df-1729-4969-b128-1f0318ac5854"),
    death_data_table=Input(rid="ri.foundry.main.dataset.5ad995c3-98fd-4005-b19b-07c96a73a88d")
)
"""
 Here rd_covid_linear_cohort is considered in place of hormonised_data_for_model table because long COVID data is not listed in enclave due to observation table issue.
"""
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import isnan, when, count, col, lit, regexp_replace, mean, desc, row_number, array, array_contains

def covid_rd_vac_longc_antiv(Covid_drug_treatment, death_data_table, Hormonised_data_rds):
    df1 = Hormonised_data_rds
    df2 = Covid_drug_treatment.select("person_id","drug_exposure_start_date","Paxlovid","Molnupiravir","Bebtelovimab","selected_antiviral_treatment")
    df3 = death_data_table    
    df4 = df1.join(df2, on = 'person_id', how = 'leftouter')
    print('size of covid plus drug table', df4.count())
    df5 = df4.join(df3, on = 'person_id', how = 'leftouter')
    print('size of covid plus death table', df5.count())
    
    # number of days survival since covid first diagnosis day
    df5 = df5.withColumn('death_days_since_diagnosis', F.datediff("death_date", "COVID_first_poslab_or_diagnosis_date"))
    
    # Ensure death_days_since_diagnosis >= 0
#    df5 = df5.filter(F.col('death_days_since_diagnosis') >= 0)

    # Filter deaths to be between 2020-01-01 and 2025-01-01
#    df5 = df5.filter(
#        (F.col('death_date') >= '2020-01-01') & (F.col('death_date') <= '2025-01-01')
#    )

    # Replace NaN with 0 except BMI.  
    df = df5.na.fill(value=0, subset=[col for col in df5.columns if col not in ('BMI')])

    # Combine 7 rare diseases classes as other rare diseases and drop them
    binary_cols = ["rare_odontologic_disease",
                   "rare_disorder_due_to_toxic_effects",
                   "rare_transplantation_disease",
                   "rare_abdominal_surgical_diseases"]

    # Create a new column that is an array of the binary columns
    df = df.withColumn("binary_array", array(*binary_cols))

    # Create a new column that is 1 if the array contains 1, and 0 otherwise
    df = df.withColumn("other_rare_diseases", array_contains("binary_array", 1).cast("int"))

    # Now drop patients with the following 7 rare disease classes
    df = df.where(F.col('other_rare_diseases') == 0)

    # Create a window specification for ordering by date
    window_spec = Window.partitionBy("person_id").orderBy(desc("COVID_first_poslab_or_diagnosis_date"))

    # Add a rank column to identify the latest date per person_id
    df_ranked = df.withColumn("rank", row_number().over(window_spec))

    # Select rows with rank = 1 (latest date)
    result_df = df_ranked.where(col("rank") == 1).drop("rank")
    
    return result_df

"""
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import isnan, when, count, col, lit, regexp_replace,mean,desc, row_number, array, array_contains
def covid_rd_vac_longc_antiv( Covid_drug_treatment, death_data_table, Hormonised_data_rds):
    df1 = Hormonised_data_rds
    df2 = Covid_drug_treatment.select("person_id","drug_exposure_start_date","Paxlovid","Molnupiravir","Bebtelovimab","selected_antiviral_treatment")
    df3 = death_data_table    
    df4 = df1.join(df2, on = 'person_id', how = 'leftouter')
    print('size of covid plus drug table', df4.count())
    df5 = df4.join(df3, on = 'person_id', how = 'leftouter')
    print('size of covid plus death table', df5.count())
    # number of days survival since covid first diagnosis day
    df5 = df5.withColumn('death_days_since_diagnosis', F.datediff("death_date", "COVID_first_poslab_or_diagnosis_date"))
    # Replace NaN with 0 except BMI.  
    df = df5.na.fill(value=0, subset = [col for col in df5.columns if col not in ('BMI')])

    # Combine 7 rare diseases classes as other rare diseases and drop them
    binary_cols = ["rare_odontologic_disease",
    "rare_disorder_due_to_toxic_effects",
    "rare_transplantation_disease",
    "rare_abdominal_surgical_diseases"]

    # Create a new column that is an array of the binary columns
    df = df.withColumn("binary_array", array(*binary_cols))

    # Create a new column that is 1 if the array contains 1, and 0 otherwise
    df = df.withColumn("other_rare_diseases", array_contains("binary_array", 1).cast("int"))

    # Now drop patients with following 7 rare disease classes
    df = df.where(F.col('other_rare_diseases') == 0)

    # Create a window specification for ordering by date
    window_spec = Window.partitionBy("person_id").orderBy(desc("COVID_first_poslab_or_diagnosis_date"))

    # Add a rank column to identify the latest date per person_id
    df_ranked = df.withColumn("rank", row_number().over(window_spec))

    # Select rows with rank = 1 (latest date)
    result_df = df_ranked.where(col("rank") == 1).drop("rank")
    
    return result_df
"""    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5ad995c3-98fd-4005-b19b-07c96a73a88d"),
    Covid_deaths=Input(rid="ri.foundry.main.dataset.03fcbd50-a1ea-42ac-8c60-6bcb87092d21")
)
# Covid_deathstable from COVID cohort workbook (visit_date is renamed from death_date in that cohort)
from pyspark.sql import functions as F
def death_data_table(Covid_deaths):
    df = Covid_deaths.select('person_id','visit_date')
    # Renaming columns name for further usage
    df = df.withColumnRenamed('visit_date', 'death_date')
    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.762d32b4-b4d8-451a-83dc-d2f2f9765d07"),
    cohort_data=Input(rid="ri.foundry.main.dataset.045b70e3-bf68-4747-8bec-2f2d33ba1776")
)

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, col
from pyspark.sql.types import FloatType
from scipy.stats import norm

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import numpy as np
from scipy.stats import norm

def death_relative_risk_incidence_rate_CI_gynecology(cohort_data):
    df = cohort_data.select("gender_MALE", "gender_FEMALE", "race_Black_or_African_American", "race_Unknown", "race_White", "race_Asian","ethnicity_Not_Hispanic_or_Latino", "ethnicity_Hispanic_or_Latino", "ethnicity_Unknown", "smoking_status_Non_smoker", "smoking_status_Current_or_Former", "hospitalized", "death", "age_1_20", "age_20_40", "age_40_65", "age_65_above", "rare_disease", "rare_bone_diseases","rare_cardiac_diseases", "rare_circulatory_system_disease","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_maxillo_facial_surgical_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease","rare_systemic_or_rheumatologic_disease","rare_urogenital_disease")

    df = df.withColumn("control", F.lit(1 - df["rare_disease"]))
    
    feature_columns = ["control", "rare_bone_diseases","rare_cardiac_diseases", "rare_circulatory_system_disease","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_maxillo_facial_surgical_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease","rare_systemic_or_rheumatologic_disease","rare_urogenital_disease"]

    feature_list = []
    total_count = []
    number_incident = []

    for feature in feature_columns:
        if feature == 'rare_gynecologic_or_obstetric_disease':
            # For 'rare_gynecologic_or_obstetric_disease', filter by gender_FEMALE == 1
            num_patients = df.filter((df["gender_FEMALE"] == 1) & (df[feature] == 1)).count()
            num_deaths = df.filter((df["gender_FEMALE"] == 1) & (df[feature] == 1) & (df["death"] == 1)).count()
        else:
            # For other features, no gender restriction
            num_patients = df.filter(df[feature] == 1).count()
            num_deaths = df.filter((df[feature] == 1) & (df["death"] == 1)).count()

        # Append results to list
        feature_list.append(feature)
        total_count.append(num_patients)
        number_incident.append(num_deaths)

    # Create dataframe result_df
    result_df = spark.createDataFrame(zip(feature_list, total_count, number_incident), schema=["features", "total_patients", "death_count"])

    # Calculate relative risk
    control_death_count = result_df.filter(col("features") == "control").select("death_count").collect()[0][0]
    control_total_patients = result_df.filter(col("features") == "control").select("total_patients").collect()[0][0]

    result_df = result_df.withColumn("relative_risk", (col("death_count") / col("total_patients")) / (control_death_count / control_total_patients))

    # Calculate p-values
    def calculate_p_value(row):
        rr = row['relative_risk']
        se_log_rr = np.sqrt((1 / row['death_count']) + (1 / control_death_count) - (1 / row['total_patients']) - (1 / control_total_patients))
        z_score = np.log(rr) / se_log_rr
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        return float(p_value)

    p_value_udf = F.udf(calculate_p_value, FloatType())
    result_df = result_df.withColumn("p_value", p_value_udf(F.struct([result_df[x] for x in result_df.columns])))

    # Calculate standard error of ln(RR)
    result_df = result_df.withColumn("SE_ln_RR", 1 / col("death_count") + 1 / control_death_count - 1 / (col("death_count") + col("total_patients")) - 1 / (control_death_count + control_total_patients)).toPandas()

    # Z-score for 95% confidence interval
    Z = 1.96  # For a 95% CI

    result_df['relative_risk'] = round(result_df['relative_risk'], 3)
    result_df['ln_RR'] = round(np.log(result_df['relative_risk']), 3)
    result_df['lower_CI'] = round(np.exp(result_df['ln_RR'] - Z * result_df['SE_ln_RR']), 3)
    result_df['upper_CI'] = round(np.exp(result_df['ln_RR'] + Z * result_df['SE_ln_RR']), 3)
    result_df['RR_CI'] = result_df.apply(lambda row: f"{row['relative_risk']} ({row['lower_CI']} - {row['upper_CI']})", axis=1)

    return result_df[["features","total_patients","death_count", "RR_CI", "p_value","relative_risk"]]

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.db5e15bb-ec7e-4774-aad9-ef4d2a555ea8"),
    cohort_data=Input(rid="ri.foundry.main.dataset.045b70e3-bf68-4747-8bec-2f2d33ba1776")
)

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, col
from pyspark.sql.types import FloatType
from scipy.stats import norm

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import numpy as np
from scipy.stats import norm

def hosp_relative_risk_incidence_rate_CI_gynecology(cohort_data):
    df = cohort_data
    df = df.withColumn("control", F.lit(1 - df["rare_disease"]))
    
    feature_columns = ["control", "rare_bone_diseases","rare_cardiac_diseases", "rare_circulatory_system_disease","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_maxillo_facial_surgical_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease","rare_systemic_or_rheumatologic_disease","rare_urogenital_disease"]

    feature_list = []
    total_count = []
    number_incident = []

    for feature in feature_columns:
        if feature == 'rare_gynecologic_or_obstetric_disease':
            # For 'rare_gynecologic_or_obstetric_disease', filter by gender_FEMALE == 1
            num_patients = df.filter((df["gender_FEMALE"] == 1) & (df[feature] == 1)).count()
            num_deaths = df.filter((df["gender_FEMALE"] == 1) & (df[feature] == 1) & (df["hospitalized"] == 1)).count()
        else:
            # For other features, no gender restriction
            num_patients = df.filter(df[feature] == 1).count()
            num_deaths = df.filter((df[feature] == 1) & (df["hospitalized"] == 1)).count()

        # Append results to list
        feature_list.append(feature)
        total_count.append(num_patients)
        number_incident.append(num_deaths)

    # Create dataframe result_df
    result_df = spark.createDataFrame(zip(feature_list, total_count, number_incident), schema=["features", "total_patients", "hosp_count"])

    # Calculate relative risk
    control_death_count = result_df.filter(F.col("features") == "control").select("hosp_count").collect()[0][0]
    control_total_patients = result_df.filter(F.col("features") == "control").select("total_patients").collect()[0][0]

    result_df = result_df.withColumn("relative_risk", (F.col("hosp_count") / F.col("total_patients")) / (control_death_count / control_total_patients))

    # Calculate p-values
    def calculate_p_value(row):
        rr = row['relative_risk']
        se_log_rr = np.sqrt((1 / row['hosp_count']) + (1 / control_death_count) - (1 / row['total_patients']) - (1 / control_total_patients))
        z_score = np.log(rr) / se_log_rr
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        return float(p_value)

    p_value_udf = F.udf(calculate_p_value, FloatType())
    result_df = result_df.withColumn("p_value", p_value_udf(F.struct([result_df[x] for x in result_df.columns])))

    # Calculate standard error of ln(RR)
    result_df = result_df.withColumn("SE_ln_RR", 1 / F.col("hosp_count") + 1 / control_death_count - 1 / (F.col("hosp_count") + F.col("total_patients")) - 1 / (control_death_count + control_total_patients)).toPandas()

    # Z-score for 95% confidence interval
    Z = 1.96  # For a 95% CI

    result_df['relative_risk'] = round(result_df['relative_risk'], 3)
    result_df['ln_RR'] = round(np.log(result_df['relative_risk']), 3)
    result_df['lower_CI'] = round(np.exp(result_df['ln_RR'] - Z * result_df['SE_ln_RR']), 3)
    result_df['upper_CI'] = round(np.exp(result_df['ln_RR'] + Z * result_df['SE_ln_RR']), 3)
    result_df['RR_CI'] = result_df.apply(lambda row: f"{row['relative_risk']} ({row['lower_CI']} - {row['upper_CI']})", axis=1)

    return result_df[["features","total_patients","hosp_count","RR_CI","p_value","relative_risk"]]

@transform_pandas(
    Output(rid="ri.vector.main.execute.8575db2c-e0bd-468c-ae90-b92d0f02320e"),
    covid_rd_cohort_data=Input(rid="ri.foundry.main.dataset.8759c318-64c1-447e-ac27-6d21a2c93ccd")
)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
from pyspark.sql.functions import expr

def hospitalized_model(covid_rd_cohort_data):
    # Select relevant columns from the PySpark DataFrame
    df = covid_rd_cohort_data.select("death", "hospitalized", "rare_disease","rare_bone_diseases","rare_cardiac_diseases", "rare_circulatory_system_disease","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_maxillo_facial_surgical_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease","rare_systemic_or_rheumatologic_disease","rare_urogenital_disease")

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['rds','Severity', 'odds_ratio', 'coefficient', 'CI_lower', 'CI_upper', 'pvalue', 'N'])

    rare_diseases = ["rare_disease","rare_bone_diseases","rare_cardiac_diseases", "rare_circulatory_system_disease","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_maxillo_facial_surgical_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease","rare_systemic_or_rheumatologic_disease","rare_urogenital_disease"] 
    for rd in rare_diseases:
        # Filter the DataFrame for the current rare disease category
        df_filtered = df.filter(expr(f"(rare_disease = 1 AND {rd} = 1) OR (rare_disease = 0 AND {rd} = 0)"))
        
        # Convert the filtered PySpark DataFrame to a Pandas DataFrame for logistic regression
        df_filtered_pd = df_filtered.toPandas()

        formula = f'hospitalized ~ {rd}'
        y, X = dmatrices(formula, data=df_filtered_pd, return_type='dataframe')
        mod = sm.Logit(y, X)
        res = mod.fit()
        
        # Extract relevant coefficients and confidence intervals
        coef = res.params[rd]
        odds_ratio = np.exp(coef)
        ci_lower, ci_upper = np.exp(res.conf_int().loc[rd])
        pvalue = res.pvalues[rd]
        N = len(df_filtered_pd)

        # Append results to the DataFrame
        results_df = results_df.append({
            'rds': rd,
            'Severity':'hospitalized',
            'odds_ratio': round(odds_ratio, 2),
            'coefficient': round(coef, 2),
            'CI_lower': round(ci_lower, 2),
            'CI_upper': round(ci_upper, 2),
            'pvalue': pvalue,
            'N': N
        }, ignore_index=True)

        results_df['odds_ratio_new'] = results_df.apply(lambda row: f"{row['odds_ratio']} ({row['CI_lower']} - {row['CI_upper']})", axis = 1)

    return results_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.4ed109de-5d14-4c29-9a4b-0059c78d90ce"),
    covid_rd_cohort_data=Input(rid="ri.foundry.main.dataset.8759c318-64c1-447e-ac27-6d21a2c93ccd")
)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
from pyspark.sql.functions import expr

def mortality_model(covid_rd_cohort_data):
    # Select relevant columns from the PySpark DataFrame
    df = covid_rd_cohort_data.select("death", "hospitalized", "rare_disease","rare_bone_diseases","rare_cardiac_diseases", "rare_circulatory_system_disease","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_maxillo_facial_surgical_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease","rare_systemic_or_rheumatologic_disease","rare_urogenital_disease")

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['rds','Severity', 'odds_ratio', 'coefficient', 'CI_lower', 'CI_upper', 'pvalue', 'N'])

    rare_diseases = ["rare_disease","rare_bone_diseases","rare_cardiac_diseases", "rare_circulatory_system_disease","rare_developmental_defect_during_embryogenesis","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_maxillo_facial_surgical_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_surgical_thoracic_disease","rare_systemic_or_rheumatologic_disease","rare_urogenital_disease"] 
    for rd in rare_diseases:
        # Filter the DataFrame for the current rare disease category
        df_filtered = df.filter(expr(f"(rare_disease = 1 AND {rd} = 1) OR (rare_disease = 0 AND {rd} = 0)"))
        
        # Convert the filtered PySpark DataFrame to a Pandas DataFrame for logistic regression
        df_filtered_pd = df_filtered.toPandas()

        formula = f'death ~ {rd}'
        y, X = dmatrices(formula, data=df_filtered_pd, return_type='dataframe')
        mod = sm.Logit(y, X)
        res = mod.fit()
        
        # Extract relevant coefficients and confidence intervals
        coef = res.params[rd]
        odds_ratio = np.exp(coef)
        ci_lower, ci_upper = np.exp(res.conf_int().loc[rd])
        pvalue = res.pvalues[rd]
        N = len(df_filtered_pd)

        # Append results to the DataFrame
        results_df = results_df.append({
            'rds': rd,
            'Severity':'Mortality',
            'odds_ratio': round(odds_ratio, 2),
            'coefficient': round(coef, 2),
            'CI_lower': round(ci_lower, 2),
            'CI_upper': round(ci_upper, 2),
            'pvalue': pvalue,
            'N': N
        }, ignore_index=True)

        results_df['odds_ratio_new'] = results_df.apply(lambda row: f"{row['odds_ratio']} ({row['CI_lower']} - {row['CI_upper']})", axis = 1)

    return results_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.0fa02bff-c2b9-41e4-9b41-58e452b6885d"),
    rd_smr_age_gende=Input(rid="ri.foundry.main.dataset.14b7a539-d6e1-4bcb-b86e-37e5e849afc8")
)

import numpy as np
import matplotlib.pyplot as plt

def rd_age_death_female(rd_smr_age_gende):
    df = rd_smr_age_gende
    df = df[df['Gender'] == 'FEMALE']
    df = df[['Age_bin','rare_disease','total_patients','total_deaths']]
    
    # Create rare disease death percentage column
    df_rd = df[df['rare_disease'] == 1]
    df_rd['Rare disease'] = (df_rd['total_deaths'] / df_rd['total_patients']) * 100
    df_rd = df_rd.sort_values(by='Age_bin', ascending=False)

    # Create with no rare disease death percentage column
    df_rd_no = df[df['rare_disease'] == 0]
    df_rd_no['Without rare disease'] = (df_rd_no['total_deaths'] / df_rd_no['total_patients']) * 100
    df_rd_no = df_rd_no.sort_values(by='Age_bin', ascending=False)

    Age_bin = df_rd['Age_bin'].tolist()
    rd_list = df_rd['Rare disease'].tolist()
    no_rd_list = df_rd_no['Without rare disease'].tolist()

    y = np.arange(len(no_rd_list))

    fig, axes = plt.subplots(ncols=2, sharey=True, figsize = (7,3))
    
    # Plotting bars for rare disease
    axes[0].barh(y, rd_list, align='center', color='gray', zorder=10)
    axes[0].set(title='Rare Disease (Mortality)')
    axes[0].set_xlim(0, max(rd_list) + 5)  # Adjusted xlim to accommodate text

    # Plotting bars for no rare disease
    axes[1].barh(y, no_rd_list, align='center', color='gray', zorder=10)
    axes[1].set(title='Without rare disease (Mortality)')
    axes[1].set_xlim(0, max(rd_list) + 5)  # Adjusted xlim to accommodate text

    # Adding text labels on bars
    for i in range(len(rd_list)):
        axes[0].text(rd_list[i] + 3.6, y[i], f'{rd_list[i]:.2f}%', va='center')
        axes[1].text(no_rd_list[i] + 0.25, y[i], f'{no_rd_list[i]:.2f}%', va='center')

    axes[0].invert_xaxis()
    axes[0].set(yticks=y, yticklabels=Age_bin)
    axes[0].yaxis.tick_right()

    for ax in axes.flat:
        ax.margins(0.02)
        ax.grid(True , linestyle='--') # 

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0)
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.eed5e94d-db2a-481b-9e0c-7a5f1675e8c4"),
    rd_hospitalized_age_gender=Input(rid="ri.foundry.main.dataset.96da0af3-adfa-4718-a668-0797b1be6b91")
)
import numpy as np
import matplotlib.pyplot as plt

def rd_age_hospitalized_female(rd_hospitalized_age_gender):
    df = rd_hospitalized_age_gender
    df = df[df['Gender'] == 'FEMALE']
    df = df[['Age_bin','rare_disease','total_patients','total_hospitalized']]
    # Create rare disease hospitalized paercentage column
    df_rd = df[df['rare_disease'] == 1]
    # Create the new column 'rd_hospitalized_percentage'
    df_rd['Rare disease'] = (df_rd['total_hospitalized'] / df_rd['total_patients']) * 100
    df_rd = df_rd.sort_values(by = 'Age_bin',ascending = False)

    # Create with no rare disease death paercentage column
    df_rd_no = df[df['rare_disease'] == 0]
    # Create the new column 'rd_hospitalized_percentage'
    df_rd_no['Without rare disease'] = (df_rd_no['total_hospitalized'] / df_rd_no['total_patients']) * 100
    df_rd_no = df_rd_no.sort_values(by = 'Age_bin',ascending = False)

    Age_bin = df_rd['Age_bin'].tolist()
    print(Age_bin)
    rd_list = df_rd['Rare disease'].tolist()
    no_rd_list = df_rd_no['Without rare disease'].tolist()
    print(no_rd_list)

    y = np.arange(len(no_rd_list))

    fig, axes = plt.subplots(ncols=2, sharey=True, figsize = (7,3))

    # Plotting bars for rare disease
    axes[0].barh(y, rd_list, align='center', color='gray', zorder=10)
    axes[0].set(title='Rare Disease (Hospitalized)')
    axes[0].set_xlim(0, max(rd_list) + 5)  # Adjusted xlim to accommodate text

    # Plotting bars for no rare disease
    axes[1].barh(y, no_rd_list, align='center', color='gray', zorder=10)
    axes[1].set(title='Without rare disease (Hospitalized)')
    axes[1].set_xlim(0, max(rd_list) + 5)  # Adjusted xlim to accommodate text

    # Adding text labels on bars
    for i in range(len(rd_list)):
        axes[0].text(rd_list[i] + 3.95, y[i], f'{rd_list[i]:.2f}%', va='center')
        axes[1].text(no_rd_list[i] + 0.25, y[i], f'{no_rd_list[i]:.2f}%', va='center')

    axes[0].invert_xaxis()
    axes[0].set(yticks=y, yticklabels=Age_bin)
    axes[0].yaxis.tick_right()

    for ax in axes.flat:
        ax.margins(0.02)
        ax.grid(True , linestyle='--') # 

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0)
    plt.show()
#    return df_rd_no

@transform_pandas(
    Output(rid="ri.vector.main.execute.0df4ebf3-0b27-4dbb-8382-245aab9b175e"),
    rd_hospitalized_age_gender=Input(rid="ri.foundry.main.dataset.96da0af3-adfa-4718-a668-0797b1be6b91")
)

import numpy as np
import matplotlib.pyplot as plt

def rd_age_hospitalized_male(rd_hospitalized_age_gender):
    df = rd_hospitalized_age_gender
    df = df[df['Gender'] == 'MALE']
    df = df[['Age_bin','rare_disease','total_patients','total_hospitalized']]
    # Create rare disease hospitalized paercentage column
    df_rd = df[df['rare_disease'] == 1]
    # Create the new column 'rd_hospitalized_percentage'
    df_rd['Rare disease'] = (df_rd['total_hospitalized'] / df_rd['total_patients']) * 100
    df_rd = df_rd.sort_values(by = 'Age_bin',ascending = False)

    # Create with no rare disease hospitalized paercentage column
    df_rd_no = df[df['rare_disease'] == 0]
    # Create the new column 'rd_hospitalized_percentage'
    df_rd_no['Without rare disease'] = (df_rd_no['total_hospitalized'] / df_rd_no['total_patients']) * 100
    df_rd_no = df_rd_no.sort_values(by = 'Age_bin',ascending = False)

    Age_bin = df_rd['Age_bin'].tolist()
    print(Age_bin)
    rd_list = df_rd['Rare disease'].tolist()
    no_rd_list = df_rd_no['Without rare disease'].tolist()
    print(no_rd_list)

    y = np.arange(len(no_rd_list))

    fig, axes = plt.subplots(ncols=2, sharey=True, figsize = (7,3))

    # Plotting bars for rare disease
    axes[0].barh(y, rd_list, align='center', color='gray', zorder=10)
    axes[0].set(title='Rare Disease (Hospitalized)')
    axes[0].set_xlim(0, max(rd_list) + 5)  # Adjusted xlim to accommodate text

    # Plotting bars for no rare disease
    axes[1].barh(y, no_rd_list, align='center', color='gray', zorder=10)
    axes[1].set(title='Without rare disease (Hospitalized)')
    axes[1].set_xlim(0, max(rd_list) + 5)  # Adjusted xlim to accommodate text

    # Adding text labels on bars
    for i in range(len(rd_list)):
        axes[0].text(rd_list[i] + 4.5, y[i], f'{rd_list[i]:.2f}%', va='center')
        axes[1].text(no_rd_list[i] + 0.25, y[i], f'{no_rd_list[i]:.2f}%', va='center')

    axes[0].invert_xaxis()
    axes[0].set(yticks=y, yticklabels=Age_bin)
    axes[0].yaxis.tick_right()

    for ax in axes.flat:
        ax.margins(0.02)
        ax.grid(True , linestyle='--') # 

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0)
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.112b5b67-b284-4670-b54f-61b438f5a668"),
    rd_smr_age_gende=Input(rid="ri.foundry.main.dataset.14b7a539-d6e1-4bcb-b86e-37e5e849afc8")
)
import numpy as np
import matplotlib.pyplot as plt

def rd_age_male_mortality_plot(rd_smr_age_gende):
    df = rd_smr_age_gende
    df = df[df['Gender'] == 'MALE']
    df = df[['Age_bin','rare_disease','total_patients','total_deaths']]
    
    # Create rare disease death percentage column
    df_rd = df[df['rare_disease'] == 1]
    df_rd['Rare disease'] = (df_rd['total_deaths'] / df_rd['total_patients']) * 100
    df_rd = df_rd.sort_values(by='Age_bin', ascending=False)

    # Create with no rare disease death percentage column
    df_rd_no = df[df['rare_disease'] == 0]
    df_rd_no['Without rare disease'] = (df_rd_no['total_deaths'] / df_rd_no['total_patients']) * 100
    df_rd_no = df_rd_no.sort_values(by='Age_bin', ascending=False)

    Age_bin = df_rd['Age_bin'].tolist()
    rd_list = df_rd['Rare disease'].tolist()
    no_rd_list = df_rd_no['Without rare disease'].tolist()

    y = np.arange(len(no_rd_list))

    fig, axes = plt.subplots(ncols=2, sharey=True, figsize = (7,3))
    
    # Plotting bars for rare disease
    axes[0].barh(y, rd_list, align='center', color='gray', zorder=10)
    axes[0].set(title='Rare Disease (Mortality)')
    axes[0].set_xlim(0, max(rd_list) + 5)  # Adjusted xlim to accommodate text

    # Plotting bars for no rare disease
    axes[1].barh(y, no_rd_list, align='center', color='gray', zorder=10)
    axes[1].set(title='Without rare disease (Mortality)')
    axes[1].set_xlim(0, max(rd_list) + 5)  # Adjusted xlim to accommodate text

    # Adding text labels on bars
    for i in range(len(rd_list)):
        axes[0].text(rd_list[i] + 3.95, y[i], f'{rd_list[i]:.2f}%', va='center')
        axes[1].text(no_rd_list[i] + 0.25, y[i], f'{no_rd_list[i]:.2f}%', va='center')

    axes[0].invert_xaxis()
    axes[0].set(yticks=y, yticklabels=Age_bin)
    axes[0].yaxis.tick_right()

    for ax in axes.flat:
        ax.margins(0.02)
        ax.grid(True , linestyle='--') # 

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0)
    plt.show()

"""
import numpy as np
import matplotlib.pyplot as plt

def rd_age_male_mortality_plot(rd_smr_age_gende):
    df = rd_smr_age_gende
    df = df[df['Gender'] == 'MALE']
    df = df[['Age_group','rare_disease','total_patients','total_deaths']]
    # Create rare disease death paercentage column
    df_rd = df[df['rare_disease'] == 1]
    # Create the new column 'rd_death_percentage'
    df_rd['Rare disease'] = (df_rd['total_deaths'] / df_rd['total_patients']) * 100
    df_rd = df_rd.sort_values(by = 'Age_group',ascending = False)

    # Create with no rare disease death paercentage column
    df_rd_no = df[df['rare_disease'] == 0]
    # Create the new column 'rd_death_percentage'
    df_rd_no['Without rare disease'] = (df_rd_no['total_deaths'] / df_rd_no['total_patients']) * 100
    df_rd_no = df_rd_no.sort_values(by = 'Age_group',ascending = False)

    Age_bin = df_rd['Age_group'].tolist()
    print(Age_bin)
    rd_list = df_rd['Rare disease'].tolist()
    no_rd_list = df_rd_no['Without rare disease'].tolist()
    print(no_rd_list)

    y = np.arange(len(no_rd_list))

    fig, axes = plt.subplots(ncols=2, sharey=True)
    axes[0].barh(y, rd_list, align='center', color='gray', zorder=10)
    axes[0].set(title='Rare Disease')
    axes[1].barh(y, no_rd_list, align='center', color='gray', zorder=10)
    axes[1].set(title='Without rare disease')

    axes[0].invert_xaxis()
    axes[0].set(yticks=y, yticklabels=Age_bin)
    axes[0].yaxis.tick_right()

    for ax in axes.flat:
        ax.margins(0.02)
        ax.grid(True, linestyle = '--')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.025)
    plt.tight_layout()
    plt.show()
"""

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.79e2e404-6e61-470d-aa1b-2f56aff3e3a5"),
    covid_rd_cohort_data=Input(rid="ri.foundry.main.dataset.8759c318-64c1-447e-ac27-6d21a2c93ccd")
)

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import expr, when, col, udf
import matplotlib.pyplot as plt
def rd_chisqr_age_gender(covid_rd_cohort_data):
    df = covid_rd_cohort_data.select('Age',"Age_bin",'Gender','death','hospitalized','rare_disease')
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7a8e4e3b-1e3a-4445-bc73-4c2f34f7f44d"),
    cohort_death=Input(rid="ri.foundry.main.dataset.3eca198a-b21a-4336-8cdb-227fbf966dff")
)
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan, when, count, col
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def rd_distribution_plot(cohort_death):
    df = cohort_death.select('features','total_patients')

    # Assuming 'df' is your DataFrame
    filtered_df = df.filter(
        col("features").isin(
            "rare_bone_diseases",
            "rare_cardiac_diseases", 
            "rare_circulatory_system_disease", 
            "rare_developmental_defect_during_embryogenesis",
            "rare_endocrine_disease",
            "rare_gastroenterologic_disease",
            "rare_gynecologic_or_obstetric_disease",
            "rare_hematologic_disease",
            "rare_hepatic_disease",
            "rare_immune_disease",
            "rare_inborn_errors_of_metabolism",
            "rare_infectious_disease",
            "rare_maxillo_facial_surgical_disease",
            "rare_neoplastic_disease",
            "rare_neurologic_disease",
            "rare_ophthalmic_disorder",
            "rare_otorhinolaryngologic_disease",
            "rare_renal_disease",
            "rare_respiratory_disease",
            "rare_skin_disease",
            "rare_surgical_thoracic_disease",
            "rare_systemic_or_rheumatologic_disease",
            "rare_urogenital_disease"
        )
    )

    return filtered_df
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.4fc03c4f-ce70-4319-89d1-0192da0dc4cc"),
    cohort_death=Input(rid="ri.foundry.main.dataset.3eca198a-b21a-4336-8cdb-227fbf966dff")
)

from pyspark.sql import functions as F
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

def rd_hospitalization_distribution_plot(cohort_death):
    df = cohort_death.select('features', 'hosp_percentage')
    # Filter DataFrame to keep only the specified features
    df = df.filter(
        col("features").isin(
            "rare_bone_diseases",
            "rare_cardiac_diseases", 
            "rare_circulatory_system_disease", 
            "rare_developmental_defect_during_embryogenesis",
            "rare_endocrine_disease",
            "rare_gastroenterologic_disease",
            "rare_gynecologic_or_obstetric_disease",
            "rare_hematologic_disease",
            "rare_hepatic_disease",
            "rare_immune_disease",
            "rare_inborn_errors_of_metabolism",
            "rare_infectious_disease",
            "rare_maxillo_facial_surgical_disease",
            "rare_neoplastic_disease",
            "rare_neurologic_disease",
            "rare_ophthalmic_disorder",
            "rare_otorhinolaryngologic_disease",
            "rare_renal_disease",
            "rare_respiratory_disease",
            "rare_skin_disease",
            "rare_surgical_thoracic_disease",
            "rare_systemic_or_rheumatologic_disease",
            "rare_urogenital_disease"
        )
    ).toPandas()
    
    # Sort and select top 23 rows
    df_sorted = df.sort_values(by='hosp_percentage', ascending=False)
    ranking1 = df_sorted.head(23)
    ranking = ranking1.sort_values(by='hosp_percentage', ascending=True)

    # Prepare data for plotting
    index1 = ranking['features'].tolist()
    index = [x.replace("_", " ").capitalize() for x in index1]
    values = ranking['hosp_percentage']

    plot_title = '' #'Rare Diseases Class'
    title_size = 20
    x_label = 'Hospitalization (%)'
    filename = 'barh-plot'
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(9, 11), dpi=600)

    # Create horizontal bar plot with black bars
    bar = ax.barh(index, values, color='black', align='center')

    # Set x-axis ticks to integer values
    max_value = int(np.ceil(max(values)))
    ax.set_xticks(range(0, max_value + 5, 5))  # Adjust range if needed

    # Format x-axis labels as integers
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Remove the box (frame) around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Place a label for each bar with percentage format
    for rect in bar:
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        ax.annotate(f"{x_value:.1f}%", (x_value, y_value), xytext=(5, 0),
                    textcoords="offset points", ha='left', va='center', fontsize=16, color='black')

    # Set labels and title
    ax.set_xlabel(x_label, fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(plot_title, fontsize=title_size, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.96da0af3-adfa-4718-a668-0797b1be6b91"),
    covid_rd_cohort_data=Input(rid="ri.foundry.main.dataset.8759c318-64c1-447e-ac27-6d21a2c93ccd")
)

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import expr, when, col, udf
import matplotlib.pyplot as plt
def rd_hospitalized_age_gender(covid_rd_cohort_data):
    df = covid_rd_cohort_data.select('Age_bin','Gender','hospitalized','rare_disease')

    # Calculate the total COVID-positive patients and deaths for each age range, gender, and rare disease status 
    total_patients = df.groupBy("Age_bin", "Gender", "rare_disease").count().withColumnRenamed("count", "total_patients")
    total_hospitalized = df.filter(df.hospitalized == 1).groupBy("Age_bin", "Gender", "rare_disease").count().withColumnRenamed("count", "total_hospitalized")

    # Join the total patients and total hospitalized DataFrames 
    joined_df = total_patients.join(total_hospitalized, ["Age_bin", "Gender", "rare_disease"], "left_outer")

  
    return joined_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.693ea4f6-7e77-4677-95c3-de2bed07596e"),
    cohort_death=Input(rid="ri.foundry.main.dataset.3eca198a-b21a-4336-8cdb-227fbf966dff")
)
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

def rd_mortality_distribution_plot(cohort_death):
    df = cohort_death.select('features', 'death_percentage')
    # Filter DataFrame to keep only the specified features
    df = df.filter(
        col("features").isin(
            "rare_bone_diseases",
            "rare_cardiac_diseases", 
            "rare_circulatory_system_disease", 
            "rare_developmental_defect_during_embryogenesis",
            "rare_endocrine_disease",
            "rare_gastroenterologic_disease",
            "rare_gynecologic_or_obstetric_disease",
            "rare_hematologic_disease",
            "rare_hepatic_disease",
            "rare_immune_disease",
            "rare_inborn_errors_of_metabolism",
            "rare_infectious_disease",
            "rare_maxillo_facial_surgical_disease",
            "rare_neoplastic_disease",
            "rare_neurologic_disease",
            "rare_ophthalmic_disorder",
            "rare_otorhinolaryngologic_disease",
            "rare_renal_disease",
            "rare_respiratory_disease",
            "rare_skin_disease",
            "rare_surgical_thoracic_disease",
            "rare_systemic_or_rheumatologic_disease",
            "rare_urogenital_disease"
        )
    ).toPandas()
    
    # Sort and select top 23 rows
    df_sorted = df.sort_values(by='death_percentage', ascending=False)
    ranking1 = df_sorted.head(23)
    ranking = ranking1.sort_values(by='death_percentage', ascending=True)

    # Prepare data for plotting
    index1 = ranking['features'].tolist()
    index = [x.replace("_", " ").capitalize() for x in index1]
    values = ranking['death_percentage']

    plot_title = '' #'Rare Diseases Class'
    title_size = 20
    x_label = 'Mortality (%)'
    filename = 'barh-plot'
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(9, 11), dpi=600)

    # Create horizontal bar plot with black bars
    bar = ax.barh(index, values, color='black', align='center')

    # Set x-axis ticks to integer values
    max_value = int(np.ceil(max(values)))
    ax.set_xticks(range(0, max_value + 5, 5))  # Adjust range if needed

    # Format x-axis labels as integers
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Remove the box (frame) around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Place a label for each bar with percentage format
    for rect in bar:
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        ax.annotate(f"{x_value:.1f}%", (x_value, y_value), xytext=(5, 0),
                    textcoords="offset points", ha='left', va='center', fontsize=16, color='black')

    # Set labels and title
    ax.set_xlabel(x_label, fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(plot_title, fontsize=title_size, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return df

"""
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

def rd_mortality_distribution_plot(cohort_death):
    df = cohort_death.select('features', 'death_percentage')
    # Filter DataFrame to keep only the specified features
    df = df.filter(
        col("features").isin(
            "rare_neoplastic_disease",
            "rare_neurologic_disease",
            "rare_ophthalmic_disorder",
            "rare_otorhinolaryngologic_disease",
            "rare_gastroenterologic_disease",
            "rare_hematologic_disease",
            "rare_hepatic_disease",
            "rare_immune_disease",
            "rare_developmental_defect_during_embryogenesis",
            "rare_endocrine_disease",
            "rare_skin_disease",
            "rare_systemic_or_rheumatologic_disease",
            "rare_transplantation_disease",
            "rare_inborn_errors_of_metabolism",
            "rare_infectious_disease",
            "rare_bone_diseases",
            "rare_cardiac_diseases",
            "rare_renal_disease",
            "rare_respiratory_disease"
        )
    ).toPandas()
    
    # Sort and select top 19 rows
    df_sorted = df.sort_values(by='death_percentage', ascending=False)
    ranking1 = df_sorted.head(19)
    ranking = ranking1.sort_values(by='death_percentage', ascending=True)

    # Prepare data for plotting
    index1 = ranking['features'].tolist()
    index = [x.replace("_", " ").capitalize() for x in index1]
    values = ranking['death_percentage']

    plot_title = '' #'Rare Diseases Class'
    title_size = 20
    x_label = '% of Patients'
    filename = 'barh-plot'
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(9, 11), dpi=600)

    # Create horizontal bar plot with black bars
    bar = ax.barh(index, values, color='black', align='center')

    # Format the x-axis to display percentages with one decimal place
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f%%'))
    
    # Remove the log scale from the x-axis
    # log scale argument removed

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Remove the box (frame) around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Place a label for each bar
    for rect in bar:
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        ax.annotate(f"{x_value:.1f}%", (x_value, y_value), xytext=(5, 0),
                    textcoords="offset points", ha='left', va='center', fontsize=12, color='black')

    # Set labels and title
    ax.set_xlabel(x_label, fontsize=18, fontweight='bold')
    # Removed y-axis label
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(plot_title, fontsize=title_size, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return df
"""

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.14b7a539-d6e1-4bcb-b86e-37e5e849afc8"),
    covid_rd_cohort_data=Input(rid="ri.foundry.main.dataset.8759c318-64c1-447e-ac27-6d21a2c93ccd")
)
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import expr, when, col, udf
import matplotlib.pyplot as plt
def rd_smr_age_gende(covid_rd_cohort_data):
    df = covid_rd_cohort_data.select('Age_bin','Age','Gender','death','rare_disease')

    # Calculate the total COVID-positive patients and deaths for each age range, gender, and rare disease status 
    total_patients = df.groupBy("Age_bin", "Gender", "rare_disease").count().withColumnRenamed("count", "total_patients")
    total_deaths = df.filter(df.death == 1).groupBy("Age_bin", "Gender", "rare_disease").count().withColumnRenamed("count", "total_deaths")

    # Join the total patients and total deaths DataFrames 
    joined_df = total_patients.join(total_deaths, ["Age_bin", "Gender", "rare_disease"], "left_outer")
  
    return joined_df

"""
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import expr, when, col, udf
import matplotlib.pyplot as plt
def rd_smr_age_gende(covid_rd_cohort_data):
    df = covid_rd_cohort_data.select('Age','Gender','death','rare_disease')
    # Bin the Age variable into specified age ranges 
    def age_binning(age):
        if age <=10: return ' 2-10'
        elif age <= 20: return '11-20'
        elif age <= 30: return '21-30'
        elif age <= 40: return '31-40'
        elif age <= 50: return '41-50'
        elif age <= 60: return '51-60'
        elif age <= 70: return '61-70'
        else: return '70+'
#    def age_binning(age):
#        if age <=20: return '1-20'
#        elif age <= 40: return '21-40'
#        elif age <= 65: return '41-65'
#        else: return '65+'
    # Register UDF
    age_binning_udf = udf(age_binning)

    # Apply age binning
    df = df.withColumn('Age_group', age_binning_udf(col('Age')))

    # Calculate the total COVID-positive patients and deaths for each age range, gender, and rare disease status 
    total_patients = df.groupBy("Age_group", "Gender", "rare_disease").count().withColumnRenamed("count", "total_patients")
    total_deaths = df.filter(df.death == 1).groupBy("Age_group", "Gender", "rare_disease").count().withColumnRenamed("count", "total_deaths")

    # Join the total patients and total deaths DataFrames 
    joined_df = total_patients.join(total_deaths, ["Age_group", "Gender", "rare_disease"], "left_outer")
  
    return joined_df

"""

@transform_pandas(
    Output(rid="ri.vector.main.execute.240d4328-25bc-4c89-bb22-d4d6621d56d7"),
    rd_distribution_plot=Input(rid="ri.foundry.main.dataset.7a8e4e3b-1e3a-4445-bc73-4c2f34f7f44d")
)
'''
Distribution plot of rare disease classess
'''
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

def top_23_rd_class_fig(rd_distribution_plot):
    df = rd_distribution_plot.toPandas()
    df_sorted = df.sort_values(by='total_patients', ascending=False)
    ranking1 = df_sorted.head(23)
    ranking = ranking1.sort_values(by='total_patients', ascending=True)

    index1 = ranking['features'].tolist()
    index = [x.replace("_", " ").capitalize() for x in index1]
    values = ranking['total_patients']

    plot_title = 'Rare diseases class'
    title_size = 20
    x_label = '# of Patients'
    filename = 'barh-plot'
    set_output_image_type("svg")
    fig, ax = plt.subplots(figsize=(9, 11), dpi=600)  # 6,6

    # Create horizontal bar plot with black bars
    bar = ax.barh(index, values, color='black', log=True, align='center')

    plt.tight_layout()
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Remove the box (frame) around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    rects = ax.patches
    # Place a label for each bar
    for rect in rects:
        # Get X and Y placement of label from rect
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        ax.annotate(f"{x_value:.0f}", (x_value, y_value), xytext=(5, 0),
                    textcoords="offset points", ha='left', fontsize=14, va='center')

    # Set labels and title
    ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
    # Removed y-axis label
    # ax.set_ylabel("Rare Diseases Class", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()

    return df

"""
######
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

def top_19_rd_class_fig(rd_distribution_plot):
    df = rd_distribution_plot.toPandas()
    df_sorted = df.sort_values(by='total_patients', ascending=False)
    ranking1 = df_sorted.head(19)
    ranking = ranking1.sort_values(by='total_patients', ascending=True)

    index1 = ranking['features'].tolist()
    index = [x.replace("_", " ").capitalize() for x in index1]
    values = ranking['total_patients']

    plot_title = 'Rare diseases class'
    title_size = 20
    subtitle = ''  # 'Patients with Rare Disease then diagnosis with COVID-19'
    x_label = '# of Patients'
    filename = 'barh-plot'
    set_output_image_type("svg")
    fig, ax = plt.subplots(figsize=(9, 11), dpi=600)  # 6,6
    mpl.pyplot.viridis()

    bar = ax.barh(index, values, log=True, align='center')  # True
    plt.tight_layout()
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.subplots_adjust(top=0.9, bottom=0.1)

    def custom_gradientbars(bars):
        grad = np.atleast_2d(np.linspace(0, 1, 256))
        ax = bars[0].axes
        lim = ax.get_xlim() + ax.get_ylim()
        for bar in bars:
            bar.set_zorder(1)
            bar.set_facecolor('none')
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            ax.imshow(grad, extent=[x + w, x, y, y + h], aspect='auto', zorder=1)
        ax.axis(lim)

    custom_gradientbars(bar)

    # Remove the box (frame) around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    rects = ax.patches
    # Place a label for each bar
    for rect in rects:
        # Get X and Y placement of label from rect
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        ax.annotate(f"{x_value:.0f}", (x_value, y_value), xytext=(5, 0),
                    textcoords="offset points", ha='left', va='center')

#    ax.set_title(plot_title, fontsize=title_size, fontweight = 'bold')
    ax.set_xlabel(x_label, fontsize= 18, fontweight = 'bold')
    ax.set_ylabel("Rare Diseases Class", fontsize= 18, fontweight = 'bold')
    plt.xticks(fontsize = 15) # fontweight='bold'
    plt.yticks(fontsize = 15)
    plt.tight_layout()
    plt.show()
    return df
######

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

print('numpy version is:',np.__version__)
print('pandas version is:',pd.__version__)
print('matplotlib version is:', mpl.__version__)
print('Seaborn version is:', sns.__version__)

def top_19_rd_class_fig(rd_distribution_plot):
    df = rd_distribution_plot.toPandas()
    df_sorted = df.sort_values(by = 'total_patients',ascending = False)
    ranking1 = df_sorted.head(19)
    ranking = ranking1.sort_values(by = 'total_patients',ascending = True)

    index1 = ranking['features'].tolist()
    index = [x.replace("_", " ").capitalize() for x in index1]
    values = ranking['total_patients']
  
    plot_title = 'Rare diseases class'
    title_size = 18
    subtitle = '' #'Patients with Rare Disease then diagnosis with COVID-19'
    x_label = '# of Patients'
    filename = 'barh-plot'
    set_output_image_type("svg")
    fig, ax = plt.subplots(figsize=(9,11),  dpi = 600) # 6,6
    mpl.pyplot.viridis()
   
    bar = ax.barh(index, values, log = True,  align = 'center') # True
    plt.tight_layout()
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.subplots_adjust(top=0.9, bottom=0.1)
    def gradientbars(bars):
        grad = np.atleast_2d(np.linspace(0,1,256))
        ax = bars[0].axes
        lim = ax.get_xlim()+ax.get_ylim()
        for bar in bars:
            bar.set_zorder(1)
            bar.set_facecolor('none')
            x,y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            ax.imshow(grad, extent=[x+w, x, y, y+h], aspect='auto', zorder=1)
        ax.axis(lim)
    gradientbars(bar)

    rects = ax.patches
    # Place a label for each bar
    for rect in rects:
        # Get X and Y placement of label from rect
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label; change to your liking
        space = 2
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: place label to the left of the bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label to the right
            ha = 'right'

        # Use X value as label and format number
        label = '{:,.0f}'.format(x_value)

        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at bar end
            xytext=(space, 0),          # Horizontally shift label by `space`
            textcoords='offset points', # Interpret `xytext` as offset in points
            va='center',                # Vertically center label
            ha=ha,                      # Horizontally align label differently for positive and negative values
            color = 'blue',fontsize = 11) #13

    #    # Set subtitle
        tfrom = ax.get_xaxis_transform()
        ann = ax.annotate(subtitle, xy=(5, 1), xycoords=tfrom, bbox=dict(boxstyle='square,pad=1.3', fc='#f0f0f0', ec='none'))
        
        #Set x-label
        ax.set_xlabel(x_label, color='k',fontweight='bold')   
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xticks(fontsize = 12) # fontweight='bold'
        plt.yticks(fontsize = 12)
        plt.style.use('classic')
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)
        #plt.xlim(500, 800000)
    plt.tight_layout()
    plt.show()
    
    return df_sorted
"""

@transform_pandas(
    Output(rid="ri.vector.main.execute.a7275f52-4372-4cfe-9575-8931eef84010"),
    rd_distribution_plot=Input(rid="ri.foundry.main.dataset.7a8e4e3b-1e3a-4445-bc73-4c2f34f7f44d")
)
import matplotlib.pyplot as plt
import matplotlib as mpl

def top_5_rd_class_fig(rd_distribution_plot):
    df = rd_distribution_plot.toPandas()
    df_sorted = df.sort_values(by='total_patients', ascending=False)
    ranking1 = df_sorted.head(5)
    ranking = ranking1.sort_values(by='total_patients', ascending=True)

    index1 = ranking['features'].tolist()
    index = [x.replace("_", " ").capitalize() for x in index1]
    values = ranking['total_patients']

    plot_title = 'Rare diseases class'
    title_size = 20
    x_label = '# of Patients'
    filename = 'barh-plot'
    set_output_image_type("svg")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=600)  # 6,6

    # Create horizontal bar plot with black bars without log scale
    bar = ax.barh(index, values, color='black', align='center')

    plt.tight_layout()

    # Set formatter to prevent scientific notation for the x-axis
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Explicitly set the x-ticks (0, 50,000, 100,000)
    ax.set_xticks([0, 50000, 100000])  # Set x-tick positions
    ax.set_xticklabels(['0', '50,000', '100,000'])  # Set x-tick labels

    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Remove the box (frame) around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    rects = ax.patches
    # Place a label for each bar
    for rect in rects:
        # Get X and Y placement of label from rect
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        ax.annotate(f"{x_value:.0f}", (x_value, y_value), xytext=(5, 0),
                    textcoords="offset points", ha='left', va='center', fontsize=16)

    # Set labels and title
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')

    # Removed y-axis label
    ax.set_ylabel(" ", fontsize=18, fontweight='bold')  # Rare Diseases Class
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

    return df

"""
import matplotlib.pyplot as plt
import matplotlib as mpl

def top_5_rd_class_fig(rd_distribution_plot):
    df = rd_distribution_plot.toPandas()
    df_sorted = df.sort_values(by='total_patients', ascending=False)
    ranking1 = df_sorted.head(5)
    ranking = ranking1.sort_values(by='total_patients', ascending=True)

    index1 = ranking['features'].tolist()
    index = [x.replace("_", " ").capitalize() for x in index1]
    values = ranking['total_patients']

    plot_title = 'Rare diseases class'
    title_size = 20
    x_label = '# of Patients'
    filename = 'barh-plot'
    set_output_image_type("svg")
    fig, ax = plt.subplots(figsize=(9, 4), dpi=600)  # 6,6

    # Create horizontal bar plot with black bars
    bar = ax.barh(index, values, color='black', log=True, align='center')

    plt.tight_layout()
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Remove the box (frame) around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    rects = ax.patches
    # Place a label for each bar
    for rect in rects:
        # Get X and Y placement of label from rect
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        ax.annotate(f"{x_value:.0f}", (x_value, y_value), xytext=(5, 0),
                    textcoords="offset points", ha='left', va='center', fontsize=16)

    # Set labels and title
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    

    # Removed y-axis label
    ax.set_ylabel(" ", fontsize=18, fontweight='bold') #Rare Diseases Class
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

    return df
"""

@transform_pandas(
    Output(rid="ri.vector.main.execute.19290d34-d521-427b-a149-495075c99d7d"),
    covid_rd_cohort_data=Input(rid="ri.foundry.main.dataset.8759c318-64c1-447e-ac27-6d21a2c93ccd")
)
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from patsy import dmatrices
import statsmodels.api as sm
import patsy

from pyspark.sql import functions as F

def univariate_death_rd(covid_rd_cohort_data):
    df = covid_rd_cohort_data.select("rare_disease","death","hospitalized","age_1_20","age_20_40","age_40_65","age_65_above","BMI_obese","BMI_over_weight","BMI_normal","BMI_under_weight","gender_MALE","gender_FEMALE","race_Black_or_African_American","race_Unknown","race_White","race_Asian","ethnicity_Not_Hispanic_or_Latino","ethnicity_Hispanic_or_Latino","ethnicity_Unknown","smoking_status_Non_smoker","smoking_status_Current_or_Former","vaccination_status").toPandas()
    df_rd = df[df["rare_disease"] == 1]

    print('Mortality patients_yes:', df_rd['death'].sum())
    print('Mortality patients_no:', df_rd.shape[0]-df_rd['death'].sum())
    print('sum of features:', df_rd.shape[0])

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['rds', 'coefficient', 'odds_ratio', 'CI_lower', 'CI_upper', 'pvalue', 'N'])
    # Loop through each rare disease
    for rd_col in ["age_1_20","age_20_40","age_40_65","age_65_above","BMI_obese","BMI_over_weight","BMI_normal","BMI_under_weight","gender_MALE","gender_FEMALE","race_Black_or_African_American","race_Unknown","race_White","race_Asian","ethnicity_Not_Hispanic_or_Latino","ethnicity_Hispanic_or_Latino","ethnicity_Unknown","smoking_status_Non_smoker","smoking_status_Current_or_Former"]:

        X = df_rd[rd_col]

        X = sm.add_constant(X)  # Add an intercept term
        y = df_rd["death"] #"death", "hospitalized",

    #    y, X = dmatrices( 'death ~ rd_col', data=df_rd, return_type='dataframe')

        # Fit logistic regression model
        model = sm.Logit(y, X).fit()

        print (model.summary()) 

        # Extract relevant coefficients and confidence intervals
        coef = model.params[rd_col]
        odds_ratio = np.exp(coef)
        ci_lower, ci_upper = np.exp(model.conf_int().loc[rd_col])
        pvalue = model.pvalues[rd_col]
        N = len(X)
       
        # Append results to the DataFrame
        results_df = results_df.append({
            'rds': rd_col,
            'coefficient': round(coef,2),
            'odds_ratio': round(odds_ratio,2),
            'CI_lower': round(ci_lower,2),
            'CI_upper': round(ci_upper,2),
            'pvalue': pvalue,
            'N': N
        }, ignore_index=True)
        results_df['odds_ratio_CI'] = results_df.apply(lambda row: f"{row['odds_ratio']} ({row['CI_lower']} - {row['CI_upper']})", axis = 1)

    return results_df
   

"""

def RD_univariate_model_lt(rd_model_data):
    df = rd_model_data.select("life_thretening","hospitalized","rare_bone_diseases","rare_cardiac_diseases","rare_developmental_defect_during_embryogenesis","rare_disorder_due_to_toxic_effects","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_systemic_or_rheumatologic_disease","rare_transplantation_disease","other_rare_diseases").toPandas()
    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['rds', 'coefficient', 'odds_ratio', 'CI_lower', 'CI_upper', 'pvalue', 'N'])
    # Loop through each rare disease
    for rd_col in ["rare_bone_diseases","rare_cardiac_diseases","rare_developmental_defect_during_embryogenesis","rare_disorder_due_to_toxic_effects","rare_endocrine_disease","rare_gastroenterologic_disease","rare_gynecologic_or_obstetric_disease","rare_hematologic_disease","rare_hepatic_disease","rare_immune_disease","rare_inborn_errors_of_metabolism","rare_infectious_disease","rare_neoplastic_disease","rare_neurologic_disease","rare_ophthalmic_disorder","rare_otorhinolaryngologic_disease","rare_renal_disease","rare_respiratory_disease","rare_skin_disease","rare_systemic_or_rheumatologic_disease","rare_transplantation_disease","other_rare_diseases"]:
        X = df[rd_col]
        X = sm.add_constant(X)  # Add an intercept term
        y = df['life_thretening']

        # Fit logistic regression model
        model = sm.Logit(y, X).fit()

        # Extract relevant coefficients and confidence intervals
        coef = model.params[rd_col]
        odds_ratio = np.exp(coef)
        ci_lower, ci_upper = np.exp(model.conf_int().loc[rd_col])
        pvalue = model.pvalues[rd_col]
        N = len(df)

        # Append results to the DataFrame
        results_df = results_df.append({
            'rds': rd_col,
            'coefficient': round(coef,2),
            'odds_ratio': round(odds_ratio,2),
            'CI_lower': round(ci_lower,2),
            'CI_upper': round(ci_upper,2),
            'pvalue': pvalue,
            'N': N
        }, ignore_index=True)
        results_df['odds_ratio_CI'] = results_df.apply(lambda row: f"{row['odds_ratio']} ({row['CI_lower']} - {row['CI_upper']})", axis = 1)

    return results_df
"""

@transform_pandas(
    Output(rid="ri.vector.main.execute.11f11f84-6992-4a8a-86c4-e80b65127277"),
    covid_rd_cohort_data=Input(rid="ri.foundry.main.dataset.8759c318-64c1-447e-ac27-6d21a2c93ccd")
)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from patsy import dmatrices
import statsmodels.api as sm
import patsy

print('numpy version is:',np.__version__)
print('pandas version is:',pd.__version__)
print('statsmodel version is:',sm.__version__)
print('patsy version is:', patsy.__version__)
print('PySpark version is:', spark.version)

from pyspark.sql import functions as F

def univariate_hospitalization_rd(covid_rd_cohort_data):
    df = covid_rd_cohort_data.select("rare_disease","death","hospitalized","age_1_20","age_20_40","age_40_65","age_65_above","BMI_obese","BMI_over_weight","BMI_normal","BMI_under_weight","gender_MALE","gender_FEMALE","race_Black_or_African_American","race_Unknown","race_White","race_Asian","ethnicity_Not_Hispanic_or_Latino","ethnicity_Hispanic_or_Latino","ethnicity_Unknown","smoking_status_Non_smoker","smoking_status_Current_or_Former","vaccination_status").toPandas()
    df_rd = df[df["rare_disease"] == 0]

    print('hospitalized patients_yes:', df_rd['hospitalized'].sum())
    print('hospitalized patients_no:', df_rd.shape[0]-df['hospitalized'].sum())
    print('total size:', df_rd.shape[0])

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['rds', 'coefficient', 'odds_ratio', 'CI_lower', 'CI_upper', 'pvalue', 'N'])
    # Loop through each rare disease
    for rd_col in ["age_1_20","age_20_40","age_40_65","age_65_above","BMI_obese","BMI_over_weight","BMI_normal","BMI_under_weight","gender_MALE","gender_FEMALE","race_Black_or_African_American","race_Unknown","race_White","race_Asian","ethnicity_Not_Hispanic_or_Latino","ethnicity_Hispanic_or_Latino","ethnicity_Unknown","smoking_status_Non_smoker","smoking_status_Current_or_Former"]:

        X = df_rd[rd_col]

        X = sm.add_constant(X)  # Add an intercept term
        y = df_rd["hospitalized"] #"death", "hospitalized",

    #    y, X = dmatrices( 'death ~ rd_col', data=df_rd, return_type='dataframe')

        # Fit logistic regression model
        model = sm.Logit(y, X).fit()

        print (model.summary()) 

        # Extract relevant coefficients and confidence intervals
        coef = model.params[rd_col]
        odds_ratio = np.exp(coef)
        ci_lower, ci_upper = np.exp(model.conf_int().loc[rd_col])
        pvalue = model.pvalues[rd_col]
        N = len(X)
       
        # Append results to the DataFrame
        results_df = results_df.append({
            'rds': rd_col,
            'coefficient': round(coef,2),
            'odds_ratio': round(odds_ratio,2),
            'CI_lower': round(ci_lower,2),
            'CI_upper': round(ci_upper,2),
            'pvalue': pvalue,
            'N': N
        }, ignore_index=True)
        results_df['odds_ratio_CI'] = results_df.apply(lambda row: f"{row['odds_ratio']} ({row['CI_lower']} - {row['CI_upper']})", axis = 1)

    return results_df




@transform_pandas(
    Output(rid="ri.foundry.main.dataset.73c37aad-8f95-4bc3-8b30-0203f8b94ac6"),
    covid_rd_cohort_data=Input(rid="ri.foundry.main.dataset.8759c318-64c1-447e-ac27-6d21a2c93ccd")
)
-- ,long_covid
SELECT rare_disease,death,hospitalized,Age_bin,BMI_category,Gender,Race,Ethnicity,smoking_status,Cancer
FROM covid_rd_cohort_data

/*,Cardiomyophaties,Cerebro_vascular_disease,Chronic_lung_disease,Coronary_artery_disease,Dementia_before,Depression,Diabetes,Heart_failure,HIV_infection,HTN,Kidney_disease,liver_disease,Myocardial_infraction,Peripheral_vascular_disease,Rheumatologic_disease,Systemic_corticosteroids,vaccination_status,COVID_reinfection,rare_disease,rare_bone_diseases,rare_cardiac_diseases,rare_developmental_defect_during_embryogenesis,
rare_endocrine_disease,rare_gastroenterologic_disease,rare_gynecologic_or_obstetric_disease,
rare_hematologic_disease,rare_hepatic_disease,rare_immune_disease,rare_inborn_errors_of_metabolism,
rare_infectious_disease,rare_neoplastic_disease,rare_neurologic_disease,rare_ophthalmic_disorder,
rare_otorhinolaryngologic_disease,rare_renal_disease,rare_respiratory_disease,rare_skin_disease,
rare_systemic_or_rheumatologic_disease,rare_transplantation_disease,other_rare_diseases
*/


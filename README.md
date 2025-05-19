## Systematic identification of rare disease patients in electronic health records enables evaluation of clinical outcomes


Full Citation:
Arjun S. Yadaw, Eric Sid, Hythem Sidky,Chenjie Zeng, Qian Zhu, Ewy Mathé."Systematic identification of rare disease patients in electronic health records enables evaluation of clinical outcomes". (MedRxiv doi: https://doi.org/10.1101/2025.05.02.25325348).

## Project Summary: 
There are over 10,000 rare diseases. In fact, rare diseases are estimated to affect more than 30 million people in the United States and more than 300 million people across the world. In the United States, a rare disease is one that fewer than 200,000 people live with. (In other words, 6 per 10,000 individuals.). Each disease may be rare individually, but people with rare diseases often face similar challenges and there are more than 30 million people affected by rare diseases in USA. These challenges may include accessing information, getting a diagnosis, and finding resources (https://rarediseases.info.nih.gov/about). 

Mapping rare diseases to EHR data is challenging because large number of rare disease are not codded or mapped to group of disorders or phenotype. In this project, we aim to create rare disease mapping to EHR system by using ICD-10 codes & SNOMED-CT codes. We constructed a cohort of COVID-19 infected individuals that had a rare disease (RDs) diagnosis prior to COVID-19 infection, as defined in our workflow figure 2.  To accomplish this, we created a curated list of unique rare diseases, defined by 12,003 GARD IDs(Genetic and Rare Diseases (GARD) Information Center; https://rarediseases.info.nih.gov/about).  These GARD_IDs were mapped with ORPHANET (July 2023 version) which mapped 9,369 rare diseases.  Of the 9,369 GARD IDs mapping to ORPHANET, 2,724 did not map to either a SNOMED_CT or ICD-10 codes and were discarded.  The remaining diseases were represented by 6,555 SNOMED-CT and 575 ICD-10 codes.  We note that ICD-10 codes represent single diseases with no descendent whereas few SNOMED-CT codes had descendent concept ids. We dropped groups of disorder and phenotype diseases from these codes and finally, there were 12,081 SNOMED-CT codes along with descendent ids, 350 ICD-10 codes.
## Purpose of this code: 
This code is designed to identify rare disease patients in electronic health record (EHR) system. We have taken semi automated approach to map rare disease patients and removing false positive patients by dropping groups of disorder, phenotype filtering and incidence filter.

## Python Libraries Used:
The following Python version and packages are required to execute this in palantir foundary:

* Python (3.10.16)
* numpy (1.26.4)
* pandas (1.5.3)
* statsmodel (0.14.4)
* patsy (1.0.1)
* matplotlib (3.9.4)
* Seaborn (0.13.2)
* Spark SQL (3.4.1.34)

## R Libraries Used:
* gtsummary (2.1.0)
* tidyverse (2.0.0)
* gt (0.11.1)
* ggplot2 (3.5.1)

## Prerequisites: 
In order to run this code(reproduce results), you will need following:

* Electronic health record data table (version 154) in the OMOP data model
* Rare disease patients cohort based on SNOMED-CT and ICD-10 codes
* Rare disease linearization classification based on ORPHANET Linearization
  
## Code for rare diseases identification & classification (Folders)
* Rare disease specific ICD-10 codes 
* Rare disease specific SNOMED-CT codes
* Rare disease classification based on ORPHANET Linearization
* Rare disease misceleneous folder

## Rare disease cohort building at N3C enclave and analysis
* Rare disease cohort 
* Rare disease analysis codes
* Cohort of COVID-19 positive patients

## Running our model:
To produce the results, One has create rare disease specific ICD-10 and SNOMED-CT codes for different rare disease classes by using (Rare disease classification based on ORPHANET Linearization) then use this code to N3C enclave to create rare disease cohort. Also create cohorts of cohort of COVID-19 positive patients (Cohort of COVID-19 positive patients) then run RD_model (Rare disease analysis codes) by using these input tables.
  

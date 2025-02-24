# Systematic identification of rare disease patients in electronic health records enables evaluation of COVID-19 or other clinical outcomes


Full Citation:
Arjun S. Yadaw, Qian Zhu, Hythem Sidky, Eric Sid, Ewy Mathé."Systematic identification of rare disease patients in electronic health records enables evaluation of      COVID-19 or other clinical outcomes". (Journal details etc).

## Project Summary: 
There are over 10,000 rare diseases. In fact, rare diseases are estimated to affect more than 30 million people in the United States and more than 300 million people across the world. In the United States, a rare disease is one that fewer than 200,000 people live with. (In other words, 6 per 10,000 individuals.).Each disease may be rare individually, but people with rare diseases often face similar challenges. These challenges may include accessing information, getting a diagnosis, and finding resources (https://rarediseases.info.nih.gov/about). 

Mapping rare diseases to EHR data is challenging because large number of rare disease are not codded or mapped to group of disorders or phenotype. In this project, we aim to create rare disease mapping to EHR system by using ICD-10 codes & SNOMED-CT codes. We constructed a cohort of COVID-19 infected individuals that had a rare disease (RDs) diagnosis prior to COVID-19 infection, as defined in our workflow figure 2.  To accomplish this, we created a curated list of unique rare diseases, defined by 12,004 GARD IDs(Genetic and Rare Diseases (GARD) Information Center; https://rarediseases.info.nih.gov/about).  These GARD_IDs were mapped with ORPHANET (July 2023 version) which mapped 9,369 rare diseases.  Of the 9,369 GARD IDs mapping to ORPHANET, 2,725 did not map to either a SNOMED_CT or ICD10 code and were discarded.  The remaining diseases were represented by 6,555 SNOMED-CT and 575 ICD10 codes.  We note that ICD10 codes represent single diseases with no descendent whereas few SNOMED-CT codes had descendent concept ids. We dropped groups of disorder and phenotype diseases from these codes and finally, there were 10,596 SNOMED-CT codes along with descendent ids, 351 ICD-10 codes.
## Purpose of this code: 
This code is designed to identify rare disease patients in electronic health record (EHR) system. We have taken semi automated approach to map rare disease patients and removing false positive patients by dropping groups of disorder, phenotype filtering and incidence filter.
## Prerequisites: 
In order to run this code(reproduce results), you will need following:

* Electronic health record data table (version 154) in the OMOP data model
* Rare disease patients cohort based on SNOMED-CT and ICD-10 codes
* Rare disease linearization classification based on ORPHANET

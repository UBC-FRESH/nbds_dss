# Decision Support System (DSS) Description

## Overview

This Decision Support System (DSS) facilitates carbon management analysis for specific case studies using Jupyter notebooks. It organizes inputs, processes data, and outputs results in various formats to evaluate scenarios related to harvested wood products and displacement effects.

## Folder Structure

### 1. **Data Folder**

This folder contains all the necessary data for the DSS.  
- Shapefiles are placed in the corresponding large data folder for each case study.  
- Each case study has its dedicated folder containing the respective Woodstock files.
- The curves folder contains the generated carbon curves for each case study.  


### 2. **Outputs Folder**

Results are saved in this folder in the form of figures, CSV files, and pickle files.


## Main Scripts

The DSS is structured with three main types of scripts:

### 1. **Generate Woodstock Files**

- For each case study, a specific notebook is created to generate Woodstock files.
- While the main structure is consistent across cases, modifications are made as needed for each case.  
- Scripts in this category are named starting with `00`.

### 2. **Generate Carbon Curves**

- The main script for generating carbon curves is named `01_dss_build_compare_carbon_curve`.  

### 3. **Build and Run Scenarios**

- The main script for running scenarios is named `02_dss_build_scenarios`.  






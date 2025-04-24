# Decision Support System (DSS) Description

## Overview

This Decision Support System (DSS) facilitates carbon management analysis for specific case studies using Jupyter notebooks. It organizes inputs, processes data, and outputs results in various formats to evaluate scenarios related to harvested wood products and displacement effects.

## Folder Structure

### 1. **Data Folder**

This folder contains all the necessary data for the DSS.  
- **Note**: Shape files for the *Equity Silver* and *Red Chris* cases are not included in the GitHub repository due to their large file sizes.  
- Each case study has its dedicated folder containing the respective Woodstock files.
- The curves folder contains the generated carbon curves for each case study.  


### 2. **Outputs Folder**

Results are saved in this folder in the form of figures, CSV files, and pickle files. Subfolders are organized as follows:

- **`hwp1_dis1`**: Includes results considering Harvested Wood Products (HWP) with displacement effects.  
- **`hwp1_dis0`**: Includes results considering HWP without displacement effects.  
- **`hwp0_immed1`**: Assumes all harvested products turn into emissions immediately after harvesting.

### 3. **Scenarios Folder**

This folder contains results for all scenarios run under different assumptions regarding HWPs and displacement effects for all cases.

## Main Scripts

The DSS is structured with three main types of scripts:

### 1. **Generate Woodstock Files**

- For each case study, a specific notebook is created to generate Woodstock files.
- While the main structure is consistent across cases, modifications are made as needed for each case.  
- Scripts in this category are named starting with `00`.

### 2. **Generate Carbon Curves**

- The main script for generating carbon curves is named `dss_build_compare_carbon_curve_redchris`.  
- Results for individual cases are saved using scripts named starting with `01`.

### 3. **Build and Run Scenarios**

- The main script for running scenarios is named `dss_build_scenarios_loop`.  
- Results for each case are generated using scripts named starting with `02-05`.






# informationUncertainty
Codes for Information Uncertainty Project. Data can be obtained CMIE. 

## Data generation
The files for generating the data to train the model are located in the **"dataGeneration"** folder. They should be executed in the following order:
- Wealth data: Execute **"0_Prepping Data for Wealth Analysis.ipynb"** to generate the wealth data and **"0_estimation_wealth_dist_longdata.do"** to estimate the initial wealth values.
- Run the files in **"NSS_IndiaSurveyAnalysis"** to estimate the wealth distribution using the NSS survey.
- Execute **"1_explore_HH.ipynb"**, **"2_explore_HHConsumption.ipynb"**, and **"3_exploreHHIncome.ipynb"** to generate household-level data.
- Run **"4_Create_Data_for_Model.ipynb"** to merge the data together and train the model.## Model Estimation 
## Model Training
- To estimate the Flow Model run **flowModelEstimation.ipynb** in **ModelEstimation** folder.
## Uncertainty Estimation
- To estimate the future life time trajectory and , run the following files in the **"UncertaintyEstimation"** folder. Finaly to plot the results run **"PlotsUncertainty.ipynb"**.

## Other
- Bin2 includes the code for the model, and helper function for the uncertainty estimation.
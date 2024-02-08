# informationUncertainty
Codes for Information Uncertainty Project

## Data generation
The files to generate the data to train the model is in "dataGeneration" folder. The files should run in the following order
- Wealth data - Run "0_Preping Data for Wealth Analysis.ipynb" to generate the wealth data and "0_estimation_wealth_dist_longdata.do" to estimate the inital wealth values
- Run the files in "NSS_IndiaSurveyAnalysis" to estimate the wealth distribution using the NSS survey 
- Run "1_explore_HH.ipynb", "3_exploreHHIncome.ipynb" ""2_explore_HHConsumption.ipynb" to generate the household level data
- Run "4_Create_Data_for_Model.ipynb" to merge data together and to train the model. 
## Model Estimation 

## Uncertainty Estimation

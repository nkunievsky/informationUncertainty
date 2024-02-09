
import pickle
import torch 
import bin2.setGlobals as gl
gl.torch_device = torch.device(type='cpu')  
from bin2.sampleFunctions import *

import torch.multiprocessing as mp
from functools import partial

#################3
# import os 
# os.chdir('D:/Dropbox/Dropbox/uchicago_fourth/uncertaintyInequality')
# os.getcwd()
########
def keep_latest_date(df, id_col, month_col):
    """
    Filters a DataFrame to keep only the latest entry for each unique ID based on the month.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the data to be filtered.
    - id_col (str): The name of the column in `df` that contains unique IDs.
    - month_col (str): The name of the column in `df` that contains month information.

    Returns:
    - pandas.DataFrame: A DataFrame containing only the latest entry for each unique ID based on the month.
    """
    df = df.sort_values(by=[id_col, month_col], ascending=[True, False])
    df = df.drop_duplicates(subset=id_col, keep='first')    
    return df

def generateSample(Age, maxAge, data, model, continiousCols, binaryCols,
                   incomeConsNormalizer, continiousNormalizer, R,
                   maxValue_Income, maxValue_Consumption, minValue_Income, minValue_Consumtpion, forceBiSecConverge=5,
                   omittedCols=[]):
    """
    Generates a sample dataset for a given age, applying transformations and filtering based on the provided model and constraints.

    Parameters:
    - Age (int): The current age to filter the sample data on.
    - maxAge (int): The maximum age considered in the analysis.
    - data (numpy.ndarray): The input data for analysis.
    - model (object): The model used for generating panel data.
    - continiousCols (list): List of column indices for continuous variables.
    - binaryCols (list): List of column indices for binary variables.
    - incomeConsNormalizer (dict), continiousNormalizer (dict): Normalizers for income/consumption and continuous variables.
    - R (float): Discount rate or rate of return used in calculations.
    - maxValue_Income, maxValue_Consumption, minValue_Income, minValue_Consumtpion (float): Upper and lower bounds for income and consumption values.
    - forceBiSecConverge (int): Number of iterations for the bisection method to force convergence.
    - omittedCols (list): Columns to be omitted from the final sample.

    Returns:
    - list: A list containing the original age and capital data, the cleaned inverse data sample, and the full inverse data sample.
    """
    analysisData = data.copy()
    cAge = int(Age)
    print("THE AGE WE ARE LOOKING AT IS", cAge)
    dataPeriods = int((analysisData.shape[1] - 2) / 2)
    periods = maxAge - dataPeriods - cAge + 1
    sample = analysisData[analysisData[:, continiousCols][:, 0] == cAge, :]
    print("Amount of People Going in", sample.shape)
    sample, OG_ageCapital, flag = genPanelData(sample, model, contiCols=continiousCols, indCols=binaryCols,
                                                incom_dic=incomeConsNormalizer, contin_dic=continiousNormalizer,
                                                R=R, colList=[-2, -1], T=periods, forceBiSecConverge=forceBiSecConverge)
    if flag == 0:
        print("Im OUT with FLAG 0")
        return None

    print("Amount of People Going to Sample Inv ", sample.shape)
    sampleINV = invData(sample, contiCols=continiousCols, indCols=binaryCols,
                        incom_dic=incomeConsNormalizer, contin_dic=continiousNormalizer,
                        incomeUB=maxValue_Income, consumptionUB=maxValue_Consumption,
                        incomeLB=minValue_Income, consumptionLB=minValue_Consumtpion)
    print("Amount of People getting out of Sample Inv ", sample.shape)

    inverted_mask = np.isin(np.arange(sampleINV.shape[1]), omittedCols, invert=True)
    sampleINV_clean = sampleINV[:, inverted_mask]

    finSample = [OG_ageCapital, sampleINV_clean, sampleINV]
    return finSample


######################################
## Import data, parametes and model ##
######################################

with open(gl.tempsFolder  + '/data_test.pkl', 'rb') as f:
    dataForSimulationTest = pickle.load(f)

with open(gl.tempsFolder  + '/data_train.pkl', 'rb') as f:
    dataForSimulationTrain = pickle.load(f)

with open(gl.tempsFolder  + '/df_analysis.pkl', 'rb') as f:
    dataForSimulation_pd = pickle.load(f)

omittedCols = [ii-1 for ii,i in enumerate(dataForSimulation_pd.columns) if i.find('TOTAL_EXPENDITURE_')>-1 or i.find('labourIncome_')>-1] 
dataForSimulation = keep_latest_date(dataForSimulation_pd, "HH_ID", "ageMonth")
dataForSimulation = dataForSimulation.to_numpy()[:,1:]

with open(gl.tempsFolder  +'/normalizers.pkl', 'rb') as f:
    normalizers = pickle.load(f)

incomeConsNormalizer = normalizers['incomeConsNormalizer'].copy()
continiousNormalizer = normalizers['continiousNormalizer'].copy()
indicNormalizer = normalizers['indicNormalizer'].copy()

with open(gl.tempsFolder  +'/parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)

binaryCols = parameters['binaryCols'].copy()
continiousCols = parameters['continiousCols'].copy()
maxValue_Income = parameters['maxValue_Income']
maxValue_Consumption = parameters['maxValue_Consumption']
minValue_Income = parameters['minValue_Income']
minValue_Consumtpion = parameters['minValue_Consumtpion']
R = gl.R
beta = gl.beta

model = torch.load(gl.modelsFolder  +'/pytorchModel.pt',map_location=torch.device('cpu'))
model.to('cpu')
model.eval()
print(next(model.parameters()).is_cuda) # retur)

def extendFucntion(x):
    global finData
    finData.extend(x)

if __name__ == '__main__':
    ############
    # Estimate #
    ############

    lastAgeInAnalysis = gl.lastAgeInAnalysis
    print('the last age we consider is',lastAgeInAnalysis)
    list_of_ages = np.unique(dataForSimulation[:,continiousCols][:,0])
    maxAge = 80*12
    a = map(partial(generateSample,maxAge = maxAge,data=dataForSimulation,model=model,continiousCols=continiousCols,binaryCols=binaryCols,
                incomeConsNormalizer=incomeConsNormalizer,continiousNormalizer=continiousNormalizer,R=R,forceBiSecConverge=5,
                maxValue_Income=maxValue_Income,maxValue_Consumption=maxValue_Consumption,
                minValue_Income=minValue_Income,minValue_Consumtpion=minValue_Consumtpion),list_of_ages[list_of_ages<=lastAgeInAnalysis])
    finData = list(a)

    emptyAges = [i for i in range(len(finData)) if finData[i] is None]
    finDataClean = [finData[i] for i in range(len(finData)) if finData[i] is not None]
    
    with open(gl.tempsFolder  + '/finData' + gl.fileNamesSuffix + '.pkl', 'wb') as f:
        pickle.dump(finDataClean,f)

    badAges = {'ageList':list_of_ages,
            'badAgesIndex':emptyAges 
    } #Notice that the indices does not need to be correct because I'm using async. But likely will be. 
    
    with open(gl.tempsFolder  + '/badAges' + gl.fileNamesSuffix + '.pkl', 'wb') as f:
        pickle.dump(badAges,f)
    
    print('finData - DONE!')

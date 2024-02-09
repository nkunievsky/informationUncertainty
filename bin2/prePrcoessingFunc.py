import numpy as np
import bin2.setGlobals as gl 
# torch_device = gl.torch_device

def logTransformation(col, inv=False):
    """
    Applies a logarithmic transformation to the input column. It supports both forward and inverse transformations.
    
    Parameters:
    - col (np.ndarray): The input column to be transformed.
    - inv (bool): Flag indicating whether to perform the inverse transformation. Default is False.
    
    Returns:
    - np.ndarray: The transformed column.
    """
    if not inv:
        col = col + 1
        col = np.log(col)
    else:
        col = np.exp(col)
        col = col - 1
        # Trimming negative values to a small positive value due to potential noise-induced negative consumption
        col[col < 1e-3] = 1e-3
    return col

def standarize(data_norm, tmean, tstd, inv=False):
    """
    Standardizes or reverse standardizes the input data based on the provided mean and standard deviation.
    
    Parameters:
    - data_norm (np.ndarray): The input data to be standardized.
    - tmean (float): The mean used for standardization.
    - tstd (float): The standard deviation used for standardization.
    - inv (bool): Flag indicating whether to perform the inverse standardization. Default is False.
    
    Returns:
    - np.ndarray: The standardized or reverse standardized data.
    """
    if not inv:
        stdData = (data_norm - tmean) / tstd
    else:
        stdData = data_norm * tstd + tmean
    return stdData

def normalize(data_norm, tmax, tmin, typed, inv=False):
    """
    Normalizes or reverse normalizes the input data based on the provided maximum and minimum values.
    
    Parameters:
    - data_norm (np.ndarray): The input data to be normalized.
    - tmax (float): The maximum value used for normalization.
    - tmin (float): The minimum value used for normalization.
    - typed (str): Specifies the normalization type ('Uniform' or other types assumed to be linear scaling).
    - inv (bool): Flag indicating whether to perform the inverse normalization. Default is False.
    
    Returns:
    - np.ndarray: The normalized or reverse normalized data.
    """
    if not inv:
        if typed == 'Uniform':
            data_norm = (data_norm - tmin) / (tmax - tmin)
        else:
            data_norm = data_norm / (tmax - tmin)
    else:
        if typed == 'Uniform':
            data_norm = data_norm * (tmax - tmin) + tmin
        else:
            data_norm = data_norm * (tmax - tmin)
    return data_norm



def preprocessData(data,logs=True,std=True,Normalize=True,bandwidth=1.2,typed="Uniform",invert=False,invertDic={}):
    """
    Preprocesses the given data with options for logarithmic transformation, standardization, and normalization.
    It can also perform the inverse of these operations based on the provided dictionary.

    Parameters:
    - data (np.ndarray): The input data to preprocess.
    - logs (bool): If True, applies logarithmic transformation. Default is True.
    - std (bool): If True, standardizes the data. Default is True.
    - Normalize (bool): If True, normalizes the data. Default is True.
    - bandwidth (float): Scaling factor for normalization bounds. Default is 1.2.
    - typed (str): The type of normalization, either "Uniform" or "Gaussian". Default is "Uniform".
    - invert (bool): If True, performs the inverse preprocessing operations. Default is False.
    - invertDic (dict): Dictionary containing parameters for inverse operations.

    Returns:
    - dict: A dictionary containing the preprocessed data and parameters used in preprocessing.
    """
    if not invert:
        resDict = {'data': data,
                   'tmean' : None,
                   'tstd' : None,
                   'tmax' :None,
                   'tmin' :None
                  }

        if logs :
            tempData = resDict['data']
            for c in range(tempData.shape[1]):
                tempData[:,c] = logTransformation(tempData[:,c])                
            resDict['data']  = tempData

        if std:        
            data_norm = resDict['data'] 
            tmean,tstd =  data_norm.mean(axis=0), data_norm.std(axis=0)
            data_norm = standarize(data_norm,tmean,tstd)
            resDict['data'] = data_norm
            resDict['tmean'] = tmean
            resDict['tstd'] = tstd

        if Normalize:
            data_norm = resDict['data'] 
            tmax,tmin =  data_norm.max(axis=0)*bandwidth, data_norm.min(axis=0)*bandwidth
            data_norm = normalize(data_norm,tmax,tmin,typed)
            resDict['data'] = data_norm
            resDict['tmax'] = tmax
            resDict['tmin'] = tmin

        return resDict
    #######
    #######
    if invert:
        if Normalize and typed == "Uniform":
            data_norm = invertDic['data']
            tmin = invertDic['tmin']
            tmax = invertDic['tmax']
            data_norm = data_norm*(tmax-tmin) + tmin

            invertDic['data'] = data_norm

        if Normalize and typed == "Gaussian":
            data_norm = invertDic['data'] 
            tmin = invertDic['tmin']
            tmax = invertDic['tmax']
            data_norm = (data_norm)*(tmax-tmin)
            invertDic['data'] = data_norm
        
        if std:        
            data_norm = invertDic['data']
            tmean = invertDic['tmean']
            tstd = invertDic['tstd']
            data_norm = (data_norm)*tstd +tmean
            invertDic['data'] = data_norm

        if logs :
            tempData = invertDic['data']
            for c in range(tempData.shape[1]):
                tempData[:,c] = np.exp(tempData[:,c])
                tempData[:,c] = tempData[:,c]-1
            invertDic['data'] = tempData
    
    return invertDic


def preprocessData_colTypes(data,incomeConsumptionCOls,indicatorCols= [],continiousCols= [],
                            logs=True,std=True,Normalize=True,bandwidth=1.2,typed="Uniform",
                            logsCont=-99,stdCont=-99,NormalizeCont=-99,bandwidthCont=-99,typedCont=-99
                            ): 
    """
    Preprocesses data with different treatments for income/consumption, continuous, and indicator columns.

    Parameters:
    - data (np.ndarray): The input data array.
    - incomeConsumptionCols (list): Column indices for income and consumption data.
    - indicatorCols (list): Column indices for indicator variables.
    - continiousCols (list): Column indices for continuous variables.
    - logs, std, Normalize (bool): Flags to apply logarithmic transformation, standardization, and normalization.
    - bandwidth, typed (float, str): Parameters for normalization.
    - logsCont, stdCont, NormalizeCont, bandwidthCont, typedCont: Specific preprocessing parameters for continuous columns.

    Returns:
    - list: A list containing dictionaries for preprocessed income/consumption, continuous, and indicator data.
    """

    #Income and consumption type columns:    
    conincData = data[:,incomeConsumptionCOls]    
    colNumber = conincData.shape[1]
    #Log transform
    for c in range(colNumber):
        conincData[:,c] = logTransformation(conincData[:,c],inv=False)
            
    #Stdartize Income 
    incSTD = np.std(conincData[:,np.arange(0,colNumber,2)])
    incMEAN = np.mean(conincData[:,np.arange(0,colNumber,2)])
    conincData[:,np.arange(0,colNumber,2)] = (conincData[:,np.arange(0,colNumber,2)]-incMEAN)/incSTD
    #Normalize income 
    incMAX = np.max(conincData[:,np.arange(0,colNumber,2)])
    incMIN = np.min(conincData[:,np.arange(0,colNumber,2)])
    if typed=="Uniform":
        conincData[:,np.arange(0,colNumber,2)]  = (conincData[:,np.arange(0,colNumber,2)] -incMIN)/(incMAX-incMIN)
    else :
        conincData[:,np.arange(0,colNumber,2)]  = (conincData[:,np.arange(0,colNumber,2)])/(incMAX-incMIN)
    

    #Stdartize Consumption
    conSTD = np.std(conincData[:,np.arange(1,colNumber,2)])
    conMEAN = np.mean(conincData[:,np.arange(1,colNumber,2)])
    conincData[:,np.arange(1,colNumber,2)] = (conincData[:,np.arange(1,colNumber,2)]-conMEAN)/conSTD
    #Normalize Consumption 
    conMAX = np.max(conincData[:,np.arange(1,colNumber,2)])*bandwidth
    conMIN = np.min(conincData[:,np.arange(1,colNumber,2)])*bandwidth
    if typed=="Uniform":
        conincData[:,np.arange(1,colNumber,2)]  = (conincData[:,np.arange(1,colNumber,2)] -conMIN)/(conMAX-conMIN)
    else :
        conincData[:,np.arange(1,colNumber,2)]  = (conincData[:,np.arange(1,colNumber,2)])/(conMAX-conMIN)
    
    incomeConsumptionDic = {'data':conincData,
                            'tstd': [incSTD,conSTD],
                            'tmean':[incMEAN,conMEAN],
                            'tmax':[incMAX,conMAX],
                            'tmin':[incMIN,conMIN]
    }

    #Standarize the other columns 
    if len(continiousCols) > 0:
        continCols = preprocessData(data[:,continiousCols],logs=False,std=True,Normalize=True,bandwidth=bandwidthCont,typed=typedCont)
    else : 
        continCols= {}

    if len(indicatorCols) > 0 :
        indicatorsDict = {'data': data[:,indicatorCols]}
    else : 
        indicatorsDict = {}

    return [incomeConsumptionDic,continCols,indicatorsDict]

 
def Winsorizing(df,c,maxValue,minValue):
    df.loc[df[c]<minValue,c] = np.nan
    df.loc[df[c]>maxValue,c] = np.nan
    return df
    
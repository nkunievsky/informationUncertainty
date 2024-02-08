import numpy as np
import bin.setGlobals as gl 
# torch_device = gl.torch_device

def logTransformation(col,inv=False):
    if not inv :
        if np.min(col+1)>0:                
            col = col +1
            col = np.log(col)
    if inv:
        col = np.exp(col)
        col = col-1
        '''
        In theory we shouldn't get negative consumption, but we might get consumption between -1 and 0 due to noise. 
        This as we take -1 from the inverse log. I'm trimming negative values here (which is a dollar difference). 
        I'm not a huge fan of this, but not sure if anything is better. Also, I don't think that this happened a lot, but it happened once :\
        '''
        col[col<1e-3]=1e-3 
    return col

def standarize(data_norm,tmean,tstd,inv=False):
    if not inv :
        stdData = (data_norm - tmean)/tstd    
    if inv : 
        stdData = data_norm*tstd + tmean    
    return stdData

def normalize(data_norm,tmax,tmin,typed,inv=False):
    if not inv :
        if typed=='Uniform':
            data_norm = (data_norm - tmin)/(tmax-tmin)
        else :
            data_norm = (data_norm)/(tmax-tmin)
    if inv : 
        if typed=='Uniform':
            data_norm = (data_norm)*(tmax-tmin) + tmin
        else :
            data_norm = (data_norm)*(tmax-tmin)
    return data_norm



def preprocessData(data,logs=True,std=True,Normalize=True,bandwidth=1.2,typed="Uniform",invert=False,invertDic={}):
    
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
        indicatorsDict = {'data': data[indicatorCols]}
    else : 
        indicatorsDict = {}

    return [incomeConsumptionDic,continCols,indicatorsDict]

 

    
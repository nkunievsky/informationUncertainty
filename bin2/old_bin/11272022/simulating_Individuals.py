
import pickle
import torch 
import bin.setGlobals as gl
gl.torch_device = torch.device(type='cpu') #In order update the device for all - in the first file first load the gl file and choose what ever you want to change. THen! load the other functions! AHA! this would update the namespaces as needed 
from bin.sampleFunctions import *

import torch.multiprocessing as mp
from functools import partial

def generateSample(Age,maxAge,data,model,continiousCols,binaryCols,
                    incomeConsNormalizer,continiousNormalizer,R,
                    maxValue_Income,maxValue_Consumption,minValue_Income,minValue_Consumtpion,forceBiSecConverge=5):
    
    cAge = int(Age)
    # print(cAge,end="\r")
    print(cAge)
    dataPeriods = int((data.shape[1]-2)/2)
    periods = maxAge-dataPeriods - cAge + 1 
    sample = data[data[:,continiousCols][:,0]==cAge,:] #Notice that the age is the first column of the continious cols 
    sample, OG_ageCapital,flag = genPanelData(sample,model,contiCols=continiousCols,indCols=binaryCols,
                            incom_dic=incomeConsNormalizer,contin_dic=continiousNormalizer,
                            R=R,colList=[-2,-1],T = periods,forceBiSecConverge=forceBiSecConverge)
    if flag == 0:
        return None 
    sampleINV = invData(sample,contiCols=continiousCols,indCols=binaryCols,
                    incom_dic=incomeConsNormalizer,contin_dic=continiousNormalizer,
                        incomeUB=maxValue_Income,consumptionUB=maxValue_Consumption,
                        incomeLB=minValue_Income,consumptionLB=minValue_Consumtpion)
    finSample = [OG_ageCapital,sampleINV]
    return finSample

if __name__ == '__main__':

    with open(gl.tempsFolder  + '/data_test.pkl', 'rb') as f:
        dataForSimulation = pickle.load(f)

    with open(gl.tempsFolder  +'/normalizers.pkl', 'rb') as f:
        normalizers = pickle.load(f)

    incomeConsNormalizer = normalizers['incomeConsNormalizer']
    continiousNormalizer = normalizers['continiousNormalizer']
    indicNormalizer = normalizers['indicNormalizer']

    with open(gl.tempsFolder  +'/parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)

    binaryCols = parameters['binaryCols']
    continiousCols = parameters['continiousCols']
    maxValue_Income = parameters['maxValue_Income']
    maxValue_Consumption = parameters['maxValue_Consumption']
    minValue_Income = parameters['minValue_Income']
    minValue_Consumtpion = parameters['minValue_Consumtpion']
    R = gl.R
    beta = gl.beta

    model = torch.load(gl.modelsFolder  +'/pytorchModel.pt')
    model.to('cpu')
    model.eval()
    print(next(model.parameters()).is_cuda) # retur)

    ############
    # Estimate #
    ############

    p = mp.Pool(2)

    lastAgeInAnalysis = 42*12#50*12
    list_of_ages = np.unique(dataForSimulation[:,continiousCols][:,0])
    maxAge = 70*12

    finData = [] 
    p.map_async(partial(generateSample,maxAge = maxAge,data=dataForSimulation,model=model,continiousCols=continiousCols,binaryCols=binaryCols,
                    incomeConsNormalizer=incomeConsNormalizer,continiousNormalizer=continiousNormalizer,R=R,forceBiSecConverge=5,
                    maxValue_Income=maxValue_Income,maxValue_Consumption=maxValue_Consumption,
                    minValue_Income=minValue_Income,minValue_Consumtpion=minValue_Consumtpion),
                    list_of_ages[list_of_ages<=lastAgeInAnalysis],callback=finData.extend)
    
    p.close()
    p.join()
    
    with open(gl.tempsFolder  + '/finData' + gl.fileNamesSuffix + '.pkl', 'wb') as f:
        pickle.dump(finData,f)

    print('finData - DONE!')
    # print(finData)
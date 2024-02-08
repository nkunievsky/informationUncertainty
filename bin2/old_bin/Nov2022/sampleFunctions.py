   
# import deepul.pytorch_util as ptu
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import statsmodels.api as sm
import torch
from tqdm import notebook
from itertools import (combinations, combinations_with_replacement,
                       permutations, product)


import bin.setGlobals as gl
from bin.prePrcoessingFunc import *

torch_device = gl.torch_device


def toVec(x):
    if isinstance(x,list):
        x = np.array(x)
        
    if len(x.shape)==2:
        return x 
    else :
        return x.reshape(-1,1)


def sampleGenerator(data,model,maxObservations=False):
    if ~maxObservations:
        b = data.shape[0]
    else :
        b= maxObservations

    model.eval()
    obs = torch.tensor(data[:b,:]).float().to(torch_device)
    sims =  model.invertSample(obs)
    return sims

def plotCompare(dataRaw,simsRAW,columns=[-1],onTop=False,bins=100,maxRow=False):
    sims = toVec(simsRAW)
    data = dataRaw.detach().cpu().numpy() if torch.is_tensor(dataRaw) else dataRaw
            
    if sims.shape[1] == 1 :
        plt.hist(sims,bins=bins);
        if onTop is False:
            plt.show()
        if maxRow:
            plt.hist(data[:maxRow,columns],bins=bins);
            plt.show()
        else :
            plt.hist(data[:,columns],bins=bins);
            plt.show()
    else : 
        for c in range(sims.shape[1]):
            plt.hist(sims[:,c],bins=bins);
            if onTop is False:
                plt.show()
            if maxRow:
                plt.hist(data[:maxRow,columns[c]],bins=bins);
                plt.show()
            else :
                plt.hist(data[:,columns[c]],bins=bins);
                plt.show()

    
def regressionComparison(dataRaw,simsRAW,predictedCols=[-1],printRegression=False,PlotCoefs=True,printR2=True,PrintCF=True,returnit=False):
    firstcol = min(predictedCols)
    sims = toVec(simsRAW)
    data = dataRaw.detach().cpu().numpy() if torch.is_tensor(dataRaw) else dataRaw

    simData = np.concatenate((data[:,:firstcol],sims),axis=1)
    
    xData = simData[:,:firstcol]
    xData = sm.add_constant(xData, prepend=True)
    for c in predictedCols:
        print(c)
        yDataSim = simData[:,-1] #the last column in the simData
        modSim = sm.OLS(yDataSim, xData)
        resSim = modSim.fit()
        
        yDataReal = data[:,c]
        modReal = sm.OLS(yDataReal, xData)
        resReal = modReal.fit()
        
        if printRegression:
            print('Simulated Result')
            print(resSim.summary())
            print('Real Results')
            print(resReal.summary())
            
        if printR2:
            print('simulated r2 is :' +  str(resSim.rsquared))
            print('Real r2 is :' +  str(resReal.rsquared))
        
        if PrintCF:
            cfIntervalsSim = resSim.conf_int(alpha=0.05)
            cfIntervals = resReal.conf_int(alpha=0.05)

            for y,bs in enumerate(cfIntervals):
                plt.plot((bs[0],bs[1]),(y,y),'o-',c='orange')
            for y,bs in enumerate(cfIntervalsSim):
                plt.plot((bs[0],bs[1]),(y,y),'o-',c='b')
            plt.show()
            
        if PlotCoefs:
            for y,bs in enumerate(zip(resReal.params,resSim.params)):
                plt.plot([bs[0]],[y],'o',c='orange')
                plt.plot([bs[1]],[y],'o',c='blue')
            plt.show()


def sampleGeneratorDynamic(data,model,finalCol,maxObservations=False):
    model.eval()
    if torch.is_tensor(data):
        if ~maxObservations:
            obs = data.float().to(torch_device)
        else :
            b= maxObservations
            obs = data[:b,:].float().to(torch_device)
    
    else: 
        if ~maxObservations:
            obs = torch.tensor(data).float().to(torch_device)
        else :
            b= maxObservations
            obs = torch.tensor(data[:b,:]).float().to(torch_device)
    
    sims,flag =  model.invertSample(obs,finalCol=finalCol)
    return sims,flag

#This function preps the test data to be inserted into the model 
def std_norm_Logrize(panelData,continiousDict_input,incomeDict_input,numColsConti=2,numColInd=0,typed='gaussian'):
    numCols = panelData.shape[1]
    numIncomeCols = numCols-numColsConti-numColInd

    number_of_panelists = panelData.shape[0]
    ageCapital = panelData[:,0:2].copy()
    OG_ageCapital = ageCapital.copy()

    incomeCons = panelData[:,-numIncomeCols:].copy()

    #STDize Normize Cont Columns 
    ageCapital = standarize(ageCapital,continiousDict_input['tmean'],continiousDict_input['tstd'])
    ageCapital = normalize(ageCapital,continiousDict_input['tmax'],continiousDict_input['tmin'],typed=typed)

    #STDize Normize Income consumption 
    #define auxilary vectors - this just due to historic reasons. 
    conincMEAN = np.array(incomeDict_input['tmean']*int(numIncomeCols/2))
    conincSTD = np.array(incomeDict_input['tstd']*int(numIncomeCols/2))
    conincMAX = np.array(incomeDict_input['tmax']*int(numIncomeCols/2))
    conincMIN = np.array(incomeDict_input['tmin']*int(numIncomeCols/2))

    for c in range(incomeCons.shape[1]):
        incomeCons[:,c] = logTransformation(incomeCons[:,c])

    incomeCons = standarize(incomeCons,conincMEAN,conincSTD)
    incomeCons = normalize(incomeCons,conincMAX,conincMIN,typed=typed)
    
    return np.concatenate((ageCapital,incomeCons),axis=1)

def plotHistDemonstration(lastCol,realData,model,incomCols=[],continCols=[],binaryCols=[],maxObservations=False,
    binNum=100,save=False,saveName=False,truncated=False):
    a = realData
    a = std_norm_Logrize(a,continCols,incomCols)
    sims,_ = sampleGeneratorDynamic(a,model,finalCol=lastCol,maxObservations=False)
    sims = sims.cpu().detach().numpy()
    if truncated:
        _,bins,_ = plt.hist(sims[sims<truncated],bins=binNum,label='Fake Data');
        plt.hist(a[(sims<truncated).squeeze(1),lastCol],bins=bins,label='Real Data');
    else :
        _,bins,_ = plt.hist(sims,bins=binNum,label="Fake Data");
        plt.hist(a[:,lastCol],bins=bins,label="Real Data");
    plt.legend()
    if save  :
        fileName = 'plots/' + saveName + '.pdf'
        plt.savefig(fileName)  
    plt.show()


def monteCarloForR2andBeta(lastCol,realData,model,incomCols=[],continCols=[],binaryCols=[],iterations=200,
    verbose=False,save=False,saveName=False,truncated=False,returnResult=False):
    adjR2_2 = []
    betas = []
    for i in notebook.tqdm(range(iterations),desc='Iteration', leave=True):
        lastCol = -2
        a = realData
        a = std_norm_Logrize(a,continCols,incomCols)
        sims,flag = sampleGeneratorDynamic(a,model,finalCol=lastCol,maxObservations=False)
        sims = sims.cpu().detach().numpy()
        
        sims = toVec(sims)
        simData = np.concatenate((a[:,:lastCol],sims),axis=1)
        xData = simData[:,:-1]
        xData = sm.add_constant(xData, prepend=True)

        yDataSim = simData[:,-1] #the last column in the simData so it's always -1
        modSim = sm.OLS(yDataSim, xData)
        resSim = modSim.fit()
        adjR2_2.append(resSim.rsquared_adj)
        betas.append(resSim.params)
    #Run regression on the real data
    yDataReal = a[:,lastCol]
    modReal = sm.OLS(yDataReal, xData)
    resReal = modReal.fit()
    if verbose :
        print(resReal.summary())

    for i in betas:
        for y,bs in enumerate(i):
            plt.plot(bs,[y],'o',c='blue',markersize=2)


    for y,bs in enumerate(resReal.params):
        plt.plot(bs,[y],'o',c='red',markersize=2)
    if save : 
        fileName = 'plots/' + saveName + '_betas.pdf'
        plt.savefig(fileName)  
    plt.show()    
    #Plotting R2
    plt.hist(np.array(adjR2_2),bins=30)
    plt.axvline(x=resReal.rsquared_adj, color='r', linestyle='-')
    if save : 
        fileName = 'plots/' + saveName + '_adjR2.pdf'
        plt.savefig(fileName)  
    plt.show()    
    if returnResult :
        return adjR2_2,betas


##############################
# Graph Generating Functions #
##############################
## Not the most flexiable thingy. 
# Add a option to add columns 

def genPanelData(panelData,model,R=(1+0.0051585),incomeDict_input=[],continiousDict_input=[],
                colList=[-2,-1],T = 5,numColsConti=2,numColInd=0,forceBiSecConverge=False
                 ):

    #Define auxilatry vars  
    numCols = panelData.shape[1]
    numIncomeCols = numCols-numColsConti-numColInd

    number_of_panelists = panelData.shape[0]
    ageCapital = panelData[:,0:2].copy()
    OG_ageCapital = ageCapital.copy()
    
    incomeCons = panelData[:,-numIncomeCols:].copy()
    
    #STDize Normize Continious Columns 
    ageCapital = standarize(ageCapital,continiousDict_input['tmean'],continiousDict_input['tstd'])
    ageCapital = normalize(ageCapital,continiousDict_input['tmax'],continiousDict_input['tmin'],typed='gaussian')
    
    #STDize Normize Income consumption 
    #Define auxilary vectors - this just due to historic reasons. 
    conincMEAN = np.array(incomeDict_input['tmean']*int(numIncomeCols/2))
    conincSTD = np.array(incomeDict_input['tstd']*int(numIncomeCols/2))
    conincMAX = np.array(incomeDict_input['tmax']*int(numIncomeCols/2))
    conincMIN = np.array(incomeDict_input['tmin']*int(numIncomeCols/2))
    
    for c in range(incomeCons.shape[1]):
        incomeCons[:,c] = logTransformation(incomeCons[:,c])
    
    incomeCons = standarize(incomeCons,conincMEAN,conincSTD)
    incomeCons = normalize(incomeCons,conincMAX,conincMIN,typed='gaussian')
    
    #Define the sample 
    sample = np.concatenate((ageCapital,incomeCons),axis=1)

    for t in range(1,T):
        ageCapital = sample[:,0:2]    
        incomeCons = sample[:,-114:] 
        placeHolders = np.zeros((number_of_panelists,2))

        a = np.concatenate((ageCapital,incomeCons,placeHolders),axis=1)
        if forceBiSecConverge:
            for c in colList:
                flag = 0 
                iter = 0 
                while flag==0 and iter <= forceBiSecConverge:
                    maxDraws = 3
                    retryDraws = 0 
                    while retryDraws < maxDraws : #Not sure this is useful because it fails due to draws acording to the last sample. Might think about changing this. 
                        try : 
                            sims,flagReturn = sampleGeneratorDynamic(a,model,finalCol=c)                    
                            retryDraws = maxDraws+1
                            flag = flagReturn
                        except :
                            retryDraws  += 1
                            flag = 0 
                            
                    if flag==0:
                        iter += 1
                        print(('entering iteration {}').format(iter) )

                sims = sims.cpu().detach().numpy()
                a[:,c] = sims.squeeze()
                
                if flag == 0 :
                    return None, None,0


        else :
            for c in colList:
                sims,flag = sampleGeneratorDynamic(a,model,finalCol=c)
                sims = sims.cpu().detach().numpy()
                a[:,c] = sims.squeeze()

        sample = np.concatenate((sample,a[:,colList]),axis=1)

        #Update age 
        ageCapital = normalize(ageCapital,continiousDict_input['tmax'],continiousDict_input['tmin'],'gaussian',inv=True)
        ageCapital = standarize(ageCapital,continiousDict_input['tmean'],continiousDict_input['tstd'],inv=True)

        ageCapital[:,0]  =ageCapital[:,0] +1 ## add one month to age

        #update capital 
        capFlows = sample[:,-116:-114]
        capFlows = normalize(capFlows,conincMAX[0:2],conincMIN[0:2],'gaussian',inv=True)
        capFlows = standarize(capFlows,conincMEAN[0:2],conincSTD[0:2],inv=True)
        for c in range(capFlows.shape[1]):
            capFlows[:,c] = logTransformation(capFlows[:,c],inv=True)
        ageCapital[:,1]  = ageCapital[:,1]*R + capFlows[:,0] - capFlows[:,1]

        #Normalize/Std again
        ageCapital = standarize(ageCapital,continiousDict_input['tmean'],continiousDict_input['tstd'],inv=False)
        ageCapital = normalize(ageCapital,continiousDict_input['tmax'],continiousDict_input['tmin'],'gaussian',inv=False)

        sample[:,0:2] = ageCapital
    
    return sample, OG_ageCapital,1


def invData(sample,continiousDict_input,incomeDict_input,OGageCapital=None,
            incomeUB=1e14,consumptionUB=1e14,incomeLB=0,consumptionLB=1000):
    '''
    gets data after standarization and returns and invert it
    '''

    ageCapital_INV = normalize(sample[:,0:2],continiousDict_input['tmax'],continiousDict_input['tmin'],'gaussian',inv=True)
    ageCapital_INV = standarize(ageCapital_INV,continiousDict_input['tmean'],continiousDict_input['tstd'],inv=True)

    #update capital 
    coninc = sample[:,2:]
    periods = int(coninc.shape[1]/2)

    conincMEAN = np.array(incomeDict_input['tmean']*periods)
    conincSTD = np.array(incomeDict_input['tstd']*periods)
    conincMAX = np.array(incomeDict_input['tmax']*periods)
    conincMIN = np.array(incomeDict_input['tmin']*periods)

    coninc_INV = normalize(coninc,conincMAX,conincMIN,'gaussian',inv=True)
    coninc_INV = standarize(coninc_INV,conincMEAN,conincSTD,inv=True)
    for c in range(coninc_INV.shape[1]):
        coninc_INV[:,c] = logTransformation(coninc_INV[:,c],inv=True)

    '''
    We trim at the top, so that the solver won't explode - might be better to restrict the sample distribution. 
    There must be a more elegent way 
    '''
    cols =  np.arange(0,coninc.shape[1],2)
    coninc_INV[:,cols] = (coninc_INV[:,cols]>incomeUB).astype(int)*incomeUB + (1-(coninc_INV[:,cols]>incomeUB).astype(int))*coninc_INV[:,cols]
    coninc_INV[:,cols] = (coninc_INV[:,cols]<incomeLB).astype(int)*incomeLB + (1-(coninc_INV[:,cols]<incomeLB).astype(int))*coninc_INV[:,cols]

    cols =  np.arange(1,coninc.shape[1],2)
    coninc_INV[:,cols] = (coninc_INV[:,cols]>consumptionUB).astype(int)*consumptionUB + (1-(coninc_INV[:,cols]>consumptionUB).astype(int))*coninc_INV[:,cols]
    coninc_INV[:,cols] = (coninc_INV[:,cols]<consumptionLB).astype(int)*consumptionLB + (1-(coninc_INV[:,cols]<consumptionLB).astype(int))*coninc_INV[:,cols]

    if OGageCapital is None:
        sample_INV = np.concatenate((ageCapital_INV,coninc_INV),axis=1)
    else :
        sample_INV = np.concatenate((OGageCapital,ageCapital_INV,coninc_INV),axis=1)
    
    return sample_INV

def plotDensity(individualHistory,model,incomCols=[],continCols=[],binaryCols=[],predictingCols=[-1,-2],minValue=0,maxValue=100000,gridPoints=100,joint=True,save=False,saveName=False):
    #Generate the 
    analysisCols = np.sort(predictingCols)
    linspace = np.linspace(minValue,maxValue,gridPoints).reshape(-1,1)
    for ci,c in enumerate(analysisCols):
        observation = np.tile(individualHistory[:c],(gridPoints,1))
        observation = np.concatenate((observation,linspace),axis=1)
        for c2 in analysisCols[ci+1:]:
            observation = np.concatenate((observation,np.zeros((gridPoints,1))),axis=1)

        gridVar =  std_norm_Logrize(observation,continCols,incomCols)
        density = model.log_prob(torch.tensor(gridVar).to(torch_device).float(),verbose=True).cpu().detach().numpy()
        plt.plot(linspace,np.exp(density[:,c]),label=c)
    if save:
        plt.legend()
        fileName = 'plots/' + saveName + '.pdf'
        plt.savefig(fileName)
    plt.show()

    if joint and len(predictingCols)==2: #Not sure what do if more than 2 columns - maybe for each combination? should think about it. 
        print('starting Joint')
        minPredictCol = np.min(analysisCols)
        combinations = np.array(list(product(*[linspace.squeeze(),linspace.squeeze()])))
        observation = np.tile(individualHistory[:minPredictCol],(len(combinations),1))
        observation = np.concatenate((observation,combinations),axis=1)

        gridVar =  std_norm_Logrize(observation,continCols,incomCols)
        density = model.log_prob(torch.tensor(gridVar).to(torch_device).float(),verbose=True).cpu().detach().numpy()
        
        jointDenistyTable = np.concatenate((combinations,density),axis=1)
        jointDensity =  np.zeros((gridPoints,gridPoints))
        for g1i,g1 in enumerate(linspace):
            for g2i,g2 in enumerate(linspace):
                ind = (jointDenistyTable[:,0] == g1) & (jointDenistyTable[:,1] == g2)
                jointDensity[g1i,g2i] = np.exp(jointDenistyTable[ind,2]+jointDenistyTable[ind,3])
        
        ax = sns.heatmap(jointDensity,xticklabels=np.arange(gridPoints),yticklabels=np.arange(gridPoints))
        ax.invert_yaxis()
        #NEEED TO FIX LABELS ON AXIS.. but it's 1400 - so im next to the next one 
        plt.show()
    # return jointDensity, jointDenistyTable,density





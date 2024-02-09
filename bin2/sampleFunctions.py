   
# import deepul.pytorch_util as ptu
import matplotlib.pyplot as plt
import seaborn as sns

##############
import pickle #Remove it later 
import random
##############

import numpy as np
import statsmodels.api as sm
import torch
from tqdm import notebook
from itertools import (combinations, combinations_with_replacement,
                       permutations, product)


import bin2.setGlobals as gl
from bin2.prePrcoessingFunc import *

torch_device = gl.torch_device

#############
## Helpers ##
#############
def toVec(x):
    if isinstance(x,list):
        x = np.array(x)
        
    if len(x.shape)==2:
        return x 
    else :
        return x.reshape(-1,1)

#######################
## Eyeball Functions ##
#######################

def sampleGenerator(data,model,maxObservations=False):
    if ~maxObservations:
        b = data.shape[0]
    else :
        b= maxObservations

    model.eval()
    obs = torch.tensor(data[:b,:]).float().to(torch_device)
    sims =  model.invertSample(obs)
    return sims


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


#This function preps the test data to be inserted into the model 
def std_norm_Logrize(panelData,continiousDict_input=[],incomeDict_input=[],indColDict=[],
                    incomeConsCol =[],contiCols=[0,1],indcols=[],typed='gaussian'):
    numCols = panelData.shape[1]
    numIncomeCols = numCols-len(contiCols)-len(indcols)
    indCols = panelData[:,indcols].copy()
    ageCapital = panelData[:,contiCols].copy()
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
    if indCols.shape[1]>0:
        return np.concatenate((indCols,ageCapital,incomeCons),axis=1)
    else :
        return np.concatenate((ageCapital,incomeCons),axis=1)

def plotHistDemonstration(lastCol,realData,model,contiCols=[0,1],indCols=[],incomeConCols=[],
                            incom_dic=[],contin_dic=[],binary_dic=[],maxObservations=False,
                            binNum=100,save=False,saveName=False,truncated=False):
    a = realData
    a = std_norm_Logrize(a,continiousDict_input=contin_dic,incomeDict_input=incom_dic,indColDict=binary_dic,
                            contiCols=contiCols,indcols=indCols)
    sims,_ = sampleGeneratorDynamic(a,model,finalCol=lastCol,maxObservations=maxObservations)
    sims = sims.cpu().detach().numpy()
    if truncated:
        _,bins,_ = plt.hist(sims[sims<truncated],bins=binNum,label='Fake Data');
        plt.hist(a[(sims<truncated).squeeze(1),lastCol],bins=bins,label='Real Data',alpha=0.5);
    else :
        _,bins,_ = plt.hist(sims,bins=binNum,label="Fake Data");
        plt.hist(a[:,lastCol],bins=bins,label="Real Data",alpha=0.5);
    plt.legend()
    if save  :
        fileName = 'plots/' + saveName + '.pdf'
        plt.savefig(fileName)  
    plt.show()


def monteCarloForR2andBeta(lastCol,realData,model,contiCols=[0,1],indCols=[],incomeConCols=[],
                            incom_dic=[],contin_dic=[],binary_dic=[],iterations=200,
                            verbose=False,save=False,saveName=False,
                            truncated=False,returnResult=False):
    adjR2_2 = []
    betas = []
    for i in notebook.tqdm(range(iterations),desc='Iteration', leave=True):
        # lastCol = -2
        a = realData
        a = std_norm_Logrize(a,continiousDict_input=contin_dic,incomeDict_input=incom_dic,indColDict=binary_dic,
                            contiCols=contiCols,indcols=indCols)
        # std_norm_Logrize(a,continCols,incomCols)
        sims,_ = sampleGeneratorDynamic(a,model,finalCol=lastCol,maxObservations=False)
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


def plotDensity(individualHistory,model,contiCols=[0,1],indCols=[],incomeConCols=[],
                                        incom_dic=[],contin_dic=[],binary_dic=[],
                                        predictingCols=[-1,-2],minValue=0,maxValue=100000,
                                        gridPoints=100,joint=True,save=False,saveName=False):
    #Generate the 
    analysisCols = np.sort(predictingCols)
    linspace = np.linspace(minValue,maxValue,gridPoints).reshape(-1,1)
    for ci,c in enumerate(analysisCols):
        observation = np.tile(individualHistory[:c],(gridPoints,1))
        observation = np.concatenate((observation,linspace),axis=1)
        for c2 in analysisCols[ci+1:]:
            observation = np.concatenate((observation,np.zeros((gridPoints,1))),axis=1)
        
        gridVar = std_norm_Logrize(observation,continiousDict_input=contin_dic,incomeDict_input=incom_dic,
                            indColDict=binary_dic,contiCols=contiCols,indcols=indCols)
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

        gridVar = std_norm_Logrize(observation,continiousDict_input=contin_dic,incomeDict_input=incom_dic,
                            indColDict=binary_dic,contiCols=contiCols,indcols=indCols)
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


###################################
# Generating Panel Data Functions #
###################################

## Not the most flexiable thingy. 
# Add a option to add columns 

def genPanelData(panelData,model,contiCols=[],indCols=[],incomeConCols=[],
                incom_dic=[],contin_dic=[],binary_dic=[],parameters={},
                R=(1+0.0051585),colList=[-2,-1],T = 5,forceBiSecConverge=False
                ):

    """
    Generates panel data by simulating future values based on a given model and initial data.

    Parameters:
    - panelData (np.ndarray): The initial panel data.
    - model: The model used for simulation.
    - contiCols (list): Indices of continuous columns.
    - indCols (list): Indices of indicator columns.
    - incomeConCols (list): Indices of income and consumption columns.
    - incom_dic, contin_dic, binary_dic (dict): Dictionaries for income/consumption, continuous, and binary data preprocessing.
    - parameters (dict): Additional model parameters.
    - R (float): The interest rate for capital updates.
    - colList (list): Column indices for simulated data updates.
    - T (int): Number of time periods to simulate.
    - forceBiSecConverge (bool): Flag to force convergence in bisection method.

    Returns:
    - tuple: (updated panel data, original age and capital data, flag indicating success or failure).
    """

    #CLIPS THE SIMULATED VALUES IF THEY ARE WAY OVER THE TOP 
    def clipper(col,maxIncome):
        return (col<maxIncome)*col + (col>=maxIncome)*maxIncome
    
    #Define auxilatry vars  
    numCols = panelData.shape[1]
    numIncomeCols = numCols-len(contiCols) - len(indCols)

    number_of_panelists = panelData.shape[0]
    ageCapital = panelData[:,contiCols].copy()
    OG_ageCapital = ageCapital.copy()
    constantIndicators = panelData[:,indCols].copy()    

    incomeCons = panelData[:,-numIncomeCols:].copy()
    
    #STDize Normize Continious Columns 
    ageCapital = standarize(ageCapital,contin_dic['tmean'],contin_dic['tstd'])
    ageCapital = normalize(ageCapital,contin_dic['tmax'],contin_dic['tmin'],typed='gaussian')
    
    #STDize Normize Income consumption 
    #Define auxilary vectors - this just due to historic reasons. 
    conincMEAN = np.array(incom_dic['tmean']*int(numIncomeCols/2))
    conincSTD = np.array(incom_dic['tstd']*int(numIncomeCols/2))
    conincMAX = np.array(incom_dic['tmax']*int(numIncomeCols/2))
    conincMIN = np.array(incom_dic['tmin']*int(numIncomeCols/2))    
    for c in range(incomeCons.shape[1]):
        incomeCons[:,c] = logTransformation(incomeCons[:,c])    
    incomeCons = standarize(incomeCons,conincMEAN,conincSTD)
    incomeCons = normalize(incomeCons,conincMAX,conincMIN,typed='gaussian')
    
    
    maxValueColMinus2 = logTransformation(parameters['maxValue_Income'])    
    maxValueColMinus2 = standarize(maxValueColMinus2,incom_dic['tmean'][0],incom_dic['tstd'][0])
    maxValueColMinus2 = normalize(maxValueColMinus2,incom_dic['tmax'][0],incom_dic['tmin'][0],typed='gaussian')

    maxValueColMinus1 = logTransformation(parameters['maxValue_Consumption'])    
    maxValueColMinus1 = standarize(maxValueColMinus1,incom_dic['tmean'][1],incom_dic['tstd'][1])
    maxValueColMinus1 = normalize(maxValueColMinus1,incom_dic['tmax'][1],incom_dic['tmin'][1],typed='gaussian')

    incConMaxValues = [maxValueColMinus2,maxValueColMinus1]

    #Define the sample 
    sample = np.concatenate((constantIndicators,ageCapital,incomeCons),axis=1)

    for t in range(1,T):
        # print('this time',t)
        ageCapital = sample[:,contiCols]
        incomeCons = sample[:,-(numIncomeCols-len(colList)):] 
        placeHolders = np.zeros((sample.shape[0],len(colList)))
        a = np.concatenate((constantIndicators,ageCapital,incomeCons,placeHolders),axis=1)

        
        mask = ~np.isreal(a)
        rows_with_nonreal = a[mask.any(axis=1)]
        if rows_with_nonreal.shape[0]>0:
            print("non real observations",rows_with_nonreal.shape,rows_with_nonreal)
        if forceBiSecConverge:
            for c in colList:
                # print('this is columns',c)
                flag = 0 
                iter = 0 
                while flag==0 and iter <= forceBiSecConverge:
                    # print('this is iter',iter)
                    maxDraws = 3
                    retryDraws = 0 
                    while retryDraws < maxDraws  : #Not sure this is useful because it fails due to draws that were made in the previous sample generated  (this tackle isseus when the model returns Nans)
                        if retryDraws>1 :
                            print('number of tries',retryDraws)
                        try : 
                            # print('am i here 1 ?')
                            sims,didConverged = sampleGeneratorDynamic(a,model,finalCol=c)
                            #Compares consumption to the max value of a column and clips values 
                            sims = clipper(sims,incConMaxValues[c])
                            retryDraws = maxDraws+1
                            # print('number of tries after update ',retryDraws)
                            flag = np.min(didConverged).astype(int) #if even one of the individuals did not converged then redo the analysis 
                            # if retryDraws >1 :
                                # print('this is the flag',flag)
                        except Exception as e:
                            #Save Data 
                            rn = random.randrange(1000)
                            # with open('D:/Dropbox/Dropbox/uchicago_fourth/uncertaintyInequality/dataInModelThatIsBad_{}.pkl'.format(rn),'wb') as f:
                            #     pickle.dump(a, f)
                            print(str(e))
                            print('am i here 2 ?')
                            retryDraws  += 1
                            flag = 0 
                    
                    if  retryDraws !=  maxDraws+1 : #If the function failed for 3 times we return nothing and move on. It reached a point of not stable 
                        return None, None,0
                    
                    if flag==0:                        
                        print('here we are' , flag,didConverged)
                        if np.ndim(didConverged) ==0 :
                            iter += 1
                            print(('entering iteration {}').format(iter) )
                            continue

                        if didConverged.shape[0]>0:
                            if a[didConverged,:].shape[0]>0:                                
                                #Drop individuals who draws outside of the support (In the future: Need to add ability to count failures for each individual)
                                a = a[didConverged,:]
                                sample = sample[didConverged,:]
                                ageCapital = sample[:,contiCols] #Will be needed later in the last iteration so it's updated  here  as well
                                constantIndicators = constantIndicators[didConverged,:]
                                OG_ageCapital = OG_ageCapital[didConverged,:]
                                print('we dropped this amount of people', np.sum(~didConverged))
                            
                        iter += 1
                        print(('entering iteration {}').format(iter) )
                
                if flag == 0 :
                    print("Im out without sucess")
                    return None, None,0

                sims = sims.cpu().detach().numpy()
                a[:,c] = sims.squeeze()

        else :
            for c in colList:
                sims,flag = sampleGeneratorDynamic(a,model,finalCol=c)
                sims = sims.cpu().detach().numpy()
                a[:,c] = sims.squeeze()


        sample = np.concatenate((sample,a[:,colList]),axis=1)

        #Update age 
        ageCapital = normalize(ageCapital,contin_dic['tmax'],contin_dic['tmin'],'gaussian',inv=True)
        ageCapital = standarize(ageCapital,contin_dic['tmean'],contin_dic['tstd'],inv=True)

        ageCapital[:,0]  =ageCapital[:,0] + 1 ## add one month to age - notice that this is not super flexiable

        # update capital 
        capFlows = sample[:,-4:-2] #add the last last two columns  The old version old -numIncomeCols:-(numIncomeCols-len(colList))
        capFlows = normalize(capFlows,conincMAX[0:len(colList)],conincMIN[0:len(colList)],'gaussian',inv=True)
        capFlows = standarize(capFlows,conincMEAN[0:len(colList)],conincSTD[0:len(colList)],inv=True)
        for c in range(capFlows.shape[1]):
            capFlows[:,c] = logTransformation(capFlows[:,c],inv=True)
        ageCapital[:,1]  = ageCapital[:,1]*R + capFlows[:,0] - capFlows[:,1] # this might be a problem down the road. but currently OK... 

        #Normalize/Std again
        ageCapital = standarize(ageCapital,contin_dic['tmean'],contin_dic['tstd'],inv=False)
        ageCapital = normalize(ageCapital,contin_dic['tmax'],contin_dic['tmin'],'gaussian',inv=False)

        sample[:,contiCols] = ageCapital
    # print('returning whats the correct thing' , sample[0,0])
    return sample, OG_ageCapital,1


def invData(sample,contiCols=[],indCols=[],incomeConCols=[],
                incom_dic=[],contin_dic=[],binary_dic=[],OGageCapital=None,
            incomeUB=1e14,consumptionUB=1e14,incomeLB=0,consumptionLB=1000):
    '''
    gets data after standarization and returns and invert it
    '''
    constantIndCols = sample[:,indCols]

    ageCapital_INV = normalize(sample[:,contiCols],contin_dic['tmax'],contin_dic['tmin'],'gaussian',inv=True)
    ageCapital_INV = standarize(ageCapital_INV,contin_dic['tmean'],contin_dic['tstd'],inv=True)

    #update capital 
    nonIncomeConsumptoinCols = len(indCols) + len(contiCols)
    coninc = sample[:,nonIncomeConsumptoinCols:]
    periods = int(coninc.shape[1]/2)

    conincMEAN = np.array(incom_dic['tmean']*periods)
    conincSTD = np.array(incom_dic['tstd']*periods)
    conincMAX = np.array(incom_dic['tmax']*periods)
    conincMIN = np.array(incom_dic['tmin']*periods)

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
        sample_INV = np.concatenate((constantIndCols,ageCapital_INV,coninc_INV),axis=1)
    else :
        sample_INV = np.concatenate((OGageCapital,constantIndCols,ageCapital_INV,coninc_INV),axis=1)
    
    return sample_INV





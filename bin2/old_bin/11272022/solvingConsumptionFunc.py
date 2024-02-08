
import cvxpy as cp
import numpy as np
from tqdm import notebook
# from bin.sampleFunctions import *

'''
Things I need to add
- check consumption is positive. If not, re-solve we positivty constraint
- Introduce death. This would make things more reasonalbe
'''
class infomrationValueClass():
    def __init__(self,dataList,targetAge=40*12,maxAge=70*12, indCols=[],continiousCols=[],
                beta=0.995,R=1.005,flowUtility=False,Solver='ECOS',verbose=False):
        self.verbose = verbose
        self.Solver=Solver
        self.beta = beta
        self.R = R
        self.finData = dataList
        self.targetAge = targetAge 
        self.maxAge = maxAge
        self.indCols = indCols
        self.continiousCols = continiousCols
        self.numNonIncomeConsumptionCols = len(self.indCols) + len(self.continiousCols)
        
        if not flowUtility:
            self.flowUtility = lambda v: (v**(1-2))/(1-2) #Default utiltiy function is v
        else :
            self.flowUtility = lambda v: flowUtility(v,numpy=False)

        if not flowUtility:
            self.flowUtilityNumpy = lambda v: (v**(1-2))/(1-2) #Default utiltiy function is v
        else :
            self.flowUtilityNumpy = lambda v: flowUtility(v,numpy=True)

        #these are defined on call
        #
        self.incomeStreams = None 
        self.consumptionStreams = None 
        self.initalCapital = None 
        self.lifecycles = None #self.incomeStreams.shape[0]
        self.periods =  None #self.incomeStreams.shape[1]
        self.discountMatrix = None #np.tile(np.power(betas,np.arange(0,self.periods)),(self.lifecycles,1))
        
        self.negativeLifeTimeAssets = None
        self.balancedData = None        
        self.optimalUtil_val = None 
        self.realizedConsumption_val  = None 
        self.informationValue_val = None
        self.optimC = None 

    def dataValidation(self):
        '''
        We check that the total assets+income over the lifetime is greater than the consumption. 
        I don't have a way to impose it directly in the estimation. 
        '''
        numCols = self.balancedData.shape[1]-self.numNonIncomeConsumptionCols
        
        assets = self.balancedData[:,self.continiousCols][:,1] 
        incomeCols = self.balancedData[:,self.numNonIncomeConsumptionCols+np.arange(0,numCols,2)]
        consumptionCols = self.balancedData[:,self.numNonIncomeConsumptionCols+np.arange(1,numCols,2)]        
        self.negativeLifeTimeAssets = (assets + np.sum(incomeCols - consumptionCols,axis=1))>=0
        self.balancedData = self.balancedData[self.negativeLifeTimeAssets,:]


    def createBalancedPanel(self):
        balancedData = []
        # for d in notebook.tqdm(self.finData,desc="Creating Balanced Panel",leave=False):            

        counter = 0 

        for d in self.finData:
            counter += 1 
            cOG_ageCapital = d[0]
            cSample = d[1]
            cAge = int(cOG_ageCapital[0,0].item())
            constantCols = cSample[:,self.indCols]

            # if  counter < 10:
            #     continue
            if cAge > self.targetAge: #self.targetAge:
                print('this is where im at right now', cAge)
                break

            #get the "future" stream of incomes and consumption 
            FutureIncomeConsumptionStream = cSample[:,self.numNonIncomeConsumptionCols+(self.targetAge -cAge)*2:] #this two is going to cause problems 
            #Aggregate the past into the assets
            incomeConsumptionStream_upToTargetAge =  cSample[:,self.numNonIncomeConsumptionCols:self.numNonIncomeConsumptionCols+(1+self.targetAge-cAge)*2] #The plus 1 here is because we include the current age - Notice this will be issue one we change the consumption
            numcols = incomeConsumptionStream_upToTargetAge.shape[1]
            incomeCols = incomeConsumptionStream_upToTargetAge[:,np.arange(0,numcols,2)]
            consumptionCols = incomeConsumptionStream_upToTargetAge[:,np.arange(1,numcols,2)]

            #March assets forward 
            assets = cOG_ageCapital[:,1]
            for t in range(incomeCols.shape[1]):
                assets = assets*self.R + (incomeCols[:,t] - consumptionCols[:,t])
            assets = assets.reshape(-1,1)
            age = np.array([self.targetAge]*cOG_ageCapital.shape[0]).reshape(-1,1)
            dataForUncEstimation = np.concatenate((constantCols,age,assets,FutureIncomeConsumptionStream),axis=1)            
            balancedData.append(dataForUncEstimation)


        self.balancedData = balancedData
        balancedData = np.concatenate( balancedData, axis=0 )
        self.balancedData = balancedData
        self.dataValidation()
        
        # Update object 
        self.initalCapital = self.balancedData[:,self.continiousCols][:,1] 
        incomeConsumptionStreams = self.balancedData[:,self.numNonIncomeConsumptionCols:]
        self.incomeStreams = incomeConsumptionStreams[:,np.arange(0,incomeConsumptionStreams.shape[1],2)]
        self.consumptionStreams = incomeConsumptionStreams[:,np.arange(1,incomeConsumptionStreams.shape[1],2)] 
        self.lifecycles = self.incomeStreams.shape[0]
        self.periods =  self.incomeStreams.shape[1]
        betas = np.ones(shape=(1,self.periods))*self.beta
        self.discountMatrix = np.tile(np.power(betas,np.arange(0,self.periods)),(self.lifecycles,1))
        

        
    def optimalConsumption(self,returnValue = False): 
    # solve for the Optimal Consumption Path 
        
        if self.incomeStreams is None :
            self.createBalancedPanel() #If data was not inititated-  iniate it
            
        #Define Program
        C = cp.Variable(shape=(self.incomeStreams.shape))         #Consumption 
        A = cp.Variable(shape=(self.incomeStreams.shape[0],self.incomeStreams.shape[1]-1)) #Assets 

        atMinus1Matrix = cp.hstack([self.initalCapital.reshape(-1,1),A]) #Setting the inital wealth
        finalAssets = np.zeros(shape=(self.lifecycles,1)) #Consuming everything at the end 
        atMatrix = cp.hstack([A,finalAssets])
        #Set constraints 
        constraints = [C + atMatrix - atMinus1Matrix*self.R - self.incomeStreams==0 ] #Equality seems to do better numerically than <= 
        #Set objective 
        obj = cp.sum(cp.multiply(self.flowUtility(C),self.discountMatrix))
        objExpression = cp.Maximize(obj)
        prob = cp.Problem(objExpression, constraints)
        #Solve
        try :
            prob.solve(solver=self.Solver,verbose=self.verbose)
        except:
            print('first try didnt wokrk' )
        if prob.status != 'optimal':
            constraints = [C + atMatrix - atMinus1Matrix*self.R - self.incomeStreams<=0 ] #Sometimes seems to help numerical issues 
            prob = cp.Problem(objExpression, constraints)
            prob.solve(solver=self.Solver,verbose=self.verbose)
        
        optimalUtil = self.flowUtilityNumpy(C.value)*self.discountMatrix
        self.optimalUtil_val = [np.mean(np.sum(optimalUtil,axis=1)),np.sum(optimalUtil,axis=1)]
        self.optimC = C.value
        if returnValue :
            return self.optimalUtil_val

    def realizedConsumption(self,returnValue = False):
        # Accumelate assets over time till the end - change consumption at last year to consume it all
        #Need to think more on death 

        #Adjusting consumption to consume everything in the last period
        assets = self.initalCapital
        for t in range(self.incomeStreams.shape[1]-1):
            assets = assets*self.R + (self.incomeStreams[:,t] - self.consumptionStreams[:,t])
        
        self.consumptionStreamsAdjusted = self.consumptionStreams
        self.consumptionStreamsAdjusted[:,-1] = np.maximum(assets,0.001)
        
        #Avg Realized Utility         
        realizedUtil = self.flowUtilityNumpy(self.consumptionStreamsAdjusted)*self.discountMatrix
        self.realizedConsumption_val = [np.mean(np.sum(realizedUtil,axis=1)),np.sum(realizedUtil,axis=1)]
        if returnValue :
            return self.realizedConsumption_val
        
    
    def informationCost(self,returnValue = False):
        # if self.optimalUtil_val is None :
        #     self.optimalConsumption(returnValue = False)
        if self.realizedConsumption_val is None :
            self.realizedConsumption(returnValue = False)
    
        self.informationValue_val = self.optimalUtil_val[0]- self.realizedConsumption_val[0]
        if returnValue:
            self.informationValue_val


############################
## ADMM SOLVER - NOT USED ##
############################
'''
## ADMM trial - didn't pan out I think. not sure why
The reaosn it didn't pan out was that it generated lower value than the observed value from consumption. 
Trying the save with a simple convex solver didn't gave me that. not sure I'm implementing wrong or is it just hard to converge
'''
# def logUtilityFunc(v):
#     return torch.log(v)


# def quadraticUtilityFunc(v):
#     return -(v-torch.mean(v))**2

# def CRRA(v):
#     return -(v**(1-2))/(1-2)

    
# class completeInformationValues(nn.Module):
#     '''
#     the algorithm is ADMM and you can read it in 
#     MBIP-book-ADMM-Best.pdf
#     '''
#     def __init__(self,initalAssets,beta,R,incomeStreams,flowUtility,consumptionStreams=[],a=100):
#         super().__init__()
#         self.a = a 
#         self.R = R 
#         self.incomeStreams = torch.tensor(incomeStreams)
#         self.consumptionStreams = torch.tensor(consumptionStreams) if len(consumptionStreams)>0 else 0 
#         self.periods = self.incomeStreams.shape[1]
#         self.lifecycles = self.incomeStreams.shape[0]
#         if initalAssets.shape[0]==1:
#             self.initalAssets = torch.ones(size=(self.lifecycles,1))*initalAssets
#         else :
#             self.initalAssets = torch.tensor(initalAssets).unsqueeze(1)
#         self.finalAssets = torch.zeros(size=(self.lifecycles,1))
#         self.flowUtility = flowUtility

#         C = torch.ones(size=self.incomeStreams.shape)*1
#         A = torch.ones(size=(self.incomeStreams.shape[0],self.incomeStreams.shape[1]-1))*0.01
#         self.C = nn.Parameter(C)
#         self.A = nn.Parameter(A)
        
#         betas = torch.ones(size=(1,self.periods))*beta
#         self.discountMatrix = torch.tile(torch.pow(betas,torch.arange(0,self.periods)),(self.lifecycles,1))

#     def realizedUtility(self):
#         utility_realized = self.flowUtility(self.consumptionStreams)*self.discountMatrix
#         return torch.sum(utility_realized,dim=1).mean(), torch.sum(utility_realized,dim=1)

        
#     def utility_and_penalty(self,u):
#         atMatrix = torch.cat((self.A,self.finalAssets),dim=1) #Matrix from 0 to T
#         atMinus1Matrix = torch.cat((self.initalAssets ,self.A),dim=1) #Matrix from 0 to T-1 
#         cost = self.C +  atMatrix - self.R*atMinus1Matrix-self.incomeStreams + u
#         return self.flowUtility(self.C)*self.discountMatrix, self.a*torch.pow(cost,2), cost
    
#     def loss(self,u):
#         utility, penalty,_ = self.utility_and_penalty(u)
#         lossValue =   -(torch.sum(utility)) + torch.sum(penalty)
#         return lossValue

#     def getErrors(self,u):
#         _, _,cost = self.utility_and_penalty(u)
#         return cost-u

#     def getFinalUtils(self,realized=False):
#         utilities = dict({
#             'meanOptimalUtil':None,
#             'vecOptimalUtil':None,
#             'meanRealizedUtil':None,
#             'vecRealizedUtil':None
#             })

#         utility, _ ,_= self.utility_and_penalty(torch.zeros(size=self.incomeStreams.shape))
#         utilities['meanOptimalUtil'] = torch.sum(utility,dim=1).mean()
#         utilities['vecOptimalUtil'] = torch.sum(utility,dim=1)

#         if realized:
#             meanRealizedUtility,vecRealizedUtility = self.realizedUtility()
#             utilities['meanRealizedUtil'] = meanRealizedUtility
#             utilities['vecRealizedUtil'] = vecRealizedUtility
            
#         return utilities



# def ADMMsolver(model,optimizer,Algo="GD",uIterations=100,sgIterations=100,verbose=False,realizedUtils=False):

#     inital_u =torch.zeros(size=model.incomeStreams.shape,requires_grad=False)
#     if Algo == "GD":
#         u  =inital_u
#         for k in notebook.tqdm(range(uIterations)):
#             for i in range(sgIterations):
#                 loss = lifeCycleSolver.loss(u)
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#             errors = lifeCycleSolver.getErrors(u)
#             u = u + errors.detach()
#             if verbose:
#                 print(print(loss.item(),end="\r"))
#     if Algo== "LBFGS":
#         '''
#         A bit more on implementing the LBFGS
#         https://soham.dev/posts/linear-regression-pytorch/
#         https://johaupt.github.io/blog/pytorch_lbfgs.html
#         '''
#         u  =inital_u
#         for k in notebook.tqdm(range(uIterations)):
#             def closure():
#                 optimizer.zero_grad()
#                 loss = lifeCycleSolver.loss(u) #Does not need to inset data, as the data is already in the model
#                 loss.backward()
#                 return loss 
#             optimizer.step(closure)
#             loss = closure()
#             if verbose:
#                 print(loss.item(),end="\r")
#             errors = lifeCycleSolver.getErrors(u)
#             u = u + errors.detach()
    
#     maxViolation = np.max(np.abs(errors.detach().cpu().numpy()))

#     utils = model.getFinalUtils(realized=realizedUtils)
#     return utils, maxViolation


# incomeStreams = sampleINV[:,np.arange(2,numCols,2)]
# consumptionStreams = sampleINV[:,np.arange(3,numCols,2)]
# OG_ageCapital[0,1]
# lifeCycleSolver = completeInformationValues(initalAssets=OG_ageCapital[0,1],beta=0.995,R=1.005,
#                                             incomeStreams=incomeStreams,flowUtility=logUtilityFunc,a=100) #quadraticUtilityFunc
# optimizer = optim.SGD(lifeCycleSolver.parameters(), lr=1e-3)
# # optimizer = optim.LBFGS(lifeCycleSolver.parameters(), max_iter=50,history_size=100)    
# inital_u =torch.zeros(size=incomeStreams.shape,requires_grad=False)
# utils ,maxViolation = ADMMsolver(lifeCycleSolver,optimizer,Algo="GD",uIterations=100)
# (utils,maxViolation)

import cvxpy as cp
import numpy as np
from tqdm import notebook
import mosek
np.random.seed(123)
# from Code.bin2.sampleFunctions import *
# /content/drive/MyDrive/UncertaintyCosts/Code/bin2/sampleFunctions.py

'''
Things I need to add
- check consumption is positive. If not, re-solve we positivty constraint
- Introduce death. This would make things more reasonalbe
'''
class infomrationValueClass():

    """
    This class is calculateד the value of information for individuals within a dataset, 
    based on their income and consumption data across different ages. It supports the estimation of 
    optimal consumption paths, realized consumption paths, and the computation of information cost 
    (difference in utility between optimal and realized consumption paths).

    Parameters:
    - dataList (list of np.ndarray): A list containing datasets for individuals or cohorts.
    - targetAge (int): The target age (in months) up to which the analysis is conducted.
    - maxAge (int): The maximum age (in months) considered in the lifespan.
    - indCols (list of int): Indices of indicator columns in the dataset.
    - continiousCols (list of int): Indices of continuous columns in the dataset.
    - beta (float): The discount factor used in utility calculations.
    - R (float): The interest rate used for capital accumulation.
    - flowUtility (callable, optional): A custom utility function, if different from the default CRRA utility.
    - Solver (str): The solver used for optimization in CVXPY.
    - verbose (bool): If True, enables verbose output during computations.
    - num_observations (int): The number of observations to consider in the analysis.

    The class provides methods to prepare data, calculate optimal consumption paths, realized consumption paths,
    and finally, the information cost.
    """
    def __init__(self,dataList,targetAge=40*12,maxAge=70*12, indCols=[],continiousCols=[],
                beta=0.995,R=1.005,flowUtility=False,Solver='ECOS',verbose=False,num_observations=100):
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

        self.num_observations = num_observations
        #these are defined on call
        #
        self.groupMembership = None 
        self.incomeStreams = None 
        self.consumptionStreams = None 
        self.initalCapital = None 
        self.lifecycles = None #self.incomeStreams.shape[0]
        self.discountMatrix = None #np.tile(np.power(betas,np.arange(0,self.periods)),(self.lifecycles,1))
        self.periods =  None
        self.boundaryConditions = None
        self.balancedData = None        
        self.optimalUtil_val = None 
        self.realizedConsumption_val  = None 
        self.informationValue_val = None
        self.optimC = None 
    
    def calculateBoundaryCondition(self,assetCol,incomeCols,ConsumptionCols,R):
        """
        Calculates the end-of-life asset levels based on initial assets, income, and consumption over time.
        
        Parameters:
        - assetCol (np.ndarray): The initial assets for each individual.
        - incomeCols (np.ndarray): Income streams for individuals.
        - ConsumptionCols (np.ndarray): Consumption streams for individuals.
        - R (float): The interest rate used for capital updates.
        
        Returns:
        - np.ndarray: The assets at the end of life for each individual.
        """

        assets = assetCol.copy()
        for t in range(incomeCols.shape[1]-1):
            assets = assets*R + (incomeCols[:,t] - ConsumptionCols[:,t])
        endOfLifeAssets = assets
        return endOfLifeAssets


    def dataValidation(self):
        '''
        Validates the dataset to ensure that the present value of assets plus income is at least as large as
        the present value of consumption. This method filters out individuals who do not meet this criterion due to noise.
        '''
        #Test Present Value Validation
        numCols = self.balancedData.shape[1]-self.numNonIncomeConsumptionCols
        assets = self.balancedData[:,self.continiousCols][:,1] 
        incomeCols = self.balancedData[:,self.numNonIncomeConsumptionCols+np.arange(0,numCols,2)]
        consumptionCols = self.balancedData[:,self.numNonIncomeConsumptionCols+np.arange(1,numCols,2)]        
        presetValuePositiveAssets = (assets + np.sum(incomeCols*self.discountMatrix - consumptionCols*self.discountMatrix,axis=1))>=0
        #Test for Future Value Validation - does it make sense that I need both? 
        endOfLifeAssets = self.calculateBoundaryCondition(assets,incomeCols,consumptionCols,self.R) #for the realized consumption
        endOfLivePositiveAssets = endOfLifeAssets>0

        self.boundaryConditions = presetValuePositiveAssets*endOfLivePositiveAssets
        
        self.balancedData = self.balancedData[self.boundaryConditions,:]


    def createBalancedPanel(self):
        """
        Prepares a balanced panel dataset that aligns income and consumption streams with the target age,
        adjusting for the accumulation of assets over time.
        """

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
            
        
        #restricting the number of units 
        balancedData = np.concatenate( balancedData, axis=0 )
        if self.num_observations is not None:
            balancedData = balancedData[np.random.choice(balancedData.shape[0],size=self.num_observations),:]
        
        # Create discount matrix for data validation 
        periods = int((balancedData[:,self.numNonIncomeConsumptionCols:].shape[1])/2)
        betas = np.ones(shape=(1,periods))*self.beta
        self.discountMatrix = np.tile(np.power(betas,np.arange(0,periods)),(balancedData.shape[0],1))
        self.balancedData = balancedData
        self.dataValidation()
        print(self.balancedData.shape)
        # Update object 
        self.groupMembership = self.balancedData[:,self.indCols]
        self.initalCapital = self.balancedData[:,self.continiousCols][:,1] 
        incomeConsumptionStreams = self.balancedData[:,self.numNonIncomeConsumptionCols:]
        self.incomeStreams = incomeConsumptionStreams[:,np.arange(0,incomeConsumptionStreams.shape[1],2)]
        self.consumptionStreams = incomeConsumptionStreams[:,np.arange(1,incomeConsumptionStreams.shape[1],2)] 
        self.lifecycles = self.incomeStreams.shape[0]
        self.periods =  self.incomeStreams.shape[1] #Updating Periods just in case 
        betas = np.ones(shape=(1,self.periods))*self.beta
        #Create new discount matrix 
        self.discountMatrix = np.tile(np.power(betas,np.arange(0,self.periods)),(self.lifecycles,1))
        

        
    def optimalConsumption(self,returnValue = False): 
        """
        Solves for the optimal consumption path that maximizes the utility of individuals given their income
        streams and initial capital, using convex optimization.
        
        Parameters:
        - returnValue (bool): If True, returns the calculated optimal utility values.
        
        Returns:
        - Optional[np.ndarray]: The optimal utility values, if returnValue is True.
        """

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
        constraints = [C + atMatrix - atMinus1Matrix*self.R - self.incomeStreams==0 ] #BC = Equality seems to do better numerically than <= 
        #Set objective 
        obj = cp.sum(cp.multiply(self.flowUtility(C),self.discountMatrix))
        objExpression = cp.Maximize(obj)
        prob = cp.Problem(objExpression, constraints)
        #Solve
        maxAttempts = 5 
        attempt = 0 
        success = "No"
        while success != 'optimal' and attempt < maxAttempts:
          if attempt == 0 :
            try :
                prob.solve(solver=self.Solver,verbose=self.verbose,mosek_params={mosek.iparam.presolve_eliminator_max_num_tries:0,mosek.iparam.intpnt_solve_form: mosek.solveform.dual}) 
            except:
                print('first try didnt wokrk' )
            if prob.status != 'optimal':
                constraints = [C + atMatrix - atMinus1Matrix*self.R - self.incomeStreams<=0 ] #Sometimes seems to help numerical issues 
                prob = cp.Problem(objExpression, constraints)
                prob.solve(solver=self.Solver,verbose=self.verbose,mosek_params={mosek.iparam.presolve_eliminator_max_num_tries:0,mosek.iparam.intpnt_solve_form: mosek.solveform.dual})

          else : 
            try :
                prob.solve(solver=self.Solver,verbose=True) 
            except:
                print('first try didnt wokrk' )
            if prob.status != 'optimal':
                constraints = [C + atMatrix - atMinus1Matrix*self.R - self.incomeStreams<=0 ] #Sometimes seems to help numerical issues 
                prob = cp.Problem(objExpression, constraints)
                prob.solve(solver=self.Solver,verbose=True)
                
          success = prob.status 
          attempt += 1
        
        print('Solved after '+ str(attempt) + ' attempts')
          
        ###################################################################
        if prob.status  != 'optimal':  
          print('I gave up')
          self.optimalUtil_val = [None,None ]
          return self.optimalUtil_val
        ################################################################
        
        optimalUtil = self.flowUtilityNumpy(C.value)*self.discountMatrix
        self.optimalUtil_val = [np.mean(np.sum(optimalUtil,axis=1)),np.sum(optimalUtil,axis=1)]
        self.optimC = C.value
        if returnValue :
            return self.optimalUtil_val

    def realizedConsumption(self,returnValue = False):
        """
        Calculates the utility of the realized consumption path, adjusting the final period consumption
        to ensure all assets are consumed.
        
        Parameters:
        - returnValue (bool): If True, returns the calculated realized utility values.
        
        Returns:
        - Optional[np.ndarray]: The realized utility values, if returnValue is True.
        """
        # Accumelate assets over time till the end - change consumption at last year to consume it all
        #Need to think more on death 
        #Adjusting consumption to consume everything in the last period

        assets = self.initalCapital
        assets= self.calculateBoundaryCondition(assets,self.incomeStreams,self.consumptionStreams,self.R)
        # for t in range(self.incomeStreams.shape[1]-1):
        #     assets = assets*self.R + (self.incomeStreams[:,t] - self.consumptionStreams[:,t])
        self.consumptionStreamsAdjusted = self.consumptionStreams
        self.consumptionStreamsAdjusted[:,-1] = np.maximum(assets,0.001)
        
        #Avg Realized Utility   
        print('am i here? calcling utilities?')      
        realizedUtil = self.flowUtilityNumpy(self.consumptionStreamsAdjusted)*self.discountMatrix
        self.realizedConsumption_val = [np.mean(np.sum(realizedUtil,axis=1)),np.sum(realizedUtil,axis=1)]
        if returnValue :
            return self.realizedConsumption_val
        
    
    def informationCost(self,returnValue = False):
        """
        Computes the information cost as the difference in utility between the optimal and realized
        consumption paths.
        
        Parameters:
        - returnValue (bool): If True, returns the information cost.
        
        Returns:
        - Optional[np.ndarray]: The information cost, if returnValue is True.
        """

        # 1==1
        if self.optimalUtil_val is None :
            self.optimalConsumption(returnValue = False)
        if self.realizedConsumption_val is None :
            self.realizedConsumption(returnValue = False)
        ############################################################
        if any(x is None for x in self.optimalUtil_val):
          self.informationValue_val = [None,None]
          if returnValue:
              return self.informationValue_val  
          return 
        ########################################################

        self.informationValue_val = [self.optimalUtil_val[0]- self.realizedConsumption_val[0],self.optimalUtil_val[1]- self.realizedConsumption_val[1]]
        if returnValue:
            return  self.informationValue_val


############################
## ADMM SOLVER - NOT USED ##
############################
'''
## ADMM trial - didn't work. 
The reaosn it didn't work was that it generated lower value than the observed value from consumption. 
Trying the save with a simple convex solver didn't gave me that. Seems that ADMM does not give good approximation
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

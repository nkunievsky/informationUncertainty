
#############################################################################
### Parallel code does not work - I'm not sure why yet 
### I'm running not in parallel on the server. Might be a bit faster ########
#############################################################################
import bin2.setGlobals as gl

import numpy as np
import pickle 
import multiprocessing as mp
from bin2.solvingConsumptionFunc import *
from functools import partial
import datetime
from mpi4py.futures import MPIPoolExecutor


lastAgeInAnalysis = gl.lastAgeInAnalysis
firstAgeAnalysis = gl.firstAgeAnalysis



# Load finData data
with open(gl.tempsFolder  + '/finData' + gl.fileNamesSuffix + '.pkl', 'rb') as inp:
    finData = pickle.load(inp)


def estimateCosts(age, dataList, indCols, continiousCols, beta, R, Solver='MOSEK', verbose=False, num_observations=100):
    """
    Estimates the cost of uncertainty associated for each age and caste

    Parameters:
    - age (int): The age for which the cost estimation is performed.
    - dataList: The dataset used for analysis.
    - indCols (list): List of column indices for individual-specific variables.
    - continiousCols (list): List of column indices for continuous variables.
    - beta (float): The discount factor used in the model.
    - R (float): The interest rate used in the model.
    - Solver (str): The solver used for optimization (default is 'MOSEK').
    - verbose (bool): Flag to control the verbosity of the output (default is False).
    - num_observations (int): The number of observations to consider in the analysis (default is 100).

    Returns:
    - list: A list containing the age, information value, optimal utility value, realized consumption value,
            optimal consumption, consumption streams, and group membership.
    """
    print("num observation is ", num_observations)
    print('hi this is the current age', age)
    informationValue = infomrationValueClass(dataList=dataList, targetAge=age, indCols=indCols, continiousCols=continiousCols,
                                             beta=beta, R=R, Solver=Solver, verbose=verbose, num_observations=num_observations)
    informationValue.informationCost()
    print('Im here now')
    now = datetime.datetime.now()
    with open(gl.procDataFolder + '/tracker/' + str(int(age)) + '.txt', 'w') as f:
        f.write(str(age))
        f.write("Current date and time : ")
        f.write(now.strftime("%Y-%m-%d %H:%M:%S"))

    print('and now Im here now')
    return [age, informationValue.informationValue_val, informationValue.optimalUtil_val, informationValue.realizedConsumption_val,
            informationValue.optimC, informationValue.consumptionStreams, informationValue.groupMembership]


if __name__ == '__main__':  

    now = datetime.datetime.now()
    with open(gl.procDataFolder + '/tracker/'+'TEST' + '.txt', 'w') as f:
        f.write('START TIME \n')
        f.write("Current date and time : \n")
        f.write(now.strftime("%Y-%m-%d %H:%M:%S"))

    print('Starting Parallel - Uncertainty Costs')  
    print(firstAgeAnalysis,lastAgeInAnalysis)

    with MPIPoolExecutor(max_workers=13) as executor3:  # 13
        m = executor3.map(partial(estimateCosts,dataList=finData,indCols=gl.binaryCols,continiousCols=gl.continiousCols,
                                        beta=gl.beta,R=gl.R,Solver='MOSEK',verbose=False,num_observations=None),range(firstAgeAnalysis ,lastAgeInAnalysis ))

    uncertaintyMeasures = list(m)

    with open(gl.tempsFolder  + '/uncertaintyMeasures' + gl.fileNamesSuffix + '.pkl', 'wb') as f:
        pickle.dump(uncertaintyMeasures,f)
    
    print(uncertaintyMeasures)
    

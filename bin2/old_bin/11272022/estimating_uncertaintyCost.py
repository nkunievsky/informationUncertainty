
import bin.setGlobals as gl

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import multiprocessing as mp
from bin.solvingConsumptionFunc import *
from functools import partial

# Load finData data
with open(gl.tempsFolder  + '/finData.pkl', 'rb') as inp:
    finData = pickle.load(inp)

def estimateCosts(age,dataList=finData,indCols=gl.binaryCols,continiousCols=gl.continiousCols,
                                    beta=gl.beta,R=gl.R,Solver='MOSEK',verbose=False):
    print(age)
    informationValue = infomrationValueClass(dataList=dataList,targetAge=age,indCols=indCols,continiousCols=continiousCols,
                                    beta=beta,R=R,Solver=Solver,verbose=verbose)
    informationValue.informationCost()
    return [age,informationValue.informationValue_val,informationValue.optimalUtil_val,informationValue.realizedConsumption_val,
            informationValue.optimC,informationValue.consumptionStreams]

if __name__ == '__main__':    
    p = mp.Pool(2) #mp.cpu_count()
    uncertaintyMeasures = []  
    p.map_async(partial(estimateCosts,dataList=finData,indCols=gl.binaryCols,continiousCols=gl.continiousCols,
                                        beta=gl.beta,R=gl.R,Solver='MOSEK',verbose=False),range(40*12,41*12),callback=uncertaintyMeasures.extend)

    p.close()
    p.join() 
    #The callback function is called only after we excute the close and join

    with open(gl.tempsFolder  + '/uncertaintyMeasures' + gl.fileNamesSuffix + '.pkl', 'wb') as f:
        pickle.dump(uncertaintyMeasures,f)
    
    plotValues = [[v[0],v[1]] for v in uncertaintyMeasures] 
    tt = [np.array(i).reshape(-1,1).T for i in plotValues]
    tt = np.concatenate(tt,axis=0)
    plt.plot(tt[:,0],tt[:,1])
    plt.show()


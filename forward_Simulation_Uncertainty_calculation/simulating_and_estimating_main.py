import os 
import numpy as np


# os.chdir('..')
print(os.getcwd())
#Gen Fin data 
print('starting Simulating')
exec(open("simulating_Individuals.py").read())
print('starting Estimating')
exec(open("estimating_uncertaintyCost_MPI.py").read())

#run The first file and save 
# run the second file and 


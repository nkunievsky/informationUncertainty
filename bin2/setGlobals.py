import os 
import torch 

# Define the working folder 

plotFolder = 'plots/'
procDataFolder = 'proc_data/'
tempsFolder = 'proc_data/temps'
modelsFolder = 'proc_data/models'

#Parameters 
binaryCols = [0,1,2,3]
continiousCols = [4,5]
beta = 0.995
R = 1.005 

lastAgeInAnalysis = 70*12
firstAgeAnalysis = 39*12

# from os.path import exists
# file_exists = exists(path_to_file)
# Load training Data and test data 
# Notice that this is used only if we run the training part seperatly, otherwise it's being overwritten 
# trainData = 
# Load Data for panel generation 
# Notice that this is used only if we run the generation file seperatly otherwise it's being overwritten 


#Outoutfile name suffix
fileNamesSuffix = 'basicModel_11272022'

#Define the working enviornment
global torch_device
torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(torch_device)

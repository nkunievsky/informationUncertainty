   
# import deepul.pytorch_util as ptu
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import bin2.setGlobals as gl 
global torch_device 
torch_device  = gl.torch_device

def safe_log(x):
    '''truncated log function'''
    return torch.log(x.clamp(min=1e-22))

def inverseLogistic(x, reverse=False): #We use this to shift 
    """Inverse logistic function."""
    if reverse:
        z = torch.sigmoid(x)
        ldj = F.softplus(x) + F.softplus(-x)
    else:
        z = -safe_log(x.reciprocal() - 1.)
        ldj = -safe_log(x) - safe_log(1. - x)

    return z, ldj


class Rescale(nn.Module): #Taken from here https://github.com/chrischute/flowplusplus/blob/master/models/flowplusplus/nn.py
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """

    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, 1))

    def forward(self, x):
        x = self.weight * x
        return x

class resnetBlock_normalDraws(nn.Module):
    """
    This class implements a ResNet block for the case where Z is generated by normal draws.
    It includes ReLU activation functions and Batch Normalization before the linear transformations, which is atypical
    but aimed at enhancing stability.

    Parameters:
    - in_dim (int): Input dimensionality of the block.
    - out_dim (int): Output dimensionality of the block.

    The block comprises a sequence of layers starting with a ReLU activation, followed by Batch Normalization, and then
    a Linear layer. This sequence is repeated twice, with the input dimensionality preserved in the first part and
    transitioned to the output dimensionality in the final Linear layer.
    """

    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.in_dim = in_dim 
        self.out_dim = out_dim
        
        self.theBlock = nn.Sequential(
            nn.ReLU(), #Switched the order here putting relu and then weights
            nn.BatchNorm1d(num_features=self.in_dim,momentum=0.1), #Seems very important for stability and preventing gradient explosion             
            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(), #Switched the order here putting relu and then weights
            nn.BatchNorm1d(num_features=self.in_dim,momentum=0.1), #Seems very important for stability and preventing gradient explosion             
            nn.Linear(self.in_dim, self.out_dim)
            )
        
    def forward(self,x):
        shortcut = x
        residual = self.theBlock(x)
        return residual + shortcut


class ResNetDynamicFLOW_normalDraws(nn.Module):
    """
    This class implements a dynamic ResNet architecture tailored for flows with normal draws. It uses a series of ResNet blocks
    followed by batch normalization and ReLU activation, aiming at enhancing model stability and preventing gradient explosion.

    Parameters:
    - input_size (int): Dimensionality of the input features.
    - n_blocks (int): Number of ResNet blocks to include in the model.
    - hidden_size (int): Dimensionality of the hidden layers within the ResNet blocks.
    - output_size (int): Dimensionality of the output layer.

    The model initializes weights using Kaiming Normal initialization for linear layers and sets batch normalization
    weights to 1 and biases to zero, which is beneficial for networks utilizing ReLU activation functions.
    """

    def __init__(self, input_size, n_blocks, hidden_size, output_size):
        super().__init__()
        #Initializing command
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu') #Good for ReLU activiation
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(n_blocks):
            layers.append(resnetBlock_normalDraws(hidden_size,hidden_size)) 
        layers.append(nn.BatchNorm1d(num_features=hidden_size,momentum=0.1)) #Seems very important for stability and preventing gradient explosion 
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)
        # Apply initializinztion 
        self.layers.apply(init_weights)
    
    def forward(self, x):
        return self.layers(x)


class resBlockAutoRegressiveDynamicFLOW_normalDraws(nn.Module):
    """
    Implements an autoregressive dynamic flow model with a configurable base distribution and mixture distribution for
    normal draws. 

    Parameters:
    - mixture_dist (str): The type of mixture distribution to use, currently supports 'gaussian'.
    - condVecSize (int): The size of the conditional vector.
    - n_components (int): The number of components in the mixture distribution.
    - mlp_hidden_size (int): The hidden layer size for the MLPs within the model.
    - n_blocks (int): The number of blocks in each MLP.
    - lastColumnToPredict (list of int): Indices of the last columns to predict, allowing for selective modeling.
    - base (str): The base distribution of the flow, supports 'Uniform' or 'gaussian'.

    The model dynamically generates a series of MLPs based on the specified last columns to predict, allowing for
    autoregressive modeling of conditional distributions. It uses a mixture of distributions for flexibility in
    modeling complex data patterns.
    """

    def __init__(self, 
                 mixture_dist='gaussian', 
                 condVecSize = 20,
                 n_components=5,
                 mlp_hidden_size=64,
                 n_blocks=3,
                 lastColumnToPredict = [-1],
                 base='Uniform'):
        super().__init__()
        
        self.lastColumnToPredict = np.sort(lastColumnToPredict)
        self.n_components = n_components        
        self.condVecSize = condVecSize
        self.rescale= Rescale()
        self.rescaleAs= torch.nn.Parameter(torch.ones(1,1))
        
        if base == "Uniform":
            self.base_dist = Uniform(torch.tensor(0.0).to(torch_device), torch.tensor(1.0).to(torch_device)) 
        if base == "gaussian":
            self.base_dist = Normal(torch.tensor(0.0).to(torch_device), torch.tensor(1.0).to(torch_device)) 
        #.to(torch_device) #I'm adding epsilon slack, becuase sometimes we have value of f(x) that gives 1 - mainly because this a cdf and numerical errors 
    
        if mixture_dist == 'gaussian':
            self.mixture_dist = Normal
            self.mixture_distName = 'gaussian'            
        elif mixture_dist == 'logit':
            raise NotImplementedError
        
        #Generate the NN for each var
        self.mlps = nn.ParameterList() #parameter llist 
        for ci, c in enumerate(self.lastColumnToPredict):
            condSize = self.condVecSize +ci 
            self.mlps.append(ResNetDynamicFLOW_normalDraws(condSize, n_blocks, mlp_hidden_size, n_components*3 +2).to(torch_device))
    #####################
    #Inversion Functions#
    #####################
    def invertSample(self,x,finalCol,lb=-20,ub=20):
        """
        Inverts a sample from the model output back to its original space, using a bisection method for inversion.
        
        Parameters:
        - x (Tensor): The input tensor to the model.
        - finalCol (int): The final column index to be predicted.
        - lb (float): The lower bound for the bisection method.
        - ub (float): The upper bound for the bisection method.
        
        Returns:
        - Tensor: The inverted sample.
        - int: A flag indicating whether the inversion was successful.
        """
 
        
        def bisectionInvert(func,objVec,a,b,*args):
            #Fixed parameters
            size = objVec.shape[0]
            eps = 1e-5 
            diff = 150 
            counter = 0
            aVec = torch.ones((size,1)).to(torch_device)*a
            bVec = torch.ones((size,1)).to(torch_device)*b
            
            while diff>eps and counter <= 150:
                c = (aVec+bVec)/2
                value = func(c,*args).unsqueeze(1)-objVec
                gt = value >0
                lt = ~gt 
                bVec = gt*c+lt*bVec
                aVec = gt*aVec+lt*c
                diff = torch.abs(value).max()
                counter += 1 
            
            if diff > eps:
                print(diff, counter,' the bisection algo did not converge')
                whoConverged = torch.abs(value).detach().cpu().numpy()<eps
                return c,whoConverged.squeeze() #Flags that it did not converge 
                
            return c,1 #Flags that it did not converge 
            
        def func(y,*args):
            self = args[0]
            loc = args[1]
            std = args[2]
            weights = args[3]            
            
            yVec = y.repeat((1,self.n_components))
            return (self.mixture_dist(loc, std).cdf(yVec)*weights).sum(dim=1)
        
        
        ###############
        #Start Module #
        ###############
        xCond = x[:,:finalCol]
        whichNN = np.where(self.lastColumnToPredict==finalCol)[0].item()
        batchSize = x.shape[0]
        with torch.no_grad():
            zs = self.base_dist.rsample(torch.tensor([batchSize])).to(torch_device).unsqueeze(1)            
            loc, log_scale, weight_logits,shifters = torch.split(self.mlps[whichNN](xCond),[self.n_components,self.n_components,self.n_components,2],dim=1)            
            weights = F.softmax(weight_logits -weight_logits.max(dim=1,keepdim=True)[0], dim=1)
#             As = shifters[:,0].exp().unsqueeze(1)
            As = (torch.tanh(shifters[:,0].unsqueeze(1))*self.rescaleAs).exp() #.unsqueeze(1)
            Bs = shifters[:,1].unsqueeze(1)
            std = self.rescale(torch.tanh(log_scale)).exp()

            zs = torch.sigmoid((zs-Bs)/As) #Get the inverse of the invSig(x)a + b which is sig((z-b)/a)
            drawnX,flag = bisectionInvert(func,zs,lb,ub,self,loc,std,weights) 
            return drawnX, flag
        
        
    ##################
    # Flow Functions #
    ##################
    def flow(self, x,verbose=False):
        """
        Processes the input through the flow model, transforming it according to the learned distribution mappings.
        
        Parameters:
        - x (Tensor): The input tensor.
        - verbose (bool): If True, returns additional diagnostic information.
        
        Returns:
        - Tensor: The transformed tensor.
        - Tensor: The log determinant of the Jacobian of the transformations.
        """

        x = x.float()
        zs = []
        log_dets = [] 
        for ci,c in enumerate(self.lastColumnToPredict):
            xCond = x[:,:c]
            y = x[:,c].unsqueeze(1)
            # individually flow on each dim          
            loc, log_scale, weight_logits,shifters = torch.split(self.mlps[ci](xCond),[self.n_components,self.n_components,self.n_components,2],dim=1)
            weights = F.softmax(weight_logits -weight_logits.max(dim=1,keepdim=True)[0], dim=1)
            std = self.rescale(torch.tanh(log_scale)).exp()
            
#             As = shifters[:,0].exp().unsqueeze(1)
            As = (torch.tanh(shifters[:,0].unsqueeze(1))*self.rescaleAs).exp() #.unsqueeze(1)
            
            Bs = shifters[:,1].unsqueeze(1)
            
            #adding the samples
            z_01 = (self.mixture_dist(loc, std).cdf(y.repeat(1, self.n_components)) * weights).sum(dim=1).unsqueeze(1)
            logistic_z,logistic_z_derivative = inverseLogistic(z_01)
            zs.append(logistic_z*As+Bs)

            #Adding samples 
            cdfDerivative = torch.clamp((self.mixture_dist(loc, std).log_prob(y.repeat(1, self.n_components)).exp() * weights).sum(dim=1).log().unsqueeze(1),min=-1e20)
            log_dets.append(cdfDerivative + logistic_z_derivative + As.log())

            
        if verbose :
            return torch.cat(zs,dim=1), torch.cat(log_dets,dim=1), As, Bs,cdfDerivative , logistic_z_derivative, z_01
        else :
            return torch.cat(zs,dim=1), torch.cat(log_dets,dim=1)

    def log_prob(self, x,verbose=False):
        """
        Computes the log probability of the input data under the model.
        """
        if verbose:
            z, log_det = self.flow(x)
            return (self.base_dist.log_prob(z) + log_det)
        else :
            z, log_det = self.flow(x,verbose=verbose)
            return (self.base_dist.log_prob(z) + log_det).mean(dim=1) # shape: [batch_size, dim] - changed thiS!

    def nll(self, x):
        """
        Computes the negative log-likelihood of the input data under the model.
        """
        return -self.log_prob(x).mean()
    

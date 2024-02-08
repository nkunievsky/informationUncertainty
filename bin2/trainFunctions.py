# import deepul.pytorch_util as ptu
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LR_scheduler
from tqdm import notebook

import bin2.setGlobals as gl
torch_device = gl.torch_device



def trainDynamic(model, train_loader, optimizer):
    model.train()
    losses = []
#     cc = 0 
    for x in train_loader:
        x = x.to(torch_device).float() #.contiguous()
#         x = x.float() #.contiguous()
        loss = model.nll(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def eval_lossDynamic(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.to(torch_device).float() #.contiguous()
#             x = x.float() #.contiguous()
            loss = model.nll(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss.item()

def train_epochsDynamic(model, train_loader, test_loader, train_args,checkPointSave=None):
    epochs, lr , schedularInd= train_args['epochs'], train_args['lr'],train_args['schedular']    
    optimizer = optim.Adam(model.parameters(), lr=lr)    
    if schedularInd :
        stepSize = 400
        scheduler = LR_scheduler.StepLR(optimizer, step_size=stepSize, gamma=0.1, last_epoch=-1, verbose=False)
    
    train_losses_eval, train_losses, test_losses = [],[], []
    test_loss = eval_lossDynamic(model, test_loader)
    test_losses.append(test_loss)  # loss at init
    for epoch in notebook.tqdm(range(epochs), desc='Epoch', leave=True):
        epoch_train_losses = trainDynamic(model, train_loader, optimizer) 
        train_losses.extend(epoch_train_losses)
        
        test_loss = eval_lossDynamic(model, test_loader)
        test_losses.append(test_loss)
        
        train_losses_e = eval_lossDynamic(model, train_loader)
        train_losses_eval.append(train_losses_e)
        
        if schedularInd :
            scheduler.step()
            if np.mod(epoch,stepSize) == 0:
                print(optimizer.state_dict()['param_groups'][0]['lr'])

        if isinstance(checkPointSave, list):
            if epoch in checkPointSave:
                filename = 'proc_data/models/modelCheckpoint_{}.pt'.format(epoch)
                torch.save(model, filename)


            
    return train_losses, test_losses, train_losses_eval
import torch 

global torch_device
torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch_device)

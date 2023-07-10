import torch
from torch.utils.data import Dataset

class ProtospacerDataset(Dataset):
    def __init__(self, X_proto, targets, xproto_dtype, ydtype=torch.float32):
        # TODO restructure the dataset to add seqids, etc..

        # for now this stands for the X_protospacer
        self.X_proto = torch.tensor(X_proto).type(xproto_dtype) 
        self.targets = torch.tensor(targets).type(ydtype)
    
    def __len__(self):
        return len(self.X_proto)
    
    def __getitem__(self, index):
        x = self.X_proto[index]
        y = self.targets[index]
        
        return x, y
    
class ProtospacerExtendedDataset(Dataset):
    def __init__(self, X_proto, X_feat, targets, xproto_dtype, xfeat_dtype, ydtype=torch.float32):
        # TODO restructure the dataset to add seqids, etc..
        
        # for now this stands for the X_protospacer
        self.X_proto = torch.tensor(X_proto).type(xproto_dtype) 
        self.X_feat = torch.tensor(X_feat).type(xfeat_dtype)
        self.targets = torch.tensor(targets).type(ydtype)
    
    def __len__(self):
        return len(self.X_proto)
    
    def __getitem__(self, index):
        x = self.X_proto[index]
        x_feat = self.X_feat[index]
        y = self.targets[index]
        
        return x, x_feat, y
    
class PartitionDataTensor(Dataset):

    def __init__(self, criscas_datatensor, partition_ids, dsettype, run_num):
        self.criscas_datatensor = criscas_datatensor  # instance of :class:`CrisCASDataTensor` or :`CrisCasSeqDataTensor`
        self.partition_ids = partition_ids  # list of sequence indices
        self.dsettype = dsettype  # string, dataset type (i.e. train, validation, test)
        self.run_num = run_num  # int, run number
        self.num_samples = len(self.partition_ids[:])  # int, number of docs in the partition

    def __getitem__(self, indx):
        target_id = self.partition_ids[indx]
        return self.criscas_datatensor[target_id]

    def __len__(self):
        return(self.num_samples)
    
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = torch.Tensor(self.data[index])
        y = torch.Tensor([self.targets[index]])
        
        return x, y
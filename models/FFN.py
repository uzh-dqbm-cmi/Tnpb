## feed forward network based model is defined here
import torch
import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader
#import torch.optim as optim
#import itertools
#from sklearn.model_selection import KFold
#from src.utils import ReaderWriter, create_directory, perfmetric_report_cont,compute_spearman_corr,compute_pearson_corr,performance_statistics
#from src.utils import ReaderWriter
#from .dataset import CustomDataset
#import numpy as np
import torch.nn.init as init
# Define neural network model
class RegressionFFNN(nn.Module):
    def __init__(self,input_size, hidden_size, mlp_embedder_config=None):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size[0])
        self.fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
        
        
        
        
        
        if mlp_embedder_config is not None:
            self.mlp_embedder = MLPEmbedder(mlp_embedder_config.input_dim, 
                                            mlp_embedder_config.embed_dim, 
                                            mlp_embed_factor=mlp_embedder_config.mlp_embed_factor,
                                            nonlin_func=mlp_embedder_config.nonlin_func, 
                                            pdropout=mlp_embedder_config.p_dropout, 
                                            num_encoder_units=mlp_embedder_config.num_encoder_units)
            
            self.lastlayer_inputdim = hidden_size[1] + mlp_embedder_config.embed_dim
        else:
            self.lastlayer_inputdim = hidden_size[1]
        
        self.fc3 = torch.nn.Linear(self.lastlayer_inputdim, 1,bias=True)
        #self.Wy = nn.Linear(self.lastlayer_inputdim, 1, bias=True)

        self._init_params_()
        
        '''
        # initialize the layers
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.1)
        '''
    def _init_params_(self):
        # Initialize the parameters with Kaiming uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
               
                init.kaiming_uniform_(m.weight)
                init.zeros_(m.bias)
            #else:
                #print('did not intialized')
        
    def forward(self, x_proto, x_feat=None):
        
        z = torch.nn.functional.relu(self.fc1(x_proto))
        z_1 = torch.nn.functional.relu(self.fc2(z))
        
        
        
        if x_feat is not None:
            z_2 = self.mlp_embedder(x_feat)
            z_final = torch.cat((z_1, z_2), dim = -1)
        else:
            z_final = z_1
            
        output = self.fc3(z_final)
        
        return output.squeeze(-1)
        #return output
    
    
    
    
class MLPBlock(nn.Module):
            
    def __init__(self,
                 input_dim,
                 embed_dim,
                 mlp_embed_factor,
                 nonlin_func, 
                 pdropout):
        
        super().__init__()
        
        assert input_dim == embed_dim

        self.layernorm_1 = nn.LayerNorm(embed_dim)

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, embed_dim*mlp_embed_factor),
            nonlin_func,
            nn.Linear(embed_dim*mlp_embed_factor, embed_dim)
        )
        self.dropout = nn.Dropout(p=pdropout)

    def forward(self, X):
        """
        Args:
            X: input tensor, (batch, sequence length, input_dim)
        """
        o = self.MLP(X)
        o = self.layernorm_1(o + X)
        o = self.dropout(o)
        return o

    
class MLPEmbedder(nn.Module):
    def __init__(self,
                 inp_dim,
                 embed_dim,
                 mlp_embed_factor=2,
                 nonlin_func=nn.ReLU, 
                 pdropout=0.3, 
                 num_encoder_units=2):
        
        super().__init__()
        
        nonlin_func = nonlin_func()
        self.We = nn.Linear(inp_dim, embed_dim, bias=True)
        encunit_layers = [MLPBlock(embed_dim,
                                   embed_dim,
                                   mlp_embed_factor,
                                   nonlin_func, 
                                   pdropout)
                          for i in range(num_encoder_units)]

        self.encunit_pipeline = nn.Sequential(*encunit_layers)

    def forward(self, X):
        """
        Args:
            X: tensor, float32, (batch, embed_dim) representing x_target
        """

        X = self.We(X)
        out = self.encunit_pipeline(X)
        return out
    
    

class FFNHyperparamConfig:
    def __init__(self, h, l2_reg, batch_size, num_epochs):
        self.h = h
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs


    def __repr__(self):
        desc = " h:{}\n l2_reg:{} \n batch_size:{} \n num_epochs: {}".format(self.h,
                                                                             self.l2_reg, 
                                                                             self.batch_size,
                                                                             self.num_epochs)
        return desc  

import torch
import torch.nn.init as init
import torch.nn as nn


class PredictionCNN(nn.Module):
    def __init__(self, k, dictionary = 4, mlp_embedder_config=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=dictionary, out_channels=32, kernel_size=k)
        #self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2)
        #self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2)
        #self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=128 * 17, out_features=64)


        if mlp_embedder_config is not None:
            self.mlp_embedder = MLPEmbedder(mlp_embedder_config.input_dim, 
                                            mlp_embedder_config.embed_dim, 
                                            mlp_embed_factor=mlp_embedder_config.mlp_embed_factor,
                                            nonlin_func=mlp_embedder_config.nonlin_func, 
                                            pdropout=mlp_embedder_config.p_dropout, 
                                            num_encoder_units=mlp_embedder_config.num_encoder_units)
            self.lastlayer_inputdim = 64 + mlp_embedder_config.embed_dim
        else:
            self.lastlayer_inputdim = 64
        

        self.Wy = nn.Linear(self.lastlayer_inputdim, 1, bias=True)

        self._init_params_()

    def _init_params_(self):
        # Initialize the parameters with Kaiming uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    
    def forward(self, x_proto, x_feat=None):
        
        # processing protospacer tensor
        x = x_proto.reshape(x_proto.shape[0], 20, 4) 
        #x = np.transpose(x, (0, 2, 1))
        x = torch.transpose(x, 1,2)
        #print(x.shape)
        #print('type of x', type(x))
        x = self.conv1(x)
        x = nn.functional.relu(x)
        
        #x = self.pool1(x)
        #print('after first conv', x.shape)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        
        #x = self.pool2(x)
        #print('after second conv', x.shape)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        #print('after third conv', x.shape)
        #x = self.pool3(x)
        x = x.view(x.size(0), -1)
        #print('after flattening', x.shape)
        x = self.fc1(x)
        z_1 = nn.functional.relu(x)

        if x_feat is not None:
            ### hanlde other features
            z_2 = self.mlp_embedder(x_feat)
            z_final = torch.cat((z_1, z_2), dim = -1)
        else:
            z_final = z_1
        
        output = self.Wy(z_final)
        
        #return output.squeeze(-1)
        return output

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

class CNNHyperparamConfig:
    def __init__(self, k, l2_reg, batch_size, num_epochs):
        self.k = k
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs


    def __repr__(self):
        desc = " k:{}\n l2_reg:{} \n batch_size:{} \n num_epochs: {}".format(self.k,
                                                                             self.l2_reg, 
                                                                             self.batch_size,
                                                                             self.num_epochs)
        return desc  




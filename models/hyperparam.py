import torch
from .FFN import FFNHyperparamConfig
from .CNN import CNNHyperparamConfig
from .RNN import RNNHyperparamConfig
from .Transformer import TrfHyperparamConfig

class MLPEmbedderHyperparamConfig:
    def __init__(self, input_dim, embed_dim, mlp_embed_factor, nonlin_func, p_dropout, num_encoder_units):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.mlp_embed_factor = mlp_embed_factor
        self.nonlin_func = nonlin_func
        self.p_dropout = p_dropout
        self.num_encoder_units = num_encoder_units



    def __repr__(self):
        desc = " embed_dim:{}\n mlp_embed_factor:{} \n nonlin_func:{} \n p_dropout:{} \n num_encoder_units:{}\n ".format(self.embed_dim,
                                                                                                                         self.mlp_embed_factor,
                                                                                                                         self.nonlin_func,
                                                                                                                         self.p_dropout,
                                                                                                                         self.num_encoder_units)
                          
        return desc

def build_config_map(model_name, optim_tup, model_tup, mlp_embedder_tup=None, fdtype=torch.float32, loss_func='MSEloss'):
    
    if model_name == 'FFN':
        #print('we are here!')
        #print(model_tup)
        model_hyperparam_config = FFNHyperparamConfig(*model_tup)
        #print(model_hyperparam_config)
        
    elif model_name == 'CNN':
        model_hyperparam_config = CNNHyperparamConfig(*model_tup)
        #print(model_hyperparam_config)
    
    elif model_name == 'RNN':
        model_hyperparam_config = RNNHyperparamConfig(*model_tup)
    elif model_name == 'Transformer':
        model_hyperparam_config = TrfHyperparamConfig(*model_tup)

    if mlp_embedder_tup is not None:
        mlp_hyperparam_config = MLPEmbedderHyperparamConfig(*mlp_embedder_tup)
    else:
        mlp_hyperparam_config = None
        
    if optim_tup:
        optim_config = {'lr':{'l0':optim_tup[0], 'lmax':optim_tup[1]},
                        'momentum':{'m0':optim_tup[2], 'mmax':optim_tup[3]},
                        'annealing_percent':optim_tup[4],
                        'stop_crit':optim_tup[5]}
    else:
        optim_config = {}
        
    run_num = -1 
    fdtype = torch.float32
    
    mconfig, options = generate_models_config((model_hyperparam_config, mlp_hyperparam_config), optim_config,
                                              run_num, fdtype,
                                              loss_func=loss_func)
    return mconfig, options

def generate_models_config(hyperparam_config, optim_config, 
                           run_num, fdtype, loss_func):
    
    model_cfg, mlp_cfg = hyperparam_config

    dataloader_config = {'batch_size': model_cfg.batch_size,
                         'num_workers': 0}
    config = {'dataloader_config': dataloader_config,
              'model_config': model_cfg,
              'mlpembedder_config':mlp_cfg,
              'optimizer_config': optim_config
             }

    options = {
               'run_num': run_num,
               'num_epochs': model_cfg.num_epochs,
               'weight_decay': model_cfg.l2_reg,
               'fdtype':fdtype,
               'to_gpu':True,
               'loss_func':loss_func}

    return config, options

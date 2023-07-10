## defining the model
from torch import nn
import torch

class NucleoPosEmbedder(nn.Module):
    def __init__(self, num_nucleotides, seq_length, embedding_dim,  pos_embed_concat_opt='sum'):
        super().__init__()
        
        if  pos_embed_concat_opt == 'stack':
            # by default we half the embedding dimension when we concat input and position
            embedding_dim = embedding_dim//2 
        
        self.nucleo_emb = nn.Embedding(num_nucleotides, embedding_dim)
        #self.nucleo_emb_other = nn.Linear(10, embedding_dim)
        self.pos_emb = nn.Embedding(seq_length, embedding_dim)
        self. pos_embed_concat_opt = pos_embed_concat_opt

    def forward(self, X):
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        X_emb = self.nucleo_emb (X)
        # X_emb = X
        bsize, seqlen, featdim = X_emb.size()
        device = X_emb.device

        # (batch, sequence length, embedding dim)
        if self.pos_embed_concat_opt in {'sum', 'stack'}:
            positions = torch.arange(seqlen).to(device)
            positions_emb = self.pos_emb(positions)[None, :, :].expand(bsize, seqlen, featdim)
            if self.pos_embed_concat_opt == 'sum':
                X_embpos = X_emb + positions_emb
            elif self.pos_embed_concat_opt == 'stack':
                X_embpos = torch.cat([X_emb, positions_emb], dim=-1)
        else:
            X_embpos = X_emb
        return X_embpos

class TransformerUnit(nn.Module):
    
    def __init__(self, input_size, num_attn_heads, mlp_embed_factor, nonlin_func, 
                 pdropout, multihead_type='Wide'):
        
        super().__init__()
        
        embed_size = input_size
        
        if multihead_type == 'Wide':
            self.multihead_attn = MH_SelfAttentionWide(input_size, num_attn_heads)
        elif multihead_type == 'Narrow':
            self.multihead_attn = MH_SelfAttentionNarrow(input_size, num_attn_heads)

        self.layernorm_1 = nn.LayerNorm(embed_size)

        # also known as position wise feed forward neural network
        self.MLP = nn.Sequential(
            nn.Linear(embed_size, embed_size*mlp_embed_factor),
            nonlin_func,
            nn.Linear(embed_size*mlp_embed_factor, embed_size)
        )
        
        self.layernorm_2 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(p=pdropout)
                
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        # z is tensor of size (batch, sequence length, input_size)
        z, attn_mhead_tensor = self.multihead_attn(X, X, X)
        # layer norm with residual connection
        z = self.layernorm_1(z + X)
        z = self.dropout(z)
        z_ff= self.MLP(z)
        z = self.layernorm_2(z_ff + z)
        z = self.dropout(z)
        
        return z, attn_mhead_tensor
    
# class MLPDecoder(nn.Module):
#     def __init__(self,
#                  inp_dim,
#                  outp_dim,
#                  seq_length):
        
#         super().__init__()
        
#         self.bias = nn.Parameter(torch.randn((seq_length, outp_dim), dtype=torch.float32), requires_grad=True)
#         self.Wy = nn.Linear(inp_dim, outp_dim, bias=False)

#     def forward(self, Z):
#         """
#         Args:
#             Z: tensor, float32, (batch, num_haplotypes, seq_len, embed_dim) representing computed from :class:`HaplotypeEncoderEncoder`
#         """
#         y = self.Wy(Z) + self.bias
#         return y
    

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

        
class PredictionTransformer(nn.Module):

    def __init__(self, input_size, embed_size=64, num_nucleotides=4, 
                 seq_length=20, num_attn_heads=8, 
                 mlp_embed_factor=2, nonlin_func=nn.ReLU, 
                 pdropout=0.3, num_transformer_units=12, pos_embed_concat_opt='sum',
                 pooling_mode='attn_perbase', multihead_type='Wide', 
                 mlp_embedder_config=None, num_classes=1):
        
        super().__init__()
        
        nonlin_func = nonlin_func()
        self.nucleopos_embedder = NucleoPosEmbedder(num_nucleotides, seq_length, embed_size,  pos_embed_concat_opt)
        
        trfunit_layers = [TransformerUnit(embed_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout, multihead_type) 
                          for i in range(num_transformer_units)]
        self.trfunit_pipeline = nn.ModuleList(trfunit_layers)        
        
        self.pooling_mode = pooling_mode
        if self.pooling_mode == 'none':
            self.penultimate_layer_inpsize = embed_size*seq_length

        elif self.pooling_mode == 'attn_perbase':
            self.pooling = PerBaseFeatureEmbAttention(embed_size, seq_length)
            # self.mlpdecoder = MLPDecoder(embed_size, 2, seq_length)
            # self.mlpdecoder = MLPDecoder(embed_size*seq_length, embed_size, embed_size//2,num_encoder_units=2)
            # self.penultimate_layer_inpsize = embed_size//2
            self.penultimate_layer_inpsize = embed_size*seq_length

        elif pooling_mode == 'attn_overall':
            self.pooling = FeatureEmbAttention(embed_size)
            self.penultimate_layer_inpsize = embed_size
            
        if input_size == 20: # encodes the length of input (i.e. protospacer only or protospacer + derived features)
        # self.bias = nn.Parameter(torch.randn((embed_size*seq_length, num_classes), dtype=torch.float32), requires_grad=True)
            self.Wy = nn.Linear(self.penultimate_layer_inpsize, num_classes, bias=True)
        else:
            # TODO: consider to make these as part of hyperparams to pass
            f_size = mlp_embedder_config.input_dim

            self.mlp_embedder = MLPEmbedder(f_size, 
                                            mlp_embedder_config.embed_dim, 
                                            mlp_embed_factor=mlp_embedder_config.mlp_embed_factor,
                                            nonlin_func=mlp_embedder_config.nonlin_func, 
                                            pdropout=mlp_embedder_config.p_dropout, 
                                            num_encoder_units=mlp_embedder_config.num_encoder_units)
            self.Wy = nn.Linear(self.penultimate_layer_inpsize + mlp_embedder_config.embed_dim, num_classes, bias=True)
            
        # self.sigmoid = torch.nn.Sigmoid()
        self._init_params_()
        
    # def _init_params_(self):
    #     for p_name, p in self.named_parameters():
            
    #         param_dim = p.dim()
    #         if param_dim > 1: # weight matrices
    #             nn.init.xavier_uniform_(p)
    #         elif param_dim == 1: # bias parameters
    #             if p_name.endswith('bias'):
    #                 nn.init.uniform_(p, a=-1.0, b=1.0)
    
    def _init_params_(self):
        for p_name, p in self.named_parameters():
            param_dim = p.dim()
            if param_dim > 1: # weight matrices
                nn.init.xavier_normal_(p)
            elif param_dim == 1: # bias parameters
                if p_name.endswith('bias'):
                    nn.init.uniform_(p, a=-0.5, b=1.0)

    # def _init_params_(self):
    #     for p_name, p in self.named_parameters():
    #         param_dim = p.dim()
    #         if param_dim > 1: # weight matrices
    #             nn.init.kaiming_uniform_(p)
    #         elif param_dim == 1: # bias parameters
    #             if p_name.endswith('bias'):
    #                 nn.init.zeros_(p)

    def forward(self, X_proto, X_f=None):
        """
        Args:
            X_proto: tensor, int64,  (batch, sequence length)
            X_f : (optional) tensor, float32 or float64, (batch, feat_dim)
        """

        #print('we are running the protospacer model')
        X_embpos = self.nucleopos_embedder(X_proto)
        bsize, num_positions, inp_dim = X_embpos.shape
        attn_tensor = X_embpos.new_zeros((bsize, num_positions, num_positions))
        xinput = X_embpos
        for trfunit in self.trfunit_pipeline:
            z, attn_mhead_tensor = trfunit(xinput)
            xinput = z
            attn_tensor += attn_mhead_tensor
            
        attn_tensor = attn_tensor/len(self.trfunit_pipeline)
        
        # print(self.pooling_mode)
        if self.pooling_mode == 'none':
            fattn_w_norm = []
            ## reshape z
            n_1, n_2, n_3 = z.shape
            #print(z.shape)
            z_concat = z.reshape(n_1, n_2*n_3)
        elif self.pooling_mode == 'attn_perbase':
            z, fattn_w_norm = self.pooling(z)
            # z = self.mlpdecoder(z)
            ## reshape z
            n_1, n_2, n_3 = z.shape
            #print(z.shape)
            z_concat = z.reshape(n_1, n_2*n_3)
            # z_concat = self.mlpdecoder(z_concat)
        elif self.pooling_mode == 'attn_overall':
            z, fattn_w_norm = self.pooling(z)
            z_concat = z
        
        if X_f is not None:
            z_f = self.mlp_embedder(X_f)
            z_concat = torch.cat((z_concat, z_f), dim = -1)

        y = self.Wy(z_concat)

        return y.squeeze(-1), fattn_w_norm, attn_tensor   

class MH_SelfAttentionNarrow(nn.Module):
    """ multi head self-attention module
    """
    def __init__(self, input_size, num_attn_heads):
        
        super().__init__()
        
        assert input_size%num_attn_heads == 0
        
        embed_size = input_size
        
        self.num_attn_heads = num_attn_heads
        self.head_dim = embed_size//num_attn_heads
        
        layers = [SH_SelfAttention(self.head_dim) for i in range(self.num_attn_heads)]
        self.multihead_pipeline = nn.ModuleList(layers)
        
        self.Wz = nn.Linear(embed_size, embed_size, bias=True)
        
        
    def forward(self, Xin_q, Xin_k, Xin_v, mask=None):
        """
        Args:
            Xin_q: query tensor, (batch, sequence length, input_size)
            Xin_k: key tensor, (batch, sequence length, input_size)
            Xin_v: value tensor, (batch, sequence length, input_size)
            mask: tensor, (batch, sequence length) with 0/1 entries
                  (default None)        """
        out = []
        # attn_dict = {}
        bsize, q_seqlen, inputsize = Xin_q.size()
        kv_seqlen = Xin_k.size(1)

        # print('Xin_q.shape', Xin_q.shape)
        # print('Xin_k.shape', Xin_k.shape)
        # print('Xin_v.shape', Xin_v.shape)
        # print('mask.shape', mask.shape)

        Xq_head = Xin_q.view(bsize, q_seqlen, self.num_attn_heads, self.head_dim)
        Xk_head = Xin_k.view(bsize, kv_seqlen, self.num_attn_heads, self.head_dim)
        Xv_head = Xin_v.view(bsize, kv_seqlen, self.num_attn_heads, self.head_dim)
        
        attn_tensor = Xq_head.new_zeros((bsize, q_seqlen, kv_seqlen))

        for count, SH_layer in enumerate(self.multihead_pipeline):
            z, attn_w = SH_layer(Xq_head[:,:,count,:],
                                 Xk_head[:,:,count,:],
                                 Xv_head[:,:,count,:],
                                 mask=mask)
            out.append(z)
            # attn_dict[f'h{count}'] = attn_w
            attn_tensor += attn_w
        attn_tensor = attn_tensor/len(self.multihead_pipeline)
        # concat on the feature dimension
        out = torch.cat(out, -1)         
        # return a unified vector mapping of the different self-attention blocks
        # for now we are returning the averaged attention matrix
        return self.Wz(out), attn_tensor

class MH_SelfAttentionWide(nn.Module):
    """ multi head self-attention module
    """
    def __init__(self, input_size, num_attn_heads):
        
        super().__init__()
        
        embed_size = input_size
        
        layers = [SH_SelfAttention(embed_size) for i in range(num_attn_heads)]
        self.multihead_pipeline = nn.ModuleList(layers)
        
        self.Wz = nn.Linear(num_attn_heads*embed_size, embed_size, bias=True)
    
    def forward(self, Xin_q, Xin_k, Xin_v, mask=None):
        """
        Args:
            Xin_q: query tensor, (batch, sequence length, input_size)
            Xin_k: key tensor, (batch, sequence length, input_size)
            Xin_v: value tensor, (batch, sequence length, input_size)
            mask: tensor, (batch, sequence length) with 0/1 entries
                  (default None)
        """
        out = []
        # attn_dict = {}
        bsize, q_seqlen, inputsize = Xin_q.size()
        kv_seqlen = Xin_k.size(1)

        attn_tensor = Xin_q.new_zeros((bsize, q_seqlen, kv_seqlen))

        for count, SH_layer in enumerate(self.multihead_pipeline):
            z, attn_w = SH_layer(Xin_q, Xin_k, Xin_v, mask=mask)
            out.append(z)
            # attn_dict[f'h{count}'] = attn_w
            attn_tensor += attn_w
        attn_tensor = attn_tensor/len(self.multihead_pipeline)
        # concat on the feature dimension
        out = torch.cat(out, -1) 
        
        # return a unified vector mapping of the different self-attention blocks
        # for now we are returning the averaged attention matrix
        return self.Wz(out), attn_tensor   
    
    
class SH_SelfAttention(nn.Module):
    """ single head self-attention module
    """
    def __init__(self, input_size):
        
        super().__init__()
        # define query, key and value transformation matrices
        # usually input_size is equal to embed_size
        self.embed_size = input_size
        self.Wq = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wk = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wv = nn.Linear(input_size, self.embed_size, bias=False)
        self.softmax = nn.Softmax(dim=2) # normalized across feature dimension
    
    def forward(self, Xin_q, Xin_k, Xin_v,mask=None):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        X_q = self.Wq(Xin_q) # queries
        X_k = self.Wk(Xin_k) # keys
        X_v = self.Wv(Xin_v) # values
        
        # scaled queries and keys by forth root 
        X_q_scaled = X_q / (self.embed_size ** (1/4))
        X_k_scaled = X_k / (self.embed_size ** (1/4))
        
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2))
        # (batch, sequence length, sequence length)
        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        
        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v)
        
        return z, attn_w_normalized
    
class PerBaseFeatureEmbAttention(nn.Module):
    """ Per base feature attention module
    """
    def __init__(self, input_dim, seq_len):
        
        super().__init__()
        # define query, key and value transformation matrices
        # usually input_size is equal to embed_size
        self.embed_size = input_dim
        self.Q = nn.Parameter(torch.randn((seq_len, input_dim), dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1) # normalized across feature dimension
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        bsize, seqlen, featdim = X.shape
        #print('bsize, seqlen, featdim', bsize, seqlen, featdim)
        X_q = self.Q[None, :, :].expand(bsize, seqlen, featdim) # queries
        X_k = X
        X_v = X
        # scaled queries and keys by forth root 
        X_q_scaled = X_q / (self.embed_size ** (1/4))
        X_k_scaled = X_k / (self.embed_size ** (1/4))
        # print(X_q_scaled.shape)
        # print(X_k_scaled.shape)
        
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2))
        # attn_w = X_q_scaled.matmul(X_k_scaled.transpose(1,0))
        # (batch, sequence length, sequence length)
        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        
        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v)
        #print('z.shape', z.shape)
        
        return z, attn_w_normalized
    

class FeatureEmbAttention(nn.Module):
    def __init__(self, input_dim):
        '''
        Args:
            input_dim: int, size of the input vector (i.e. feature vector)
        '''

        super().__init__()
        self.input_dim = input_dim
        # use this as query vector against the transformer outputs
        self.queryv = nn.Parameter(torch.randn(input_dim, dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=1) # normalized across seqlen

    def forward(self, X):
        '''Performs forward computation
        Args:
            X: torch.Tensor, (bsize, seqlen, feature_dim), dtype=torch.float32
        '''

        X_scaled = X / (self.input_dim ** (1/4))
        queryv_scaled = self.queryv / (self.input_dim ** (1/4))
        # using  matmul to compute tensor vector multiplication
        
        # (bsize, seqlen)
        attn_weights = X_scaled.matmul(queryv_scaled)

        # softmax
        attn_weights_norm = self.softmax(attn_weights)

        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, seqlen)
        # perform batch multiplication with X that has shape (bsize, seqlen, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = attn_weights_norm.unsqueeze(1).bmm(X).squeeze(1)
        
        # returns (bsize, feat_dim), (bsize, seqlen)
        return z, attn_weights_norm
    

class TrfHyperparamConfig:
    def __init__(self, embed_dim, num_attn_heads, num_transformer_units, 
                p_dropout, nonlin_func, mlp_embed_factor, multihead_type,
                pos_embed_concat_opt, pooling_opt, l2_reg, batch_size, num_epochs):
        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.num_transformer_units = num_transformer_units
        self.p_dropout = p_dropout
        self.nonlin_func = nonlin_func
        self.mlp_embed_factor = mlp_embed_factor
        self.multihead_type = multihead_type
        self.pos_embed_concat_opt = pos_embed_concat_opt
        self.pooling_opt = pooling_opt
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs


    def __repr__(self):
        desc = " embed_dim:{}\n num_attn_heads:{}\n num_transformer_units:{}\n p_dropout:{} \n " \
               "nonlin_func:{} \n mlp_embed_factor:{} \n multihead_type:{} \n pos_embed_concat_opt:{} \n pooling_opt:{} \n" \
               "l2_reg:{} \n batch_size:{} \n num_epochs: {}".format(self.embed_dim,
                                                                     self.num_attn_heads,
                                                                     self.num_transformer_units,
                                                                     self.p_dropout, 
                                                                     self.nonlin_func,
                                                                     self.mlp_embed_factor,
                                                                     self.multihead_type,
                                                                     self.pos_embed_concat_opt,
                                                                     self.pooling_opt,
                                                                     self.l2_reg, 
                                                                     self.batch_size,
                                                                     self.num_epochs)
        return desc

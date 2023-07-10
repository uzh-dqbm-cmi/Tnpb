import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class PredictionRNN(nn.Module):
    def __init__(self, 
                 input_dim,
                 embed_dim,
                 hidden_dim, 
                 z_dim,
                 outp_dim,
                 seq_len,
                 device,
                 num_hiddenlayers=1, 
                 bidirection= False, 
                 rnn_pdropout=0., 
                 rnn_class=nn.LSTM, 
                 nonlinear_func=nn.ReLU,
                 pooling_mode='attn_overall',
                 mlp_embedder_config=None,
                 fdtype = torch.float32):
        
        super().__init__()

        self.fdtype = fdtype
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_hiddenlayers = num_hiddenlayers
        self.rnn_pdropout = rnn_pdropout
        self.device = device


        nonlinear_func = nonlinear_func()
        # embed protospacer sequence
        self.Wproto = nn.Embedding(4+1, embed_dim, padding_idx=4)
        self.rnninput_dim = self.embed_dim
        if num_hiddenlayers == 1:
            rnn_pdropout = 0
        self.rnn = rnn_class(self.rnninput_dim, 
                             hidden_dim, 
                             num_layers=num_hiddenlayers, 
                             dropout=rnn_pdropout, 
                             bidirectional=bidirection,
                             batch_first=True)
        if(bidirection):
            self.num_directions = 2
        else:
            self.num_directions = 1
   
        self.Wz = nn.Linear(self.num_directions*hidden_dim, self.z_dim)
        self.nonlinear_func = nonlinear_func

        # pooling options
        self.pooling_mode = pooling_mode
        if self.pooling_mode == 'none':
            self.penultimate_layer_inpsize = self.z_dim*seq_len

        elif self.pooling_mode == 'attn_perbase':
            self.pooling = PerBaseFeatureEmbAttention(self.z_dim, seq_len)
            # self.mlpdecoder = MLPDecoder(embed_size, 2, seq_length)
            # self.mlpdecoder = MLPDecoder(embed_size*seq_length, embed_size, embed_size//2,num_encoder_units=2)
            # self.penultimate_layer_inpsize = embed_size//2
            self.penultimate_layer_inpsize = self.z_dim*seq_len

        elif pooling_mode == 'attn_overall':
            self.pooling = FeatureEmbAttention(self.z_dim)
            self.penultimate_layer_inpsize = self.z_dim

        if input_dim == 20: # encodes the length of input (i.e. protospacer only or protospacer + derived features)
        # self.bias = nn.Parameter(torch.randn((embed_size*seq_length, num_classes), dtype=torch.float32), requires_grad=True)
            self.Wy = nn.Linear(self.penultimate_layer_inpsize, outp_dim, bias=True)

        else:
            # TODO: consider to make these as part of hyperparams to pass
            # f_size = input_dim - seq_len
            f_size = mlp_embedder_config.input_dim

            self.mlp_embedder = MLPEmbedder(f_size, 
                                            mlp_embedder_config.embed_dim, 
                                            mlp_embed_factor=mlp_embedder_config.mlp_embed_factor,
                                            nonlin_func=mlp_embedder_config.nonlin_func, 
                                            pdropout=mlp_embedder_config.p_dropout, 
                                            num_encoder_units=mlp_embedder_config.num_encoder_units)
            self.Wy = nn.Linear(self.penultimate_layer_inpsize + mlp_embedder_config.embed_dim, outp_dim, bias=True)
            
        
    def init_hidden(self, batch_size, requires_grad=True):
        """initialize hidden vectors at t=0
        
        Args:
            batch_size: int, the size of the current evaluated batch
        """
        device = self.device
        # a hidden vector has the shape (num_layers*num_directions, batch, hidden_dim)
        h0=torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim).type(self.fdtype)
        h0.requires_grad=requires_grad
        h0 = h0.to(device)
        if(isinstance(self.rnn, nn.LSTM)):
            c0=torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim).type(self.fdtype)
            c0.requires_grad=requires_grad
            c0 = c0.to(device)
            hiddenvec = (h0,c0)
        else:
            hiddenvec = h0
        return(hiddenvec)
    
    
    def forward(self, X_proto, X_f=None, requires_grad=True):
        """ perform forward computation
        
            Args:
                batch_seqs: tensor, shape (batch, seqlen, input_dim)
                seqs_len: np.array, (batch,), comprising length of the sequences in the batch
        """

        # print('batch_seqs.shape', batch_seqs.shape)
        # (bsize, seq_len, embed_dim)
        batch_seqs = self.Wproto(X_proto)
        # print('embeddeed batch_seqs.shape', batch_seqs.shape)

        seqs_len = np.array([20]*batch_seqs.shape[0])
        # init hidden
        hidden = self.init_hidden(batch_seqs.size(0), requires_grad=requires_grad)
        # print('hidden.shape:', hidden.shape)
        # pack the batch
        packed_embeds = pack_padded_sequence(batch_seqs, seqs_len, batch_first=True, enforce_sorted=False)
        packed_rnn_out, hidden = self.rnn(packed_embeds, hidden)

        # we need to unpack sequences
        unpacked_output, out_seqlen = pad_packed_sequence(packed_rnn_out, batch_first=True)
        # print('unpacked_output.shape:', unpacked_output.shape)
        z_logit = self.nonlinear_func(self.Wz(unpacked_output))
        # print('z_logit.shape:', z_logit.shape)
        if self.pooling_mode == 'none':
            d1,d2,d3 = z_logit.shape
            z_resh = z_logit.reshape(d1, d2*d3)
            z_f = z_resh
        elif self.pooling_mode == 'attn_perbase':
            p, __ = self.pooling(z_logit)
            # print('pooled shape:', p.shape)
            d1,d2,d3 = p.shape
            p_resh = p.reshape(d1, d2*d3)
            # print('pooled reshaped shape:', p_resh.shape)
            z_f = p_resh
        elif self.pooling_mode == 'attn_overall':
            p, __ = self.pooling(z_logit)
            z_f = p
            # print('pooled shape:', p.shape)

        if X_f is not None:
            z_mlp = self.mlp_embedder(X_f)
            z_f = torch.cat((z_f, z_mlp), dim = -1)

        outp = self.Wy(z_f)

        #return outp.squeeze(-1)
        return outp
    
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
        self.neg_inf = -1e6

    def forward(self, X, mask=None):
        '''Performs forward computation
        Args:
            X: torch.Tensor, (bsize, seqlen, feature_dim), dtype=torch.float32
        '''

        X_scaled = X / (self.input_dim ** (1/4))
        queryv_scaled = self.queryv / (self.input_dim ** (1/4))
        # using  matmul to compute tensor vector multiplication
        
        # (bsize, seqlen)
        attn_w = X_scaled.matmul(queryv_scaled)


        if mask is not None:
            # (batch, seqlen)
            # fill with neginf where mask == 0  
            attn_w = attn_w.masked_fill(mask == 0, self.neg_inf)
            # print('attn_w masked:\n', attn_w)

        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        # print('attn_w_normalized masked:\n', attn_w_normalized)
        
        if mask is not None:
            # for cases where the mask is all 0 in a row
            attn_w_normalized = attn_w_normalized * mask


        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, seqlen)
        # perform batch multiplication with X that has shape (bsize, seqlen, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = attn_w_normalized.unsqueeze(1).bmm(X).squeeze(1)
        
        # returns (bsize, feat_dim), (bsize, seqlen)
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
    
    
class RNNHyperparamConfig:

    def __init__(self, 
                 embed_dim,
                 hidden_dim, 
                 z_dim,
                 num_hidden_layers, 
                 bidirection, 
                 p_dropout,     
                 rnn_class,
                 nonlin_func, 
                 pooling_mode,
                 l2_reg, 
                 batch_size,
                num_epochs):

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_hidden_layers = num_hidden_layers
        self.bidirection = bidirection
        self.p_dropout = p_dropout
        self.rnn_class = rnn_class
        self.nonlin_func = nonlin_func
        self.pooling_mode = pooling_mode
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs


    def __repr__(self):
        desc = f" embed_dim:{self.embed_dim}\n hidden_dim:{self.embed_dim}\n z_dim:{self.z_dim}\n" \
               f" num_hidden_layers:{self.num_hidden_layers}\n " \
              f"  bidirection:{self.bidirection}\n " \
               f" p_dropout:{self.p_dropout} \n rnn_class:{self.rnn_class} \n nonlin_func:{self.nonlin_func} \n " \
               f" pooling_mode:{self.pooling_mode} \n l2_reg:{self.l2_reg} \n batch_size:{self.batch_size} \n num_epochs: {self.num_epochs}"
        return desc
    
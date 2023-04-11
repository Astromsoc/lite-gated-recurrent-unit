"""
    Numpy-based and PyTorch-based GRU model(s) in use.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 11, 2023
"""

import numpy as np

import torch
import torch.nn as nn

from src.utils import ParamsObject




class sigmoid(object):

    def forward(self, X):
        self.Xout = np.true_divide(1, 1 + np.exp(-X))
        return self.Xout
    
    def backward(self):
        return np.multiply(self.Xout, 1 - self.Xout)



class tanh(object):
    
    def forward(self, X):
        self.Xout = np.tanh(X)
        return self.Xout
    
    def backward(self):
        return 1 - np.square(self.Xout)




class LiteGatedRecurrentUnit(object):

    def __init__(self, configs: ParamsObject):
        # archiving
        self.hidden_dim = configs.hidden_dim
        self.input_dim = configs.input_dim
        self.inith_wz = True if configs.inith_opt.startswith('zero') else False
        self.inith_bp = False if configs.inith_opt.endswith('frozen') else True
        # build up weights & biases
        """
            to compute gates
        """
        self.Whr = np.zeros((self.hidden_dim, self.hidden_dim))
        self.bhr = np.zeros((self.hidden_dim, ))
        self.Wxr = np.zeros((self.input_dim, self.hidden_dim))
        self.bxr = np.zeros((self.hidden_dim, ))
        self.Whu = np.zeros((self.hidden_dim, self.hidden_dim))
        self.bhu = np.zeros((self.hidden_dim, ))
        self.Wxu = np.zeros((self.input_dim, self.hidden_dim))
        self.bxu = np.zeros((self.hidden_dim, ))
        """
            to obtain affine values
        """
        self.Wha = np.zeros((self.hidden_dim, self.hidden_dim))
        self.bha = np.zeros((self.hidden_dim, ))
        self.Wxa = np.zeros((self.input_dim, self.hidden_dim))
        self.bxa = np.zeros((self.hidden_dim, ))
        """
            activations
        """
        self.sigmoid_r = sigmoid()
        self.sigmoid_u = sigmoid()
        self.tanh_a = tanh()
    

    def init_inith(self):
        assert hasattr(self, 'batch_size')
        self.inith = (np.zeros((self.hidden_dim, )) if self.inith_wz else
                      np.random.random((self.hidden_dim, )))
        self.dinith = np.zeros((self.hidden_dim, ))


    def zero_grad(self):
        """
            build up gradients & clear derivatives
        """
        self.dWxa = np.zeros((self.input_dim, self.hidden_dim))
        self.dbxa = np.zeros((self.hidden_dim, ))
        self.dWha = np.zeros((self.hidden_dim, self.hidden_dim))
        self.dbha = np.zeros((self.hidden_dim, ))
        self.dWxr = np.zeros((self.input_dim, self.hidden_dim))
        self.dbxr = np.zeros((self.hidden_dim, ))
        self.dWhr = np.zeros((self.hidden_dim, self.hidden_dim))
        self.dbhr = np.zeros((self.hidden_dim, ))
        self.dWxu = np.zeros((self.input_dim, self.hidden_dim))
        self.dbxu = np.zeros((self.hidden_dim, ))
        self.dWhu = np.zeros((self.hidden_dim, self.hidden_dim))
        self.dbhu = np.zeros((self.hidden_dim, ))
    

    def forward(self, x: np.ndarray, hprev: np.ndarray=None):
        """
            Args:
                x (np.ndarray): (B, input_dim) input 
                hprev (np.ndarray): (B, hidden_dim) hidden state from the previous time step
        """
        # update batch size: B
        self.batch_size = x.shape[0]
        # initiate h_0 if hprev is None:
        if hprev is None: 
            self.init_inith()
            # expand initial hprev to match batch size
            hprev = np.broadcast_to(self.inith, (self.batch_size, self.hidden_dim))
        # archiving for backward use
        self.x, self.hprev = x, hprev
        # [1] remember gate: (B, hidden_dim)
        self.gr = self.sigmoid_r.forward(
            self.x.dot(self.Wxr) + self.bxr + self.hprev.dot(self.Whr) + self.bhr
        )
        # [2] update gate: (B, hidden_dim)
        self.gu = self.sigmoid_r.forward(
            self.x.dot(self.Wxu) + self.bxu + self.hprev.dot(self.Whu) + self.bhu
        )
        # [3] activated affine combination before weighted sum with update gate: (B, hidden_dim)
        self.a_from_hprev = self.hprev.dot(self.Wha) + self.bha
        self.aa = self.tanh_a.forward(
            self.x.dot(self.Wxa) + self.bxa + np.multiply(
                self.gr, self.a_from_hprev
            )
        )
        # [4] weighted sum of prev hidden state and current input: (B, hidden_dim)
        self.h = np.multiply(1 - self.gu, self.aa) + np.multiply(self.gu, self.hprev)
        # finally return new hidden state
        return self.h
    
    
    def backward(self):
        pass


    def __call__(self, x: np.ndarray, hprev: np.ndarray):
        return self.forward(x, hprev)




class TrainableLiteGatedRecurrentUnit(LiteGatedRecurrentUnit):

    def __init__(self, configs: ParamsObject):
        super().__init__(configs)
    

    def backward(self, dh: np.ndarray):
        """
            Args:
                dh (np.ndarray): (B, hidden_dim) derivative for previously forwarded hidden state (self.h)
        """
        # from [4]:     self.h <- (- self.gu * self.aa)
        daa = np.multiply(1 - self.gu, dh)
        #               self.h <- (- self.gu * self.aa + self.gu * self.hprev)
        dgu = np.multiply(- self.aa + self.hprev, dh)
        #               self.h <- (self.gu * self.hprev)
        self.dhprev += np.multiply(self.gu, dh)

        # from [3]: activation backward first, da.shape = (B, hidden_dim)
        da = np.multiply(self.tanh_a.backward(), daa)
        #               (before act) a <- Wxa * x + bxa
        self.dWxa += self.x.dot(da) / self.batch_size
        self.dx += da.dot(self.Wxa.T)
        self.dbxa += da.mean(axis=0)
        #               (before act) a <- self.gr * da_from_prev
        dgr = np.multiply(da, self.a_from_hprev)
        da_from_prev = np.multiply(da, self.gr)
        #               (before act) a_from_prev <- Wha * h + bha
        self.dWha += self.hprev.dot(da_from_prev) / self.batch_size
        self.dhprev += da_from_prev.dot(self.Wha.T)
        self.dbha += da_from_prev.mean(axis=0)

        # from [2]: activation backward first, dgu_addine.shape = (B, hidden_dim)
        dgu_affine = np.multiply(self.sigmoid_u.backward(), dgu)
        #               gu_affine <- Wxu * x + bxu
        self.dWxu += self.x.dot(dgu_affine) / self.batch_size
        self.dx += dgu_affine.dot(self.Wxu.T)
        self.dbxu += dgu_affine.mean(axis=0)
        #               gu_affine <- Whu * h + bhu
        self.dWhu += self.hprev.dot(dgu_affine) / self.batch_size
        self.dhprev += dgu_affine.dot(self.Whu.T)
        self.dbhu += dgu_affine.mean(axis=0)


        # from [1]: activation backward first, dgr_addine.shape = (B, hidden_dim)
        dgr_affine = np.multiply(self.sigmoid_r.backward(), dgr)
        #               gr_affine <- Wxr * x + bxr
        self.dWxr += self.x.dot(dgr_affine) / self.batch_size
        self.dx += dgr_affine.dot(self.Wxr.T)
        self.dbxr += dgr_affine.mean(axis=0)
        #               gr_affine <- Whr * h + bhr
        self.dWhr += self.hprev.dot(dgr_affine) / self.batch_size
        self.dhprev += dgr_affine.dot(self.Whr.T)
        self.dbhr += dgr_affine.mean(axis=0)

        # output gradients for x and h_prev to pass on backprop chain
        return self.dx, self.dhprev




class TorchL2ROneLayerEncDecGruSeqPred(nn.Module):
    """
        PyTorch based model trained on sequential prediction.

        Note: 
            for inference simplicity, only a lightweight encoder unidirectional GRU and a decoder
            unidirectional GRU is used (more complex architecture could lead to better performance 
            at the cost of larger models & potentially difficulty in numpy based replication)
    """
    def __init__(self, configs: ParamsObject):
        super().__init__()
        # archiving
        self.configs = configs
        # frequenly used attributes
        self.max_steps = self.configs.max_steps
        self.init_dec_idx = self.configs.init_dec_idx
        self.take_last = self.configs.enc_take_last
        # build model layers
        self.enc_emb = nn.Embedding(num_embeddings=self.configs.num_inp,
                                    embedding_dim=self.configs.enc_emb_dim,
                                    padding_idx=self.configs.enc_pad_idx)
        self.dec_emb = nn.Embedding(num_embeddings=self.configs.num_cls,
                                    embedding_dim=self.configs.dec_emb_dim,
                                    padding_idx=self.configs.dec_pad_idx)
        self.enc_gru = nn.GRU(num_layers=1, 
                              input_size=self.configs.enc_emb_dim,
                              bidirectional=False,
                              **self.configs.encgru.__dict__)
        self.dec_gru = nn.GRUCell(input_size=self.configs.dec_emb_dim,
                                  hidden_size=self.configs.decgru.hidden_size)
        self.cls = nn.Sequential(
            nn.Linear(in_features=self.configs.decgru.hidden_size,
                      out_features=self.configs.cls_lin_dim),
            nn.GELU(),
            nn.Linear(in_features=self.configs.cls_lin_dim,
                      out_features=self.configs.num_cls)
        )
    

    def forward(self, 
                input_ids: torch.tensor,
                input_lens: torch.tensor,
                golden_output_ids: torch.tensor=None,
                tf_rate: float=0.0):
        """
            Forward input: 
                input_ids: input token ids, eg. character ids of words
                    shape: (batch_size, padded_input_seq_len)
                input_lens: input token lengths, eg. character counts of words
                golden_output_ids: output token ids, eg. (grapheme, phoneme) ids
                    shape: (batch_size, padded_output_seq_len)
                tf_fate: teacher forcing rate
                    dtype: float
        """
        # ENC: obtain input encodings (last step)
        enc = self.enc_emb(input_ids)
        # (batch_size, padded_input_seq_len, emb_dim)
        enc = self.enc_gru(enc)[0]
        # (batch_size * emb_dim, padded_input_seq_len)
        # OPTION 1: take out the last time step given input lengths
        if self.take_last:
            enc = torch.stack([enc[i, input_lens[i] - 1, :] for i in range(len(enc))], dim=0)
        # OPTION 2: take out the last time step (with paddings convoluted potentially)
        else:
            enc = enc[:, -1, :]
        # (batch_size, encgru_hidden_size)

        # set maximum number of decoding iterations
        NUMSTEPS = golden_output_ids.size(1) if self.training else self.configs.max_steps

        # DEC: obtain (g, p) index for each time step
        full_dec_logits = list()
        # initial hidden state: hidden information conpressed to the last time step
        dec_h = enc
        # initial inputs: definitely initial idx for whatever mode / tf_rate
        dec_x = torch.ones_like(input_ids[:, 0], requires_grad=False) * self.init_dec_idx
        # (batch_size, )
        for t in range(NUMSTEPS - 1):
            # obtain the embeddings of current input (last-step gp idx)
            dec_xemb = self.dec_emb(dec_x)
            # (batch_size, dec_emb_dim)
            dec_h = self.dec_gru(dec_xemb, dec_h)
            # (batch_size, dec_hid_dim)
            full_dec_logits.append(self.cls(dec_h))
            # (batch_size, num_cls)

            # obtain gp idx as inputs to next step
            if self.training and golden_output_ids is not None and torch.rand(1).item() < tf_rate:
                dec_x = golden_output_ids[:, t + 1]
            else:
                dec_x = full_dec_logits[-1].argmax(dim=-1)
            # (batch_size, )
        
        # output shape: (batch_size, dec_time_steps, num_cls)
        return torch.stack(full_dec_logits, dim=1)

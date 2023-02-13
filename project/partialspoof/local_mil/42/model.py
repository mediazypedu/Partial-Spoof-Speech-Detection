#!/usr/bin/env python
"""
model.py

Self defined model definition.
Usage:

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn

import torch.nn.functional as torch_nn_func

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end
import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk
import core_modules.p2sgrad as nii_p2sgrad
import config as prj_conf

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

##############
## util
##############

def protocol_parse(protocol_filepath):
    """ Parse protocol of ASVspoof2019 and get bonafide/spoof for each trial

    input:
    -----
      protocol_filepath: string, path to the protocol file
        for convenience, I put train/dev/eval trials into a single protocol file

    output:
    -------
      data_buffer: dic, data_bufer[filename] -> 1 (bonafide), 0 (spoof)
    """
    # data_buffer = {}
    # temp_buffer = np.loadtxt(protocol_filepath, dtype='str')
    # for row in temp_buffer:
    #     if row[-1] == 'bonafide':
    #         data_buffer[row[1]] = 0
    #     else:
    #         data_buffer[row[1]] = 1
    """
    ===./database/segment_labels/{train,dev,eval}_seglab_0.16.npy ===
    This segment_labels folder contains the segmental-level label in .npy format for each set.
    Labels are generated with frame length of 0.16s and frame shift of 0.00s.
    The labels are: '0' for spoof, and '1' for bona fide.
    Here is the commands to load numpy file of the labels.
    >>> import numpy as np
    >>> train_seglab=np.load("train_seglab_0.16.npy", allow_pickle=True).item()
    >>> train_seglab['CON_T_0000000']
    array(['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
           '0', '0', '0', '1', '1'], dtype='<U21')
    >>> len(train_seglab)
    25380
    """
    import numpy as np
    bag_buffer = {}
    instance_buffer = np.load(protocol_filepath, allow_pickle=True).item()
    for key in instance_buffer.keys():
        if key[0:3] == 'CON':
            bag_buffer[key] = 1
        else:
            bag_buffer[key] = 0
    del instance_buffer
    return bag_buffer
    #return instance_buffer

##############
## FOR MODEL
##############

from encoder import ConformerEncoder

class Model(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, mean_std=None):
        super(Model, self).__init__()

        ##### required part, no need to change #####

        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         args, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        
        # a flag for debugging (by default False)
        #self.model_debug = False
        #self.validation = False
        #####
        
        ####
        # on input waveform and output target
        ####
        # Load protocol and prepare the target data for network training
        protocol_file = prj_conf.optional_argument[0]
        self.protocol_parser = protocol_parse(protocol_file)
        
        # Working sampling rate
        #  torchaudio may be used to change sampling rate
        self.m_target_sr = 16000

        ####
        # optional configs (not used)
        ####                
        # re-sampling (optional)
        #self.m_resampler = torchaudio.transforms.Resample(
        #    prj_conf.wav_samp_rate, self.m_target_sr)

        # vad (optional)
        #self.m_vad = torchaudio.transforms.Vad(sample_rate = self.m_target_sr)
        
        # flag for balanced class (temporary use)
        #self.v_flag = 1

        ####
        # front-end configuration
        #  multiple front-end configurations may be used
        #  by default, use a single front-end
        ####    
        # frame shift (number of waveform points)
        self.frame_hops = [160]
        # frame length
        self.frame_lens = [320]
        # FFT length
        self.fft_n = [512]

        # LFCC dim (base component)
        self.lfcc_dim = [20]
        self.lfcc_with_delta = True

        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating (not used)
        self.amp_floor = 0.00001
        
        # number of frames to be kept for each trial
        # no truncation
        self.v_truncate_lens = [None for x in self.frame_hops]


        # number of sub-models (by default, a single model)
        self.v_submodels = len(self.frame_lens)        

        # dimension of embedding vectors
        # here, the embedding is just the activation before sigmoid()
        self.v_emd_dim = 1
        ####
        # create network
        ####
        # 1st part of the classifier
        self.m_transform = []
        # 
        self.m_before_pooling = []
        # 2nd part of the classifier
        self.m_output_act = []
        # front-end
        self.m_frontend = []
        # final part on training


        # it can handle models with multiple front-end configuration
        # by default, only a single front-end
        for idx, (trunc_len, fft_n, lfcc_dim) in enumerate(zip(
                self.v_truncate_lens, self.fft_n, self.lfcc_dim)):
            
            fft_n_bins = fft_n // 2 + 1
            if self.lfcc_with_delta:
                lfcc_dim = lfcc_dim * 3
            
            self.m_transform.append(
                ConformerEncoder(input_dim=lfcc_dim)
            )

            self.m_before_pooling.append(
                torch_nn.Sequential(
                    nii_nn.BLSTMLayer((lfcc_dim // 16) * 32, (lfcc_dim // 16) * 32),
                    nii_nn.BLSTMLayer((lfcc_dim // 16) * 32, (lfcc_dim // 16) * 32)
                )
            )

            self.m_output_act.append(
                # 144:conformer编码器的输出维度
                #torch_nn.AdaptiveAvgPool1d(1),
                torch_nn.Linear(96, self.v_emd_dim),
                # torch_nn.Linear((lfcc_dim // 16) * 32, self.v_emd_dim)
            )

            self.m_frontend.append(
                nii_front_end.LFCC(self.frame_lens[idx],
                                   self.frame_hops[idx],
                                   self.fft_n[idx],
                                   self.m_target_sr,
                                   self.lfcc_dim[idx],
                                   with_energy=True)
            )
        self.m_frontend = torch_nn.ModuleList(self.m_frontend)
        self.m_transform = torch_nn.ModuleList(self.m_transform)
        self.m_output_act = torch_nn.ModuleList(self.m_output_act)
        self.m_before_pooling = torch_nn.ModuleList(self.m_before_pooling)
        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, data_mean_std=None):
        """ prepare mean and std for data processing
        This is required for the Pytorch project, but not relevant to this code
        """
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.ones([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.ones([out_dim])
            
        return in_m, in_s, out_m, out_s
        
    def normalize_input(self, x):
        """ normalizing the input data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        This is required for the Pytorch project, but not relevant to this code
        """
        return y * self.output_std + self.output_mean

    def get_batch_weights_from_local_attnmap(self, actual_len, map_list):
        """ attnmap:list[Tensor(BxHxFxW)...]
            do F->actual_len
        return weight(BxF)
        """
        map_0 = map_list[0][:, :, :actual_len, :]
        B, H, F, W = map_0.size()[0], map_0.size()[1], map_0.size()[2], map_0.size()[3]
        pad_tensor = torch.zeros(B, H, W // 2, W).to(map_0.device)
        # weight = torch.zeros(B, F)
        weights = []
        for attn_map in map_list:
            attn_map = attn_map[:, :, :actual_len, :]
            attn_map = torch.cat([pad_tensor, attn_map, pad_tensor], dim=2)
            for i in range(F):
                temp = torch.cat([attn_map[:, :, i + k: i + k + 1, W - 1 - k: W - k] for k in range(W)], dim=3)
                if i == 0:
                    weight = temp
                else:
                    weight = torch.cat([weight, temp], dim=2)
            # weight:BxHxFxW
            weight = torch.sum(torch.sum(weight, dim=3, keepdim=False), dim=1, keepdim=False)
            # weight:BxF
            weights.append(weight)
        # BxLxF
        weights = torch.stack([weight for weight in weights], dim=1)
        weights = torch.sum(weights, dim=1, keepdim=False)
        max = torch.max(weights, dim=1, keepdim=True)[0]
        min = torch.min(weights, dim=1, keepdim=True)[0]
        # 归一化
        weights = (weights - min) / (max - min)
        # 和为1
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        # mean = weights.mean(dim=1).unsqueeze(dim=1) #32x1
        # std = weights.std(dim=1, unbiased=False).unsqueeze(dim=1)
        # weights = (weights - mean) / std
        return weights
    def get_batch_weights_from_globl_attnmap(self, map_list):
        """ attnmap:list[Tensor(BxHxFxF)...]
            W=F do F->actual_len
        return weight(BxF)
        """
        weights = []
        for attn_map in map_list:
            #attn_map = attn_map[:, :, :actual_len, :actual_len]
            # weight:BxHxFxF
            weight = torch.sum(torch.sum(attn_map, dim=2, keepdim=False), dim=1, keepdim=False)
            # weight:BxF
            weights.append(weight)
        # BxLxF
        weights = torch.stack([weight for weight in weights], dim=1)
        weights = torch.sum(weights, dim=1, keepdim=False)
        max = torch.max(weights, dim=1, keepdim=True)[0]
        min = torch.min(weights, dim=1, keepdim=True)[0]
        # 归一化
        weights = (weights - min) / (max - min)
        # 和为1
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        import matplotlib.pyplot as plt
        plt.plot(weights[0].cpu().numpy())
        plt.show()
        # mean = weights.mean(dim=1).unsqueeze(dim=1) #32x1
        # std = weights.std(dim=1, unbiased=False).unsqueeze(dim=1)
        # weights = (weights - mean) / std
        return weights

    def get_batch_weights_from_instance_score(self, instance_score):
        """ instance_score:Tensor(BxFx2)...
        return weight(BxF)
        """
        instance_score = instance_score[:, :, 0]
        max = torch.max(instance_score, dim=1, keepdim=True)[0]
        min = torch.min(instance_score, dim=1, keepdim=True)[0]
        # 归一化
        weight = (instance_score - min) / (max - min)
        # 和为1
        weight = weight / torch.sum(weight, dim=1, keepdim=True)

        return weight
    def _front_end(self, wav, idx, trunc_len, datalength):
        """ simple fixed front-end to extract features
        
        input:
        ------
          wav: waveform
          idx: idx of the trial in mini-batch
          trunc_len: number of frames to be kept after truncation
          datalength: list of data length in mini-batch

        output:
        -------
          x_sp_amp: front-end featues, (batch, frame_num, frame_feat_dim)
        """
        #分帧时会在最后补一帧,即使整除帧移,也会完全补一帧
        with torch.no_grad():
            x_sp_amp = self.m_frontend[idx](wav.squeeze(-1))

        # return
        return x_sp_amp

    def _compute_embedding(self, x, datalength):
        """ definition of forward method 
        Assume x (batchsize, length, dim)
        Output x (batchsize * number_filter, output_dim)
        """
        # resample if necessary
        #x = self.m_resampler(x.squeeze(-1)).unsqueeze(-1)
        
        # number of sub models
        batch_size = x.shape[0]

        # buffer to store output scores from sub-models
        # output_emb = torch.zeros([batch_size * self.v_submodels,
        #                           self.v_emd_dim],
        #                           device=x.device, dtype=x.dtype)
        
        # compute scores for each sub-models
        for idx, (fs, fl, fn, trunc_len, m_trans, m_be_pool, m_output) in \
                enumerate(
                    zip(self.frame_hops, self.frame_lens, self.fft_n,
                        self.v_truncate_lens, self.m_transform,
                        self.m_before_pooling, self.m_output_act)):
            
            # extract front-end feature
            x_sp_amp = self._front_end(x, idx, trunc_len, datalength)
            #print(x_sp_amp.size())
            # compute scores
            #  1. unsqueeze to (batch, 1, frame_length, fft_bin)
            #  2. compute hidden features
            hidden_features, _ = m_trans(x_sp_amp.unsqueeze(1))
            #instance_features = m_output(hidden_features)
            hidden_features_lstm = m_be_pool(hidden_features)
            instance_features = m_output(hidden_features_lstm + hidden_features)
        return instance_features, None

    def _compute_score(self, instance_features, inference=False):
        """
        """
        # instance_features is [BxFx1]
        instance_scores = torch.sigmoid(instance_features).squeeze(-1)
        r = 9
        neg_scores = torch.log(torch.mean(torch.exp(r * instance_scores), dim=1)) / r
        pos_scores = torch.max(instance_scores, dim=1)[0]
        if inference:
            return pos_scores, instance_scores
        else:
            return pos_scores, neg_scores


    def _get_target(self, filenames):
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)
    def forward(self, x, fileinfo):
        
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        if self.training:
            instance_features, attn_maps = self._compute_embedding(x, datalength)#BxFxD,BxHxFxF
            pos_scores, neg_scores = self._compute_score(instance_features, False)
            # target
            target = self._get_target(filenames)
            target_vec = torch.tensor(target,
                                      device=x.device, dtype=x.dtype)
            return [pos_scores, neg_scores, target_vec, True]

        else:
            instance_features, attn_maps = self._compute_embedding(x, datalength)
            scores, instance_score = self._compute_score(instance_features, True)
            target = self._get_target(filenames)
            print("Output, %s, %d, %f" % (filenames[0],
                                          target[0], scores.mean()))

            return [scores, instance_score]


class Loss():
    """ Wrapper to define loss function 
    """
    def __init__(self, args):
        """
        """
        #self.alpha = torch.Tensor([0.1]).cuda()
        #self.b_loss = torch_nn.BCEWithLogitsLoss(pos_weight=self.alpha)
        self.b_loss = torch_nn.BCELoss(reduction='none')
        self.alpha = 10
    def compute(self, outputs, target):
        """
        """
        pos_scores = outputs[0]
        neg_scores = outputs[1]
        target = outputs[2]
        with torch.no_grad():
            pos_mask = target
            temp = torch.ones_like(target)
            neg_mask = temp - pos_mask
        pos_loss = (self.b_loss(pos_scores, target) * pos_mask).sum()
        neg_loss = (self.b_loss(neg_scores, target) * neg_mask).sum()
        return pos_loss + self.alpha * neg_loss
    def compute_for_test(self, outputs, target):
        pos_scores = outputs[0]
        neg_scores = outputs[1]
        target = outputs[2]
        with torch.no_grad():
            pos_mask = target
            temp = torch.ones_like(target)
            neg_mask = temp - pos_mask
        pos_loss = (self.b_loss(pos_scores, target) * pos_mask).sum()
        neg_loss = (self.b_loss(neg_scores, target) * neg_mask).sum()
        return pos_loss +  neg_loss
        #return pos_loss + self.alpha * neg_loss

    
if __name__ == "__main__":
    print("Definition of model")

    

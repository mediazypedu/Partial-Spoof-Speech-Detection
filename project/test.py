# # srcfile 需要复制、移动的文件
# # dstpath 目的地址
# import shutil
# import os
# def mycopyfile(srcfile, dstpath):  # 复制函数
#     if not os.path.isfile(srcfile):
#         print("%s not exist!" % (srcfile))
#     else:
#         fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
#         if not os.path.exists(dstpath):
#             os.makedirs(dstpath)  # 创建路径
#         shutil.copy(srcfile, dstpath + fname)  # 复制文件
#         print("copy %s -> %s" % (srcfile, dstpath + fname))
#
# path=os.getcwd()
# srcfile = os.path.join(path,'evaluate.py')
# dst_dir = os.path.join(path,'partialspoof/local_mil/')
# dist_dirs = os.listdir(dst_dir)
# for dist_dir in dist_dirs:
#     mycopyfile(srcfile, dist_dir)  # 复制文件


import numpy as np
import random
#
# train_dev_eval_labels = np.load('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof/segment_labels' + \
#                                 '/train_dev_eval_seglab_0.16.npy', allow_pickle=True).item()
# print(train_dev_eval_labels['CON_E_0000099'])
#CON_E_0000002 [0. 0. 0. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#0.84646255,0.2727478,0.9692266,0.998777,0.94596463,0.7749254,0.91267467,0.8763028,0.012243835,0.00040231628,0.00018197908,0.00054304296,0.0009449152,0.01088644,0.0170612,0.02896496,0.0028813162,0.00034513357,0.008921964,0.034327235,0.013794593,0.020379039,0.01142019,0.014699213
#CON_E_0000001 [0. 0. 0. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
#0.83648527,0.45631966,0.99126196,0.9988726,0.6785029,0.004784134,0.0004508688,0.00043345214,0.0016622879,0.84797055,0.69319564,0.19595724,0.11150587,0.09370159,0.04687846,0.5911434
#0.85591173,0.29964823,0.6474535,0.9958448,0.74161965,0.6078001,0.1674229,0.011574941,0.019979684,0.47452322,0.31543937,0.018440481,0.0017411511,0.0006799721,6.610596e-05,0.0006315805
# import sandbox.util_frontend as nii_front_end
#
# _front_end=nii_front_end.LFCC(320,
#                    160,
#                    512,
#                    16000,
#                    20,
#                    with_energy=True)
from scipy.io import wavfile
import numpy as np
import librosa
_wav_file_='/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof/train_dev_eval/con_wav/'+ \
                                'CON_E_0000002.wav'
# (sig, rate) = librosa.load(_wav_file_, sr=16000)
#x=torch.tensor(sig).unsqueeze(0)
#x_sp_amp = _front_end(x)
#LFCC=x_sp_amp.squeeze(0).numpy()

# -*- coding:utf-8 -*-
import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
def oscillogram_spectrum(audio_path):
    """
    画出音频文件audio_path的声波图和频谱图
    :param audio_path:音频文件路径
    :return:
    """
    # 读取wav文件
    filename = audio_path
    wavefile = wave.open(filename, 'r')  # open for writing
    # 读取wav文件的四种信息的函数。期中numframes表示一共读取了几个frames。
    nchannels = wavefile.getnchannels()
    sample_width = wavefile.getsampwidth()
    framerate = wavefile.getframerate()
    numframes = wavefile.getnframes()
    print("channel", nchannels)
    print("sample_width", sample_width)
    print("framerate", framerate)
    print("numframes", numframes)
    # 建一个y的数列，用来保存后面读的每个frame的amplitude。
    y = np.zeros(numframes)
    # for循环，readframe(1)每次读一个frame，取其前两位，是左声道的信息。右声道就是后两位啦。
    # unpack是struct里的一个函数，简单说来就是把＃packed的string转换成原来的数据，无论是什么样的数据都返回一个tuple。这里返回的是长度为一的一个
    # tuple，所以我们取它的第零位。
    for i in range(numframes):
        val = wavefile.readframes(1)
        left = val[0:2]
        # right = val[2:4]
        v = struct.unpack('h', left)[0]
        y[i] = v
    # framerate就是声音的采用率，文件初读取的值。
    Fs = framerate
    time = np.arange(0, numframes) * (1.0 / framerate)
    time = time[8000:3*16000]
    y = y[8000:3*16000]
    # 显示时域图(波形图)
    plt.figure(dpi=300, figsize=(24, 8))
    #plt.subplot(211)
    #plt.xlim(time)
    plt.plot(time, y)
    plt.show()
    # 显示频域图(频谱图)
    plt.figure(dpi=300, figsize=(24, 8))
    #plt.subplot(212)
    plt.specgram(y, NFFT=512, Fs=Fs, noverlap=160)
    plt.show()
if __name__ == '__main__':
    audio_path = _wav_file_
    oscillogram_spectrum(audio_path)





# import torch
# checkpoint = torch.load('/data/git_repository/research/ASVSpoof/partialspoof/partialspoof_MIL/project/'
#                         'partialspoof//lse-lse/mil/01/output/checkpoints/epoch_024.pt')
# print()
# tmp_best_name='/data/git_repository/research/ASVSpoof/partialspoof/partialspoof_MIL/project/'+\
#                         'partialspoof/local_mil/44/output/trained_network.pt'
# torch.save(checkpoint['state_dict'], tmp_best_name)
# print()
# import torch
# instance_features = torch.tensor([0.5,0.9,0.9,0.99])
# r = 9
# bag_features = torch.log(torch.mean(torch.exp(r * instance_features))) / r
# print(bag_features)
# import torch
# import torch.nn as nn
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# a=torch.tensor([1.0,2.0])
# b=torch.Tensor([0,1])
# loss_function = nn.BCEWithLogitsLoss()
# loss_function1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0,1.0]))
# loss_function2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0,10.0]))
# loss_function3 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0,1.0]))
# loss_function4 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0,100.0]))
# loss = loss_function(a,b)
# loss1 = loss_function1(a,b)
# loss2 = loss_function2(a,b)
# loss3 = loss_function3(a,b)
# loss4 = loss_function4(a,b)
# print(loss,loss1,loss2,loss3,loss4)


# import torch
# tensor1=torch.tensor([1,2,3,4,5])
# tensor2=torch.flip(tensor1, dims=[0])
# print(tensor2)
# import core_scripts.other_tools.str_tools as nii_str_tool
# import core_scripts.other_tools.list_tools as nii_list_tool
# tmp='/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof/protocols/PartialSpoof_LA_cm_protocols/'
# trn_list1=tmp+'PartialSpoof.LA.cm.train.trl.txt'
# trn_list2=tmp+'PartialSpoof.LA.cm.eval.trl.txt'
# trn_list3=tmp+'PartialSpoof.LA.cm.dev.trl.txt'
# lst=tmp+'PartialSpoof.LA.cm.train_dev_eval.trl.txt'
#
# trn_lst1 = nii_list_tool.read_list_from_text(trn_list1)
# trn_lst2 = nii_list_tool.read_list_from_text(trn_list2)
# trn_lst3 = nii_list_tool.read_list_from_text(trn_list3)
# list=trn_lst1+trn_lst2+trn_lst3
# nii_list_tool.write_list_to_text_file(list,lst)

# import numpy as np
# tmp = '/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'
#
# protocol_filepath=tmp + '/segment_labels/train_dev_eval_seglab_0.16.npy'
# data_buffer = np.load(protocol_filepath, allow_pickle=True).item()
# temp=data_buffer.keys()
# for key in temp:
#     print(key[:3])

# import numpy as np
# temppath='/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof/protocols/PartialSpoof_LA_cm_protocols/'
# protocol_filepaths=[
#     temppath+'PartialSpoof.LA.cm.dev.trl.txt',
#     temppath+'PartialSpoof.LA.cm.eval.trl.txt',
#     temppath+'PartialSpoof.LA.cm.train.trl.txt',
# ]
# temp_buffer0 = np.loadtxt(protocol_filepaths[0], dtype='str')
# temp_buffer1 = np.loadtxt(protocol_filepaths[1], dtype='str')
# temp_buffer2 = np.loadtxt(protocol_filepaths[2], dtype='str')
# temp_buffer=np.vstack((temp_buffer0, temp_buffer1,temp_buffer2))
# txt_file_name = temppath+'PartialSpoof.LA.cm.train_dev_eval.trl.txt'
# np.savetxt(txt_file_name,temp_buffer)


# for i in range(4,8):
#     print(i)
# import torch
#
# tensor1 = torch.tensor([[[3, 1], [2, 5], [2131, 12]], [[141, 545], [75, 12], [46, 35]]]).cuda()
# scores = torch.zeros_like(tensor1)
# tensor2= torch.zeros((1, 2), device=tensor1.device)
# sha=tensor1.shape[0]
# print()
# import torch
# tensor1=torch.tensor([[[3,1],[2,5],[2131,12]],[[141,545],[75,12],[46,35]]])
# #tensor2=torch.min(tensor1,dim=2)
# min_index=torch.argmin(tensor1[:,:,1],dim=1)
# index1=min_index[0].item()
# index2=min_index[1].item()
# batch1=tensor1[0,index1,:]
# batch2=tensor1[1,index2,:]
# batch=torch.stack([batch1,batch2],dim=0).unsqueeze(dim=1)
# print()

# res=torch.sigmoid(tensor1)
#
# mins = torch.min(tensor1, dim=1, keepdim=False)[0]
# val=mins.cpu().numpy()[0]
# print(val)


# import os
# import sys
# import torch
# #sh执行时,需要加入以下指令  当前sh所在位置等效于pycharm source一下上级目录
# #加入当前main所在路径,为了引入当前目录的程序如config
# sys.path.append('.')
# #加入项目路径,为了引入一级目录定义的程序包 core_scripts等
# sys.path.append('/data/git_repository/research/ASVSpoof/partialspoof/partialspoof_MIL')
# print(sys.path,'\n')#
# print(os.getcwd(),'\n')#输出的是sh当前目录
# print(os.path.abspath('.'),'\n')#输出的是sh当前目录
# print(os.path.abspath('..'),'\n')#获得当前工作目录的父目录
# print(os.path.abspath(os.curdir),'\n')#获得当前工作目录
# from transformer_models.modules import Linear
# model=Linear(1,1,True)
# torch.save(model.state_dict(), './01/output/checkpoints/trained_network.pt')


# import torch
# import numpy as np
#
# def get_batch_weights_from_attnmap(map_list):
#     """ attnmap:list[Tensor(BxHxFxW)...]
#     return weight(BxF)
#     """
#     map_0 = map_list[0]
#     B, H, F, W = map_0.size()[0], map_0.size()[1], map_0.size()[2], map_0.size()[3]
#     #weight = torch.zeros(B, F)
#     weights = []
#     for attn_map in map_list:
#         pad_tensor = torch.zeros(B, H, W // 2, W)
#         attn_map = torch.cat([pad_tensor, attn_map, pad_tensor], dim=2)
#         for i in range(F):
#             temp = torch.cat([attn_map[:, :, i + k: i + k + 1, W - 1 - k: W - k] for k in range(W)], dim=3)
#             if i == 0:
#                 weight = temp
#             else:
#                 weight = torch.cat([weight, temp], dim=2)
#         #weight:BxHxFxW
#         weight = torch.sum(torch.sum(weight, dim=3, keepdim=False), dim=1, keepdim=False)
#         # weight:BxF
#         weights.append(weight)
#     #BxLxF
#     weights = torch.stack([weight for weight in weights], dim=1)
#     weights = torch.sum(weights, dim=1, keepdim=False)
#     max = torch.max(weights, dim=1,keepdim=True)[0]
#     min = torch.min(weights, dim=1,keepdim=True)[0]
#     #归一化
#     weights = (weights - min) / (max - min)
#     #和为1
#     weights = weights / torch.sum(weights, dim=1, keepdim=True)
#     #mean = weights.mean(dim=1).unsqueeze(dim=1) #32x1
#     #std = weights.std(dim=1, unbiased=False).unsqueeze(dim=1)
#     #weights = (weights - mean) / std
#     return weights
#
# from matplotlib import pyplot as plt
# B=2
# H=2
# F=10
# W=3 #W//2 = 1
# tensor1 = torch.randn(B, H, F, W)
# tensor2 = torch.randn(B, H, F, W)
# tensor3 = torch.randn(B, H, F, W)
# map_list = [tensor1, tensor2, tensor3]
# for l in range(len(map_list)):
#     attn_map = map_list[l]
#     for b in range(B):
#         for h in range(H):
#             plot_map = attn_map[b, h, :, :].numpy()
#             plt.imshow(plot_map)  # 用imshow方法画图numpy数组
#             plt.title('%s,%s,%s'%(l,b,h))
#             plt.show()
#
# weights = get_batch_weights_from_attnmap(map_list)
# B,F = weights.size()[0],weights.size()[1]
# for i in range(B):
#     value = weights[i,:].numpy()
#     plt.plot(value,label="weight")
#     plt.show()
# import torch
# def get_batch_weights_from_globl_attnmap(actual_len, map_list):
#     """ attnmap:list[Tensor(BxHxFxF)...]
#         W=F do F->actual_len
#     return weight(BxF)
#     """
#     map_0 = map_list[0][:, :, :actual_len, :actual_len]
#     B, H, F, W = map_0.size()[0], map_0.size()[1], map_0.size()[2], map_0.size()[3]
#     # weight = torch.zeros(B, F)
#     weights = []
#     for attn_map in map_list:
#         attn_map=attn_map[:, :, :actual_len, :actual_len]
#         # weight:BxHxFxF
#         weight = torch.sum(torch.sum(attn_map, dim=3, keepdim=False), dim=1, keepdim=False)
#         # weight:BxF
#         weights.append(weight)
#     # BxLxF
#     weights = torch.stack([weight for weight in weights], dim=1)
#     weights = torch.sum(weights, dim=1, keepdim=False)
#     max = torch.max(weights, dim=1, keepdim=True)[0]
#     min = torch.min(weights, dim=1, keepdim=True)[0]
#     # 归一化
#     weights = (weights - min) / (max - min)
#     # 和为1
#     weights = weights / torch.sum(weights, dim=1, keepdim=True)
#     # mean = weights.mean(dim=1).unsqueeze(dim=1) #32x1
#     # std = weights.std(dim=1, unbiased=False).unsqueeze(dim=1)
#     # weights = (weights - mean) / std
#     return weights
#
# from matplotlib import pyplot as plt
# B=2
# H=2
# F=10
# W=10 #W//2 = 1
# tensor1 = torch.randn(B, H, F, W)
# tensor2 = torch.randn(B, H, F, W)
# tensor3 = torch.randn(B, H, F, W)
# map_list = [tensor1, tensor2, tensor3]
# for l in range(len(map_list)):
#     attn_map = map_list[l]
#     for b in range(B):
#         for h in range(H):
#             plot_map = attn_map[b, h, :, :].numpy()
#             plt.imshow(plot_map)  # 用imshow方法画图numpy数组
#             plt.title('%s,%s,%s'%(l,b,h))
#             plt.show()

# weights = get_batch_weights_from_globl_attnmap(8,map_list)
# B,F = weights.size()[0],weights.size()[1]
# for i in range(B):
#     value = weights[i,:].numpy()
#     plt.plot(value,label="weight")
#     plt.show()

# import numpy as np
#
# from matplotlib import pyplot as plt
#
# fig = plt.figure(figsize=(12,8), dpi=72)
# x = np.arange(0, 1, 0.01)
#
# y = np.log2(1 + np.exp(1-1/(x+np.finfo(np.float32).eps)))
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title("milpooling")
# plt.plot(x, y,label="milpooling")
# plt.show()
# res1 = np.log2(1 + np.exp(1-1/(0.5 + np.finfo(np.float32).eps)))
# print(res1)
# 将label改为float32
import numpy as np
# data_buffer1 = np.load('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'+\
#                       '/segment_labels/eval_seglab_0.16.npy', allow_pickle=True).item()
# data_buffer2 = np.load('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'+\
#                       '/segment_labels/train_dev_seglab_0.16.npy', allow_pickle=True).item()
# train_dev_eval_dev={}
# for key in data_buffer1.keys():
#     train_dev_eval_dev[key] = data_buffer1[key].astype(np.float32)
# for key in data_buffer2.keys():
#     train_dev_eval_dev[key] = data_buffer2[key].astype(np.float32)
# np.save('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'+\
#                       '/segment_labels/train_dev_eval_seglab_0.16.npy', train_dev_eval_dev)
# train_dev_eval_dev=np.load('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'+\
#                       '/segment_labels/train_dev_eval_seglab_0.16.npy', allow_pickle=True).item()
# print(1)

# data_buffer = np.load('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'+\
#                       '/segment_labels/train_dev_eval_seglab_0.16(old).npy', allow_pickle=True).item()
# train_dev_eval_dev={}
# time=0
# for key in data_buffer.keys():
#     time=time+1
#     length=data_buffer[key].size
#     val = np.ones((length,),dtype=np.float32)
#     train_dev_eval_dev[key] = val - data_buffer[key]
# np.save('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'+\
#                       '/segment_labels/train_dev_eval_seglab_0.16.npy', train_dev_eval_dev)
# print(time)

# data_buffer1 = np.load('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'+\
#                        '/segment_labels/train_dev_seglab_0.16(old).npy', allow_pickle=True).item()
# data_buffer2 = np.load('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'+\
#                        '/segment_labels/train_dev_eval_seglab_0.16.npy', allow_pickle=True).item()


# data1,data2={},{}
# for i in range(10000):
#     data1[str(i)] = np.ones((10,),dtype=np.float32)
#     data2[str(i)] = np.zeros((10,), dtype=np.float32)
#
# np.save('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'+\
#                        '/segment_labels/data1.npy', data1)
# np.save('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof'+\
#                        '/segment_labels/data2.npy', data2)

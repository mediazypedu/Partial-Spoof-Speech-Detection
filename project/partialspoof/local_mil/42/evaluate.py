#!/usr/bin/python
""" 
Wrapper to parse the score file and compute EER and min tDCF

Usage:
python 00_evaluate.py log_file

"""

import os
import sys

# sh执行时,需要加入以下指令  当前sh所在位置等效于pycharm source一下上级目录
# 加入当前main所在路径,为了引入当前目录的程序如config
sys.path.append('.')
# 加入项目路径,为了引入一级目录定义的程序包 core_scripts等
sys.path.append('/data/git_repository/research/ASVSpoof/partialspoof/partialspoof_MIL')
import numpy as np
from sandbox import eval_asvspoof
from sandbox.eval_asvspoof import compute_eer
import t_DCF.eval_metrics as em



#
def parse_txt_seg(file_path):
    bonafide = []
    spoofed = []
    train_dev_eval_labels = np.load('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof' + \
                                    '/segment_labels/train_dev_eval_seglab_0.16.npy', allow_pickle=True).item()
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            line = line.replace("\n", "")
            types = line.split('|')
            info = types[0].split(',')
            # utt = types[1].split(',')
            segs = types[2].split(',')
            key = info[1]
            label = train_dev_eval_labels[key]
            #length = (int(info[3]) // 160 + 1) // 16
            # length = len(segs)
            # assert len(segs)==(len(label)+1)
            for i in range(len(label)):
                flag = int(label[i])
                #计算eer,tdcf默认假为0真为1
                score = 1 - float(segs[i])
                if flag:
                    spoofed.append(score)
                else:
                    bonafide.append(score)
    bonafide = np.array(bonafide)
    spoofed = np.array(spoofed)
    return bonafide, spoofed
def parse_txt_utt(file_path):
    """
    基础版本,查询所有utt的eer
    """
    bonafide = []
    spoofed = []

    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            line = line.replace("\n", "")
            types = line.split('|')
            info = types[0].split(',')
            #############################
            score = 1 - float(types[1])
            #############################
            key = info[1]
            if key.startswith('CON'):
                spoofed.append(score)
            else:
                bonafide.append(score)
    bonafide = np.array(bonafide)
    spoofed = np.array(spoofed)
    return bonafide, spoofed
def get_ratios_form_label(labels):
    bonafide, spoofed = 0., 0.
    for i in labels:
        if i == 0:
            bonafide = bonafide + 1
        else:
            spoofed = spoofed + 1
    ratio = spoofed / len(labels)
    if ratio == 0:
        return 0
    elif ratio > 0. and ratio <= 0.1:
        return 1
    elif ratio > 0.1 and ratio <= 0.2:
        return 2
    elif ratio > 0.2 and ratio <= 0.3:
        return 3
    elif ratio > 0.3 and ratio <= 0.4:
        return 4
    elif ratio > 0.4 and ratio <= 0.5:
        return 5
    elif ratio > 0.5 and ratio <= 0.6:
        return 6
    elif ratio > 0.6 and ratio <= 0.7:
        return 7
    elif ratio > 0.7 and ratio <= 0.8:
        return 8
    elif ratio > 0.8 and ratio <= 0.9:
        return 9
    elif ratio > 0.9 and ratio <= 1.0:
        return 10
def parse_txt_utt_with_diffRatios(file_path):
    """
    查询不同spoof_radios的utteer
    0.65/47.63/99.80%
    0.72/46.78/99.77%
    0.23/41.97/99.81%
    暂固定为0~5%,5~10%.....95~100% key=index
    """
    bonafide = []
    spoofed = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    train_dev_eval_labels = np.load('/data/git_repository/research/ASVSpoof/partialspoof/dataset/partialspoof' + \
                                    '/segment_labels/train_dev_eval_seglab_0.16.npy', allow_pickle=True).item()
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            line = line.replace("\n", "")
            types = line.split('|')
            info = types[0].split(',')
            #############################
            score = 1 - float(types[1])
            #############################
            key = info[1]
            flag = key.startswith('CON')
            if flag:
                seg_labels = train_dev_eval_labels[key]
                radio_index = get_ratios_form_label(seg_labels)
                spoofed[radio_index].append(score)
            else:
                bonafide.append(score)

    bonafide = np.array(bonafide)
    for key in spoofed.keys():
        spoofed[key] = np.array(spoofed[key])
    return bonafide, spoofed

def get_min_tdcf(bonafide,spoofed, asv_score_file):
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }
    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv, non_asv, spoof_asv = [], [], []
    length = asv_data.shape[0]
    for i in range(length):
        if asv_sources[i] == 'spoof':
            spoof_asv.append(asv_scores[i])
        else:
            if asv_keys[i] == 'target':
                tar_asv.append(asv_scores[i])
            else:
                non_asv.append(asv_scores[i])
    tar_asv = np.array(tar_asv)
    non_asv = np.array(non_asv)
    spoof_asv = np.array(spoof_asv)

    # Extract bona fide (real bonafide) and spoof scores from the CM scores
    # bona_cm = cm_scores[cm_keys == 'bonafide']
    # spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)
    # Compute t-DCF
    tDCF_curve, CM_thresholds = em.compute_tDCF(bonafide, spoofed, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model,False)
    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]
    print('min-tDCF= {}'.format(min_tDCF))


import argparse

parser = argparse.ArgumentParser(description='General argument parse')
parser.add_argument('--eval_score', type=str, default="./output/eval_score.txt")
parser.add_argument('--dev_score', type=str, default="./output/dev_score.txt")
args = parser.parse_args()
if __name__ == "__main__":
    cm_score_file = args.dev_score
    bonafide, spoofed = parse_txt_seg(cm_score_file)
    eer_cm, eer_threshold = compute_eer(bonafide, spoofed)
    print("DEV_SEG_EER: %2.3f %%\tThreshold: %f" % (eer_cm * 100, eer_threshold))

    bonafide, spoofed = parse_txt_utt(cm_score_file)
    eer_cm, eer_threshold = compute_eer(bonafide, spoofed)
    print("DEV_UTT_EER: %2.3f %%\tThreshold: %f" % (eer_cm * 100, eer_threshold))

    asv_score_file = '/data/git_repository/research/ASVSpoof/partialspoof/partialspoof_MIL/t_DCF/' \
                     'ASV_scores/PartialSpoof_LA_asv_scores/PartialSpoof.LA.asv.dev.gi.trl.scores.txt'
    get_min_tdcf(bonafide,spoofed,asv_score_file)

    cm_score_file = args.eval_score
    bonafide, spoofed = parse_txt_seg(cm_score_file)
    eer_cm, eer_threshold = compute_eer(bonafide, spoofed)
    print("EVAL_SEG_EER: %2.3f %%\tThreshold: %f" % (eer_cm * 100, eer_threshold))

    bonafide, spoofed = parse_txt_utt(cm_score_file)
    eer_cm, eer_threshold = compute_eer(bonafide, spoofed)
    print("EVAL_UTT_EER: %2.3f %%\tThreshold: %f" % (eer_cm * 100, eer_threshold))

    asv_score_file = '/data/git_repository/research/ASVSpoof/partialspoof/partialspoof_MIL/t_DCF/' \
                     '/ASV_scores/PartialSpoof_LA_asv_scores/PartialSpoof.LA.asv.eval.gi.trl.scores.txt'
    get_min_tdcf(bonafide,spoofed, asv_score_file)
    ########################################################################################
    cm_score_file = args.dev_score
    bonafide, spoofed = parse_txt_utt_with_diffRatios(cm_score_file)
    for key in spoofed.keys():
        #print(key,len(spoofed[key]))
        eer_cm, eer_threshold = compute_eer(bonafide, spoofed[key])
        print("%s DEV_UTT_EER: %2.3f %%\tThreshold: %f" % (key, eer_cm * 100, eer_threshold))

    cm_score_file = args.eval_score
    bonafide, spoofed = parse_txt_utt_with_diffRatios(cm_score_file)
    for key in spoofed.keys():
        #print(key, len(spoofed[key]))
        eer_cm, eer_threshold = compute_eer(bonafide, spoofed[key])
        print("%s EVAL_UTT_EER: %2.3f %%\tThreshold: %f" % (key, eer_cm * 100, eer_threshold))

    # mintDCF, eer, threshold = eval_asvspoof.tDCF_wrapper(bonafide, spoofed)
    # print("mintDCF: %f\tEER: %2.3f %%\tThreshold: %f" % (mintDCF, eer * 100,threshold))

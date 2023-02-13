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
            length = (int(info[3]) // 160 + 1) // 16
            # length = len(segs)
            # assert len(segs)==(len(label)+1)
            for i in range(length):
                flag = int(label[i])
                if flag:
                    spoofed.append(float(segs[i]))
                else:
                    bonafide.append(float(segs[i]))
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
            score = float(types[1])
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
            score = float(types[1])
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


import argparse

parser = argparse.ArgumentParser(description='General argument parse')
parser.add_argument('--eval_score', type=str, default="./output/eval_score.txt")
parser.add_argument('--dev_score', type=str, default="./output/dev_score.txt")
args = parser.parse_args()
if __name__ == "__main__":
    data_path = args.dev_score
    bonafide, spoofed = parse_txt_seg(data_path)
    eer_cm, eer_threshold = compute_eer(spoofed, bonafide)
    print("DEV_SEG_EER: %2.3f %%\tThreshold: %f" % (eer_cm * 100, eer_threshold))

    bonafide, spoofed = parse_txt_utt(data_path)
    eer_cm, eer_threshold = compute_eer(spoofed, bonafide)
    print("DEV_UTT_EER: %2.3f %%\tThreshold: %f" % (eer_cm * 100, eer_threshold))

    data_path = args.eval_score
    bonafide, spoofed = parse_txt_seg(data_path)
    eer_cm, eer_threshold = compute_eer(spoofed, bonafide)
    print("EVAL_SEG_EER: %2.3f %%\tThreshold: %f" % (eer_cm * 100, eer_threshold))

    bonafide, spoofed = parse_txt_utt(data_path)
    eer_cm, eer_threshold = compute_eer(spoofed, bonafide)
    print("EVAL_UTT_EER: %2.3f %%\tThreshold: %f" % (eer_cm * 100, eer_threshold))
    ########################################################################################
    data_path = args.dev_score
    bonafide, spoofed = parse_txt_utt_with_diffRatios(data_path)
    for key in spoofed.keys():
        print(key,len(spoofed[key]))
        eer_cm, eer_threshold = compute_eer(spoofed[key], bonafide)
        print("%s DEV_UTT_EER: %2.3f %%\tThreshold: %f" % (key, eer_cm * 100, eer_threshold))

    data_path = args.eval_score
    bonafide, spoofed = parse_txt_utt_with_diffRatios(data_path)
    for key in spoofed.keys():
        print(key, len(spoofed[key]))
        eer_cm, eer_threshold = compute_eer(spoofed[key], bonafide)
        print("%s EVAL_UTT_EER: %2.3f %%\tThreshold: %f" % (key, eer_cm * 100, eer_threshold))

    # mintDCF, eer, threshold = eval_asvspoof.tDCF_wrapper(bonafide, spoofed)
    # print("mintDCF: %f\tEER: %2.3f %%\tThreshold: %f" % (mintDCF, eer * 100,threshold))

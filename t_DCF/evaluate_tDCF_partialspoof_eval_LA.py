import sys
import numpy as np
import t_DCF.eval_metrics as em
import matplotlib.pyplot as plt


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


# Replace CM scores with your own scores or provide score file as the first argument.
#cm_scores_file =  './cm_scores/dev_score_mil_partial.txt'
cm_scores_file =  './cm_scores/eval_score_mil_partial.txt'
#cm_scores_file =  './cm_scores/dev_score_local_mil_partial.txt'
#cm_scores_file =  './cm_scores/eval_score_local_mil_partial.txt'
#cm_scores_file =  './cm_scores/dev_score_mil_2019topartial.txt'
#cm_scores_file =  './cm_scores/eval_score_mil_2019topartial.txt'
#cm_scores_file =  './cm_scores/dev_score_local_mil_2019topartial.txt'
#cm_scores_file =  './cm_scores/eval_score_local_mil_2019topartial.txt'
# Replace ASV scores with organizers' scores or provide score file as the second argument.
#asv_score_file = './ASV_scores/PartialSpoof_LA_asv_scores/PartialSpoof.LA.asv.dev.gi.trl.scores.txt'
asv_score_file = './ASV_scores/PartialSpoof_LA_asv_scores/PartialSpoof.LA.asv.eval.gi.trl.scores.txt'
# Fix tandem detection cost function (t-DCF) parameters
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

# Load CM scores
# cm_data = np.genfromtxt(cm_scores_file, dtype=str)
# cm_utt_id = cm_data[:, 0]
# cm_sources = cm_data[:, 1]
# cm_keys = cm_data[:, 2]
# cm_scores = cm_data[:, 3].astype(np.float)
bona_cm, spoof_cm = parse_txt_utt(cm_scores_file)
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
eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
[Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

# Compute t-DCF
tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)

# Minimum t-DCF
min_tDCF_index = np.argmin(tDCF_curve)
min_tDCF = tDCF_curve[min_tDCF_index]

print('ASV SYSTEM')
print('   EER            = {:8.5f} % (Equal error rate (target vs. nontarget discrimination)'.format(eer_asv * 100))
print('   Pfa            = {:8.5f} % (False acceptance rate of nontargets)'.format(Pfa_asv * 100))
print('   Pmiss          = {:8.5f} % (False rejection rate of targets)'.format(Pmiss_asv * 100))
print('   1-Pmiss,spoof  = {:8.5f} % (Spoof false acceptance rate)'.format((1 - Pmiss_spoof_asv) * 100))

print('\nCM SYSTEM')
print('   EER            = {:8.9f} % (Equal error rate for countermeasure)'.format(eer_cm * 100))

print('\nTANDEM')
print('   min-tDCF       = {:8.9f}'.format(min_tDCF))

# Visualize ASV scores and CM scores
plt.figure()
ax = plt.subplot(121)
plt.hist(tar_asv, histtype='step', density=True, bins=50, label='Target')
plt.hist(non_asv, histtype='step', density=True, bins=50, label='Nontarget')
plt.hist(spoof_asv, histtype='step', density=True, bins=50, label='Spoof')
plt.plot(asv_threshold, 0, 'o', markersize=10, mfc='none', mew=2, clip_on=False, label='EER threshold')
plt.legend()
plt.xlabel('ASV score')
plt.ylabel('Density')
plt.title('ASV score histogram')

ax = plt.subplot(122)
plt.hist(bona_cm, histtype='step', density=True, bins=50, label='Bona fide')
plt.hist(spoof_cm, histtype='step', density=True, bins=50, label='Spoof')
plt.legend()
plt.xlabel('CM score')
# plt.ylabel('Density')
plt.title('CM score histogram')

# Plot t-DCF as function of the CM threshold.
plt.figure()
plt.plot(CM_thresholds, tDCF_curve)
plt.plot(CM_thresholds[min_tDCF_index], min_tDCF, 'o', markersize=10, mfc='none', mew=2)
plt.xlabel('CM threshold index (operating point)')
plt.ylabel('Norm t-DCF')
plt.title('Normalized tandem t-DCF')
plt.plot([np.min(CM_thresholds), np.max(CM_thresholds)], [1, 1], '--', color='black')
plt.legend(('t-DCF', 'min t-DCF ({:.9f})'.format(min_tDCF), 'Arbitrarily bad CM (Norm t-DCF=1)'))
plt.xlim([np.min(CM_thresholds), np.max(CM_thresholds)])
plt.ylim([0, 1.5])

plt.show()

"""
Author@Xinzi Xu

This code is released for academic and research purposes only, as part of the publication:
"CardioLike-Net: An Edge-end Inter-patient Arrhythmia Classifier with Quantization-aware-training for Wearable ECG Applications"

Permission is granted to use, copy, and modify the code **for non-commercial research purposes**, provided that proper citation is given to the above paper.

Any commercial use of this code or any derivative work is strictly prohibited without explicit written permission from the author.

If you find this work useful in your research, please consider citing the following paper: XXX
"""
import numpy as np
import scipy.io as scio
import random
import os
# from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt
import torch
# from tensorflow.keras.metrics import MeanIoU
from typing import List
import os.path
L = 256
LEADS = ['avf', 'avl', 'avr', 'i', 'ii',
         'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
lued_path = './seglued_250/'
qtdb_path = './segqtdb_250/'
mitdb_path = './segmitdb/'
ptb_path = './segptbdb_250/'
ptbxl_path = './segptbxldb_250/'

np.random.seed(2)




mit_id = np.array([100, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
                  100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234], dtype=int)


def quan_xbit (input, bit):
    print('input', input)
    output = torch.floor(input*(2**bit))
    print('output', output/(2**bit))
    return output
def quan_data(bit, data):
    ECG_max = np.max(data)
    # print('ECG_max',ECG_max)
    ECG_min = np.min(data)
    # print('ECG_min',ECG_min)
    ECG_minmax = (data - ECG_min) / (ECG_max - ECG_min)
    data_quan = np.round((ECG_minmax - 0.5) * 2**bit * 0.9) + 2**(bit - 1)
    return data_quan



def preprocessing_0phase(ecg, lf_enable, hf_enable):
    # High-pass filter parameters
    fc = 0.6
    si = 4  # sampling interval (ms)
    delta = 0.6
    g = np.tan(np.pi * fc * si * 0.001)

    g1 = 1 + 2 * delta * g + g**2
    g2 = 2 * g**2 - 2
    g3 = 1 - 2 * delta * g + g**2

    b1 = g**2 / g1
    b3 = b1
    b2 = 2 * b1
    a2 = g2 / g1
    a3 = g3 / g1

    b = [b1, b2, b3]
    a = [1, a2, a3]

    dec_bit = 16

    ecg_e = ecg * (2 ** dec_bit)

    ecg_bl = ecg_e.copy()
    ecg_bl[0] = int((b[0] + b[1] + b[2] - a[1] - a[2]) * ecg_e[0] / a[0])
    ecg_bl[1] = int((b[0] * ecg_e[1] + (b[1] + b[2] - a[2]) * ecg_e[0] - a[1] * ecg_bl[0]) / a[0])

    for i in range(2, len(ecg_e)):
        ecg_bl[i] = int((b[0] * ecg_e[i] + b[1] * ecg_e[i - 1] + b[2] * ecg_e[i - 2] - 
                         a[1] * ecg_bl[i - 1] - a[2] * ecg_bl[i - 2]) / a[0])

    ecg_hf = ecg_e - ecg_bl

    ecg_hf = ecg_hf/int(2 ** dec_bit)

    if hf_enable:
        ecg_hf = ecg_hf
    else:
        ecg_hf = ecg

    # Low-pass filter parameters
    fc = 30
    si = 4
    delta = 0.6
    g = np.tan(np.pi * fc * si * 0.001)

    g1 = 1 + 2 * delta * g + g**2
    g2 = 2 * g**2 - 2
    g3 = 1 - 2 * delta * g + g**2

    b1 = g**2 / g1
    b3 = b1
    b2 = 2 * b1
    a2 = g2 / g1
    a3 = g3 / g1

    b = [b1, b2, b3]
    a = [1, a2, a3]

    dec_bit = 16

    ecg_e = ecg_hf * (2 ** dec_bit)

    ecg_bl = ecg_e.copy()
    ecg_bl[0] = int((b[0] + b[1] + b[2] - a[1] - a[2]) * ecg_e[0] / a[0])
    ecg_bl[1] = int((b[0] * ecg_e[1] + (b[1] + b[2] - a[2]) * ecg_e[0] - a[1] * ecg_bl[0]) / a[0])

    for i in range(2, len(ecg_e)):
        ecg_bl[i] = int((b[0] * ecg_e[i] + b[1] * ecg_e[i - 1] + b[2] * ecg_e[i - 2] - 
                         a[1] * ecg_bl[i - 1] - a[2] * ecg_bl[i - 2]) / a[0])

    ecg_lf = ecg_bl / (2 ** dec_bit)

    if lf_enable:
        ecg_lf = ecg_lf
    else:
        ecg_lf = ecg_hf

    # Smoothing filter
    window = np.ones(8) / 8
    ecg_f = np.convolve(ecg_lf, window, mode='same').astype(int)
    ecg_f = ecg_f[:len(ecg)]

    return ecg_f

def segmentation(data, rpeaks):
    pre = 106
    post = 150
    data_segs = []
    feas = []


    for j in range(1, rpeaks.shape[0] - 1):  # MATLAB uses 1-based indexing
        # too long RR
        pre_RR = rpeaks[j] - rpeaks[j - 1]
        post_RR = rpeaks[j + 1] - rpeaks[j]

        if pre_RR > 500 or post_RR > 500:
            continue

        peak = rpeaks[j]
        seg = data[peak - pre:peak + post]
        
        fea = np.expand_dims(np.array([post_RR,pre_RR ]),axis = 0)
        # print('fea', fea.shape)
        data_segs.append(seg)
        feas.append(fea)
    data_segs = np.concatenate(np.expand_dims(data_segs,axis=0), axis=0)
    # print('data_segs', data_segs.shape)
    feas =  np.concatenate(feas, axis=0)
    # print('feas', feas.shape)
    return data_segs, feas


def load_mit_matfile(ids, base_path, thres=0, feas_en=False, dict_en=False):
    if dict_en:
        data = {}
        labels = {}
        ids = [ids]
        for id in ids:
            data_mat = os.path.join(base_path, str(id)+"_seg.mat")
            mat = np.array(scio.loadmat(data_mat)['segs'])
            mat = np.transpose(mat, axes=(2, 1, 0))
            if id == 114:
                mat[:, :, [0, 1]] = mat[:, :, [1, 0]]
            data[str(id)] = mat
            label_mat = os.path.join(base_path, str(id)+"_labelseg.mat")
            mat = np.array(scio.loadmat(label_mat)['labels'])
            labels[str(id)] = mat
        return data, labels

    else:
        labels = []
        data = []
        feas = []

        ids = [ids]
        for id in ids:
            if thres != 0:
                label_mat = os.path.join(base_path, str(
                    id)+"_labelseg_"+str(thres)+".mat")
            else:
                label_mat = os.path.join(base_path, str(id)+"_labelseg.mat")
            mat = np.array(scio.loadmat(label_mat)['labels'])
            labels.append(mat)
            if thres != 0:
                data_mat = os.path.join(
                    base_path, str(id)+"_seg_"+str(thres)+".mat")
            else:
                data_mat = os.path.join(base_path, str(id)+"_seg.mat")
            mat = np.array(scio.loadmat(data_mat)['segs'])
            mat = np.transpose(mat, axes=(2, 1, 0))
            if id == 114:
                mat[:, :, [0, 1]] = mat[:, :, [1, 0]]
            data.append(mat)
            if feas_en:
                fea_mat = os.path.join(base_path, str(id)+"_feaseg.mat")
                mat = np.array(scio.loadmat(fea_mat)['fea_plus'])
                feas.append(mat)

        labels = np.concatenate(labels, axis=1).T
        data = np.concatenate(data, axis=0)
        if feas_en:
            feas = np.concatenate(feas, axis=0)
            return data, labels, feas
        else:
            return data, labels


def find_wave_onset(wave_category: list) -> np.ndarray:
    onsets = []
    prev = 0
    for i, val in enumerate(wave_category):
        if val != 0 and prev == 0:
            onsets.append(i)
        prev = val
    return np.array(onsets)


def find_wave_offset(wave_category: list) -> np.ndarray:
    offsets = []
    prev = 0
    for i, val in enumerate(wave_category):
        if val == 0 and prev != 0:
            offsets.append(i)
        prev = val
    return np.array(offsets)


def find_wave(label):
    onset = find_wave_onset(label)
    offset = find_wave_offset(label)
    if len(onset) > len(offset):
        assert len(onset) == len(offset) + 1
        if onset[-1] == L-1:
            onset = np.delete(onset, -1)
        else:
            offset = np.append(offset, len(label)-1)
    assert len(onset) == len(offset)
    wave_info = np.vstack((onset, offset)).T
    wave_info = np.int32(wave_info)
    return wave_info


def trans_one_hot_label(label):
    num_classes = 4
    one_hot_codes = np.eye(num_classes)
    one_hot_label = np.zeros((label.shape[0], num_classes, label.shape[1]))

    for i in range(label.shape[0]):
        for j in range(label.shape[1]):

            if (label[i, j] == 0):
                one_hot_label[i, :, j] = one_hot_codes[0]
            if (label[i, j] == 1):
                one_hot_label[i, :, j] = one_hot_codes[1]
            if (label[i, j] == 2):
                one_hot_label[i, :, j] = one_hot_codes[2]
            if (label[i, j] == 3):
                one_hot_label[i, :, j] = one_hot_codes[3]
    return one_hot_label


def mask_to_onehot(mask):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    palette = [0, 1, 2, 3]
    semantic_map = []
    for label in palette:
        equality = np.equal(mask, label)
        # print(equality.shape)
        # class_map = np.all(equality, axis=0)
        semantic_map.append(equality)
        # print(class_map.shape)
    semantic_map = np.stack(semantic_map, axis=1).astype(np.float32)
    # print(semantic_map.shape)

    return semantic_map


def get_inference_data():
    data = load_matfile(new_luedid, lued_path,
                        label=False, leads_combine=False)
    label = mask_to_onehot(load_matfile(
        new_luedid, lued_path, label=True, leads_combine=False))

    return data, label



class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_seg(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = target.size(0)
    input_length = target.size(2)

    target = torch.argmax(target, 1)
    if output.dim() == 3:
        pred = torch.argmax(output, 1)
    else:
        pred = output

    # print(pred)
    # print(target)
    # print(pred[0])
    # print(target[0])
    # print('pred',pred.shape)
    # print('target',target.shape)
    correct = (pred == target).sum().item()
    # print(correct)
    res = correct*100/(input_length * batch_size)

    return res


def accuracy_clf(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = output.size(0)

    if output.dim() == 2:
        pred = torch.argmax(output, 1)
    else:
        pred = output

    # print(pred.shape)
    # print(target.shape)
    correct = (pred == target.squeeze()).sum().item()
    # print(correct)
    res = correct*100/(batch_size)

    return res


def cal_statistic(cm, class_num):
    total_pred = cm.sum(0)
    total_true = cm.sum(1)
    # total_true = np.array([17703,   491,  1357,   159])
    # special acc, abnormal inlcuded only
    acc_SP = sum([cm[i, i] for i in range(1, class_num)]) / \
        total_pred[1:class_num].sum()
    pre_i = [cm[i, i] / total_pred[i] for i in range(class_num)]
    rec_i = [cm[i, i] / total_true[i] for i in range(class_num)]
    F1_i = [2 * pre_i[i] * rec_i[i] / (pre_i[i] + rec_i[i])
            for i in range(class_num)]

    pre_i = np.array(pre_i)
    rec_i = np.array(rec_i)
    F1_i = np.array(F1_i)
    pre_i[np.isnan(pre_i)] = 0
    rec_i[np.isnan(rec_i)] = 0
    F1_i[np.isnan(F1_i)] = 0

    return acc_SP, list(pre_i), list(rec_i), list(F1_i)


def mask2time(mask):
    time_change = [0]
    for i in range(1, mask.shape[0]):
        if (mask[i] != mask[i-1]):
            time_change.append(i)
    time_change.append(mask.shape[0])
    return time_change





def get_probably_matching_timepoints(shorter: np.ndarray, longer: np.ndarray):

    indices_minimizind_distancies = [
        np.argmin(row) for row in np.abs(np.subtract.outer(shorter, longer))
    ]

    return longer[indices_minimizind_distancies]


def get_true_false_errs_from_mismatching_timepoints(shorter: np.ndarray, longer: np.ndarray, toll: int):
    if len(shorter) == 0:
        return len(longer), 0, []

    falses = len(longer) - len(shorter)

    matched_longer = get_probably_matching_timepoints(shorter, longer)

    dists = np.abs(matched_longer - shorter)

    trues = np.sum(dists <= toll)
    errs = list(dists[dists < toll])

    return falses, trues, errs


WAVE_TO_COLUMN = {
    "None": 0,
    "P": 1,
    "QRS": 2,
    "T": 3,
}

WAVE_DATA = {
    "None": None,
    "P": {},
    "QRS": {},
    "T": {},
}





def post_processing(label, min_length):
    post_label = label
    for i in range(label.shape[0]):
        bg_duration = 0
        for j in range(label.shape[1]-1):
            if label[i, j] == 0:
                bg_duration = bg_duration + 1
            else:
                if bg_duration <= min_length:

                    if (j - bg_duration > 0) and (label[i, j-bg_duration-1] == label[i, j]):

                        post_label[i, j-bg_duration: j] = label[i,
                                                                j-bg_duration-1]
                        bg_duration = 0
                    else:
                        bg_duration = 0
                else:
                    bg_duration = 0

    for i in range(label.shape[0]):
        bg_duration = 0
        for j in range(label.shape[1]-1):
            if label[i, j] == 1:
                bg_duration = bg_duration + 1
            else:
                if bg_duration <= 3:

                    if (j - bg_duration > 0) and (label[i, j-bg_duration-1] == label[i, j]):

                        post_label[i, j-bg_duration: j] = label[i,
                                                                j-bg_duration-1]
                        bg_duration = 0
                    else:
                        bg_duration = 0
                else:
                    bg_duration = 0

    for i in range(label.shape[0]):
        bg_duration = 0
        for j in range(label.shape[1]-1):
            if label[i, j] == 2:
                bg_duration = bg_duration + 1
            else:
                if bg_duration <= 3:

                    if (j - bg_duration > 0) and (label[i, j-bg_duration-1] == label[i, j]):

                        post_label[i, j-bg_duration: j] = label[i,
                                                                j-bg_duration-1]
                        bg_duration = 0
                    else:
                        bg_duration = 0
                else:
                    bg_duration = 0

    for i in range(label.shape[0]):
        bg_duration = 0
        for j in range(label.shape[1]-1):
            if label[i, j] == 3:
                bg_duration = bg_duration + 1
            else:
                if bg_duration <= 3:

                    if (j - bg_duration > 0) and (label[i, j-bg_duration-1] == label[i, j]):

                        post_label[i, j-bg_duration: j] = label[i,
                                                                j-bg_duration-1]
                        bg_duration = 0
                    else:
                        bg_duration = 0
                else:
                    bg_duration = 0


    return post_label


def prepare_dirs(config):
    for path in [config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    model_name = config.model_name
    filename = model_name + '_params.json'
    param_path = os.path.join(config.ckpt_dir, filename)
    print(model_name)
    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def transition_matrix(transitions, N, dir):
    transitions = np.int32(transitions)
    if transitions.shape[1] == 4:
        transitions = np.argmax(transitions, axis=1)
    total_batch = transitions.shape[0]
    length = transitions.shape[1]

    num_class = 4  # number of states

    M = np.zeros((num_class, num_class))

    if dir == 1:
        for n_batch in range(total_batch):
            for (i, j) in zip(transitions[n_batch, length//2:], transitions[n_batch, length//2+N:]):

                M[i, j] += 1
    else:
        for n_batch in range(total_batch):
            for (i, j) in zip(transitions[n_batch, N:length//2], transitions[n_batch, 0:length//2-N]):
                M[i, j] += 1

    # print(M)
    # now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M


def probability_layer(output, label, step, loss_weight):

    p1_forward = transition_matrix(label, step[0], 1)
    print('p1_forward', p1_forward)
    p1_backward = transition_matrix(label, step[0], -1)
    print('p1_backward', p1_backward)
    p5_forward = transition_matrix(label, step[1], 1)
    print('p5_forward', p5_forward)
    p5_backward = transition_matrix(label, step[1], -1)
    print('p5_backward', p5_backward)
    p10_forward = transition_matrix(label, step[2], 1)
    print('p10_forward', p10_forward)
    p10_backward = transition_matrix(label, step[2], -1)
    print('p10_backward', p10_backward)
    B, C, L = output.shape[0], output.shape[1], output.shape[2]
    output_post = output
    for b in range(B):
        for l in range(step[2], L-step[2]):
            if l < 256:
                out = output[b, :, l]

                output_post[b, :, l-step[2]] = loss_weight[0]*output[b, :, l-step[2]] + loss_weight[1] * np.matmul(
                    p1_backward, out[:, None]).T + loss_weight[2]*np.matmul(p5_backward, out[:, None]).T + loss_weight[3]*np.matmul(p10_backward, out[:, None]).T
            else:

               output_post[b, :, l+step[2]] = loss_weight[0]*output[b, :, l+step[2]] + loss_weight[1] * np.matmul(
                    p1_forward, out[:, None]).T + loss_weight[2] * np.matmul(p5_forward, out[:, None]).T + loss_weight[3]*np.matmul(p10_forward, out[:, None]).T
    return output_post


def determin_peak(start, finish, ecg, TH_max, TH_min):
    peak = np.max(ecg[start: finish])
    peak_loc = np.argmax(ecg[start: finish]) + start
    valley = np.min(ecg[start: finish])
    valley_loc = np.argmin(ecg[start: finish]) + start
    if peak < TH_max and valley < TH_min:
        return valley_loc
    else:
        return peak_loc


def plot_with_points(ecg, label, points, index):
    time_change = mask2time(label)

    # print(time_change)
    fig = plt.figure(figsize=(28, 20))
    ax = plt.subplot(211)
    ax.plot(ecg)
    for i in range(1, len(time_change)):
        if label[time_change[i]-1] == 0:
            # print('black',i)
            seg = np.arange(time_change[i-1], time_change[i])
            ax.plot(seg, ecg[seg], color='black', linewidth=5)

        elif label[time_change[i]-1] == 1:
            seg = np.arange(time_change[i-1], time_change[i])
            ax.plot(seg, ecg[seg], color='red', linewidth=5)
            # print('red',i)
        elif label[time_change[i]-1] == 2:
            seg = np.arange(time_change[i-1], time_change[i])
            ax.plot(seg, ecg[seg], color='green', linewidth=5)
            # print('green',i)
        elif label[time_change[i]-1] == 3:
            seg = np.arange(time_change[i-1], time_change[i])
            ax.plot(seg, ecg[seg], color='blue', linewidth=5)
            # print('blue',i)
    ax2 = plt.subplot(212)

    mark = ['o', '*', 'o', 'o', '*', 'o', 'o', '*', 'o']
    col = ['g', 'g', 'g', 'r', 'r', 'r', 'b', 'b', 'b']

    ax2.plot(ecg)
    for i in range(len(points)):
        if points[i] != -3000:
            point = int(points[i])
            ax2.scatter(points[i], ecg[point], marker=mark[i], c=col[i], s=50)

    plt.show()


def determin_tpeak(start, finish, ecg, TH_max, TH_min):
    peak = np.max(ecg[start: finish])
    peak_loc = np.argmax(ecg[start: finish]) + start
    valley = np.min(ecg[start: finish])
    valley_loc = np.argmin(ecg[start: finish]) + start
    peak_rel = abs(peak-TH_max)
    valley_rel = abs(TH_min-valley)
    # print('peak_rel', peak_rel)
    # print('valley_rel', valley_rel)
    peak_in = True
    if peak_rel < 20 and valley_rel < 20:

        if finish < len(ecg)-1:

            for j in range(finish, len(ecg)-1):

                if (ecg[j+1]-ecg[j])*(ecg[j]-ecg[j-1]) <= 0:
                    peak_in = False
                    break

            return j, peak_in
        else:
            return peak_loc, peak_in
    else:
        if abs(peak-TH_max) >= abs(TH_min-valley):
            return peak_loc, peak_in
        else:
            return valley_loc, peak_in


def determin_ppeak(start, finish, ecg, TH_max, TH_min):
    peak = np.max(ecg[start: finish])
    peak_loc = np.argmax(ecg[start: finish]) + start
    valley = np.min(ecg[start: finish])
    valley_loc = np.argmin(ecg[start: finish]) + start
    peak_rel = abs(peak-TH_max)
    valley_rel = abs(TH_min-valley)

    peak_in = True

    if abs(peak-TH_max) >= abs(TH_min-valley):
        return peak_loc, peak_in
    else:
        return valley_loc, peak_in


def determin_peak(start, finish, ecg, TH_max, TH_min):
    peak = np.max(ecg[start: finish])
    peak_loc = np.argmax(ecg[start: finish]) + start
    valley = np.min(ecg[start: finish])
    valley_loc = np.argmin(ecg[start: finish]) + start
    peak_rel = abs(peak-TH_max)
    valley_rel = abs(TH_min-valley)

    if abs(peak-TH_max) >= abs(TH_min-valley):
        return peak_loc
    else:
        return valley_loc


def cal_slope(x, i):
    slope = abs(((x[i+1]-x[i]) - (x[i]-x[i-1])))
    return slope


def slope_max(ecg, loc, min_len):
    slope_max = 0
    slope_max_loc = loc
    if min_len > 0:
        if loc - 2 > 0 & loc+min_len+2 < len(ecg)-1:
            for i in range(loc, loc+min_len+1):
                slope = cal_slope(ecg, i)
                print(slope)
                if slope > slope_max:
                    slope_max_loc = i
                    slope_max = slope
                else:
                    slope_max_loc = slope_max_loc
        else:
            # print(loc)
            slope_max_loc = loc

    return slope_max_loc


def cal_isoline(points, ecg):
    if (points[-1] != -3000) and (points[5] != -3000):

        iso_line = np.median(ecg[int(points[6])] - ecg[int(points[5])])
    elif points[5] != -3000:
        iso_line = 0
    else:
        iso_line = 0
    return iso_line


def fiducial_points(label, ecg, id):

    if label.shape[1] != 4:
        label_onehot = trans_one_hot_label(label)
    # print(ecg.shape)
    B, L = label_onehot.shape[0], label_onehot.shape[2]
    # ecg = ecg[:,:,0]
    ecg = np.squeeze(ecg)

    points = np.zeros((B, 9))
    iso_line = np.zeros((B, 1))
    for b in range(B):

        qrs_info = find_wave(label_onehot[b, 2, :])
        # print('qrs_info',qrs_info)
        p_info = find_wave(label_onehot[b, 1, :])
        # print('p_info',p_info)
        t_info = find_wave(label_onehot[b, 3, :])
        # print('t_info',t_info)

        if len(qrs_info) != 0:
            for qrs_i in range(qrs_info.shape[0]):

                if (((106) <= qrs_info[qrs_i, 1]) and ((106) >= qrs_info[qrs_i, 0])):
                    # print('o',qrs_info[qrs_i, 0])
                    # print('1',qrs_info[qrs_i, 1])
                    q_on = qrs_info[qrs_i, 0]
                    # q_on = slope_max(ecg[b,:], q_on, 5)
                    s_off = qrs_info[qrs_i, 1]
                    # s_off = slope_max(ecg[b,:], s_off, 20)
                    # print('q_on', q_on)
                    # print('s_off',s_off)
                    ############################# trt -> r #################################
                    if (len(t_info) != 0):
                        delete = []
                        for k in range(len(t_info)):

                            if (qrs_info[qrs_i, 0] == t_info[k, 1]) and (t_info[k, 1]-t_info[k, 0] < 15):
                                q_on = t_info[k, 0]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                            if (qrs_info[qrs_i, 1] == t_info[k, 0]) and (t_info[k, 1]-t_info[k, 0] < 15):
                                s_off = t_info[k, 1]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                        t_info = np.delete(t_info, delete, axis=0)
                    ########################################################################
                    break

                else:
                    dis = np.min(np.abs(qrs_info - L//2), axis=1)
                    qrs_i = np.argmin(dis)
                    q_on = qrs_info[qrs_i, 0]
                    s_off = qrs_info[qrs_i, 1]

                    ############################# trt -> r #################################
                    if (len(t_info) != 0):
                        delete = []
                        for k in range(len(t_info)):

                            if (qrs_info[qrs_i, 0] == t_info[k, 1]) and (t_info[k, 1]-t_info[k, 0] < 15):

                                q_on = t_info[k, 0]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                            if (qrs_info[qrs_i, 1] == t_info[k, 0]) and (t_info[k, 1]-t_info[k, 0] < 15):
                                print(t_info[k, 1])
                                s_off = t_info[k, 1]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                        t_info = np.delete(t_info, delete, axis=0)
                    ########################################################################
                    break
            # print(q_on, s_off)
                    # s_off = slope_max(ecg[b,:], s_off, 20)
            r = determin_peak(q_on, s_off, ecg[b, :], TH_max=max(
                ecg[b, q_on], ecg[b, s_off]), TH_min=min(ecg[b, q_on], ecg[b, s_off]))
            if len(t_info) != 0:
                t_i = np.argmin(np.abs(np.min((t_info - s_off), axis=1)))
                t_on = t_info[t_i, 0]
                # t_on = slope_max(ecg[b,:], t_on, 5)
                t_off = t_info[t_i, 1]
                # t_off = slope_max(ecg[b,:], t_off, 5)

                if (t_on - s_off) > 150 or (t_on < s_off):
                    t_on = -3000
                    t_off = -3000
                    t = -3000
                    print(b, 't is detected but not in the right place')
                else:

                    t, peak_in = determin_tpeak(t_on, t_off, ecg[b, :], TH_max=max(
                        ecg[b, t_on], ecg[b, t_off]), TH_min=min(ecg[b, t_on], ecg[b, t_off]))

                    if not peak_in:
                        t_off = min(t + 15, L-1)

            else:
                t_on = -3000
                t_off = -3000
                t = -3000
                print(b, 'no t is detected')

            if len(p_info) != 0:

                p_i = np.argmin(np.abs(np.min((p_info - q_on), axis=1)))
                p_on = p_info[p_i, 0]
                # p_on = slope_max(ecg[b,:], p_on, 5)
                p_off = p_info[p_i, 1]
                # p_off = slope_max(ecg[b,:], p_off, 5)

                if (q_on - p_off > 150) or (q_on < p_off):
                    p_on = -3000
                    p_off = -3000
                    p = -3000
                    print(b, 'p is detected but not in the right place')
                else:

                    p, peak_in = determin_ppeak(p_on, p_off, ecg[b, :], TH_max=max(
                        ecg[b, p_on], ecg[b, p_off]), TH_min=min(ecg[b, p_on], ecg[b, p_off]))
                    # print(peak_in)
                    if (not peak_in):
                        p_on = -3000
                        p_off = -3000
                        p = -3000
                        print(b, 'no p is detected')
            else:
                p_on = -3000
                p_off = -3000
                p = -3000
                print(b, 'no p is detected')
        else:
            q_on = -3000
            s_off = -3000
            r = -3000
            p_on = -3000
            p_off = -3000
            p = -3000
            t_on = -3000
            t_off = -3000
            t = -3000
            print(b, 'no qrs is detected')
            points[b, :] = [p_on, p, p_off, q_on, r, s_off, t_on, t, t_off]
            # plot_with_points(ecg[b,:], label[b,:],points[b,:], b)

        points[b, :] = [p_on, p, p_off, q_on, r, s_off, t_on, t, t_off]
        # iso_line [b] = cal_isoline(points[b,:], ecg[b,:])
        # print(points[b,:])
        # plot_with_points(ecg[b,:], label[b,:],points[b,:], b,id)
    return points


def embedding(ecg, em_len=30, em_thres=8):  # num, 256
    # print(ecg.shape)
    ###################### old #######################
    # l = ecg.shape[0]
    # thres = 8
    # ecg_em = np.zeros((em_len,))
    # # print(ecg_em.shape)

    # for j in range(1,min(l,em_len)):
    #     if (ecg[j] - ecg[j-1])>thres:
    #         ecg_em[ j] = 1
    #     elif (ecg[j] - ecg[j-1])<-thres:
    #         ecg_em[j] =-1
    #     else:
    #         ecg_em[j] = 0

    ###################### new #######################
    l = ecg.shape[0]
    thres = em_thres
    ecg_em = []
    # print(ecg_em.shape)

    for j in range(1, min(l, em_len)):
        if (ecg[j] - ecg[j-1]) > thres:
            ecg_em.append(1)
        elif (ecg[j] - ecg[j-1]) < -thres:
            ecg_em.append(-1)
    if (len(ecg_em)):
        ecg_em = np.array(ecg_em)
        if len(ecg_em) > em_len:
            ecg_em = ecg_em[0:em_len]
        else:
            ecg_em = np.concatenate((ecg_em, np.zeros((em_len-len(ecg_em),))))
    else:
        ecg_em = np.zeros((em_len,))

    return ecg_em



def rbf(x1, x2, gamma):
    return np.exp(-gamma*np.linalg.norm(x1-x2)**2)


def predict_self(clf, input_data, gamma):
    support_vectors = clf.support_vectors_
    n_SV = support_vectors.shape[0]
    dual_coef = clf.dual_coef_
    intercept = clf.intercept_
    decision_value = np.ones((input_data.shape[0], 1))
    decision_result = np.ones((input_data.shape[0], 1))
    predict_label = np.ones((input_data.shape[0], 1))
    for i in range(input_data.shape[0]):
        kernel_value = np.ones((n_SV, 1))
        for j in range(n_SV):
            kernel_value[j] = float(
                rbf(support_vectors[j, :], input_data[i, :], gamma))
        decision_value[i] = float(np.dot(dual_coef, kernel_value)+intercept)
        decision_result[i] = np.sign(decision_value[i])

        if (decision_result[i] == 0) or (decision_result[i] == -1):
            predict_label[i] = 0
    return decision_value, predict_label


def judge_mi(preds, preds_probas, mode='direct'):
    if mode == 'direct':
        pred_sum = np.sum(preds, axis=1)
        pred_final = np.zeros_like(pred_sum)
        for num in range(pred_sum.shape[0]):
            if pred_sum[num] > 9:
                pred_final[num] = 1
        pred_final = np.squeeze(pred_final)
    elif mode == 'prob':
        # for b in range(preds_probas.shape[0]):
        #     for lead in range(preds_probas.shape[2]):
        #         if preds_probas[b, 0, lead] > 0.5:
        # print(b, lead, preds_probas[b, 0, lead])
        pred_sum = np.sum(preds_probas, axis=2)
        pred_sum = np.squeeze(pred_sum)
        pred_final = np.zeros(pred_sum.shape[0])

        for num in range(pred_sum.shape[0]):
            print(num)
            print(pred_sum[num, 0])
            print(pred_sum[num, 1])
            if pred_sum[num, 1] > pred_sum[num, 0]:
                pred_final[num] = 1
    # elif mode == 'prob_confident':
    #     for num in range():

    return pred_final


def label_eval(points, label):
    label_post = label

    B, L = label.shape[0], label.shape[1]
    for b in range(B):
        if (points[b, 0] != -3000):
            start = points[b, 0]
        elif (points[b, 3] != -3000):
            start = points[b, 3]
        elif (points[b, 6] != -3000):
            start = points[b, 6]
        else:
            start = None
        if (points[b, 8] != -3000):
            finish = points[b, 8]
        elif (points[b, 5] != -3000):
            finish = points[b, 5]
        elif (points[b, 2] != -3000):
            finish = points[b, 2]
        else:
            finish = None
        finish = np.int32(finish)
        start = np.int32(start)
        # print('finish', finish)
        # print('start', start)
        # assert (finish == None) and (start == None)
        print(b)
        print(start)
        print(finish)
        if finish != None:
            label_post[b, 0:start] = 0
            label_post[b, finish:L] = 0
        else:
            label_post[b, :] = 0
        print(label_post[b, :])
    return label_post

def dec2bnr(dec: int, lenth: int = 12) -> str:
    """十进制数转指定长度二进制补码(BNR)
    Args:
        dec (int): 十进制数
        length (int, optional): 指定长度(正数高位补0，负数高位补1). 

    Raises:
        TypeError: 输入非十进制整数！
        OverflowError: 输入十进制整数过大，超过指定补码长度

    Returns:
        str: 返回二进制补码字符串
    """
    if not isinstance(dec, int):
        raise TypeError("输入非十进制整数！")

    # 计算十进制转化为二进制后的位数
    digits = (len(bin(dec)) - 3) if dec < 0 else (len(bin(dec)) - 2)

    if digits > lenth:

        raise OverflowError("输入十进制整数过大，超过指定补码长度")

    # Note: dec & 相同位数的0b111...强制转换为补码形式
    pattern = f"{dec & int('0b' + '1' * lenth, 2):0{lenth}b}"

    return pattern


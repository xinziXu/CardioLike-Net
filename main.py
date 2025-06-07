"""
Author@Xinzi Xu

This code is released for academic and research purposes only, as part of the publication:
"CardioLike-Net: An Edge-end Inter-patient Arrhythmia Classifier with Quantization-aware-training for Wearable ECG Applications"

Permission is granted to use, copy, and modify the code **for non-commercial research purposes**, provided that proper citation is given to the above paper.

Any commercial use of this code or any derivative work is strictly prohibited without explicit written permission from the author.

If you find this work useful in your research, please consider citing the following paper: XXX
"""


import numpy as np
from utils import *
from CardioVI.modules_torch import *
from CardioVI.network import *
import torch
from CardioCOG.features import Features
from CardioCOG.utils import *
from CardioCOG.network import *
from scipy.interpolate import interp1d


present_samplerate = 250  # Specify the desired sampling rate
original_samplerate = 360  # Assuming the original sampling rate of the MIT-BIH database




####################### load input from database ######################y
import wfdb
data_file = f'./Samples/mitdb/100'
data, fields = wfdb.rdsamp(data_file, sampfrom=0, channels=[0])  # Load ECG data, assuming lead II is used

####################### resample ######################y
sampletime = np.arange(len(data)) / original_samplerate
# resampletime = np.arange(len(data)) / present_samplerate
new_length = int(len(data) * present_samplerate / original_samplerate)
resampletime = np.arange(new_length) / present_samplerate
interp_func = interp1d(sampletime, data.flatten(), kind='linear',bounds_error=False, fill_value="extrapolate" )
data = interp_func(resampletime)
####################### Quantization to 12-bit resolution ######################y 
ECG_12bit = quan_data(12, data)

####################### preprocessing ######################
data_f = preprocessing_0phase(ECG_12bit, 1, 1) 
####################### R-peak detection ######################
import numpy as np
from biosppy.signals import ecg

# Assuming ecg_signal is your ECG signal and sampling_rate is the sampling rate
rpeaks = ecg.ecg(signal=data_f, sampling_rate=present_samplerate, show=False)['rpeaks']

###################### Segmentation ######################
data ,rr = segmentation(data_f, rpeaks)


####################################CardioLike-Net##############################################################################

ckpt_path = './CardioVI/bilstm_q1_onlyiiTrue_transTruebce_bound_model_best_qat.pth'
ckpt = torch.load(ckpt_path, map_location= lambda storage, loc:storage)
state = ckpt['model_state']

dec2_2 = CardioVI_forward_int(data,state)
####################################################FEATURE_MAP#######################################################################
predicted = torch.argmax(dec2_2, 1)
# print(predicted)
# np.save('predicted.npy',predicted)
predicted_post = post_processing(predicted, 10)
# print('predicted_post',predicted_post.shape)
points, dirs = fiducial_points_vary(predicted_post, data)

points_dirs_id =  np.concatenate((points,dirs), axis =1)
FeatureExtraction = Features(data, points_dirs_id, rr)
features = FeatureExtraction.feature_map()

####################################################CardioCOG#######################################################################
ckpt_path = './CardioCOG/ckpt/ann_model_best.pth'
ckpt = torch.load(ckpt_path,map_location= lambda storage, loc:storage)
state = ckpt['model_state']

input = torch.Tensor(features)
fc2 = CardioCOG_forward(input,state)

result = np.array(torch.argmax(fc2, axis = 1))
Type = {0:'N', 1:'S',2:'V', 3:'F',4:'Q' }

'p_on, p, p_off, pq, q, r, s, st, t_on, t, t_off, p_dir, t_dir'
for index in range(rr.shape[0]):
    print('第',index,'个心拍:')
    print('前一个RR间隔为:',rr[index,0])
    print('后一个RR间隔为:',rr[index,1])
    if index > 8:
        print('心律失常类型为:',Type[result[index-9]])
        print('p_on所在位置:',points_dirs_id[index-9,0])
        print('p所在位置:',points_dirs_id[index-9,1])
        print('p_off所在位置:',points_dirs_id[index-9,2])
        print('pq所在位置:',points_dirs_id[index-9,3])
        print('q所在位置:',points_dirs_id[index-9,4])
        print('r所在位置:',points_dirs_id[index-9,5])
        print('s所在位置:',points_dirs_id[index-9,6])
        print('st所在位置:',points_dirs_id[index-9,7])
        print('t_on所在位置:',points_dirs_id[index-9,8])
        print('t所在位置:',points_dirs_id[index-9,9])
        print('t_off所在位置:',points_dirs_id[index-9,10])
        print('p方向:',points_dirs_id[index-9,11])
        print('t方向:',points_dirs_id[index-9,12])

###########################################################################################################################
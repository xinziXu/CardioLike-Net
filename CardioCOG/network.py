"""
Author@Xinzi Xu

This code is released for academic and research purposes only, as part of the publication:
"CardioLike-Net: An Edge-end Inter-patient Arrhythmia Classifier with Quantization-aware-training for Wearable ECG Applications"

Permission is granted to use, copy, and modify the code **for non-commercial research purposes**, provided that proper citation is given to the above paper.

Any commercial use of this code or any derivative work is strictly prohibited without explicit written permission from the author.

If you find this work useful in your research, please consider citing the following paper: XXX
"""
import numpy as np
import torch
import os
# from modules_npy import conv1d_npy, bn1d_npy, relu_npy, conv1dtrans_npy,lstm_npy,conv1d_npy_quan,conv1dtrans_npy_quan,lstm_npy_quan,bn1d_npy_quan,bit, cnn_bn_fusion

bit = 10
def Quant(input, scale, zero_point):

    x_q = torch.clamp(torch.round(input/scale + zero_point),0,128 )

    return x_q

def DeQuant(input, scale, zero_point):

    # print('scale',scale)
    # print('zp', zero_point)
    # input = input.int()
    # print(input[0,0,0])
    # print(input[0,0,0] - zero_point)
    x_dq = scale * (input - zero_point)
    # print(x_dq[0,0,0])
    return x_dq

def quan_xbit (input, bit = bit):
    output = torch.floor(input*(2**bit))
    return output

def Linear_Relu(input, W, bias ):
    '''
    input: [batch_size, in_features]
    W: [out_features, in_features]
    bias: [out_features]
    output: [batch_size, out_features]
    '''
    
    batch_size = input.shape[0]
    out_features = W.shape[0]
    bias_expand = bias.unsqueeze(0)
    bias_expand.repeat(batch_size,out_features)
    output = torch.matmul(input, W.T) + bias_expand
    zero = torch.zeros_like(output)
    output = torch.max(output, zero)
    return output

def Linear_Relu_quan(input, W, bias, is_fl = False, save_weight = False, layer_id = 1):
    '''
    input: [batch_size, in_features]
    W: [out_features, in_features]
    bias: [out_features]
    output: [batch_size, out_features]
    '''
    # input = quan_xbit(input)
    
    
    W = quan_xbit(W)
    bias = quan_xbit(bias)
    # print('W',W[0,:])
    # print('b',bias[0])

    if (save_weight):
        route_w = '../text/output_dec/ann'+str(layer_id)+'_16_weight.txt'
        with open(route_w, 'a') as txt:
            np.savetxt(route_w, W.detach().cpu().flatten(), delimiter='\n')  
        route_b = '../text/output_dec/ann'+str(layer_id)+'_16_bias.txt'
        with open(route_b, 'a') as txt:
            np.savetxt(route_b, bias.detach().cpu().flatten(), delimiter='\n')          
    batch_size = input.shape[0]
    out_features = W.shape[0]
    bias_expand = bias.unsqueeze(0)
    bias_expand.repeat(batch_size,out_features)

    if is_fl:
        # print('input[:9]',input[:23])
        # print('W[0,:9].T',W[13,:23].T)
        # aa = torch.matmul(input[:23], W[13,:23].T)
        # print(aa)
        output = torch.matmul(input, W.T) + bias_expand
        # print('output',output)
    else:
        # print('input[:9]',input[0,:16])
        # print('W[0,:9].T',W[0,:16].T)
        # aa = torch.matmul(input[0,:14], W[0,:14].T)
        # print(aa)
        output = torch.floor(torch.matmul(input, W.T)/(2**bit)) + bias_expand
        # print(output)
    zero = torch.zeros_like(output)
    output = torch.max(output, zero)
    return output

def CardioCOG_forward(input,state):
    
    fc1 = Linear_Relu_quan(input, state['ann.linear1.weight'], state['ann.linear1.bias'], is_fl = True)

    fc2 = Linear_Relu_quan(fc1, state['ann.linear2.weight'], state['ann.linear2.bias'])

    # route = './data_torch/' + 'fc2_self_quan.txt'
    # tmp = fc2[0].detach().cpu().numpy()
    # with open(route, 'a') as txt:
    #     np.savetxt(txt, tmp, delimiter='\n')     
    return fc2





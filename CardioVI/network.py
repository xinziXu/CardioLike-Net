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
# from SegNetwork.modules_npy import conv1d_npy, bn1d_npy, relu_npy, conv1dtrans_npy,lstm_npy,conv1d_npy_quan,conv1dtrans_npy_quan,lstm_npy_quan,bn1d_npy_quan,bit, cnn_bn_fusion
from SegNetwork.modules_torch import ConvTrans_torch_int, ConvFuse_torch_int, ConvBnReLU_torch, Lstm_torch,Quant, DeQuant,ConvTrans_torch,Lstm_int



def reshape_to_bias(input):
    return input.reshape(-1)
def reshape_to_weight(input):
    return input.reshape(-1, 1, 1) 
def fuse_bias(beta, running_mean, gamma, running_var, eps=1e-05 ):
    bias_fused = reshape_to_bias(beta - running_mean * gamma / torch.sqrt (running_var + eps))
    return bias_fused

def fuse_weight(weight, gamma, running_var, eps=1e-05):
    # print('weight',weight.shape)
    # print('gamma',gamma.shape)
    # print('running_var',running_var.shape)
    weight_fused = weight *reshape_to_weight(( gamma / torch.sqrt(running_var + eps)))
    # print('weight_fused',weight_fused.shape)
    return weight_fused

def CardioVI_forward_int(data,state):
    input = torch.from_numpy(data)
    input = torch.unsqueeze(input, 1) # expand for channel in
    # input = torch.unsqueeze(input, 0) # expand for batchsize
    # input = torch.unsqueeze(input, 0) # expand for batchsize
    # print('input',input)
    cnn_out = ConvBnReLU_torch(input, state['encoder.conv1.weight'], state['encoder.bn1.weight'], state['encoder.bn1.bias'], state['encoder.bn1.running_mean'], state['encoder.bn1.running_var'], padding = 3, stride = 4)
    # print('cnn_out',cnn_out)


    cnn_out = torch.permute(cnn_out, (0,2,1))
    cnn_out  = cnn_out/ (2**31)

    lstm_feature_int = Lstm_int(cnn_out, state, num_layers = 2)
    # print('lstm_feature_int',lstm_feature_int.shape)
    lstm_feature_int = torch.permute(lstm_feature_int, (0,2,1))
    dec1 = ConvTrans_torch_int(lstm_feature_int, state['lstm.ht_quantizer.scale'],\
                                                    state['decoder.dec1_1_conv.activation_quantizer.scale'],\
                                                    state['decoder.upsample_1.weight_quantizer.scale'],\
                                                    state['decoder.upsample_1.weight'],\
                                                    None,\
                                                    padding = 3, stride = 2,layer_id = 1)
    # print('dec1',dec1.shape)

    fused_bias = fuse_bias(state['decoder.dec1_1_conv.beta'],\
                            state['decoder.dec1_1_conv.running_mean'],\
                            state['decoder.dec1_1_conv.gamma'],\
                            state['decoder.dec1_1_conv.running_var'])

    fused_weight = fuse_weight(state['decoder.dec1_1_conv.weight'],\
                            state['decoder.dec1_1_conv.gamma'],\
                            state['decoder.dec1_1_conv.running_var'])
    dec1_1 = ConvFuse_torch_int(dec1, state['decoder.dec1_1_conv.activation_quantizer.scale'],\
                                        state['decoder.dec1_2_conv.activation_quantizer.scale'],\
                                        state['decoder.dec1_1_conv.weight_quantizer.scale'],\
                                        fused_weight,\
                                        fused_bias,\
                                        padding = 2, stride = 1, layer_id = 11)

    # print('dec1_1',dec1_1.shape)
    fused_bias = fuse_bias(state['decoder.dec1_2_conv.beta'],\
                            state['decoder.dec1_2_conv.running_mean'],\
                            state['decoder.dec1_2_conv.gamma'],\
                            state['decoder.dec1_2_conv.running_var'])
                            
    fused_weight = fuse_weight(state['decoder.dec1_2_conv.weight'],\
                            state['decoder.dec1_2_conv.gamma'],\
                            state['decoder.dec1_2_conv.running_var'])
    dec1_2 = ConvFuse_torch_int(dec1_1, state['decoder.dec1_2_conv.activation_quantizer.scale'],\
                                        state['decoder.upsample_2.activation_quantizer.scale'],\
                                        state['decoder.dec1_2_conv.weight_quantizer.scale'],\
                                        fused_weight,\
                                        fused_bias,\
                                        padding = 2, stride = 1,layer_id = 12)

    # print('dec1_2',dec1_2.shape)
    dec2 = ConvTrans_torch_int(dec1_2, state['decoder.upsample_2.activation_quantizer.scale'],\
                                        state['decoder.dec2_1_conv.activation_quantizer.scale'],\
                                        state['decoder.upsample_2.weight_quantizer.scale'],\
                                        state['decoder.upsample_2.weight'],\
                                        None,
                                        padding = 3, stride = 2, layer_id  =2)



    # print('dec2',dec2.shape)
    fused_bias = fuse_bias(state['decoder.dec2_1_conv.beta'],\
                            state['decoder.dec2_1_conv.running_mean'],\
                            state['decoder.dec2_1_conv.gamma'],\
                            state['decoder.dec2_1_conv.running_var'])
    fused_weight = fuse_weight(state['decoder.dec2_1_conv.weight'],\
                            state['decoder.dec2_1_conv.gamma'],\
                            state['decoder.dec2_1_conv.running_var'])

    dec2_1 = ConvFuse_torch_int(dec2, state['decoder.dec2_1_conv.activation_quantizer.scale'],\
                                        state['decoder.dec2_2_conv.activation_quantizer.scale'],\
                                        state['decoder.dec2_1_conv.weight_quantizer.scale'],\
                                        fused_weight,\
                                        fused_bias,\
                                        padding = 2, stride = 1, layer_id = 21 )

    # print('dec2_1',dec2_1.shape)



    fused_bias = fuse_bias(state['decoder.dec2_2_conv.beta'],\
                            state['decoder.dec2_2_conv.running_mean'],\
                            state['decoder.dec2_2_conv.gamma'],\
                            state['decoder.dec2_2_conv.running_var'])
    fused_weight = fuse_weight(state['decoder.dec2_2_conv.weight'],\
                            state['decoder.dec2_2_conv.gamma'],\
                            state['decoder.dec2_2_conv.running_var'])   
                
    dec2_2 = ConvFuse_torch_int(dec2_1, state['decoder.dec2_2_conv.activation_quantizer.scale'],\
                                        state['decoder.dec2_2_conv.last_activation_quantizer.scale'],\
                                        state['decoder.dec2_2_conv.weight_quantizer.scale'],\
                                        fused_weight,\
                                        fused_bias,\
                                        padding = 2, stride = 1,layer_id=22)

    # output = DeQuant(dec2_2, state['decoder.dec2_2_conv.last_activation_quantizer.scale'], torch.Tensor([0]))
    return dec2_2


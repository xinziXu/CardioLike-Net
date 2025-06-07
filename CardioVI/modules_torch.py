"""
Author@Xinzi Xu

This code is released for academic and research purposes only, as part of the publication:
"CardioLike-Net: An Edge-end Inter-patient Arrhythmia Classifier with Quantization-aware-training for Wearable ECG Applications"

Permission is granted to use, copy, and modify the code **for non-commercial research purposes**, provided that proper citation is given to the above paper.

Any commercial use of this code or any derivative work is strictly prohibited without explicit written permission from the author.

If you find this work useful in your research, please consider citing the following paper: XXX
"""
import torch
import numpy as np

from quantizer import SymmetricQuantizer, AsymmetricQuantizer,MovingAverageMinMaxObserver,MinMaxObserver
from torch.nn.quantized import Quantize
# from AP_Error import MA
from utils import *
'''
Fucntions including:
ConvBnReLU_torch
Lstm_torch
ConvTrans_torch_int
ConvFuse_torch_int
Conv1d_torch
'''
# P = np.load('../Approx_product.npy')

def save_as_txt(torch_data, route,save_name):
    # route = '/home/xuxinzi/ECG_CLF/code/code_matlab/text/'
    file_path = route + save_name
    numpy_data = torch_data.numpy()
    numpy_data_flatten = numpy_data.flatten()
    with open(save_name, 'a') as txt:
        np.savetxt(file_path, numpy_data_flatten, delimiter = '\n')

def re_order(weight, weight_type):
    '''
    weight [input, 4* hidden_size]
    block_size depends on the spad length 
    weight_type: weight, unit, bias
    '''
    if weight_type == 'weight':
        pe_num = 32
        spad_depth = 8
        input_size = weight.shape[0]
        hidden_size =  int(weight.shape[1]/4)
        block_size = int(input_size/spad_depth)
        pe_times = int(pe_num/2) # half for the weight, half for the unit
        gate_times = int(4*hidden_size/(pe_num/2)) #8 according to verilog        
        
        hidden_size = int(weight.shape[1]/4)
        weight_reorder =  torch.zeros((input_size*hidden_size*4,))
        # print(weight[:,:hidden_size])
        weight = torch.cat((weight[:,hidden_size:2*hidden_size], weight[:,:hidden_size],weight[:,2*hidden_size:]), dim = 1)# ft ahead
        
        weight = weight.T
        # print(weight[16,0:8])
        ids_row = range(pe_times)
        ids_column = range(block_size)
        cnt = 0
        for cnt_gt in range(gate_times):
            for cnt_bt in range(block_size):
                for cnt_pe in range(pe_times):
                    cnt = cnt + 1
                    id_row = ids_row[cnt_pe]
                    remainder = cnt_pe % block_size
                    id_column = ids_column[remainder]
                    idx_weight_reorder = cnt - 1
                    # print('id_row',id_row,'id_column',id_column,'idx_weight_reorder', idx_weight_reorder)
                    # print('id_column',id_column)
                    # print('idx_weight_reorder', idx_weight_reorder)
                    # print('weight_row',id_row+cnt_gt*pe_times,'weight_column',id_column*spad_depth)

                    # 
                    # print('weight_column',id_column*spad_depth)
                    weight_reorder [idx_weight_reorder*spad_depth: (idx_weight_reorder+1)*spad_depth] \
                    =  weight[id_row+cnt_gt*pe_times,id_column*spad_depth:id_column*spad_depth+spad_depth]
                    # print(weight_reorder[4*16:4*16+8])
                ########### LOOP SHIFTER ##################
                if block_size == 4:
                    ids_row = [ids_row[3]] + list(ids_row[0:3])+\
                            [ids_row[7]] + list(ids_row[4:7])+\
                            [ids_row[11]] + list(ids_row[8:11])+\
                            [ids_row[15]] + list(ids_row[12:15])
                elif block_size == 8:
                    ids_row = [ids_row[7]] + list(ids_row[0:7])+\
                              [ids_row[15]] + list(ids_row[8:15]) 
                else:
                    print('ERROR')
                                               
        
        # print(weight_reorder[4*16*8:4*16*8+8]) 
        return weight_reorder        
    elif weight_type == 'bias':
        hidden_size = int(weight.shape[0]/4)
        # print('bias',weight.shape)
        weight_reorder = torch.cat((weight[hidden_size:2*hidden_size], weight[:hidden_size],weight[2*hidden_size:]))# ft ahead
        return weight_reorder       
    else:
        print('you need enter a type of weight')

                        

bit = 31 # 1-bit for signed
bit_scale_lstm = 31
bit_scale_cnn = 31
def quan_xbit (input, bit):
    output = torch.floor(input*(2**bit))
    return output

def Conv1d_torch(input, Conv_W, Conv_B, padding, stride):
    '''
    input: [batch_size, channel_in, length]
    Conv_W: [channel_out, channel_in, kernel_size]
    Conv_B: [channel_out]
    output: [batch_size, channel_out, length_out]
    '''
    
    # print('Conv_W',Conv_W.shape)
    batch_size = input.shape[0]
    channel_in = input.shape[1]
    length_in = input.shape[2]
    channel_out = Conv_W.shape[0]
    kernel_size = Conv_W.shape[2]

    length_out = np.int32(np.floor((length_in + 2* padding -kernel_size )/stride)+1)
    output = torch.zeros((batch_size, channel_out, length_out))
    weight = quan_xbit(Conv_W, bit)
    bias = quan_xbit(Conv_B, bit)
    # weight = Conv_W
    # bias = Conv_B
    
    for bs in range(batch_size):
        data = input[bs]
        channel_sum = torch.zeros((length_out, channel_out))
        for cin in range(channel_in):
            data_ch = data[cin]
            data_pad = torch.nn.functional.pad(data_ch, (padding, padding), 'constant', 0)
            # print('data_pad', data_pad.shape)
            data_matrix = torch.zeros((length_out,kernel_size))
            
            for lo in range(length_out):
                # print('lo* stride +kernel_size',lo* stride +kernel_size)
                data_matrix[lo] = data_pad [lo * stride : lo* stride +kernel_size]
            kernel = weight[:, cin, :].T
            # print('kernel',kernel[:,0])
            # print('data_matrix', data_matrix[0])
            # print('data_matrix',data_matrix.device)
            output_ci = torch.matmul(data_matrix, kernel) # length_out * channel_out
            # print('output_ci',output_ci[2,1])
            channel_sum = channel_sum + output_ci
            # print('channel_sum',channel_sum.shape)
        channel_sum = torch.transpose(channel_sum, 0,1) # channel_out * length_out
        bias_expand = bias.unsqueeze(1)
        bias_expand.repeat(channel_out, length_out)
        output[bs] = channel_sum + bias_expand

    return output

def Relu_torch(input):
    zero = torch.zeros_like(input)
    return torch.maximum(input, zero)


def Sigmoid_linear_torch_quan(input):
    output = torch.zeros_like(input, dtype=torch.float32)

    for i in range(input.shape[0]):
        if input[i] <-1* (2**bit):
            output[i] =  -1* (2**bit)
        elif input[i] > 1* (2**bit):
            output[i] = 1* (2**bit)
        else:
            output[i] = input[i]
    return output

def sigmoid_linear_torch(input):
    output = torch.zeros_like(input)
    if len(input.size()) ==1:

        for i in range(input.shape[0]):
            if input[i] <-1:
                output[i] = -1
            elif input[i] >1:
                output[i] = 1
            else:
                output[i] = input[i]
    else: # bs, hs
        output = input.clamp(-1, 1)        
    # print('input',input)
    # print('output',output)
    return output

def Hardsigmoid_int(input, s1,  z1 = 0): # s1 = s2, z1 = z2
    one_div_s1 = 1/s1
    one_div_s1 = torch.tensor(one_div_s1, dtype = torch.float32)
    input = torch.tensor(input, dtype = torch.float32)
    if s1 >= 0:

        if input > (one_div_s1 + z1):
            output = torch.clamp(torch.round(one_div_s1), -127, 127)
        elif input < (-one_div_s1 + z1):
            output = torch.clamp(torch.round(-one_div_s1), -127, 127)
        else:
            output = torch.clamp(torch.round(input), -127, 127)


    else:
        if input < (one_div_s1 + z1):
            output = torch.clamp(torch.round(one_div_s1), -127, 127)
        elif input > (-one_div_s1+z1):
            output = torch.clamp(torch.round(-one_div_s1), -127, 127)
        else:
            output = torch.clamp(torch.round(input), -127, 127)
    
    return output

def Quant_Symmetric(input, scale, zero_point):
    
    x_q = torch.clamp(torch.round(input/scale + zero_point),-127,127 )
    return x_q

def Quant(input, scale, zero_point):

    x_q = torch.clamp(torch.round(input/scale + zero_point),0,128 )

    return x_q

def Quant_bias(input, scale, zero_point):
    x_q = torch.clamp(torch.round(input/scale + zero_point),-2**31,2**31-1 )
    return x_q
def DeQuant(input, scale, zero_point):
    scale = scale.to(input.device)
    zero_point = zero_point.to(input.device)
    # print('scale',scale)
    # print('zp', zero_point)
    # input = input.int()
    # print(input[0,0,0])
    # print(input[0,0,0] - zero_point)
    x_dq = scale * (input - zero_point)
    # print(x_dq[0,0,0])
    return x_dq

def ConvBnReLU_torch(input, Conv_W, Bn_W, Bn_B, Bn_M, Bn_V, padding, stride):
    '''
    input: [batch_size, channel_in, length]
    Conv_W: [channel_out, channel_in, kernel_size]
    Bn_W, Bn_B, Bn_M, Bn_V: [channel_out]
    output: [batch_size, channel_out, length_out]
    '''
    channel_out = Conv_W.shape[0]
    channel_in = Conv_W.shape[1]
    ks = Conv_W.shape[2]
    m_bn = torch.div(Bn_W,(torch.sqrt(Bn_V+ 0.0001)))
    m_bn_matrix = torch.diag(m_bn)
    
    n_bn = Bn_B - Bn_W * Bn_M / (torch.sqrt( Bn_V + 0.0001))
    # print('Conv_W', Conv_W.shape)
    Conv_W = torch.reshape(Conv_W, (channel_out, -1))
    # print('Conv_W', Conv_W.shape)
    Conv_Fuse_W = torch.matmul(m_bn_matrix , Conv_W)
    Conv_Fuse_B = n_bn
    Conv_Fuse_W =  Conv_Fuse_W.reshape((channel_out, channel_in, ks))
                      
    cnn_out = Conv1d_torch(input, Conv_Fuse_W, Conv_Fuse_B, padding = padding, stride = stride)

    cnn_out = Relu_torch(cnn_out)


    return output



def Lstm_int(input, parameters, num_layers, bidirectional = True):
    '''
    input [batch_size, length, input_size]
    weight [input, 4* hidden_size]
    unit [hidden_size, 4*hidden_size]
    bias [4 * hidden_size]
    '''
    torch.set_printoptions(precision=32)
    weight_name = [['lstm.weight_W0','lstm.weight_W0_reverse'],['lstm.weight_W1','lstm.weight_W1_reverse']]
    unit_name = [['lstm.weight_U0','lstm.weight_U0_reverse'],['lstm.weight_U1','lstm.weight_U1_reverse']]
    bias_name = [['lstm.bias_0','lstm.bias_0_reverse'],['lstm.bias_1','lstm.bias_1_reverse']]
    weight_scale_name = [['lstm.quantizer_W0.scale','lstm.quantizer_W0_reverse.scale'],['lstm.quantizer_W1.scale','lstm.quantizer_W1_reverse.scale']]
    weight_zp_name = [['lstm.quantizer_W0.zero_point','lstm.quantizer_W0_reverse.zero_point'],['lstm.quantizer_W1.zero_point','lstm.quantizer_W1_reverse.zero_point']]
    unit_scale_name = [['lstm.quantizer_U0.scale','lstm.quantizer_U0_reverse.scale'],['lstm.quantizer_U1.scale','lstm.quantizer_U1_reverse.scale']]
    unit_zp_name = [['lstm.quantizer_U0.zero_point','lstm.quantizer_U0_reverse.zero_point'],['lstm.quantizer_U1.zero_point','lstm.quantizer_U1_reverse.zero_point']]

    batch_size = input.shape[0]
    length = input.shape[1] 
    input_size = input.shape[2] 

    hidden_size = np.int32(parameters[weight_name[0][0]].shape[1]/4)
    num_direction = 2 if bidirectional else 1
    xt_scale = parameters['lstm.xt_quantizer.scale'] # fp16
    xt_scale  = torch.tensor(xt_scale, dtype=torch.float32)
    xt_zp = parameters['lstm.xt_quantizer.zero_point']

    ht_scale = parameters['lstm.ht_quantizer.scale'] # fp16
    ht_scale = torch.tensor(ht_scale, dtype=torch.float32)
    ht_zp = parameters['lstm.ht_quantizer.zero_point']

    gates_scale = parameters['lstm.gates_quantizer.scale'] # fp16
    gates_scale = torch.tensor(gates_scale, dtype=torch.float32)
    gates_zp = parameters['lstm.gates_quantizer.zero_point']

    ct_scale = parameters['lstm.ct_quantizer.scale'] # fp16
    ct_scale = torch.tensor(ct_scale, dtype=torch.float32)
    ct_zp = parameters['lstm.ct_quantizer.zero_point']
    # gt_scale = parameters['lstm.gt_quantizer.scale']
    # gt_zp = parameters['lstm.gt_quantizer.zero_point']


    # print('xt_scale', xt_scale)
    # print('ht_scale', ht_scale)
    # print('gates_scale', gates_scale)
    # print('ct_scale', ct_scale)
    # print('xt_zp', xt_zp)
    # print('ht_zp', ht_zp)
    # print('gates_zp', gates_zp)
    # print('ct_zp', ct_zp)    


    output = torch.zeros((batch_size, length, num_direction*hidden_size))
    
    
    inputs_q = Quant_Symmetric(input, xt_scale, xt_zp)
    # print('inputs_q', inputs_q[0])
    for bs in range(batch_size):
        h_t = torch.ones((hidden_size,1)) * ht_zp
        c_t = torch.ones((hidden_size,1)) * ct_zp


        for layer in range(num_layers):
            if layer == 0:
                x = inputs_q[bs] 
                xt_scale_use = xt_scale
            
            else:
                # print(hidden_seq.shape)
                x = hidden_seq
 
                xt_scale_use = ht_scale

            # hidden_seq = []
            # hidden_seq_reverse = []
            for direction in range(num_direction):
                # if (layer == 1) and (direction == 1):
                #     print('lstm11',x)
                #     print(x.shape)
                #     c_t = torch.ones((hidden_size,1)) * ct_zp

                weight = parameters[weight_name[layer][direction]]
                unit = parameters[unit_name[layer][direction]]
                bias = parameters[bias_name[layer][direction]]  
                weight_s = parameters[weight_scale_name[layer][direction]]
                weight_zp = parameters[weight_zp_name[layer][direction]]
                unit_s =  parameters[unit_scale_name[layer][direction]]
                unit_zp = parameters[unit_zp_name[layer][direction]]
                
                weight_q = Quant_Symmetric(weight, weight_s, weight_zp)
                # weight_dq = DeQuant(weight_q, weight_s, weight_zp)
                # weight_qcpu = weight_q.detach().cpu().numpy()
                # print('weight[0,26]',weight[0,26].dtype)
                # print('weight_s.type',weight_s.dtype)
                # print(weight_s)
                # print('weight[0,26]_s',weight[0,26]/weight_s)
                # np.save('weight_qcpu.npy', weight_qcpu)
                # print('unit',unit[:,0])
                # print('unit_s',unit_s)
                unit_q = Quant_Symmetric(unit, unit_s, unit_zp)
                # print('unit_q',unit_q[:,0])
                bias_q = Quant_bias(bias, weight_s * xt_scale_use, weight_zp)
                # print('weight41',weight_q[0:7,35])

                # bias_dq = DeQuant(bias_q, weight_s * xt_scale_use, weight_zp)
                SwSx_Sg = weight_s * xt_scale_use / gates_scale
                SwSx_Sg_quan = torch.floor(SwSx_Sg*(2**bit_scale_lstm))
                SwSx_Sg_quan =  torch.tensor(SwSx_Sg_quan, dtype = torch.int64)
                
                print('SwSx_Sg', SwSx_Sg_quan)

                SuSh_Sg = unit_s * ht_scale / gates_scale
                SuSh_Sg_quan = torch.floor(SuSh_Sg*(2**bit_scale_lstm))
                SuSh_Sg_quan =  torch.tensor(SuSh_Sg_quan, dtype = torch.int64)
                print('SuSh_Sg', SuSh_Sg_quan)
                
                Sb_Sg = weight_s * xt_scale_use / gates_scale
                Sb_Sg_quan = torch.floor(Sb_Sg*(2**bit_scale_lstm))
                Sb_Sg_quan =  torch.tensor(Sb_Sg_quan, dtype = torch.int64)
                print('Sb_Sg',Sb_Sg_quan) 

                one_div_gate = 1/gates_scale
                one_div_gate_round = torch.clamp(torch.round(one_div_gate), -127, 127)
                one_div_gate_quan = torch.floor(one_div_gate*(2**16)) # original 16
                one_div_gate_quan =  torch.tensor(one_div_gate_quan, dtype = torch.int64)


                one_div_ct = 1/ct_scale   
                one_div_ct_round = torch.clamp(torch.round(one_div_ct), -127, 127)        
                one_div_ct_quan = torch.floor(one_div_ct*(2**18)) # original 18
                one_div_ct_quan =  torch.tensor(one_div_ct_quan, dtype = torch.int64)  

                SiSg_Sc = gates_scale * gates_scale / ct_scale
                SiSg_Sc_quan = torch.floor(SiSg_Sc*(2**bit_scale_lstm))
                SiSg_Sc_quan =  torch.tensor(SiSg_Sc_quan, dtype = torch.int64)

                SoSc_Sh = gates_scale * ct_scale/(ht_scale)
                SoSc_Sh_quan = torch.floor(SoSc_Sh*(2**bit_scale_lstm))
                SoSc_Sh_quan =  torch.tensor(SoSc_Sh_quan, dtype =  torch.int64)
                


                gates_scale_quan = torch.floor(gates_scale*(2**bit_scale_lstm))
                gates_scale_quan =  torch.tensor(gates_scale_quan, dtype = torch.int64)
                # if ((layer == 1) and (direction == 0 )):
                #     # print('h_t', h_t) 
                #     # print('c_t', c_t)   
                #     h_t = torch.ones((hidden_size,1)) * ht_zp
                #     c_t = torch.ones((hidden_size,1)) * ct_zp
                # print('bias_dq',bias_dq)               
                for t in range(length):

                    x_t = x[t]
                    


                    gates = torch.zeros((4*hidden_size,1), dtype = torch.int64)
                    i_t = torch.zeros(hidden_size, 1)
                    f_t = torch.zeros((hidden_size, 1))
                    g_t = torch.zeros((hidden_size, 1))
                    o_t = torch.zeros((hidden_size, 1))
                    

                    for hs in range(int(4*hidden_size)):
                        qwqx_sum = torch.tensor(0,dtype = torch.int64)
                        for in_s in range(x_t.shape[0]):
                            qw  = weight_q[in_s, hs]

                            qx = x_t[in_s]
                            

                            qwqx = qw * qx
                            
                            
                            qwqx_sum = qwqx_sum + qwqx

                            # if ((direction == 0) and (layer  == 0) and (t == 0) and (hs==33)):
                            #     print('qw',in_s,qw)
                            #     print('qx',in_s,qx)
                            #     print('qwqx',qwqx)
                            #     print('qwqx_sum',in_s,qwqx_sum)
                        

                            # print('SwSx_Sg_quan * qwqx_sum + SuSh_Sg_quan * quqh_sum +  bias_div_scale_quan',hs,SwSx_Sg_quan * qwqx_sum + SuSh_Sg_quan * quqh_sum +  bias_div_scale_quan)
                        quqh_sum = torch.tensor(0,dtype = torch.int64)
                        for hi_s in range(hidden_size):
                            qu  = unit_q[hi_s, hs]
                            qh = h_t[hi_s]

                            quqh = qu * qh
                            quqh_sum = quqh_sum + quqh
                            # if ((direction == 0) and (layer  == 0) and (t == 0) and (hs==0)):
                            #     print('qu',hi_s,qu)
                            #     print('qh',hi_s,qh)
                            #     print('quqh',hi_s,quqh)
                            #     print('quqh_sum',hi_s,quqh_sum)
                        # print('quqh_sum', hs,quqh_sum )
                        # sign = torch.sign(SwSx_Sg * qwqx_sum + SuSh_Sg * quqh_sum + Sb_Sg *bias_q[hs])
                        # gates[hs] = sign * torch.floor(torch.abs(SwSx_Sg * qwqx_sum + SuSh_Sg * quqh_sum + Sb_Sg *bias_q[hs]) + 0.5)
                        bias_div_scale  = bias[hs]/ gates_scale
                        bias_div_scale = torch.floor(bias_div_scale*(2**24))/(2**24) # orogina 24
                        bias_div_scale_quan = torch.tensor(bias_div_scale*(2**bit_scale_lstm), dtype = torch.int64)
                        qwqx_sum = torch.tensor(qwqx_sum, dtype = torch.int64)
                        quqh_sum = torch.tensor(quqh_sum, dtype = torch.int64)
                        # if((t == 0) and (direction == 0) and (layer  == 0)):
                            # print('qwqx_sum', hs, qwqx_sum)
                            # print('SwSx_Sg_quan', hs, SwSx_Sg_quan)
                            # print('SwSx_Sg_quan * qwqx_sum',hs,SwSx_Sg_quan * qwqx_sum)
                            # print('qwqx_sum', hs, quqh_sum)
                            # print('SuSh_Sg_quan', hs, SuSh_Sg_quan)
                            # print('SuSh_Sg_quan * quqh_sum',hs,SuSh_Sg_quan * quqh_sum)
                            # print('bias_div_scale_quan',hs,bias_div_scale_quan)
                            # print(hs)
                            # print('SwSx_Sg_quan * qwqx_sum + SuSh_Sg_quan * quqh_sum +  bias_div_scale_quan',SwSx_Sg_quan * qwqx_sum + SuSh_Sg_quan * quqh_sum +  bias_div_scale_quan)
                        gates[hs] = torch.clamp(torch.round((SwSx_Sg_quan * qwqx_sum + SuSh_Sg_quan * quqh_sum +  bias_div_scale_quan)/(2**bit_scale_lstm)), -127, 127)
                        # if ((direction == 0) and (layer  == 0) and (t == 0) ):
                        #     print(hs,gates[hs])
                        # if (hs == 48) and (t == 0):
                            
                        #     print('qwqx_sum',qwqx_sum)
                        #     print('SwSx_Sg',SwSx_Sg)
                        #     print('qwqx_sum',qwqx_sum)
                            # print('sss',(SwSx_Sg_quan * qwqx_sum + SuSh_Sg_quan * quqh_sum +  bias_div_scale_quan)/(2**bit_scale_lstm))

                        #     print('gates[hs]',hs,gates[hs])
                        # gates[hs] = torch.clamp(torch.round(SwSx_Sg * qwqx_sum), -127, 127) +  torch.clamp(torch.round( SuSh_Sg * quqh_sum), -127, 127) +  torch.clamp(torch.round( bias_div_scale), -127, 127)
                        # if ((t == 1) and (direction == 0) and (layer  == 1)):
                        #     print('gates[hs]', hs, gates[hs])


                        if hs < hidden_size:
      
                            i_t[hs] = Hardsigmoid_int(gates[hs], gates_scale)
                        elif hs < 2 * hidden_size:
                            f_t[hs-hidden_size] = Hardsigmoid_int(gates[hs], gates_scale)
                        elif hs < 3 * hidden_size:
                            # print(g_t[hs-2*hidden_size])
                            # print('torch.max(gates[hs],0)', torch.max(gates[hs],0) )
                            g_t[hs-2*hidden_size] = torch.max(gates[hs], gates_zp)
                            # print(g_t[hs-2*hidden_size])
                        elif hs < 4 * hidden_size:
                            o_t[hs-3*hidden_size] = Hardsigmoid_int(gates[hs], gates_scale)
                        # if  layer == 0 and direction == 0 and t == 0:
                        #     print('f_t',hs-hidden_size, f_t[hs-hidden_size])
                            
                            # print(qwqxo_sum)o
                            # print(unit_s * ht_scale * quqh_sum)
                            # print(bias[hs])

                    #     g_t_dq = DeQuant(g_t, gates_scale, gates_zp)
                    #     print('f_t_dq',g_t_dq )
                    # if layer == 0 and direction == 0 and t == 6:
                    #     print('o_t', o_t.T)
                    #     print('i_t', i_t.T)
                    #     print('g_t', g_t.T)
                    #     print('f_t', f_t.T)
                    #     print('c_t', c_t.T)





                    ct_scale_quan = torch.floor(ct_scale*(2**bit_scale_lstm))/(2**bit_scale_lstm)
                    # if layer == 1 and direction == 0 and t == 1:
                    #     print(hidden_seq[0])
                    # f_t = f_t.to(int)
                    # c_t = c_t.to(int)
                    # i_t = i_t.to(int)
                    # o_t = o_t.to(int)
                    for hs in range(hidden_size):
                        # print('f_t', hs,   f_t[hs])
                        # print('c_t', hs,   c_t[hs])
                        # print('i_t', hs,   i_t[hs])
                        # print('g_t', hs,   g_t[hs])
                        # print(' f_t[hs] * c_t[hs]', hs,  gates_scale_quan*(2**15) * f_t[hs] * c_t[hs] )
                        # print('sum', hs, gates_scale_quan*(2**15) * f_t[hs] * c_t[hs] + SiSg_Sc_quan * i_t[hs] * g_t[hs])
                        # if (hs == 23):
                        #     print('MA(  f_t[hs], c_t[hs] )',t,MA(  f_t[hs], c_t[hs] ))
                        #     print('MA(  i_t[hs], g_t[hs] )',t,MA(  i_t[hs], g_t[hs] ))
                        #     print('gates_scale_quan*MA(  f_t[hs], c_t[hs] )',gates_scale_quan*MA(  f_t[hs], c_t[hs] )/(2**bit_scale_lstm))

                        c_t[hs] = torch.clamp(torch.round((gates_scale_quan * f_t[hs] * c_t[hs] + SiSg_Sc_quan * i_t[hs] * g_t[hs])/(2**bit_scale_lstm)),-127, 127)

                        # print(' i_t[hs] * g_t[hs]', hs,  MA(  f_t[hs], c_t[hs] ))
                        # print(' SiSg_Sc_quan *  i_t[hs] * g_t[hs]', hs,   SiSg_Sc_quan * MA(  i_t[hs], g_t[hs] ))
                        # print(' i_t[hs] * g_t[hs]', hs,  MA(  i_t[hs], g_t[hs] ))
                        # print('c_t', hs,   c_t[hs])
                        # print('o_t', hs,   o_t[hs])
                        # print(SoSc_Sh_quan* MA( o_t[hs], Hardsigmoid_int(c_t[hs],ct_scale_quan) ))

                        h_t[hs] = torch.clamp(torch.round((SoSc_Sh_quan* o_t[hs] * Hardsigmoid_int(c_t[hs],ct_scale_quan))/(2**bit_scale_lstm)), -127, 127)
                        # print(' h_t[hs]',hs,  h_t[hs])
                        # if layer == 1 and direction == 0 and t == 1:
                            
                        #     print(' Hardsigmoid_int(c_t[hs],ct_scale_quan)',hs,  Hardsigmoid_int(c_t[hs],ct_scale_quan))
                        # print(' o_t[hs] * Hardsigmoid_int(c_t[hs],ct_scale_quan',hs,  o_t[hs] * Hardsigmoid_int(c_t[hs],ct_scale_quan))
                    # if layer == 1 and direction == 1 and t == 63:
                    #     # print('t', t)
                    #     print('h_t', h_t.T)
                    #     print('c_t', c_t.T)
                    if direction == 0:
                        if t == 0:
                            hidden_seq = h_t.T.clone()
                            # print('aaa',t, hidden_seq[0])
                        else:

                            # print('bbb',t, hidden_seq[0])
                            hidden_seq = torch.cat((hidden_seq, h_t.T), dim = 0)
                            # print('ccc',hidden_seq[0])
                    else:
                        if t == 0:
                            hidden_seq_reverse = h_t.T.clone()
                        else:
                            hidden_seq_reverse = torch.cat((hidden_seq_reverse, h_t.T), dim= 0)
                
                x = torch.flip(x, [0])
                if direction == 1:
                    # hidden_seq_reverse = hidden_seq_reverse[::-1,:]
                    hidden_seq_reverse = torch.flip(hidden_seq_reverse, [0])
                    hidden_seq = torch.cat((hidden_seq, hidden_seq_reverse), dim=1)
                # if bs == 0 and layer == 0 and direction == 0:
                #     print(hidden_seq)
                #     print(hidden_seq.shape)
                    
        output[bs] = hidden_seq
    return output
import torch.nn as nn
import torch.nn.functional as F                       


def ConvTrans_torch_int(input, activation_pre_scale, activation_post_scale, weight_scale, weight, bias, padding, stride = 2, layer_id = 1):
    '''
    input: int [batch_size, channel_in, length]
    weight: int [channel_in, channel_out, kernel_size]
    bias: int32 [channel_out]
    output : int [batch_size, channel_out, length_out]
    activation_pre_scale = s1, activation_post_scale = s3, weight_scale =s2, input=q1, weight=q2, output = q3, bias_scale = s4, bias_zp =z4
    q3(a, b) = s1*s2/s3 * (SUMcin(SUMj(q1(c,a,j)q2(j,b))) - C*z1*SUMj(q2(j,b))) + s4/s3 *(bias(b)*-z4)  + z3 
    '''

    # print('activation_pre_scale',activation_pre_scale)
    # print('activation_post_scale',activation_post_scale)
    # print('weight_scale', weight_scale)
    weight = Quant_Symmetric(weight,weight_scale,0 )
    batch_size = input.shape[0]
    kernel_size = weight.shape[2]
    padding_new = kernel_size -1 - padding
    channel_in = input.shape[1]
    length_in = input.shape[2]
    channel_in = weight.shape[0]
    channel_out = weight.shape[1]
    kernel_size = weight.shape[2]
    
    activation_pre_scale = torch.tensor(activation_pre_scale, dtype = torch.float32)
    activation_post_scale = torch.tensor(activation_post_scale, dtype = torch.float32)
    weight_scale = torch.tensor(weight_scale, dtype = torch.float32)

    M0 = activation_pre_scale * weight_scale/activation_post_scale
    # M0 = torch.tensor(M0, dtype = torch.float32)
    # print(M0)
    M0 = torch.floor(M0*(2**bit_scale_cnn))

    # print('dcnn'+str(layer_id)+'_scale', M0)
    # print('activation_pre_zp', activation_pre_zp)
    # print('activation_pre_zp', activation_pre_zp.int())
    # print('activation_pre_zp', activation_pre_zp.item())

    length_out = np.int32(stride * (length_in - 1) -  2 * padding + kernel_size)
    output = torch.zeros((batch_size, channel_out, length_out))
    weight_matrix = torch.flip(weight, [2])


    data_matrix = torch.zeros((channel_in, length_out, kernel_size))
    data_insert = torch.zeros((channel_in, stride * length_in - 1))
    # print(data_insert)
    for bs in range(batch_size):
        data = input[bs]
        
        for column in range(length_in):
            data_insert [:, stride * column] = data[ :,column]
        # print(data_insert[0])
        for lo in range(length_out):
            data_pad = torch.nn.functional.pad(data_insert, (padding_new, padding_new), 'constant',value = 0 )            
            # print('data_pad',data_pad[0])
            data_matrix[:, lo, :] = data_pad[:, lo :lo+kernel_size] 

            for co in range(channel_out):
                # if co==0:
                # #     print('data_matrix',data_matrix[:,0,:])
                #     print('weight_matrix', weight_matrix[:,0,:])
                sum_q1q2 = 0
                for cin in range(channel_in):
                    for ks in range(kernel_size):
                        q1 = data_matrix[cin, lo, ks]
                        # print('q1', q1)
                        q2 = weight_matrix[cin, co, ks]
                        # print('q2', q2)


                        q1q2 = MA(q1, q2)
                        # print('q1q2', q1q2)
                        # print('activation_pre_zp',activation_pre_zp)
                        sum_q1q2 = sum_q1q2 + q1q2
                    # if (cin < 32) & (co == 0) & (layer_id==1) & (lo ==0) :
                    #     print('data_matrix',data_matrix[cin,0,:])
                    #     print('weight_matrix', weight_matrix[cin,0,:])
                    #     print('sum_q1q2', sum_q1q2)
                # if (co == 0) & (layer_id==1)  :
                #     print('sum_q1q2', sum_q1q2)
                output_loco = torch.clamp(torch.round((M0 * (sum_q1q2))/(2**bit_scale_cnn)),-127,127)
                
                output[bs, co, lo] = output_loco
    # print('output_loco',output[0,0,:])
    return output
def ConvFuse_torch_int(input,activation_pre_scale, activation_post_scale, weight_scale, weight, bias, padding, stride = 1, layer_id =11):
    '''
    input: int [batch_size, channel_in, length]
    weight: int [channel_out, channel_in, kernel_size]
    bias: int32 [channel_out]
    output : int [batch_size, channel_out, length_out]
    activation_pre_scale = s1, activation_post_scale = s3, weight_scale =s2, input=q1, weight=q2, output = q3, bias_scale = s4, bias_zp =z4
    q3(a, b) = s1*s2/s3 * (SUMcin(SUMj(q1(c,a,j)q2(j,b))) - C*z1*SUMj(q2(j,b))) + s4/s3 *(bias(b)*-z4)  + z3 
    '''
    activation_pre_scale = torch.tensor(activation_pre_scale, dtype = torch.float32)
    activation_post_scale = torch.tensor(activation_post_scale, dtype = torch.float32)
    weight_scale = torch.tensor(weight_scale, dtype = torch.float32)    

    weight = Quant_Symmetric(weight,weight_scale,0 )
    # bias_scale = torch.abs(torch.max(bias))/127
    bias_scale = activation_pre_scale * weight_scale

    bias_zp = 0
    bias_quant = Quant_bias(bias, bias_scale, bias_zp)
    # print('bias_quant',bias_quant)
    # print('bias_dequant',bias_scale*(bias_quant-bias_zp))
    
    # print('bias_q',bias_q.int_repr())
    batch_size = input.shape[0]
    channel_in = input.shape[1]
    length_in = input.shape[2]
    channel_out = weight.shape[0]
    kernel_size = weight.shape[2]
    
    M0 = activation_pre_scale * weight_scale/activation_post_scale
    M0 = torch.floor(M0*(2**bit_scale_cnn))
    # print('cnn'+str(layer_id)+'_scale',M0)


    length_out = np.int32(np.floor((length_in + 2* padding -kernel_size )/stride)+1)
    output = torch.zeros((batch_size, channel_out, length_out))
    weight_matrix = weight
    data_matrix = torch.zeros((channel_in, length_out, kernel_size))
    for bs in range(batch_size):
        data = input[bs]
        # print('bs', bs)
        # print('data', data.shape)
        for lo in range(length_out):
            data_pad = torch.nn.functional.pad(data, (padding, padding), 'constant',value = 0 )            
            data_matrix[:, lo, :] = data_pad[:,lo :lo+kernel_size]  
            for co in range(channel_out):
                sum_q1q2 = 0
                for cin in range(channel_in):
                    for ks in range(kernel_size):
                        q1 = data_matrix[cin, lo, ks]
                        q2 = weight_matrix[co, cin, ks]

                        q1q2 = MA(q1, q2)
                        sum_q1q2 = sum_q1q2 + q1q2
                #     if (cin <32 ) & (co == 0)  &(lo == 0) &(layer_id == 21) :

                #         print('data_matrix',data_matrix[cin, 0,:])
                #         print('weight_matrix', weight_matrix[0,cin,:])
                #         print('sum_q1q2', sum_q1q2)
                # if  (co == 0) &(layer_id == 21):
                #     print('sum_q1q2', sum_q1q2)
                #     print(' bias_quant[co]', bias_quant[co])
                #     print(' M0*(sum_q1q2 + bias_quant[co])', M0*(sum_q1q2 + bias_quant[co]))
                output_loco = torch.round(( M0*(sum_q1q2 + bias_quant[co]) )/(2**bit_scale_cnn))
                 
                # output_loco = torch.round( M0*sum_q1q2 )+torch.round (bias_quant[co]* M0)
                output[bs, co, lo] = torch.max(output_loco, torch.Tensor([0]))
    # print('output',output[0,4,:])
    return output


                





def Lstm_torch(input, parameters, num_layers, bidirectional = True):
    '''
    input [batch_size, length, input_size]
    weight [input_size, 4 * hidden_size]
    unit [hidden_size, 4 * hidden_size]
    bias [4 * hidden_size]    
    '''
    weight_name = [['lstm.weight_W0','lstm.weight_W0_reverse'],['lstm.weight_W1','lstm.weight_W1_reverse']]
    unit_name = [['lstm.weight_U0','lstm.weight_U0_reverse'],['lstm.weight_U1','lstm.weight_U1_reverse']]
    bias_name = [['lstm.bias_0','lstm.bias_0_reverse'],['lstm.bias_1','lstm.bias_1_reverse']]

    batch_size = input.shape[0]
    length = input.shape[1]
    
    hidden_size = np.int32(parameters[weight_name[0][0]].shape[1]/4)
    num_direction = 2 if bidirectional else 1

    h_t = torch.zeros((hidden_size,1))
    c_t = torch.zeros((hidden_size,1))

    output = torch.zeros((batch_size, length, num_direction*hidden_size))
    for bs in range(batch_size):
        for layer in range(num_layers):

            if layer == 0:
                x = input[bs]
            else:
                x = hidden_seq
            hidden_seq = []
            hidden_seq_reverse = []                
            for direction in range(num_direction):       
                for t in range(length):
                    # print(layer,direction)
                    x_t = x[t]
                    weight = parameters[weight_name[layer][direction]]
                    unit = parameters[unit_name[layer][direction]]
                    bias = parameters[bias_name[layer][direction]]
                    # weight = quan_xbit(weight, bit)
                    # unit = quan_xbit(unit, bit)
                    # bias = quan_xbit(bias, bit)
                    gates = torch.matmul(weight.T,torch.unsqueeze(x_t,1)) + torch.matmul(unit.T, h_t) + torch.unsqueeze(bias,1)
                    # gates = torch.floor(torch.matmul(weight.T,torch.unsqueeze(x_t,1)) /(2**bit))+ torch.floor(torch.matmul(unit.T, h_t)/(2**bit)) + torch.unsqueeze(bias,1)

                    # if bs == 0 and t == 0 and layer == 0 and direction == 0:
                    #     print('gates_npy',gates[0:10])
                    #     print('x_t_numpy',x_t[0:10])
                    #     print('h_t_numpy',h_t[0:10])
                        # print('weight_numpy',parameters[weight_name[layer][direction]])
                    # i_t, f_t, g_t, o_t = (
                    #         Sigmoid_linear_torch_quan(gates[:hidden_size]),
                    #         Sigmoid_linear_torch_quan(gates[hidden_size:hidden_size*2]),
                    #         Relu_torch(gates[hidden_size*2:hidden_size*3]),
                    #         Sigmoid_linear_torch_quan(gates[hidden_size*3:]),
                    #         )
                    i_t, f_t, g_t, o_t = (
                            sigmoid_linear_torch(gates[:hidden_size]),
                            sigmoid_linear_torch(gates[hidden_size:hidden_size*2]),
                            Relu_torch(gates[hidden_size*2:hidden_size*3]),
                            sigmoid_linear_torch(gates[hidden_size*3:]),
                            )
                                        


                    # if bs == 0 and t == 1 and layer == 0 and direction == 0:
                    #     print('pre c_t',c_t_32)    
                    c_t = f_t * c_t + i_t * g_t
                                                            
                    # c_t = torch.floor((f_t * c_t)/(2**bit)) + torch.floor((i_t * g_t)/(2**bit))
                    
                    # if quan == True:
                    #     c_t_32 = np.floor(c_t_32/(2**bit))
                    
                    # if bs == 0 and t == 0 and layer == 0 and direction == 0:
                    #     print('i_t',i_t[0:10])
                    #     print('f_t',f_t[0:10])
                    #     print('g_t',g_t[0:10])
                    #     print('o_t',o_t[0:10])
                    #     print('c_t',c_t[0:10])
                    h_t = o_t * sigmoid_linear_torch(c_t)                     
                    
                    # if quan == True:
                    #     h_t_32 = np.floor(c_t_32/(2**bit))
                    # if bs == 0 and t == 63 and layer == 1 and direction == 1:
                    #     print('h_t_numpy',h_t[0:10])
                    if direction == 0:
                        if isinstance(hidden_seq,list):
                            hidden_seq = h_t.T
                        else:
                            hidden_seq = torch.cat((hidden_seq,h_t.T), dim = 0)
                    else:
                        if isinstance(hidden_seq_reverse, list):
                            hidden_seq_reverse = h_t.T
                        else:
                            hidden_seq_reverse = torch.cat((hidden_seq_reverse, h_t.T), dim= 0)

                # x = x[::-1,:]
                x = torch.flip(x, [0])
                if direction == 1:
                    # hidden_seq_reverse = hidden_seq_reverse[::-1,:]
                    hidden_seq_reverse = torch.flip(hidden_seq_reverse, [0])
                    hidden_seq = torch.cat((hidden_seq, hidden_seq_reverse), dim=1)
            # print(hidden_seq.shape)
            # if bs == 0 and layer == 1 and direction == 1:
            #     print(hidden_seq)
        output[bs] = hidden_seq
    return output

def ConvTrans_torch(input, weight, bias, padding, stride =2):
    weight_scale = weight.q_scale()
    weight_zp = weight.q_zero_point()
    # print('weight_scale',weight_scale)
    # print('weight_zp', weight_zp)

    weight_dq = DeQuant(weight.int_repr(), weight_scale, weight_zp)


    batch_size = input.shape[0]
    kernel_size = weight.shape[2]
    padding_new = kernel_size -1 - padding
    channel_in = input.shape[1]
    length_in = input.shape[2]
    channel_in = weight_dq.shape[0]
    channel_out = weight_dq.shape[1]
    kernel_size = weight_dq.shape[2]

    length_out = np.int32(stride * (length_in - 1) -  2 * padding + kernel_size)
    output = torch.zeros((batch_size, channel_out, length_out))

    weight_matrix = torch.flip(weight_dq, [2])
    data_matrix = torch.zeros((channel_in, length_out, kernel_size))
    data_insert = torch.zeros((channel_in, stride * length_in - 1))

    # for bs in range(batch_size):
    #     data = input[bs]
    #     for cout in range(channel_out):
    #         channel_sum = 0
    #         for cin in range(channel_in):
    #             data_ch = data[cin]
    #             data_insert = torch.zeros((stride * length_in - 1))
    #             for column in range(length_in):
    #                 data_insert [2*column] = data_ch[column]
    #             # data_insert = np.insert(data_ch,range(1,len(data_ch)),0,axis = 0)
    #             data_pad =  torch.nn.functional.pad(data_insert,(padding_new,padding_new),'constant',value=0)
    #             data_matrix = torch.zeros((length_out,kernel_size))
                
    #             for lo in range(length_out):
    #                 data_matrix[lo] = data_pad[lo  : lo  + kernel_size]
    #             kernel =  torch.flip(weight_dq[cin, cout], [0])
                
    #             kernel = torch.unsqueeze(kernel, 1)
                
    #             output_ci = torch.matmul(data_matrix, kernel) # adder kernel_size; multiplier length_out * kernel_size
    #             output_ci = torch.squeeze(output_ci)

    #             channel_sum = channel_sum + output_ci # adder channel_in * length_out

    #         output[bs,cout] = channel_sum
    for bs in range(batch_size):
        data = input[bs]
        channel_sum = 0
        for cin in range(channel_in):
            for column in range(length_in):
                # print(data[0, 0:2])
                data_insert [:, 2*column] = data[ :,column]
                # print(data_insert[0, 0:4])
            for lo in range(length_out):
                data_pad = torch.nn.functional.pad(data_insert, (padding_new, padding_new), 'constant',value = 0)            
                data_matrix[:, lo, :] = data_pad[:, lo :lo+kernel_size] 
                # print(data_matrix[0]) 
                
            output_ch = torch.matmul(data_matrix[cin,:,:], weight_matrix[cin,:,:].T).T
            channel_sum = channel_sum + output_ch

        output[bs] = channel_sum           
    return output    
###############################original torch method######################################
# def ConvTrans_torch_int(input, activation_pre_scale, activation_pre_zp, activation_post_scale, activation_post_zp, weight, bias, padding, stride = 2, is_first_layer = False):
#     '''
#     input: int [batch_size, channel_in, length]
#     weight: int [channel_in, channel_out, kernel_size]
#     bias: int32 [channel_out]
#     output : int [batch_size, channel_out, length_out]
#     activation_pre_scale = s1, activation_post_scale = s3, weight_scale =s2, input=q1, weight=q2, output = q3, bias_scale = s4, bias_zp =z4
#     q3(a, b) = s1*s2/s3 * (SUMcin(SUMj(q1(c,a,j)q2(j,b))) - C*z1*SUMj(q2(j,b))) + s4/s3 *(bias(b)*-z4)  + z3 
#     '''
#     weight_scale = weight.q_scale()
#     weight_zp = weight.q_zero_point()
#     if bias is not None:
#         bias_scale = bias.q_scale()
#         bias_zp = bias.q_zero_point()

#     batch_size = input.shape[0]
#     kernel_size = weight.shape[2]
#     padding_new = kernel_size -1 - padding
#     channel_in = input.shape[1]
#     length_in = input.shape[2]
#     channel_in = weight.shape[0]
#     channel_out = weight.shape[1]
#     kernel_size = weight.shape[2]
    

#     M0 = activation_pre_scale * weight_scale/activation_post_scale
#     # print('M0', M0)
#     # print('activation_pre_zp', activation_pre_zp)
#     # print('activation_pre_zp', activation_pre_zp.int())
#     # print('activation_pre_zp', activation_pre_zp.item())

#     length_out = np.int32(stride * (length_in - 1) -  2 * padding + kernel_size)
#     output = torch.zeros((batch_size, channel_out, length_out))
#     weight_matrix = torch.flip(weight, [2])
#     data_matrix = torch.zeros((channel_in, length_out, kernel_size))
#     data_insert = torch.ones((channel_in, stride * length_in - 1))* activation_pre_zp.item()
#     # print(data_insert)
#     for bs in range(batch_size):
#         data = input[bs]
        
#         for column in range(length_in):
#             data_insert [:, stride * column] = data[ :,column]
#         for lo in range(length_out):
#             data_pad = torch.nn.functional.pad(data_insert, (padding_new, padding_new), 'constant',value = activation_pre_zp.item())            
#             data_matrix[:, lo, :] = data_pad[:, lo :lo+kernel_size]  
#             for co in range(channel_out):
#                 sum_q1q2_z1q2 = 0
#                 for cin in range(channel_in):
#                     for ks in range(kernel_size):
#                         q1 = data_matrix[cin, lo, ks].int()
#                         # print('q1', q1)
#                         q2 = weight_matrix[cin, co, ks].int_repr().int()
#                         # print('q2', q2)


#                         q1q2 = MA(q1, q2)
#                         # print('q1q2', q1q2)
#                         # print('activation_pre_zp',activation_pre_zp)

#                         z1q2 = activation_pre_zp*q2 # int*int
#                         q1q2_z1q2 = q1q2 - z1q2
#                         sum_q1q2_z1q2 = sum_q1q2_z1q2 + q1q2_z1q2
#                 # print('sum_q1q2_z1q2', sum_q1q2_z1q2)
#                 # print(sum_q1q2-sum_z1q2 )
#                 output_loco = torch.clamp(torch.round((M0 * (sum_q1q2_z1q2) + activation_post_zp)),0,128)
#                 # print('output_loco',output_loco)
#                 output[bs, co, lo] = output_loco
#     return output

# def ConvFuse_torch_int(input,activation_pre_scale, activation_pre_zp, activation_post_scale, activation_post_zp, weight, bias, padding, stride = 1,is_MA = False):
#     '''
#     input: int [batch_size, channel_in, length]
#     weight: int [channel_out, channel_in, kernel_size]
#     bias: int32 [channel_out]
#     output : int [batch_size, channel_out, length_out]
#     activation_pre_scale = s1, activation_post_scale = s3, weight_scale =s2, input=q1, weight=q2, output = q3, bias_scale = s4, bias_zp =z4
#     q3(a, b) = s1*s2/s3 * (SUMcin(SUMj(q1(c,a,j)q2(j,b))) - C*z1*SUMj(q2(j,b))) + s4/s3 *(bias(b)*-z4)  + z3 
#     '''
#     weight_scale = weight.q_scale()
#     weight_zp = weight.q_zero_point()
#     # bias_scale = torch.abs(torch.max(bias))/127
#     bias_scale = activation_pre_scale * weight_scale
#     print('activation_pre', activation_pre_zp)
#     print('activation_post', activation_post_zp)
#     # print('biasaaaaaaassasasasasasas',bias)
#     bias_zp = 0
#     bias_quant = Quant_bias(bias, bias_scale, bias_zp)
#     # print('bias_quant',bias_quant)
#     # print('bias_dequant',bias_scale*(bias_quant-bias_zp))
#     bias_q = torch.quantize_per_tensor(bias, scale=bias_scale, zero_point=0, dtype=torch.qint32)
#     # print('bias_q',bias_q.int_repr())
#     batch_size = input.shape[0]
#     channel_in = input.shape[1]
#     length_in = input.shape[2]
#     channel_out = weight.shape[0]
#     kernel_size = weight.shape[2]
    
#     M0 = activation_pre_scale * weight_scale/activation_post_scale
#     M1 = bias_scale/activation_post_scale
#     # print('M0', M0)
#     # print('M1', M1)

#     length_out = np.int32(np.floor((length_in + 2* padding -kernel_size )/stride)+1)
#     output = torch.zeros((batch_size, channel_out, length_out))
#     weight_matrix = weight
#     data_matrix = torch.zeros((channel_in, length_out, kernel_size))
#     for bs in range(batch_size):
#         data = input[bs]
#         # print('bs', bs)
#         # print('data', data.shape)

#         for lo in range(length_out):
#             data_pad = torch.nn.functional.pad(data, (padding, padding), 'constant',value = activation_pre_zp.item())            
#             data_matrix[:, lo, :] = data_pad[:,lo :lo+kernel_size]  
#             for co in range(channel_out):
#                 sum_q1q2 = 0
#                 sum_z1q2 = 0
#                 for cin in range(channel_in):
#                     for ks in range(kernel_size):
#                         q1 = data_matrix[cin, lo, ks].int()
#                         q2 = weight_matrix[co, cin, ks].int_repr().int()
#                         if not is_MA:
#                             q1q2 = q1*q2 # int * int
#                         else:
#                             q1q2 = MA(q1, q2)
#                         z1q2 = activation_pre_zp*q2 # int*int8
#                         sum_q1q2 = sum_q1q2 + q1q2
#                         sum_z1q2 = sum_z1q2 + z1q2
#                 output_loco = torch.round(( M0*(sum_q1q2 - sum_z1q2) + ((bias_quant[co]) * bias_scale/activation_post_scale) + activation_post_zp))
#                 output[bs, co, lo] = torch.max(output_loco, activation_post_zp)
#     return output



        
            
            





            
            


            





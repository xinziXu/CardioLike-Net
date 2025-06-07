
"""
Author@Xinzi Xu

This code is released for academic and research purposes only, as part of the publication:
"CardioLike-Net: An Edge-end Inter-patient Arrhythmia Classifier with Quantization-aware-training for Wearable ECG Applications"

Permission is granted to use, copy, and modify the code **for non-commercial research purposes**, provided that proper citation is given to the above paper.

Any commercial use of this code or any derivative work is strictly prohibited without explicit written permission from the author.

If you find this work useful in your research, please consider citing the following paper: XXX
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from quantizer import *
class CNN_BiLSTM(nn.Module):
    def __init__(self, class_n, layer_n ):
        super().__init__()
        # self.quant = torch.quantization.QuantStub()
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(1, layer_n, kernel_size=8,stride=4, padding=int((8 - 1) / 2),bias = False)),
            ('bn1', nn.BatchNorm1d(layer_n)),
            ('relu1', nn.ReLU())])
        )

        self.lstm = nn.LSTM(layer_n, layer_n, dropout = 0.1, batch_first=True, num_layers=2, bidirectional=True)
        self.decoder = nn.Sequential(OrderedDict([
            ('upsample_1', nn.ConvTranspose1d(2*layer_n,layer_n,kernel_size=8,stride=2,padding=3, bias=False)),
            ('dec1_1_conv',nn.Conv1d(layer_n, layer_n, kernel_size=5,stride=1, padding=2, bias = False)),
            ('dec1_1_bn',nn.BatchNorm1d(layer_n)),
            ('dec1_1_relu', nn.ReLU()),
            ('dec1_2_conv', nn.Conv1d(layer_n, layer_n//2, kernel_size=5,stride=1, padding=2,bias = False)),
            ('dec1_2_bn',nn.BatchNorm1d(layer_n//2)),
            ('dec1_2_relu',nn.ReLU()),
            ('upsample_2', nn.ConvTranspose1d(layer_n//2,layer_n//4,kernel_size=8,stride=2,padding=3,bias=False)),
            ('dec2_1_conv', nn.Conv1d(layer_n//4, layer_n//4, kernel_size=5,stride=1, padding=2, bias = False)),
            ('dec2_1_bn', nn.BatchNorm1d(layer_n//4)),
            ('dec2_1_relu', nn.ReLU()),
            ('dec2_2_conv', nn.Conv1d(layer_n//4, class_n, kernel_size=5, stride=1, padding=2, bias = False)),
            ('dec2_2_bn', nn.BatchNorm1d(class_n)),
            ('dec2_2_relu', nn.ReLU())   ])                                  
    )
        # self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        cnn_out = self.encoder(x)  
        cnn_out1 = cnn_out.transpose(1,2)
        lstm_feature,_ = self.lstm(cnn_out1)
        lstm_feature = lstm_feature.transpose(1,2)

        output = self.decoder(lstm_feature)
        
        return output

class CNN_BiLSTM_AttenQ(nn.Module):
    def __init__(self, class_n, layer_n):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(1, layer_n, kernel_size=8,stride=4, padding=int((8 - 1) / 2),bias = False)),
            ('bn1', nn.BatchNorm1d(layer_n)),
            ('relu1', nn.ReLU())])
        )

        self.lstm = Quantized_LSTM(layer_n, layer_n, num_layers=2, bidirectional=True, is_pretrain=is_pretrain,q_type=0,a_bits=a_bits,w_bits=w_bits)
        self.decoder = nn.Sequential(OrderedDict([
            ('upsample_1', nn.ConvTranspose1d(2*layer_n,layer_n,kernel_size=8,stride=2,padding=3, bias=False)),
            ('dec1_1_conv',nn.Conv1d(layer_n, layer_n, kernel_size=5,stride=1, padding=2, bias = False)),
            ('dec1_1_bn',nn.BatchNorm1d(layer_n)),
            ('dec1_1_relu', nn.ReLU()),
            ('dec1_2_conv', nn.Conv1d(layer_n, layer_n//2, kernel_size=5,stride=1, padding=2,bias = False)),
            ('dec1_2_bn',nn.BatchNorm1d(layer_n//2)),
            ('dec1_2_relu',nn.ReLU()),
            ('upsample_2', nn.ConvTranspose1d(layer_n//2,layer_n//4,kernel_size=8,stride=2,padding=3,bias=False)),
            ('dec2_1_conv', nn.Conv1d(layer_n//4, layer_n//4, kernel_size=5,stride=1, padding=2, bias = False)),
            ('dec2_1_bn', nn.BatchNorm1d(layer_n//4)),
            ('dec2_1_relu', nn.ReLU()),
            ('dec2_2_conv', nn.Conv1d(layer_n//4, class_n, kernel_size=5, stride=1, padding=2, bias = False)),
            ('dec2_2_bn', nn.BatchNorm1d(class_n)),
            ('dec2_2_relu', nn.ReLU())   ])                                  
    )
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        cnn_out = self.encoder(x)  
        cnn_out1 = cnn_out.transpose(1,2)
        lstm_feature,_ = self.lstm(cnn_out1)
        lstm_feature = lstm_feature.transpose(1,2)

        output = self.decoder(lstm_feature)
        
        return output
class Quantized_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, bidirectional,is_pretrain =True, q_type=0,a_bits=8,w_bits=8 ):
        super(Quantized_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.is_pretrain = is_pretrain
        self.num_directions = 2 if bidirectional else 1

        self.param_names = []
        self.quantizer_names = []
        for layer in range(self.num_layers):
            self.param_names.append([])
            self.quantizer_names.append([])
            for direction in range(self.num_directions):
                self.input_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions

                W = nn.Parameter(torch.Tensor(self.input_size , self.hidden_size * 4))
                U = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size * 4))
                b = nn.Parameter(torch.Tensor(self.hidden_size * 4))
                
                W_quantizer = SymmetricQuantizer(
                                    bits=w_bits,
                                    observer=MinMaxObserver(),

                                )
                U_quantizer = SymmetricQuantizer(
                                    bits=w_bits,
                                    observer=MinMaxObserver(),

                                )
                b_quantizer = SymmetricQuantizer(
                                    bits=32,
                                    observer=MinMaxObserver(),

                )
                layer_params = (W, U, b)
                layer_quantizers = (W_quantizer,U_quantizer, b_quantizer)
                suffix = '_reverse' if direction == 1 else ''
                self.param_name = ['weight_W{}{}', 'weight_U{}{}', 'bias_{}{}']
                self.quantizer_name = ['quantizer_W{}{}', 'quantizer_U{}{}', 'quantizer_{}{}']
                self.param_name = [x.format(layer, suffix) for x in self.param_name]
                self.quantizer_name = [x.format(layer, suffix) for x in self.quantizer_name]
                for name, param in zip(self.param_name, layer_params):
                    setattr(self, name, param)
                for name, param in zip(self.quantizer_name, layer_quantizers):
                    setattr(self, name, param)                
                self.param_names[layer].append(self.param_name)
                self.quantizer_names[layer].append(self.quantizer_name)
        
        self.all_weights  = [[[getattr(self, weight) for weight in weights]
                        for weights in weights_layer] for weights_layer in self.param_names]
        self.all_quantizers  = [[[getattr(self, quantizer) for quantizer in quantizers]
                        for quantizers in quantizers_layer] for quantizers_layer in self.quantizer_names]
        self.init_weights()
        self.dropout = nn.Dropout(p=0.1)
        self.q_type = q_type
        

        if q_type == 0:
            self.xt_quantizer = SymmetricQuantizer(
                        bits=a_bits,
                        observer=MovingAverageMinMaxObserver(),

                    )
            self.ht_quantizer = SymmetricQuantizer(
                        bits=a_bits,
                        observer=MovingAverageMinMaxObserver(),

                    )
            self.gates_quantizer = SymmetricQuantizer(
                        bits=a_bits,
                        observer=MovingAverageMinMaxObserver(),

                    ) 
            self.ct_quantizer = SymmetricQuantizer(
                        bits=a_bits,
                        observer=MovingAverageMinMaxObserver(),

                    ) 
        
        else:
            self.xt_quantizer = AsymmetricQuantizer(
                        bits=a_bits,
                        observer=MovingAverageMinMaxObserver(),
                    ) 
            self.ht_quantizer = AsymmetricQuantizer(
                        bits=a_bits,
                        observer=MovingAverageMinMaxObserver(),
                    )
            self.gates_quantizer = AsymmetricQuantizer(
                        bits=a_bits,
                        observer=MovingAverageMinMaxObserver(),
                    ) 
            self.ct_quantizer = AsymmetricQuantizer(
                        bits=a_bits,
                        observer=MovingAverageMinMaxObserver(),
                    ) 

                      
                                    


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def convert(self):
        
        for state_name in self.state_dict():
            
            print(state_name,'\t',self.state_dict()[state_name])
            if state_name.startswith('weight'):
                # print(state_name,'\t',self.state_dict()[state_name])
                scale_name = 'quantizer_' + state_name.split('_')[1] + '.scale'
                zp_name = 'quantizer_' + state_name.split('_')[1] + '.zero_point'
                if self.q_type == 0:
                    q_dtype = torch.qint8
                else:
                    q_dtype = torch.quint8
                
                qm = Quantize(self.state_dict()[scale_name], (self.state_dict()[zp_name]).int(), q_dtype)
                a = qm(self.state_dict()[state_name])

                self.register_buffer(
                state_name+"_q",
                a,
                )
                      

    def forward(self, x):
        batch_size, seq_sz, _ = x.shape
        num_directions = 2 if self.bidirectional else 1
        
        # h_t, c_t = (torch.zeros(self.num_layers * num_directions,batch_size,self.hidden_size).to(x.device),
        #             torch.zeros(self.num_layers * num_directions,batch_size,self.hidden_size).to(x.device))
        h_t, c_t = (torch.zeros(batch_size,self.hidden_size).to(x.device),
                    torch.zeros(batch_size,self.hidden_size).to(x.device))

        for layer in range(self.num_layers):
            hidden_seq = []
            hidden_seq_reverse = []
            self.weight_layer = self.all_weights[layer]
            self.quantizer_layer = self.all_quantizers[layer]

            for direction in range(self.num_directions):
                self.weight = self.weight_layer[direction]
                self.quantizer = self.quantizer_layer[direction]
                
                HS = self.hidden_size
                
                # h_t, c_t = h_t[0],c_t[0]

                # if layer == 0 and direction == 1:
                #     print('HT',h_t)
                # ht_q = torch.clamp(
                # torch.round(
                #     h_t / self.ht_quantizer.scale + self.ht_quantizer.zero_point
                # ),
                # self.ht_quantizer.quant_min_val,
                # self.ht_quantizer.quant_max_val,
                # )
                quant_weight_0 = self.quantizer[0](self.weight[0])
                # # print('quant_weight_0', quant_weight_0)
                quant_weight_1 = self.quantizer[1](self.weight[1])   
                quant_weight_2 = self.quantizer[2](self.weight[2])

                
                # if direction == 0 and layer == 0:
                    # print(self.quantizer[0].scale) 
                    # print('quant_weight_2', self.weight[2])
                for t in range(seq_sz):
                    x_t = x[:, t, :]
                    
                    if self.is_pretrain:
                        x_t = x_t
                        h_t = h_t
                        gates = x_t @ self.weight[0] + h_t @ self.weight[1] \
                                + self.weight[2]
                    else:
                        
                        if layer == 0:
                            x_t = self.xt_quantizer(x_t)
                        else:
                            x_t = self.ht_quantizer(x_t)

                        h_t = self.ht_quantizer(h_t)
                        # if layer  == 0 and direction == 1 and t == 0:
                        #     print('x_t',x_t.shape)
                        #     print('quant_weight_0',quant_weight_0.shape)
                        #     print('h_t', h_t.shape)
                        #     print('quant_weight_1', quant_weight_1.shape)
                        #     print('quant_weight_2',quant_weight_2.shape)
                        gates = x_t @ quant_weight_0 + h_t @ quant_weight_1 \
                                + quant_weight_2
                    # gates = gates[0]
                    # if layer == 0 and direction == 0 and t == 0:

                    #     x_q = torch.clamp(
                    #         torch.round(
                    #             x_t / self.xt_quantizer.scale + self.xt_quantizer.zero_point
                    #         ),
                    #         self.xt_quantizer.quant_min_val,
                    #         self.xt_quantizer.quant_max_val,
                    #     )
                    #     print('x_q',x_q[1])
                    #     print('x_t', x_t[1])
                    #     print('h_t', h_t[0,1])
                    #     print('c_t',c_t[0,1])
                    #     print('quant_weight_0', quant_weight_0)
                    #     print('quant_weight_1',quant_weight_1)
                    #     print('quant_weight_2',quant_weight_2)
                    #     gates_q = torch.clamp(
                    #         torch.round(
                    #             gates / self.gates_quantizer.scale + self.gates_quantizer.zero_point
                    #         ),
                    #         self.gates_quantizer.quant_min_val,
                    #         self.gates_quantizer.quant_max_val,
                    #     )                        
                    #     print('gates',gates_q[1])
                    # gates = gates[0]

                    if self.is_pretrain:
                        gates = gates
                    else:
                        gates = self.gates_quantizer(gates)

                    i_t, f_t, g_t, o_t = (
                    # torch.sigmoid(gates[:, :HS]), # input
                    # torch.sigmoid(gates[:, HS:HS*2]), # forget
                    sigmoidlinear(gates[:, :HS]),
                    sigmoidlinear(gates[:, HS:HS*2]),
                    torch.relu(gates[:, HS*2:HS*3]),
                    # torch.sigmoid(gates[:, HS*3:]), # output
                    sigmoidlinear(gates[:, HS*3:]),
                    )      

                    if self.is_pretrain:  
                        f_t = f_t
                        c_t = c_t
                        i_t = i_t
                        g_t = g_t

                    else:
                        f_t = self.gates_quantizer(f_t)
                        c_t = self.ct_quantizer(c_t)
                        i_t = self.gates_quantizer(i_t)
                        g_t = self.gates_quantizer(g_t)


                    # if layer == 0 and direction == 0 and t == 1:
                        # print(' pre c_t',c_t[0,1])   
                #     ht_q = torch.clamp(
                #     torch.round(
                #         h_t / self.ht_quantizer.scale + self.ht_quantizer.zero_point
                #     ),
                #     self.ht_quantizer.quant_min_val,
                #     self.ht_quantizer.quant_max_val,
                # )
                #     xt_q = torch.clamp(
                #     torch.round(
                #         x_t / self.ht_quantizer.scale + self.ht_quantizer.zero_point
                #     ),
                #     self.ht_quantizer.quant_min_val,
                #     self.ht_quantizer.quant_max_val,
                # )                
                #     gates_q = torch.clamp(
                #     torch.round(
                #         gates / self.gates_quantizer.scale + self.gates_quantizer.zero_point
                #     ),
                #     self.gates_quantizer.quant_min_val,
                #     self.gates_quantizer.quant_max_val,
                # ) 
                    # if layer ==1 and direction == 0 and t == 0:            
                    #     print('ht_q',ht_q )
                    #     print('xt_q', xt_q)
                    #     print('gates_q', gates_q)

                    f_t_c_t = f_t * c_t
                    i_t_g_t = i_t * g_t
 
                    c_t = f_t_c_t + i_t_g_t

                    if not self.is_pretrain:
                        c_t = self.ct_quantizer(c_t)
                 
                    c_t_sigmoid = sigmoidlinear(c_t)
                    if not self.is_pretrain:
                        o_t= self.gates_quantizer(o_t)
                        c_t_sigmoid = self.ct_quantizer(c_t_sigmoid)                    
                    
                    # print('c_t_torch',c_t.shape)
  
                    
                    # print('o_t', o_t.shape)
                    # print('c_t_sigmoid',c_t_sigmoid.shape)
                    h_t = o_t * c_t_sigmoid
                    # print('h_t',h_t.shape)
                #     ct_q = torch.clamp(
                #     torch.round(
                #         c_t / self.ct_quantizer.scale + self.ct_quantizer.zero_point
                #     ),
                #     self.ct_quantizer.quant_min_val,
                #     self.ct_quantizer.quant_max_val,
                # )
                #     ft_q = torch.clamp(
                #     torch.round(
                #         f_t / self.gates_quantizer.scale + self.gates_quantizer.zero_point
                #     ),
                #     self.gates_quantizer.quant_min_val,
                #     self.gates_quantizer.quant_max_val,
                # )
                #     ot_q = torch.clamp(
                #     torch.round(
                #         o_t / self.gates_quantizer.scale + self.gates_quantizer.zero_point
                #     ),
                #     self.gates_quantizer.quant_min_val,
                #     self.gates_quantizer.quant_max_val,
                # )  
                #     gt_q = torch.clamp(
                #     torch.round(
                #         g_t / self.gates_quantizer.scale + self.gates_quantizer.zero_point
                #     ),
                #     self.gates_quantizer.quant_min_val,
                #     self.gates_quantizer.quant_max_val,
                # )                              
                #     ht_q = torch.clamp(
                #     torch.round(
                #         h_t / self.ht_quantizer.scale + self.ht_quantizer.zero_point
                #     ),
                #     self.ht_quantizer.quant_min_val,
                #     self.ht_quantizer.quant_max_val,
                # )
                    if (layer == (self.num_layers -1)) and (not self.is_pretrain):
                        h_t = self.ht_quantizer(h_t)

                    # if layer == 1 and direction == 1 and t == 63:
                    #     print('ht_q', ht_q)              
                        # print('ct_q', ct_q) 
                        # print('ft_q', ft_q) 
                        # print('ot_q', ot_q) 
                        # print('gt_q', gt_q)            
                        # print('qmin',self.ht_quantizer.quant_min_val)
                        # print('qmax',self.ht_quantizer.quant_max_val)
                        # print('scale', self.ht_quantizer.scale)
                    if direction == 0:
                            if isinstance(hidden_seq, list):
                                hidden_seq = h_t.unsqueeze(1)
                            else:
                                hidden_seq = torch.concat((hidden_seq, h_t.unsqueeze(1)), axis=1)

                    if direction == 1:
                        if isinstance(hidden_seq_reverse, list):
                            hidden_seq_reverse = h_t.unsqueeze(1)
                        else:
                            hidden_seq_reverse = torch.concat((hidden_seq_reverse, h_t.unsqueeze(1)), axis=1)
                x = torch.flip(x, dims=[1])
                # x = x.detach().cpu().numpy()[:,::-1,:]
                # x = torch.tensor(x.copy(),device = config.device)
      
                if direction == 1:
                    hidden_seq_reverse = torch.flip(hidden_seq_reverse, dims=[1])
                    # hidden_seq_reverse = hidden_seq_reverse.detach().cpu().numpy()[:, ::-1, :]
                    # hidden_seq_reverse = torch.tensor(hidden_seq_reverse.copy(), device = config.device)
                    hidden_seq = torch.concat((hidden_seq, hidden_seq_reverse),axis=2)
                # if layer == 0 and direction == 0:
                #     x_q = torch.clamp(
                #         torch.round(
                #             hidden_seq / self.ht_quantizer.scale + self.ht_quantizer.zero_point
                #         ),
                #         self.ht_quantizer.quant_min_val,
                #         self.ht_quantizer.quant_max_val,
                #     )
                #     print('after_dropout', x_q[1])
            # if layer == 1 and direction == 1:
            #     print(hidden_seq[0])                       
            if layer != (self.num_layers-1):
                hidden_seq = self.dropout(hidden_seq)
            x = hidden_seq

 
        # return hidden_seq, self.ht_quantizer.scale
        return hidden_seq, (h_t, c_t)
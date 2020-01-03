"""
modified from https://drive.google.com/drive/folders/1KQSe91znWWwaUPG1TSpqUyfyw0o0VgMV by Tianyuan Zhang
"""

import torch
import torchvision
import numpy as np
import os

def get_layer_name(name:str):
    '''
    :param name: e.g. group0/block1/conv2/W
    :return:  e.g. layer1
    '''
    name_fields = name.split('/')

    layer_name = 'layer' + str((int(name_fields[0][-1])+1))

    return layer_name

def get_block_name(name:str):
    '''
    :param name: e.g. group0/block1/conv2/W
    :return:  e.g.  block1
    '''
    name_fields = name.split('/')
    block_name = name_fields[1][5:]

    return block_name

def parse_conv_name(name:str):
    '''
    :param name: e.g. group0/block1/conv2/W
    :return:  e.g.  conv2.weights
    '''
    name_fields = name.split('/')
    conv_name = name_fields[-2]

    if conv_name == 'convshortcut':
        conv_name = 'downsample.0'

    conv_weight_name = conv_name + '.weight'
    return conv_weight_name

def parser_bn_name(name:str):
    '''
    :param name: e.g. group2/block23/conv3/bn/beta
    :return:  e.g. False, bn3.bias
    '''
    is_buffer = False

    name_fields = name.split('/')
    conv_name = name_fields[2]

    if conv_name == 'convshortcut':
        bn_name = 'downsample.1'
    else:
        bn_name = 'bn' + conv_name[-1]

    if name.find('EMA') is not -1:
        # Buffer
        is_buffer = True
        if name_fields[-2] == 'variance':
            bn_name = bn_name + '.running_var'
        else:
            bn_name = bn_name + '.running_mean'
    else:
        if name_fields[-1] == 'gamma':
            bn_name = bn_name + '.weight'
        if name_fields[-1] == 'beta':
            bn_name = bn_name + '.bias'

    return is_buffer, bn_name

def parser_fc_name(name:str):
    fc_name = 'fc'
    if name[-1] == 'b':
        fc_name = fc_name + '.bias'
    else:
        fc_name = fc_name + '.weight'
    return fc_name

def parse_weight_dict(dic):
    '''
    change the tensorflow type of parameter dict to that of pytorch versrion
    '''
    torch_weight_dic = {}
    torch_buffer_dic = {}
    for key in dic.keys():
        if key.find('linear') is not -1:
            fc_name = parser_fc_name(key)
            torch_weight_dic[fc_name] = dic[key]

            continue
        if key[:5] == 'conv0':
            if key == 'conv0/W':
                torch_weight_dic['conv1.weight'] = dic[key]
            else:
                new_key = 'conv1/bn1' + key[8:]
                fake_key = '0/0/' + new_key
                is_buffer, bn_name = parser_bn_name(fake_key)

                if is_buffer:
                    torch_buffer_dic[bn_name] = dic[key]
                else:
                    torch_weight_dic[bn_name] = dic[key]

            continue


        layer_name = get_layer_name(key)
        block_name = get_block_name(key)
        is_buffer = False

        if key.find('W') is not -1 and key.find('conv') is not -1:
            name = parse_conv_name(key)
        else:
            is_buffer, name = parser_bn_name(key)

        print('layer: {}  ---  block: {}  ---  name: {} '.format(layer_name, block_name, name))
        name = layer_name + '.' + block_name + '.' + name

        if is_buffer:
            torch_buffer_dic[name] = dic[key]
        else:
            torch_weight_dic[name] = dic[key]

    return torch_weight_dic, torch_buffer_dic


if __name__ == '__main__':
    dic = np.load('./adv_denoise_model/R152.npz')

    # current code does load BN statistics.
    torch_weight_dic, torch_buffer_dic = parse_weight_dict(dic)
    net = torchvision.models.resnet152()

    # check the weights
    num_param = 0
    for name, param in net.named_parameters():
        num_param += 1
        tf_param = torch_weight_dic[name]

        if name.find('conv') is not -1 or name.find('downsample.0') is not -1:
            tf_param = tf_param.transpose(3,2,0, 1)
            pass

        elif name.find('fc') is not -1 and name.find('weight') is not -1:
            tf_param = tf_param.transpose()
            pass

        tf_param = torch.tensor(tf_param, dtype = param.dtype)
        torch_weight_dic[name] = tf_param
        print(name, 'weight shape:{} - {},  is the shape right:{}'.format(param.shape, tf_param.shape, param.shape == tf_param.shape))
        if not param.shape == tf_param.shape:
            print('wrong!,  two kinds of shapes not match')

    # chceck the number of weights.
    print(len(torch_weight_dic), num_param)

    # transform numpy to torch.tensor
    for name, param in torch_buffer_dic.items():
        torch_buffer_dic[name] = torch.tensor(torch_buffer_dic[name])

    # combine the buffer part and weight part
    weights_and_buffer = dict(torch_weight_dic,  **torch_buffer_dic) 
    net.load_state_dict(weights_and_buffer, strict=False)
    torch.save(net.state_dict(), './adv_denoise_model/res152-adv.checkpoint')




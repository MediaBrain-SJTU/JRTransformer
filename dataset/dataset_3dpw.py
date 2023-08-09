import os
import numpy as np
import json
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl

def batch_denormalization(data, para):
    '''
    :data: [B, T, N, J, 6] or [B, T, J, 3]
    :para: [B, 3]
    '''
    if data.shape[2]==2:
        data[..., :3] += para[:, None, None, None, :]
    else:
        data += para[:, None, None, :]
    return data

def normalize(data):
    '''
    Notice: without batch operation.
    '''
    mean_pose = np.mean(data[0, 0, :], axis=0)
    # shape: 3
    data = data - mean_pose[None, None, None, :]
    return data, mean_pose   

def rotate_Y(input, beta):
    '''
    beta: angle
    '''
    output = np.zeros_like(input)
    beta = beta * (np.pi / 180) # angle to radian
    output[:, :, :, 0] = np.cos(beta)*input[:, :, :, 0] + np.sin(beta)*input[:, :, :, 2]
    output[:, :, :, 2] = -np.sin(beta)*input[:, :, :, 0] + np.cos(beta)*input[:, :, :, 2]
    output[:, :, :, 1] = input[:, :, :, 1]
    return output

class SoMoFDataset_3dpw(Dataset):
    def __init__(self, dset_path, seq_len, N, J, split_name='train'):
        self.seq_len = seq_len

        with open(dset_path, 'rb') as input:
            oridata = pkl.load(input)['{}'.format(split_name)]
            p_oridata = np.roll(np.array(oridata), 1, axis=1)
            oridata = np.concatenate((oridata, p_oridata), axis=0)

        self.data = []
        self.data_para = []
        
        videoNumIn = len(oridata)
        agentsNum = N
        timeStepsNum = seq_len
        jointsNum = J
        coordsNum = 3  # x y z
        self.dim = 6
        
        for i in range(videoNumIn):
            temp_data = np.array(oridata[i])
            temp_data = temp_data.reshape((agentsNum, timeStepsNum, jointsNum, coordsNum))  # [N, T, J, 3]
            temp_ = temp_data.copy()
            curr_data, curr_data_para = normalize(temp_data)  
            vel_data = np.zeros((agentsNum, timeStepsNum, jointsNum, coordsNum))
            vel_data[:,1:,:,:] = (np.roll(curr_data, -1, axis=1) - curr_data)[:,:-1,:,:]
            data = np.concatenate((curr_data, vel_data), axis=3)
            self.data.append(data)
            self.data_para.append(curr_data_para)

            if split_name == 'train':
                # rotate
                rotate_data = rotate_Y(temp_, 120)
                rotate_data, rotate_data_para = normalize(rotate_data)
                vel_data = np.zeros((agentsNum, timeStepsNum, jointsNum, coordsNum))
                vel_data[:,1:,:,:] = (np.roll(rotate_data, -1, axis=1) - rotate_data)[:,:-1,:,:]
                data = np.concatenate((rotate_data, vel_data), axis=3)
                self.data.append(data)
                self.data_para.append(rotate_data_para)

                # reverse
                reverse_data = np.flip(temp_, axis=2)
                reverse_data, reverse_data_para = normalize(reverse_data) 
                vel_data = np.zeros((agentsNum, timeStepsNum, jointsNum, coordsNum))
                vel_data[:,1:,:,:] = (np.roll(reverse_data, -1, axis=1) - reverse_data)[:,:-1,:,:]
                data = np.concatenate((reverse_data, vel_data), axis=3)
                self.data.append(data)
                self.data_para.append(reverse_data_para)
        

    def __getitem__(self, idx: int):
        data = self.data[idx].transpose((1, 0, 2, 3))  # [T, N, J, 3]
        para = self.data_para[idx]
        return data, para
    
    def __len__(self):
        return len(self.data)

class SoMoFDataset_3dpw_test(Dataset):
    def __init__(self, dset_path, seq_len, N, J):
        self.seq_len = seq_len

        oridata = []
        jsfile = open(dset_path, 'r')
        oridata_in = json.load(jsfile)

        for i in range(len(oridata_in)):
            oridata.append(oridata_in[i])
            oridata.append(np.roll(np.array(oridata_in[i]), 1, axis=0))
        
        jsfile.close()

        self.data = []
        self.data_para = []
        videoNumIn = len(oridata)
        agentsNum = N
        timeStepsNum = seq_len
        jointsNum = J
        coordsNum = 3  # x y z
        
        for i in range(videoNumIn):
            curr_data = np.zeros((agentsNum, timeStepsNum, jointsNum, coordsNum))
            curr_data_para = np.zeros((7, 1))

            temp_data = np.array(oridata[i])  
            curr_data = temp_data.reshape((agentsNum, timeStepsNum, jointsNum, coordsNum))  # [N, t, J, 3]
            curr_data, curr_data_para = normalize(curr_data) 
            vel_data = np.zeros((agentsNum, timeStepsNum, jointsNum, coordsNum))
            vel_data[:,1:,:,:] = (np.roll(curr_data, -1, axis=1) - curr_data)[:,:-1,:,:]
            data = np.concatenate((curr_data, vel_data), axis=3)
            self.data.append(data)
            self.data_para.append(curr_data_para)
    
    def __getitem__(self, idx: int):
        data = self.data[idx].transpose((1, 0, 2, 3))  # [t, N, J, 3]
        para = self.data_para[idx]
        return data, para

    def __len__(self):
        return len(self.data)
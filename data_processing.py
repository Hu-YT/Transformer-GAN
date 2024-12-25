from scipy.io import loadmat
import numpy as np
import torch
from sympy.codegen.ast import complex128


# initial
# filepath = 'car1.mat'

def data_process(filepath):
    print('Data loading...')
    data = loadmat(filepath)['data'].reshape(-1, 1)
    # empty_tensor = torch.zeros((72, 17, 10, 5), dtype=torch.complex128)
    # for i in range(72):
    #     for j in range(17):
    #         empty_tensor[i][j] = torch.from_numpy(data[i][j])
    empty_tensor = torch.zeros((1224, 10, 5), dtype=torch.float32)
    for i in range(1224):
            empty_tensor[i] = torch.from_numpy(data[i][0])
    data = empty_tensor
    # data[:,:,4] = torch.log(data[:,:,4])  #对功率取log
    # print(data.shape)
    min_values = data.min(dim=0)[0].min(dim=0)[0]
    max_values = data.max(dim=0)[0].max(dim=0)[0]
    min_values[:3] = min_values[:3].min()
    max_values[:3] = max_values[:3].max()
    # print('min_values', min_values)
    # print('max_values', max_values)
    scaled_data = 2 * (data - min_values) / (max_values - min_values) - 1
    return scaled_data
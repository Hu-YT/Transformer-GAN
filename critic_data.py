import torch
import torch.nn as nn
from scipy.io import loadmat
from models import TransformerCritic
from torch.utils.data import DataLoader, TensorDataset


batch_size = 72
num_samples = 1224
num_points = 10
feature_dim = 5
noise_dim = 128
d_model = 512

random_data = loadmat('random_data.mat')['data_random'] # (1224, 10, 5)
# random_data = loadmat('car1.mat')['data'].reshape(-1, 1)
empty_tensor = torch.zeros((num_samples, 10, 5), dtype=torch.float32)
for i in range(num_samples):
    empty_tensor[i] = torch.from_numpy(random_data[i][0])
random_data = empty_tensor
# print(random_data.shape)

critic = TransformerCritic(feature_dim=feature_dim, num_points=num_points, d_model=d_model)
critic.load_state_dict(torch.load('wgan_critic.pth'))

critic_list = []
labels = torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)  # Binary labels

dataset = TensorDataset(random_data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for i, (random_data, _) in enumerate(dataloader):
    critic_value = critic(random_data)
    critic_list += [torch.mean(critic_value).detach().item()]

print(critic_list)
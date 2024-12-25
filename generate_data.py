import torch
import torch.nn as nn
from scipy.io import savemat
from models import TransformerGenerator


batch_size = 72
num_samples = 1224
num_points = 10
feature_dim = 5
noise_dim = 128
d_model = 512

generator = TransformerGenerator(noise_dim=noise_dim, feature_dim=feature_dim, num_points=num_points, d_model=d_model)
generator.load_state_dict(torch.load('wgan_generator_initial.pth'))

for i in range(204):
    noise = torch.randn(batch_size, noise_dim)
    fake_data = generator(noise).detach()[0]
    fake_data = fake_data.unsqueeze(0)
    if i > 0 :
        data = torch.cat((data, fake_data), dim=0)
    else :
        data = fake_data


savemat('generated_data_initial.mat', {'generated_data': data})
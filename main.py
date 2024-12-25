import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_processing import data_process
from models import  TransformerCritic,TransformerGenerator
import os
from scipy.io import savemat


# Example dataset creation
batch_size = 72
num_samples = 1224
num_points = 10
feature_dim = 5
noise_dim = 128
d_model = 512

# Generate random dataset
filepath = 'car1.mat'
data = data_process(filepath)
# print(data.shape)
labels = torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)  # Binary labels
# labels will not be used in training, just to satisfy the need of TensorData

dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize WGAN models
generator = TransformerGenerator(noise_dim=noise_dim, feature_dim=feature_dim, num_points=num_points, d_model=d_model)
critic = TransformerCritic(feature_dim=feature_dim, num_points=num_points, d_model=d_model)

# Optimizers
g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=0.00005)
d_optimizer = torch.optim.RMSprop(critic.parameters(), lr=0.00005)

torch.save(generator.state_dict(), "wgan_generator_initial.pth")

# Training loop
epochs = 2000
lambda_gp = 10
d_loss_list = []
g_loss_list = []
r_values_list = []
g_values_list = []
for epoch in range(epochs):
    for i, (real_data, _) in enumerate(dataloader):
        batch_size = real_data.size(0)
        # Train Critic
        noise = torch.randn(batch_size, noise_dim)
        fake_data = generator(noise).detach()

        real_validity = critic(real_data)
        fake_validity = critic(fake_data)

        # Gradient penalty
        alpha = torch.rand(batch_size, 1, 1)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        d_interpolates = critic(interpolates)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(d_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()



        # Train Generator
        if i % 5 == 0:
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise)
            g_loss = -torch.mean(critic(fake_data))
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

    r_values_list += [torch.mean(real_validity).detach().item()]
    g_values_list += [torch.mean(fake_validity).detach().item()]
    d_loss_list += [d_loss.item()]
    g_loss_list += [g_loss.item()]
    print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Save models
torch.save(generator.state_dict(), "wgan_generator.pth")
torch.save(critic.state_dict(), "wgan_critic.pth")

output_path = os.path.join(os.getcwd(), 'losses', f'critic_loss.mat')
savemat(output_path, {'critic_losses': d_loss_list})
output_path = os.path.join(os.getcwd(), 'losses', f'generator_loss.mat')
savemat(output_path, {'generator_losses': g_loss_list})
output_path = os.path.join(os.getcwd(), 'losses', f'critic_real_values.mat')
savemat(output_path, {'critic_real_values': r_values_list})
output_path = os.path.join(os.getcwd(), 'losses', f'critic_fake_values.mat')
savemat(output_path, {'critic_fake_values': g_values_list})
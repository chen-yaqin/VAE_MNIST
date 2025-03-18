import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vq import VectorQuantizer, VectorQuantizerEMA

class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        res = x  
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += res 
        return F.relu(out)
    
class ResBlock_Conv(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock_Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        res = x  
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += res  
        return F.relu(out)
    
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.resblock1 = ResBlock(hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log-variance
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.resblock2 = ResBlock(hidden_dim)
        self.fc4 = nn.Sequential(
                    nn.Linear(hidden_dim, input_dim),
                    # nn.BatchNorm1d(input_dim)  # 归一化
                    )


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.resblock1(h1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = self.resblock2(h3)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KL


class VAE_Conv(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE_Conv, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (28x28) -> (14x14)
            nn.ReLU(),
            # ResBlock_Conv(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (14x14) -> (7x7)
            nn.ReLU(),
            # ResBlock_Conv(64),
            nn.Flatten(),  
            nn.Linear(64*7*7, 256),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 64*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)), 
            # ResBlock_Conv(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (7x7) -> (14x14)
            nn.ReLU(),
            # ResBlock_Conv(32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (14x14) -> (28x28)
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(-1, 1, 28, 28)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        return self.decoder(h.view(h.shape[0], -1))  

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        x = x.view(-1, 1, 28, 28)  
        MSE = F.mse_loss(recon_x, x, reduction='sum') 
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KL


class beta_VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, beta=1):
        super(beta_VAE, self).__init__()
        self.beta = beta
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.resblock1 = ResBlock(hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log-variance
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.resblock2 = ResBlock(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        # h1 = self.resblock1(h1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # h3 = self.resblock2(h3)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # print(f'beta = {self.beta}')
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + self.beta*KL
    

class WAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, lambda_mmd=10.0):

        super(WAE, self).__init__()
        self.lambda_mmd = lambda_mmd

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x.view(-1, 784))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z

    def loss_function(self, recon_x, x, z):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  
        MMD = self.mmd_loss(z)  # MMD Loss
        return BCE + self.lambda_mmd * MMD

    def mmd_loss(self, z):
        prior_z = torch.randn_like(z)  
        return torch.mean((z - prior_z) ** 2)

class GM_VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_components=10):

        super(GM_VAE, self).__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim

        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  
        )

        
        self.gmm_means = nn.Parameter(torch.randn(num_components, latent_dim))  
        self.gmm_logvars = nn.Parameter(torch.zeros(num_components, latent_dim))  

        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 784))
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar, z):

        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        prior_mu = self.gmm_means  
        prior_var = torch.exp(self.gmm_logvars)  

        log_p_z = -0.5 * torch.sum((z.unsqueeze(1) - prior_mu) ** 2 / prior_var, dim=-1)
        log_p_z = torch.logsumexp(log_p_z, dim=1) - torch.log(torch.tensor(self.num_components))

        log_q_z = -0.5 * torch.sum(logvar + (z - mu) ** 2 / torch.exp(logvar), dim=-1)
        KL = torch.mean(log_q_z - log_p_z)

        return BCE + KL

# class VQ_VAE(nn.Module):

#     def __init__(self, input_dim=784, hidden_dim=400, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
#         super(VQ_VAE, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, latent_dim)
#         )

#         self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim),
#             nn.Sigmoid()  
#         )

#     def forward(self, x):
#         z_e = self.encoder(x.view(-1, 784))  # 784 -> (hidden_dim) -> latent_dim
#         z_q, vq_loss = self.vq_layer(z_e)  
#         recon_x = self.decoder(z_q) 
#         return recon_x, vq_loss

#     def loss_function(self, recon_x, x, vq_loss):
#         # print(f'recon_x size = {recon_x.size()}')
#         # print(f'x size = {x.size()}')
#         BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  
#         return BCE + vq_loss  

class VQ_VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
        
        super(VQ_VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=4, stride=2, padding=1),  # (28x28) -> (14x14)
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # (14x14) -> (7x7)
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(hidden_dim * 7 * 7, latent_dim)
        )

        self.vq_layer = VectorQuantizerEMA(num_embeddings, latent_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim, 7, 7)),  
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # (7x7) -> (14x14)
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, 1, kernel_size=4, stride=2, padding=1),  # (14x14) -> (28x28)
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        z_e = self.encoder(x)  # (batch_size, latent_dim)
        z_q, vq_loss = self.vq_layer(z_e)  
        recon_x = self.decoder(z_q)  # (batch_size, 1, 28, 28)
        recon_x = recon_x.view(-1, 1, 28, 28)
        return recon_x, vq_loss

    def loss_function(self, recon_x, x, vq_loss):
        BCE = F.binary_cross_entropy(recon_x.view(-1,784), x, reduction='sum')
        return BCE + vq_loss
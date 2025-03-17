import argparse
import torch
import yaml
import torch.utils.data
from torchvision import datasets, transforms
from models.vae import VAE, VAE_Conv, beta_VAE, WAE, GM_VAE, VQ_VAE
from train import train, test


parser = argparse.ArgumentParser(description='VAE MNIST Training')
parser.add_argument('--config', type=str, default='configs/vae_std_z10.yaml')
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)
    
device = torch.device("cpu") 


torch.manual_seed(config['seed'])

model_classes = {"vae_std": VAE, "vae_conv": VAE_Conv, "beta_vae": beta_VAE, "wae": WAE, "gm_vae": GM_VAE, "vq_vae": VQ_VAE} 
ModelClass = model_classes[config["model_name"]]  

if config["model_name"] == "beta_vae":
    model = ModelClass(input_dim=784, hidden_dim=config["hidden_dim"], latent_dim=config["z_dim"], beta=config["beta"])
elif config["model_name"] == "wae":
    model = ModelClass(input_dim=784, hidden_dim=config["hidden_dim"], latent_dim=config["z_dim"], lambda_mmd=config["lambda_mmd"])
elif config["model_name"] == "gm_vae":
    model = ModelClass(input_dim=784, hidden_dim=config["hidden_dim"], latent_dim=config["z_dim"], num_components=config["num_components"])
elif config["model_name"] == "vq_vae":
    model = ModelClass(input_dim=784, hidden_dim=config["hidden_dim"], latent_dim=config["z_dim"],
                       num_embeddings=config["num_embeddings"], commitment_cost=config["commitment_cost"])
else:
    model = ModelClass(input_dim=784, hidden_dim=config["hidden_dim"], latent_dim=config["z_dim"])


train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()), batch_size=config["batch_size"], shuffle=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor()), batch_size=config["batch_size"], shuffle=True)

# model = ModelClass(input_dim=784, hidden_dim=config["hidden_dim"], latent_dim=config["z_dim"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])


if __name__ == "__main__":
    for epoch in range(1, config["epochs"] + 1):
        train(model, train_loader, optimizer, epoch, device, config["log_interval"])
        test(model, test_loader, device, epoch)
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from models.vae import VAE, beta_VAE, WAE, GM_VAE, VQ_VAE
import os

def save_checkpoint(model, optimizer, epoch, path="checkpoints"):
    os.makedirs(path, exist_ok=True) 
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, f"{path}/vae_epoch_{epoch}.pth")
    print(f"Checkpoint saved at {path}/vae_epoch_{epoch}.pth")

def train(model, train_loader, optimizer, epoch, device,log_interval=10):
    model.train()
    train_loss = 0
    train_mse_loss = 0
    for batch_idx, (data,_) in enumerate(train_loader):
        data = data.view(-1, 784) # delete if use conv
        optimizer.zero_grad()
        
        if isinstance(model, VQ_VAE):
            recon_batch, vq_loss = model(data)
            loss = model.loss_function(recon_batch, data, vq_loss)

        elif isinstance(model, WAE):
            recon_batch, z = model(data) 
            loss = model.loss_function(recon_batch, data, z)

        elif isinstance(model, GM_VAE):
            recon_batch, mu, logvar, z = model(data) 
            loss = model.loss_function(recon_batch, data, mu, logvar, z)

        else:  # VAE / Beta-VAE
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            
        # mse_loss = ((recon_batch - data).abs().mean())**2
        mse_loss = torch.mean((recon_batch-data)**2)
        # print(f"MSE Loss: {mse_loss.item()}")
        # diff = (recon_batch - data).abs().mean().item()
        # print(f"Mean absolute difference: {diff}")
        # print(f"Data Shape: {data.shape}, Recon Shape: {recon_batch.shape}")
        # print(f"Data Min/Max: {data.min().item()} / {data.max().item()}")
        # print(f"Recon Min/Max: {recon_batch.min().item()} / {recon_batch.max().item()}")

# MSE between reconstructed & original images
        train_mse_loss += mse_loss.item()
        # print(f"train_mse_loss: {train_mse_loss}")
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch}[{batch_idx*len(data)}/{len(train_loader.dataset)} '
                f'({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item()/len(data):.4f}')
    # print(f"train_loader.dataset: {len(train_loader.dataset)}")        
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_mse_loss = train_mse_loss / len(train_loader)
    
    print(f'===> Epoch: {epoch} Average Total Loss: {avg_train_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}')
    
    save_checkpoint(model, optimizer, epoch)
    
def test(model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    test_mse_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.view(-1, 784)
            
            if isinstance(model, VQ_VAE):
                recon_batch, vq_loss = model(data)
                loss = model.loss_function(recon_batch, data, vq_loss)

            elif isinstance(model, WAE):
                recon_batch, z = model(data) 
                loss = model.loss_function(recon_batch, data, z)

            elif isinstance(model, GM_VAE):
                recon_batch, mu, logvar, z = model(data) 
                loss = model.loss_function(recon_batch, data, mu, logvar, z)

            else:  # VAE / Beta-VAE
                recon_batch, mu, logvar = model(data)
                loss = model.loss_function(recon_batch, data, mu, logvar)
            
            test_loss += loss.item()
            mse_loss = torch.mean((recon_batch-data)**2)
            test_mse_loss += mse_loss.item()
            # recon_batch, mu, logvar = model(data)
            # test_loss += model.loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n].view(-1, 1, 28, 28), 
                    recon_batch.view(-1, 1, 28, 28)[:n]
                ])
                save_image(comparison.cpu(),f'results/reconstruction_{epoch}.png', nrow=n)
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_mse_loss = test_mse_loss / len(test_loader)

    print(f'====> Test set Average Total Loss: {avg_test_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}')
    
    

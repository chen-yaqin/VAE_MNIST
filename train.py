import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from models.vae import VAE, beta_VAE, WAE, GM_VAE, VQ_VAE

def train(model, train_loader, optimizer, epoch, device,log_interval=10):
    model.train()
    train_loss = 0
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
        

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch}[{batch_idx*len(data)}/{len(train_loader.dataset)} '
                f'({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item()/len(data):.4f}')
            
    print(f'===> Epoch: {epoch} Average loss: {train_loss/len(train_loader.dataset):.4f}')
    
def test(model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
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
            # recon_batch, mu, logvar = model(data)
            # test_loss += model.loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n].view(-1, 1, 28, 28), 
                    recon_batch.view(-1, 1, 28, 28)[:n]
                ])
                save_image(comparison.cpu(),f'results/reconstruction_{epoch}.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    
    

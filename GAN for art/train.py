from main import discriminator, generator
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

scaler = GradScaler()
class Trainer():
    def __init__(self, batch_size, latent_size, device):
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.device = device

    def train_discriminator(self, real_images, opt_d, use_amp = True):
        # Clear discriminator gradients
        opt_d.zero_grad(set_to_none=True)

        # Pass real images through discriminator
        with autocast(enabled=use_amp):
            real_preds = discriminator(real_images)
            real_targets = torch.ones(real_images.size(0), 1, device=self.device)
            #real_loss = F.binary_cross_entropy(real_preds, real_targets)
            real_loss = F.binary_cross_entropy_with_logits(real_preds, real_targets)
            real_score = torch.mean(real_preds).item()
            
            # Generate fake images
            latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
            fake_images = generator(latent)

            # Pass fake images through discriminator
            fake_targets = torch.zeros(fake_images.size(0), 1, device=self.device)
            fake_preds = discriminator(fake_images)
            #fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
            fake_loss = F.binary_cross_entropy_with_logits(fake_preds, fake_targets)
            fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        scaler.scale(loss).backward()
        scaler.step(opt_d)
        scaler.update()
        
        return loss.item(), real_score, fake_score


    def train_generator(self, opt_g, use_amp = True):
        # Clear generator gradients
        opt_g.zero_grad(set_to_none=True)
        
        # Generate fake images
        with autocast(enabled = use_amp):
            latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
            fake_images = generator(latent)
            
            # Try to fool the discriminator
            preds = discriminator(fake_images)
            targets = torch.ones(self.batch_size, 1, device=self.device)
            #loss = F.binary_cross_entropy(preds, targets)
            loss = F.binary_cross_entropy_with_logits(preds, targets)
        
        # Update generator weights
        scaler.scale(loss).backward()
        scaler.step(opt_g)
        scaler.update()

        return loss.item()

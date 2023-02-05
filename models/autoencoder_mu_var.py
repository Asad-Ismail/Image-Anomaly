import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch import Tensor
# Torchvision
import torchvision
from torchvision import transforms
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import timm
from tqdm import tqdm


def relative_euclidean_distance(pred, gt,eps=1e-7):
    return (pred-gt).norm(2,dim=1) / (pred.norm(2,dim=1)+eps)


class Encoder(nn.Module):
    def __init__(self,dims=512):
        super().__init__()
        self.encoder = timm.create_model('vgg19_bn',features_only=True, pretrained=True)
        self.fc_mu = nn.Linear(dims, dims)
        self.fc_var = nn.Linear(dims, dims)
    def forward(self, x):
        x=self.encoder(x)[-1]
        x=x.mean((2,3))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return (x,mu,log_var)

class Decoder(nn.Module):
    """Decoder"""
    def __init__(self,
                 num_input_channels : int,
                 c_hid : int,
                 upsample: int=4,
                 interlayers:int=3,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        
        channels=[c_hid,c_hid//2,c_hid//4,c_hid//8,c_hid//16]
        ## Convs which are to be embedded in between upsampling layers
        layers=[]
        for i in range(upsample):
            # Two upsampling layers
            layers.append(nn.ConvTranspose2d(channels[i], channels[i], kernel_size=2, stride=2))
            layers.append(act_fn())
            layers.append(nn.ConvTranspose2d(channels[i], channels[i], kernel_size=2, stride=2))
            layers.append(act_fn())
            ## Conv Layers
            midconvs=[]
            for j in range(interlayers):
                #Last layer
                if i==upsample-1:
                    midconvs.append(nn.Conv2d(channels[i], num_input_channels, kernel_size=3, padding=1))
                    midconvs.append(nn.Sigmoid())
                    break
                else:
                    # first layer change the number of channels
                    if j==0:
                        midconvs.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1))
                        midconvs.append(act_fn())
                    else:
                        midconvs.append(nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1))
                        midconvs.append(act_fn())
            layers.extend(midconvs)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class Autoencoder(pl.LightningModule):

    def __init__(self,
                 latent_dim: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3,
                 width:int= 128,
                 height:int= 128,
                 is_training=False):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class()
        self.decoder = decoder_class(num_input_channels, latent_dim)
        print(self.decoder)
        #self.inception= timm.create_model('vgg19_bn',features_only=True, pretrained=True)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
        self.is_training=is_training
    
    @torch.no_grad()
    def get_test_features(self,enc,x,xhat):
        #rec_euclidean = relative_euclidean_distance(x.flatten(start_dim=1), xhat.flatten(start_dim=1))
        rec_euclidean = F.mse_loss(xhat.flatten(start_dim=1), x.flatten(start_dim=1), reduction="none").sum(dim=[1])
        #return rec_euclidean.reshape(-1, 1)
        #rec_cosine = F.cosine_similarity(x.flatten(start_dim=1), xhat.flatten(start_dim=1),dim=1)
        enc = torch.cat([enc, rec_euclidean.unsqueeze(-1)], dim=1)
         # return rec_cosine.unsqueeze(-1)
        return enc

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        enc,mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        zreshaped= z.reshape(z.shape[0],-1,1,1)
        print(f"Input to decoder is {zreshaped.shape}")
        x_hat = self.decoder(zreshaped)
        print(f"Decoder shape is {x_hat.shape}")
        if self.training:
            return x_hat,mu,log_var
        else:
            #This one is just hack for not chaning the training code while logging validation loss
            if self.is_training:
                return x_hat,mu,log_var
            else:
                return self.get_test_features(enc,x,x_hat)


    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat,mu,log_var = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        recons_loss  = loss.sum(dim=[1,2,3]).mean(dim=[0])

        kld_weight = 1.0 # Account for the minibatch samples from the dataset
        #recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss

        return loss

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=10,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        #loss1 = self._get_reconstruction_loss(batch)
        #loss2 = self._get_inception_loss(batch)
        #loss=loss1+loss2
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)

if __name__=="__main__":
    #model=Autoencoder(latent_dim=512)
    x=torch.rand(2,3,256,256)
    model=Autoencoder(512)
    model.eval()
    #y,mu,var=model(x)
    enc=model(x)
    print(enc.shape)
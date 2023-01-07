import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision import transforms
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import timm



class Encoder(nn.Module):
    def __init__(self,minvalue=9e-1):
        super().__init__()
        self.encoder = timm.create_model('vgg19_bn',features_only=True, pretrained=True)
        self.minvalue=minvalue
    def forward(self, x):
        x=self.encoder(x)[-1]
        x=x.mean((2,3),keepdim=True)
        ## Rescale min values so the deterrminant of matrix does not becomes zero
        return x+self.minvalue

class Decoder(nn.Module):
    """Decoder"""
    def __init__(self,
                 num_input_channels : int,
                 c_hid : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        
        inter1=c_hid//2
        inter2=inter1//2
        inter3=inter2//2

        hidconvs=[]
        for _ in range(5):
            hidconvs.append(nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1))
            hidconvs.append(act_fn())
        inter1convs=[]
        for _ in range(5):
            inter1convs.append(nn.Conv2d(inter1, inter1, kernel_size=3, padding=1))
            inter1convs.append(act_fn())
        inter2convs=[]
        for _ in range(5):
            inter2convs.append(nn.Conv2d(inter2, inter2, kernel_size=3, padding=1))
            inter2convs.append(act_fn())
        inter3convs=[]
        for _ in range(5):
            inter3convs.append(nn.Conv2d(inter3, inter3, kernel_size=3, padding=1))
            inter3convs.append(act_fn())
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            act_fn(),
            *hidconvs,
            nn.ConvTranspose2d(c_hid, inter1, kernel_size=3, output_padding=1, padding=1, stride=2), # 32x32 => 64x64
            act_fn(),
            *inter1convs,
            nn.ConvTranspose2d( inter1, inter2, kernel_size=3, output_padding=1, padding=1, stride=2), # 64x64 => 128x128
            act_fn(),
            *inter2convs,
            nn.ConvTranspose2d(inter2, inter3, kernel_size=3, output_padding=1, padding=1, stride=2), # 128x128 => 256x256
            act_fn(),
            *inter3convs,
            nn.ConvTranspose2d(inter3, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 256x256 => 512x512
            act_fn(),
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1),
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

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
                 height:int= 128):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class()
        self.decoder = decoder_class(num_input_channels, latent_dim)
        self.inception= timm.create_model('vgg19_bn',features_only=True, pretrained=True)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss
    
    @torch.no_grad()
    def _get_inception(self,x):
        z=self.inception(x)[-1]
        z=z.mean((2,3),keepdim=True)+self.encoder.minvalue
        return z
    
    def _get_inception_loss(self,batch):
        x, _ = batch # We do not need the labels
        z_hat = self.encoder(x)
        z=self._get_inception(x)
        loss = F.mse_loss(z, z_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss

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
        loss1 = self._get_reconstruction_loss(batch)
        loss2 = self._get_inception_loss(batch)
        loss=loss1+loss2
        #loss = self._get_reconstruction_loss(batch)
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
    x=torch.rand(1,3,128,128)
    model=Autoencoder(512)
    y=model(x)
    print(y.shape)
    #y=model(x)
    #model_names = timm.list_models('*vgg*')
    #print(model_names)
    #encoder = timm.create_model('vgg19_bn',features_only=True, pretrained=True)
    #o = encoder(torch.randn(2, 3, 128, 128))
    #for x in o:
    #    print(x.shape)
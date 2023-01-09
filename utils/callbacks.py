import pytorch_lightning as pl
import torch
import torchvision

class GenerateCallback(pl.Callback):
    """
    A PyTorch Lightning callback for reconstructing and logging images during training.

    This callback reconstructs a set of input images using the model at the end of each training epoch,
    and logs the reconstructions to TensorBoard. The reconstructions are only logged every N epochs,
    where N is specified by the every_n_epochs parameter.

    Args:
    input_imgs (torch.Tensor): Images to reconstruct during training.
    every_n_epochs (int, optional): Only save reconstructions every N epochs. Default is 1.
    """

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs,mu,logvar = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)
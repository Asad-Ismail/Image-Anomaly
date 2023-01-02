from models.autoencoder_vgg_512 import *
from utils.callbacks import *
from data.dataloader import *
from utils.utils import *
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="hub://aismail2/cucumber_OD",help="Activeloop data path")
parser.add_argument("--size",default=512,type=int,help="Image size used for training model")
parser.add_argument("--epochs",default=1000,type=int,help="Image size used for training model")
parser.add_argument("--device", default="cuda",help="Device to Train Model")
parser.add_argument("--batch_sz", default=42,type=int,help="Device to Train Model")
parser.add_argument("--model_arch",default="repvggplus", choices=['resnet', 'repvgg','repvggplus'],type=str,help="Model Architecture")
parser.add_argument("--ckpt_path",default="./ckpts",type=str,help="Output of weights")

args = parser.parse_args()
data_path = args.dataset
img_sz = args.size
device = args.device
ckpt_path=args.ckpt_path
epochs=args.epochs
modelarch=args.model_arch
batch_sz=args.batch_sz


def train_Anamoly(latent_dim,train_loader,val_loader,vis_dataset):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(ckpt_path, f"anamoly_road_{latent_dim}"),
                         accelerator="cuda" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(get_train_images(vis_dataset,10), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(ckpt_path, f"anamoly_road_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(latent_dim=latent_dim)
    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    #test_result = trainer.test(model, test_loader, verbose=False)
    result = {"val": val_result}
    return model, result

if __name__=="__main__":
    CHECKPOINT_PATH="anamoly_checkpoints"
    train_loader,val_loader,vis_dataset=get_data(batch_sz)
    model,result=train_Anamoly(512,train_loader,val_loader,vis_dataset)
from models.autoencoder_vgg import Autoencoder

def train_Anamoly(latent_dim,CHECKPOINT_PATH):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"anamoly_road_{latent_dim}"),
                         accelerator="cuda" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=1000,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(get_train_images(8), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"anamoly_road_{latent_dim}.ckpt")
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
    CHECKPOINT_PATH=".Image_Anomoly_CKPTS"
    train_Anamoly(latent_dim=512,CHECKPOINT_PATH=CHECKPOINT_PATH)
from utils.utils import *
from data.dataloader import *
from models.autoencoder_mu_var import *
import numpy as np
from tqdm import tqdm
import scipy
from scipy import stats
from sklearn.neighbors import KernelDensity
import pickle
import os
import argparse


torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir",default="dataset/bottle/train",help="Train dir with only normal images")
parser.add_argument("--val_dir",default="dataset/bottle/test",help="Val dir with only normal and anamolus images")
args = parser.parse_args()
train_dir = args.train_dir
val_dir = args.val_dir

def get_features(model,dloader):
    ## Train features for checking 
    features_mu=[]
    features_std=[]
    features=[]
    labels=[]
    model.cuda()
    model.eval()
    for i,item in tqdm(enumerate(tqdm(dloader))):
        #if i>10:
        #    break
        x,label=item[0],item[1]
        model.cuda()
        x=x.cuda()
        enc=model(x)
        enc=enc.detach().cpu().numpy()
        labels.append(label)
        features.append(enc)
    
    features=np.concatenate(features,axis=0)
    labels=np.concatenate(labels,axis=0)
    # Flipping labels as Ananmoly gets 0 by default
    labels = (~labels.astype(bool)).astype(int)
    return features,labels


def vis_results(pred_images,labels,dists,imgs):
    pred_images=torch.concat(pred_images,axis=0)
    mask=labels==1
    anamolies=pred_images[mask,...]
    minimgs=min(anamolies.shape[0],imgs)
    anamolies=anamolies[:minimgs,...]
    anamolygrid = torchvision.utils.make_grid(anamolies, nrow=5, normalize=True, range=(-1,1))
    #plt.figure(figsize=(20,20))
    plt.imshow(np.transpose(anamolygrid, (1, 2, 0)),aspect='auto')
    plt.savefig("vis_imgs/anamolies.png")
    np.savetxt('vis_imgs/anamolies_dist.txt', dists[mask], delimiter='\n')
    mask=labels==0
    nonanamolies=pred_images[mask,...]
    minimgs=min(nonanamolies.shape[0],imgs)
    nonanamolies=nonanamolies[:minimgs,...]
    nonanamolygrid = torchvision.utils.make_grid(nonanamolies , nrow=5, normalize=True, range=(-1,1))
    #plt.figure(figsize=(20,20))
    plt.imshow(np.transpose(nonanamolygrid, (1, 2, 0)),aspect='auto')
    plt.savefig("vis_imgs/nonanamolies.png")
    np.savetxt('vis_imgs/Nonanamolies_dist.txt', dists[mask], delimiter='\n')

def get_recons_loss(model,dloader,save_examples=True,imgs=20):
    dists=[]
    labels=[]
    pred_images=[]
    for i,item in tqdm(enumerate(tqdm(dloader))):
        x,label=item[0],item[1]
        if (save_examples and i>imgs):
            break
        x=x.cuda()
        preds=model(x).detach().cpu()
        # Last entry is the distance
        dist=preds[:,-1]
        dists.append(dist.detach().cpu().numpy())
        labels.append(label.numpy())
        if save_examples:
            pred_images.append(pred_img)
    dists=np.concatenate(dists,axis=0)
    labels=np.concatenate(labels,axis=0)
    # Flipping labels as originally 0 is non anamoly
    labels = (~labels.astype(bool)).astype(int)
    print(f"Distance anamoly min and max are {dists[labels==1].min()}, {dists[labels==1].max()}")
    print(f"Distance normal min and max are {dists[labels==0].min()}, {dists[labels==0].max()}")
    if save_examples:
        vis_results(pred_images,labels,dists,imgs=imgs)
    return dists,labels

def get_reconstruction_dist(model,save_examples=True):
    """
    Compute the reconstruction loss for a given model and data.

    This function computes the reconstruction loss for a given model on a given dataset, and returns the
    losses and labels. Optionally, it can also save examples of the reconstructions.

    Args:
    model (torch.nn.Module): The model to use for reconstruction.
    save_examples (bool, optional): Whether to save examples of the reconstructions. Default is True.

    Returns:
    tuple: A tuple containing the following elements:
    - dists (List[float]): A list of reconstruction losses.
    - labels (List[int]): A list of labels for the data (0 for normal, 1 for anomalous).
    """
    train_loader,val_loader,_=get_data(train_dir,val_dir,32)
    dists,labels=get_recons_loss(model,val_loader,save_examples=save_examples)
    if not save_examples:
        epsilon, F1 = select_exact_threshold(labels, dists,reverse=True)
        print('Best epsilon found using cross-validation: %e' % epsilon)
        print('Best F1 on Cross Validation Set: %f' % F1)
    return dists,labels



def get_kde_probs(model,save_examples=False,use_cache=True):
    train_loader,val_loader,_=get_data(train_dir,val_dir,32)
    if use_cache and os.path.exists("train_features.npy"):
        print(f"Using cached training features!!")
        train_features=np.load("train_features.npy")
    else:
        print(f"Train features not found calculating!!")
        train_features,_=get_features(model,train_loader)
        print(f"Saving train features!")
        np.save("train_features.npy", train_features)
    #train_features,_=get_features(model,train_loader)
    val_features,val_labels=get_features(model,val_loader)
    ## Fit KDE on Training features
    print(f"Fitting gaussian on trianing data")
    kde = KernelDensity(kernel='gaussian',bandwidth=0.1).fit(train_features)
    print(f"Val features shape is {val_features.shape})")
    probs =  kde.score_samples(val_features)

    #print(f"Min and Max Probs of Normal {probs[val_labels==0].min()}, {probs[val_labels==0].max()} ")
    #print(f"Min and Max Probs of Anamoly {probs[val_labels==1].min()}, {probs[val_labels==1].max()} ")
    epsilon, F1 = select_exact_threshold(val_labels, probs,reverse=False)
    print('Best epsilon found using cross-validation: %e' % epsilon)
    print('Best F1 on Cross Validation Set: %f' % F1)
    return epsilon,F1




if __name__=="__main__":
    print(f"Getting Model!!")
    weights="./ckpts/bottle_512/lightning_logs/version_0/checkpoints/epoch=299-step=1800.ckpt"
    model = Autoencoder.load_from_checkpoint(weights,is_training=False)
    model.eval()
    model.cuda()
    get_reconstruction_dist(model,save_examples=False)
    #get_kde_probs(model,save_examples=False,use_cache=True)
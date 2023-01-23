from utils.utils import *
from data.dataloader import *
from models.autoencoder_mu_var import *
import numpy as np
from tqdm import tqdm
import scipy
from scipy import stats
from sklearn.neighbors import KernelDensity
import pickle


torch.multiprocessing.set_sharing_strategy('file_system')

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
        model.eval()
        x=x.cuda()
        enc=model(x)
        enc=enc.detach().cpu().numpy()
        labels.append(label)
        features.append(enc)
    
    features=np.concatenate(features,axis=0)
    labels=np.concatenate(labels,axis=0)
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
        model.cuda()
        model.eval()
        x=x.cuda()
        pred_img=model(x).detach().cpu()
        dist=F.l1_loss(pred_img.flatten(start_dim=1),x.detach().cpu().flatten(start_dim=1),reduction="none").mean(axis=1)
        dists.append(dist.detach().cpu().numpy())
        labels.append(label.numpy())
        if save_examples:
            pred_images.append(pred_img)
    dists=np.concatenate(dists,axis=0)
    labels=np.concatenate(labels,axis=0)
    # Flipping labels as originally 1 is non anamoly
    labels = (~labels.astype(bool)).astype(int)
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
    train_loader,val_loader,_=get_data(128)
    dists,labels=get_recons_loss(model,val_loader,save_examples=save_examples)
    if not save_examples:
        epsilon, F1 = select_threshold(labels, dists,reverse=True)
        print('Best epsilon found using cross-validation: %e' % epsilon)
        print('Best F1 on Cross Validation Set: %f' % F1)
    return dists,labels



def get_kde_probs(model,save_examples=False,use_cache=True):
    #import seaborn as sns
    train_loader,val_loader,_=get_data(2)
    if use_cache and os.path.exists("train_features.npy"):
        train_features=np.load("train_features.npy")
    else:
        print(f"Train features not found calculating!!")
        train_features,_=get_features(model,train_loader)
        print(f"Saving train features!")
        np.save("train_features.npy", train_features)
    train_features,_=get_features(model,train_loader)
    val_features,val_labels=get_features(model,val_loader)
    ## Fit KDE on Training features
    kde = KernelDensity(kernel='gaussian').fit(train_features)
    #kde = KernelDensity(kernel='cosine').fit(train_features)
    ## Evalue on val loader
    probs =  kde.score_samples(val_features)

    print(f"Min and Max Probs of Anamoly {probs[val_labels==1].min()}, {probs[val_labels==1].max()} ")

    print(f"Min and Max Probs of Normal {probs[val_labels==0].min()}, {probs[val_labels==0].max()} ")

    #s = len(val_labels)
    #c = np.sum(val_labels==1)
    #g = c/s
    #thresh = np.percentile(probs, int(g*100))
    #pred = (probs < thresh).astype(int)
    
    #F1=get_F1(pred,val_labels)
    #print(f"Threshold is {thresh}")
    #print(f"F1 is {F1}")


    epsilon, F1 = select_threshold(val_labels, probs,reverse=False)
    print('Best epsilon found using cross-validation: %e' % epsilon)
    print('Best F1 on Cross Validation Set: %f' % F1)
    return epsilon,F1




if __name__=="__main__":
    print(f"Getting Model!!")
    weights="./ckpts/anamoly_road_512/lightning_logs/version_1/checkpoints/epoch=209-step=202020.ckpt"
    model = Autoencoder.load_from_checkpoint(weights)
    get_kde_probs(model,save_examples=False)
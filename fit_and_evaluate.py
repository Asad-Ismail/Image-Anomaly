from utils.utils import *
from data.dataloader import *
from models.autoencoder_mu_var import *
import numpy as np
from tqdm import tqdm
import scipy
from scipy import stats

def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m, n = X.shape
    mu= np.mean(X,axis=0)
    var= np.var(X,axis=0) 
    return mu, var

def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
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

def get_features(model,dloader,keepk=None):
    ## Train features for checking 
    features_mu=[]
    features_std=[]
    labels=[]
    for i,item in tqdm(enumerate(tqdm(dloader))):
        x,label=item[0],item[1]
        if (i>50):
            break
        model.cuda()
        model.eval()
        x=x.cuda()
        #x=x.unsqueeze(0)
        mu,logvar=model.encoder(x)
        std = torch.exp(0.5 * logvar)
        #z=reparameterize(mu, logvar).detach().cpu()
        features_mu.append(mu.flatten(start_dim=1).detach().cpu().numpy())
        features_std.append(std.flatten(start_dim=1).detach().cpu().numpy())
        labels.append(label)
    features_mu=np.concatenate(features_mu,axis=0)
    features_std=np.concatenate(features_std,axis=0)
    labels=np.concatenate(labels,axis=0)

    mean=np.average(features_mu , axis=0)
    std=np.average(features_std , axis=0)

    print(f"Mean mean shape is {mean.shape} ")
    print(f"Std mean shape is {std.shape} ")
    return mean,std,labels

def fit_gaussian_latent(model):
    """Proper way to do it with gaussian fitting if determiant of variance is well defined"""
    print(f"Getting Data!!")
    train_loader,val_loader,_=get_data(32)
    print(f"Getting Train features!!")
    train_features,_=get_features(model,train_loader)
    print(f"Train features shape is {train_features.shape}")
    mu, var = estimate_gaussian(train_features)  
    print(f"Estimated mean is {mu}")
    print(f"Estimated var is {var}")
    p_train = multivariate_gaussian(train_features, mu, var)
    #p_train=multivariate_normal.pdf(train_features, mean=mu, cov=var)
    print(f"Getting Val features!!")
    val_features,y_val=get_features(model,val_loader)
    print(f"Val features shape is {val_features.shape}")
    print(f"Val Labels sizes are {y_val.shape}")
    p_val = multivariate_gaussian(val_features, mu, var)
    epsilon, F1 = select_threshold(y_val, p_val)
    print('Best epsilon found using cross-validation: %e' % epsilon)
    print('Best F1 on Cross Validation Set: %f' % F1)


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
    print(f"Getting Data!!")
    train_loader,val_loader,_=get_data(128)
    dists,labels=get_recons_loss(model,val_loader,save_examples=save_examples)
    if not save_examples:
        epsilon, F1 = select_threshold(labels, dists,reverse=True)
        print('Best epsilon found using cross-validation: %e' % epsilon)
        print('Best F1 on Cross Validation Set: %f' % F1)
    return dists,labels


def get_kdes(model,dloader,kernel,save_examples=True,imgs=200000):
    probs=[]
    labels=[]
    pred_images=[]
    for i,item in tqdm(enumerate(tqdm(dloader))):
        x,label=item[0],item[1]
        if (save_examples and i>imgs):
            break
        model.cuda()
        model.eval()
        x=x.cuda()

        mu,logvar=model.encoder(x)
        z=reparameterize(mu, logvar).detach().cpu()
        pred_image=model(x).detach().cpu()

        if isinstance(kernel,tuple):
            mu,var=kernel
            prob=multivariate_gaussian(z, mu, var)
            print("prob shape is {prob.shape}")
        else:
            z=z.transpose(1,0)
            prob=kernel(z)
        probs.append(prob)
        labels.append(label.numpy())
        if save_examples:
            pred_images.append(pred_img)
    probs=np.concatenate(probs,axis=0)
    labels=np.concatenate(labels,axis=0)
    # Flipping labels as originally 1 is non anamoly
    labels = (~labels.astype(bool)).astype(int)
    if save_examples:
        vis_results(pred_images,labels,probs,imgs=imgs)
    return probs,labels


def get_kde_probs(model,save_examples=True):
    print(f"Getting Data!!")
    import seaborn as sns
    train_loader,val_loader,_=get_data(128)
    features,_=get_features(model,train_loader)
    print(f"Features shape is {features.shape}")
    sns.histplot(features[:,30],binrange=(-5,5))
    plt.savefig("test.png")
    
    #mu, var = estimate_gaussian(train_features)  
    #p_train = multivariate_gaussian(train_features, mu, var)
    features=features.transpose(1,0)
    print(f"Features shape is {features.shape}")
    #print(f"Train features min and max are {train_features.min(),train_features.max()}")
    #print(f"Train features shape is {train_features.shape}")
    #print(f"Fitting KDE Kernel!!")
    #train_features=train_features.transpose(1,0)
    kernel = stats.gaussian_kde(features)
    print(f"Fitting complete of KDE Kernel!!")
    prob=kernel(features)
    print(f"Probs shape is {prob.shape}")

    probs,labels=get_kdes(model,val_loader,kernel,save_examples=save_examples)
    return
    probs,labels=get_kdes(model,val_loader,(mu,var),save_examples=save_examples)
    if not save_examples:
        epsilon, F1 = select_threshold(labels, probs,reverse=False)
        print('Best epsilon found using cross-validation: %e' % epsilon)
        print('Best F1 on Cross Validation Set: %f' % F1)
    return dists,labels




if __name__=="__main__":
    print(f"Getting Model!!")
    weights="./ckpts/anamoly_road_512/lightning_logs/version_1/checkpoints/epoch=209-step=202020.ckpt"
    model = Autoencoder.load_from_checkpoint(weights)
    #dists,labels=get_reconstruction_dist(model,save_examples=False)
    #dists,labels=get_reconstruction_dist(model,save_examples=True)

    get_kde_probs(model,save_examples=False)


    #y=np.arange(len(dists))
    #print(f"Min and Max dists are {dists.min(),dists.max()}")
    #color= ['green' if l == 0 else 'red' for l in labels]
    #plt.scatter(dists,y, color=color)
    #plt.savefig("test.png")
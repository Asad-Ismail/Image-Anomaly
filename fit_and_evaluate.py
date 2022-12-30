from utils.utils import *
import numpy as np
from tqdm import tqdm

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

def get_features(dloader):
    ## Train features for checking 
    features=[]
    ys=[]
    for i,item in tqdm(enumerate(tqdm(dloader))):
        x,y=item[0],item[1]
        model.cuda()
        model.eval()
        x=x.cuda()
        #x=x.unsqueeze(0)
        y=model.encoder(x).detach().cpu()
        features.append(y.flatten(start_dim=1).numpy())
        ys.append(y)
    features=np.concatenate(train_features,axis=0)
    ys=np.concatenate(ys,axis=0)
    return features,ys
    

if __name__=="__main__":
    train_dir = '../AnamolyData/train/images' 
    val_dir= '../AnamolyData/val/images'
    train_loader,val_loader,_=get_data(32)
    print(f"Getting Train features!!")
    train_features,_=get_features(train_loader)
    print(f"Train features sizes are {train_features.shape}")
    mu, var = estimate_gaussian(train_features)  
    p_train = multivariate_gaussian(train_features, mu, var)
    print(f"Getting Val features!!")
    val_features,y_val=get_features(val_loader)
    print(f"Val features sizes are {train_features.shape}")
    print(f"Val Labels sizes are {y_val.shape}")
    p_val = multivariate_gaussian(val_features, mu, var)
    epsilon, F1 = select_threshold(y_val, p_val)
    print('Best epsilon found using cross-validation: %e' % epsilon)
    print('Best F1 on Cross Validation Set: %f' % F1)
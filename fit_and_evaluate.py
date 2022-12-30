from utils import *
import numpy as np

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
    for i,item in enumerate(tqdm(dloader)):
        x,y=item[0],item[1]
        model.cuda()
        model.eval()
        x=x.cuda()
        #x=x.unsqueeze(0)
        y=model.encoder(x).detach().cpu()
        features.append(y.flatten(start_dim=1).numpy())
        ys.append(y)
    features=np.concatenate(train_features)
    ys=np.concatenate(ys)
    return features,ys
    

if __name__=="__main__":
    train_dir = '../AnamolyData/train/images' 
    val_dir= '../AnamolyData/val/images'
    transform = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=18, shuffle=True,drop_last=True) 
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=18, shuffle=True,drop_last=True)
    train_features,_=get_features(train_loader)
    mu, var = estimate_gaussian(train_features)  
    p_train = multivariate_gaussian(train_features, mu, var)
    val_features,y_val=get_features(val_loader)
    p_val = multivariate_gaussian(val_features, mu, var)
    epsilon, F1 = select_threshold(y_val, p_val)
    print('Best epsilon found using cross-validation: %e' % epsilon)
    print('Best F1 on Cross Validation Set: %f' % F1)
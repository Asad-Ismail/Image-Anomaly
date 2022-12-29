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
    train_features=[]
    for i,item in enumerate(tqdm(dloader)):
        x,y=item[0],item[1]
        model.cuda()
        model.eval()
        x=x.cuda()
        #x=x.unsqueeze(0)
        y=model.encoder(x).detach().cpu()
        train_features.append(y.flatten(start_dim=1))
    return train_features
    

if __name__=="__main__":
    train_dir = '../AnamolyData/train/images' # load from Kaggle
    val_dir= '../AnamolyData/val/images'
    transform = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=18, shuffle=True,drop_last=True) 
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=18, shuffle=True,drop_last=True)
    train_features=get_features(train_loader)
    mu, var = estimate_gaussian(train_features)  
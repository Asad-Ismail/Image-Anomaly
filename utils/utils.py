import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def get_train_images(dataset,num):
    """
    Extract a specified number of images from a given dataset.

    This function extracts a specified number of images from a given dataset and returns them as a stack of
    tensors.

    Args:
    dataset (torch.utils.data.Dataset): The dataset to extract images from.
    num (int): The number of images to extract.

    Returns:
    torch.Tensor: A stack of the extracted images, with size (num, C, H, W) where C is the number of
    channels, H is the image height, and W is the image width.
    """
    return torch.stack([dataset[i][0] for i in range(num)], dim=0)


def multivariate_gaussian(X, mu, var):
    """
    Computes the probability 
    density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and var. If var is a matrix, it is
    treated as the covariance matrix. If var is a vector, it is treated
    as the var values of the variances in each dimension (a diagonal
    covariance matrix
    """
    
    k = len(mu)

    if var.ndim == 1:
        var = np.diag(var)
    
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p
        
def visualize_fit(X, mu, var):
    """
    This visualization shows you the 
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """
    
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
        
    # Set the title
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    # Set the y-axis label
    plt.ylabel('Throughput (mb/s)')
    # Set the x-axis label
    plt.xlabel('Latency (ms)')

def get_F1(preds,gts):
    tp= sum(np.logical_and(gts==1, preds==True))
    fp= sum(np.logical_and(gts==0,preds==True))
    fn= sum(np.logical_and(gts==1, preds==False))
    prec=tp/(tp+fp)
    rec=tp/(tp+fn)
    F1=(2*(prec*rec))/(prec+rec)
    return F1


def select_threshold(y_val, p_val,reverse=False): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        if reverse:
            pred= (p_val>epsilon)
        else:
            pred= (p_val<epsilon)

        F1=get_F1(pred,y_val)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1

def select_exact_threshold(y_val, p_val,reverse=False): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    #p_val=sorted(p_val)
    
    print(f"Calculating best threshold!")

    for epsilon in tqdm(p_val):
        if reverse:
            pred= (p_val>epsilon)
        else:
            pred= (p_val<epsilon)

        F1=get_F1(pred,y_val)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1
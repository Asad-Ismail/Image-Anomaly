from torchvision import datasets, transforms
import torch
from torchvision.transforms import InterpolationMode

def get_data(batch_sz=32):
    """
    Load and preprocess image data for training and validation.

    This function loads image data from the 'train/images' and 'val/images' directories, applies some
    preprocessing transformations, and returns the resulting data loaders and datasets. The preprocessing
    transformations include resizing the images to 128x128 pixels, converting them to tensors, and normalizing
    the pixel values.

    Args:
    batch_sz (int, optional): batch size for the data loaders. Default is 32.

    Returns:
    tuple: A tuple containing the following elements:
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
    - train_dataset (torch.utils.data.Dataset): Dataset for the training data.
    """
    train_dir = '../AnamolyData/train/images'
    val_dir= '../AnamolyData/val/images'
    transform = transforms.Compose([transforms.Resize((128,128),interpolation=InterpolationMode.NEAREST),
                                    transforms.ToTensor(),
                                 ])
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz,num_workers=8,shuffle=True,drop_last=True) 
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, num_workers=8,shuffle=True,drop_last=False)
    return train_loader,val_loader,train_dataset
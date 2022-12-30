from torchvision import datasets, transforms
import torch

def get_data(batch_sz=32):
    train_dir = '../AnamolyData/train/images'
    val_dir= '../AnamolyData/val/images'
    transform = transforms.Compose([transforms.Resize((128,128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], 
                                                         [0.5, 0.5, 0.5])
                                ])
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True,drop_last=True) 
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, shuffle=True,drop_last=True)
    return train_loader,val_loader,train_dataset
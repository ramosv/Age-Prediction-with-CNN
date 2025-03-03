import torchvision.transforms as T

def get_train_transforms():
    return T.Compose([
        T.Resize((110, 110)),
        T.RandomRotation(degrees=10),
        T.RandomCrop((100, 100)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return T.Compose([
        T.Resize((100, 100)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

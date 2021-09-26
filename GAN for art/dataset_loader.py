import os
import opendatasets as od
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataset(batch_size, image_size, stats):
    dataset_url = 'https://www.kaggle.com/ikarus777/best-artworks-of-all-time'
    od.download(dataset_url)

    DATA_DIR = './best-artworks-of-all-time'
    print(os.listdir(DATA_DIR))

    batch_size = 128
    image_size = (64,64)
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    transform_ds = transforms.Compose([transforms.Resize(image_size),
    #                                    transforms.RandomCrop(32, padding=2),
    #                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*stats)
                                    ])

    train_ds = torchvision.datasets.ImageFolder(root="./best-artworks-of-all-time/resized",
                                        transform=transform_ds)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

    return train_dl

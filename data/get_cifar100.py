import os
from torchvision.datasets import CIFAR100
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def export_cifar100_to_imagefolder(root_dir='cifar100'):
    # Define transforms
    transform = transforms.ToPILImage()

    # Load CIFAR-100 dataset
    train_set = CIFAR100(root=root_dir, train=True, download=True)
    test_set = CIFAR100(root=root_dir, train=False, download=True)

    # Create directories
    for split, dataset in [('train', train_set), ('val', test_set)]:
        for idx, (img, label) in enumerate(tqdm(dataset, desc=f'Processing {split} set')):
            class_name = dataset.classes[label]
            class_dir = os.path.join(root_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            img_path = os.path.join(class_dir, f'{idx}.jpg')
            img.save(img_path)


export_cifar100_to_imagefolder('cifar100')


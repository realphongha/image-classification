import os
from torchvision.datasets import Imagenette
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def export_imagenette_to_imagefolder(root_dir='imagenette', size="320px"):
    transform = transforms.ToPILImage()

    train_set = Imagenette(root=root_dir, split="train", download=True, size=size)
    test_set = Imagenette(root=root_dir, split="val", download=True, size=size)
    classes = [cls[0] for cls in train_set.classes]
    print(classes)

    for split, dataset in [('train', train_set), ('val', test_set)]:
        for idx, (img, label) in enumerate(tqdm(dataset, desc=f'Processing {split} set')):
            class_name = dataset.classes[label][0]
            class_dir = os.path.join(root_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            img_path = os.path.join(class_dir, f'{idx}.jpg')
            img.save(img_path)


export_imagenette_to_imagefolder('imagenette', "320px")


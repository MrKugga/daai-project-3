import os, random, json
import torch.utils.data as torch_data
from torchvision.datasets import VisionDataset
from torch import from_numpy
from PIL import Image
import numpy as np

class Gta5(VisionDataset):
    
    def __init__(self, 
                 root: str, 
                 split: list,
                 transform=None, 
                 target_transform=None,
                 fda_style_transform=None
                ):
        
        self.root=root
        self.images_dir = os.path.join(self.root, "images")
        self.labels_dir = os.path.join(self.root, "labels")
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.fda_style_transform = fda_style_transform
        self.images = []
        self.labels = []
        
        self.return_unprocessed_image = False
        
        for image in self.split:
            label = image
            label_dir = os.path.join(self.labels_dir, label)
            img_dir = os.path.join(self.images_dir, image)

            self.images.append(img_dir)
            self.labels.append(label_dir)
        
        if target_transform is None:
            with open(os.path.join(root, "info.json")) as f:
                jdict = json.load(f)
            
            # Mapping from GTA5 to Cityscapes
            mapping = np.zeros((256,), dtype=np.int64)
            for i, cl in jdict["label2train"]:
                mapping[i] = cl
                
            self.target_transform = lambda x: from_numpy(mapping[x])
                        
            
            
        
    def __getitem__(self, idx):
        
        image = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(self.labels[idx])
        
        if self.return_unprocessed_image:
            return image
        
        if self.fda_style_transform is not None:
            image = self.fda_style_transform(image)

        if self.transform is not None:
            image, label = self.transform(image, label)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
 
        return image, label
    
    
    def __len__(self):
        length = len(self.images)
        return length
    
    def get_paths(self):
        return self.images, self.labels

            
        
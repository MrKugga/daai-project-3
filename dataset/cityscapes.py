import os, random
import torch.utils.data as torch_data
from torchvision.datasets import VisionDataset
from torch import from_numpy
from PIL import Image
import numpy as np

eval_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

'''
map_classes = {
    7: "road",  # 1
    8: "sidewalk",  # 2
    9: "parking",
    10: "rail truck",
    11: "building",  # 3
    12: "wall",  # 4
    13: "fence",  # 5
    14: "guard_rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",  # 6
    18: "pole_group",
    19: "light",  # 7
    20: "sign",  # 8
    21: "vegetation",  # 9
    22: "terrain",  # 10
    23: "sky",  # 11
    24: "person",  # 12
    25: "rider",  # 13
    26: "car",  # 14
    27: "truck",  # 15
    28: "bus",  # 16
    29: "caravan",
    30: "trailer",
    31: "train",  # 17
    32: "motorcycle",  # 18
    33: "bicycle"  # 19
}
'''

map_classes = {
    7: "road",  # 0
    8: "sidewalk",  # 1
    11: "building",  # 2
    12: "wall",  # 3
    13: "fence",  # 4
    17: "pole",  # 5
    19: "light",  # 6
    20: "sign",  # 7
    21: "vegetation",  # 8
    22: "terrain",  # 9
    23: "sky",  # 10
    24: "person",  # 11
    25: "rider",  # 12
    26: "car",  # 13
    27: "truck",  # 14
    28: "bus",  # 15
    31: "train",  # 16
    32: "motorcycle",  # 17
    33: "bicycle"  # 18
}

class Cityscapes(VisionDataset):
    
    def __init__(self, 
                 root: str, 
                 split: list,
                 transform=None, 
                 target_transform=None,
                 fda_style_transform=None,
                 cl19: bool=False
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
        self.map_classes = map_classes
        self.return_unprocessed_image = False
        
        for image in self.split:
            spl = image.split('_') 
            spl[-1] = "gtFine_labelIds.png"
            label = '_'.join(spl)

            label_dir = os.path.join(self.labels_dir, label)
            img_dir = os.path.join(self.images_dir, image)

            self.images.append(img_dir)
            self.labels.append(label_dir)
            
        '''
        Mapping the classes we want to evaluate. target_transform() will be passed when returning img and lable with __getitem__:
        Label will be a tensor with numbers from 0 to 18 for each pixel (indexes of list eval_classes) and 255 if class not in eval_classes.
        When calling CrossEntropyLoss, we will ignore index 255
        '''
        if cl19 and target_transform is None:
            classes = eval_classes
            mapping = np.zeros((256,), dtype=np.int64) + 255
            for i, cl in enumerate(classes):
                mapping[cl] = i
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
    
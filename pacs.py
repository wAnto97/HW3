import pandas as pd
from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


class PACS():
    def __init__(self,root,transform,data='photo'):
        
        self.sourceData = datasets.ImageFolder(root = root + '/' + data,transform=transform)
        self.train_indexes,self.val_indexes = train_test_split([i for i,(img,target) in enumerate(self.sourceData)],test_size=0.2,\
             stratify = self.sourceData.targets,random_state=41)

    def get_train_val_subset(self):
        return Subset(self.sourceData,self.train_indexes),Subset(self.sourceData,self.val_indexes)
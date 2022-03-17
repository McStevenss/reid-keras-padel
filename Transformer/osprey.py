import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


# device = initilize_device("cpu")
# We had to recreate the get_id() func since they assume the pictures are named in a specific manner. 
def get_id_padel(img_path):

  labels = []
  for path, v in img_path:
      filename = os.path.basename(path)

      label = filename.split('_')[0]
      labels.append(int(label))
  return labels
  
def extract_feature(model,dataloaders):
    
    features =  torch.FloatTensor()
    count = 0
    idx = 0
    for data in tqdm(dataloaders):
        img, label = data
        img, label = img.to(device), label.to(device)

        output = model(img)

        n, c, h, w = img.size()
        
        count += n
        features = torch.cat((features, output.detach().cpu()), 0)
        idx += 1
    return features

def image_loader(data_dir_path):    
    image_datasets = {}
    data_dir = "data/The_OspreyChallengerSet"
    data_dir = data_dir_path

    image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'),
                                            data_transforms['query'])
    image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'),
                                            data_transforms['gallery'])
    query_loader = DataLoader(dataset = image_datasets['query'], batch_size=batch_size, shuffle=False)
    gallery_loader = DataLoader(dataset = image_datasets['gallery'], batch_size=batch_size, shuffle=False)

    return query_loader, gallery_loader, image_datasets

def feature_extraction(model, query_loader):
    # Extract Query Features
    query_feature = extract_feature(model, query_loader)

    # Extract Gallery Features
    gallery_feature = extract_feature(model, gallery_loader)

    return query_feature, gallery_feature

def get_labels(image_datasets):
    #Retrieve labels
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs
    gallery_label = get_id_padel(gallery_path)
    query_label = get_id_padel(query_path)

    return gallery_label, query_label

def calc_gelt_feature(query_feature):
    concatenated_query_vectors = []
    for query in tqdm(query_feature):  
        fnorm = torch.norm(query, p=2, dim=1, keepdim=True)*np.sqrt(14) 
        query_norm = query.div(fnorm.expand_as(query))        
        concatenated_query_vectors.append(query_norm.view((-1))) # 14*768 -> 10752
    return concatenated_query_vectors

def calc_gelt_gallery(gallery_feature):
    concatenated_gallery_vectors = []
    for gallery in tqdm(gallery_feature):  
        fnorm = torch.norm(gallery, p=2, dim=1, keepdim=True) *np.sqrt(14)  
        gallery_norm = gallery.div(fnorm.expand_as(gallery))      
        concatenated_gallery_vectors.append(gallery_norm.view((-1))) # 14*768 -> 10752  
    return concatenated_gallery_vectors

def calc_faiss(concatenated_gallery_vectors, gallery_label):
    index = faiss.IndexIDMap(faiss.IndexFlatIP(10752))
    index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]), np.array(gallery_label).astype('int64'))  # original 
    return index

def search(query: str, k=1):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    return top_k

class LATransformerTest(nn.Module):
    def __init__(self, model, lmbd ):
        super(LATransformerTest, self).__init__()
        
        self.class_num = 751
        self.part = 14 # We cut the pool5 to sqrt(N) parts
        self.num_blocks = 12
        self.model = model
        self.model.head.requires_grad_ = False 
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,768))
        self.dropout = nn.Dropout(p=0.5)
        self.lmbd = lmbd
#         for i in range(self.part):
#             name = 'classifier'+str(i)
#             setattr(self, name, ClassBlock(768, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        

    def forward(self,x):
        
        # Divide input image into patch embeddings and add position embeddings
        x = self.model.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.pos_embed)
        
        # Feed forward through transformer blocks
        for i in range(self.num_blocks):
            x = self.model.blocks[i](x)
        x = self.model.norm(x)
        
        # extract the cls token
        cls_token_out = x[:, 0].unsqueeze(1)
        
        # Average pool
        x = self.avgpool(x[:, 1:])
        
        # Add global cls token to each local token 
#         for i in range(self.part):
#             out = torch.mul(x[:, i, :], self.lmbd)
#             x[:,i,:] = torch.div(torch.add(cls_token_out.squeeze(),out), 1+self.lmbd)

        return x.cpu()
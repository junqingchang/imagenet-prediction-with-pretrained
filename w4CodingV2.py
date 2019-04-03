
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform
import xml.etree.ElementTree as ET
import torchvision.models as models
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo


# In[2]:


# Tested on Python version 3.6.8 and PyTorch version 1.0.0
# Machine used: Macbook with no CUDA

SYNSET_WORDS = 'imagenet_first2500/synset_words.txt'
# ASSUMES IMAGES ARE JPEG FORMAT
IMAGE_DIRECTORY = 'imagenet_first2500/imagespart/'
XML_DIRECTORY = 'imagenet_first2500/DONOTEXPAND/'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# In[3]:


# Dataset
class ImageNetData(Dataset):
    def __init__(self, synset, image_directory, xml_directory, transform = None):
        f = open(synset,"r")
        self.label = {}
        ctr = 0
        self.index_to_class = {}
        self.name_to_index = {}
        for line in f:
            self.label[line[0:9]] = line[10:-1]
            self.index_to_class[ctr] = line[10:-1]
            self.name_to_index[line[0:9]] = ctr
            ctr += 1
        self.image_list = os.listdir(image_directory)
        self.image_path = image_directory
        self.xml_path = xml_directory
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx): 
        filename, file_extension = os.path.splitext(self.image_list[idx])
        item = self.from_xml(filename + '.xml')
        image = io.imread(self.image_path+ filename + '.JPEG')
        target = self.name_to_index[item]
        if(len(image.shape) == 2):
            image = np.stack((image,)*3, axis=-1)
        
        if self.transform:
            image = self.transform(image)
            
        return (image, target)
        
    def from_xml(self, reference):
        root = ET.parse(self.xml_path + reference).getroot()
        for obj in root.findall('object'):
            for name in obj.findall('name'):
                output_name = name.text 
        return output_name
    
    def nm_to_idx(self, nm):
        return self.name_to_index[nm]
    
    def idx_to_class(self, idx):
        return self.index_to_class[idx]


# In[4]:


def test(model, device, test_loader, log_freq = 100):
    model.eval()
    correct = 0
    ctr = 0
    is_multi = False
    num_crop = 1
    total_num_data = len(test_loader.dataset)
#     expanded_set = 1
    with torch.no_grad():
        for data, target in test_loader:
            if len(data.shape) > 4:
                num_crop = data.shape[1]
                is_multi = True
#                 target = target.view(-1, 1).repeat(1, data.shape[1]).view(1,-1)
#                 expanded_set = data.shape[1]
                data = data.view([-1,data.shape[-3],data.shape[-2],data.shape[-1]])  
            data, target = data.to(device), target.to(device)
            output = model(data)
            if is_multi:
                output = output.view(test_loader.batch_size, num_crop, -1)
                output = output.mean(1)
            pred = output.argmax(dim=1, keepdim=True)
        
            correct += pred.eq(target.view_as(pred)).sum().item()
            ctr += 1
            if ctr % log_freq == 0:
                print("Currently on Image {}".format(ctr*test_loader.num_workers))
                 
              # Break at 500 images  
#             if ctr*test_loader.batch_size >= 500:
#                 break
                
#     total_num_data *= expanded_set
    accuracy = 100. * correct / total_num_data
    

    print('\nPerformance: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, total_num_data, accuracy))
    return accuracy


# In[5]:


squeezenet = models.squeezenet1_1(pretrained = True)


# In[6]:


centercrop_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
transformed_dataset = ImageNetData(SYNSET_WORDS,IMAGE_DIRECTORY,XML_DIRECTORY, centercrop_transform)
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
print("Center Crop with no Normalize")
test(squeezenet, device, dataloader)


# In[7]:


centercropnorm_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
transformed_dataset = ImageNetData(SYNSET_WORDS,IMAGE_DIRECTORY,XML_DIRECTORY, centercropnorm_transform)
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
print("Center Crop with Normalize")
test(squeezenet, device, dataloader)


# In[8]:


fivecrop_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(280),
        transforms.FiveCrop(224), 
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(norm) for norm in norms]))
    ])
transformed_dataset = ImageNetData(SYNSET_WORDS,IMAGE_DIRECTORY,XML_DIRECTORY, fivecrop_transform)
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
print("Five Crop with Normalize")
test(squeezenet, device, dataloader)


# In[9]:


class Squeeze330(nn.Module):
    def __init__(self, num_classes=1000):
        super(Squeeze330,self).__init__()
        self.num_classes = num_classes
        self.newPool = nn.AdaptiveAvgPool2d((224,224))
        self.reference = models.squeezenet1_1(pretrained = True)
    def forward(self, x):
        x = self.newPool(x)
        x = self.reference.features(x)
        x = self.reference.classifier(x)
        return x.view(x.size(0), self.num_classes)


# In[10]:


squeeze330 = Squeeze330()


# In[11]:


task3_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(380),
        transforms.FiveCrop(330), 
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])(norm) for norm in norms]))
    ])
transformed_dataset = ImageNetData(SYNSET_WORDS,IMAGE_DIRECTORY,XML_DIRECTORY, task3_transform)
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
print("Input Size 330 with SqueezeNet")
test(squeeze330, device, dataloader)


# In[12]:


class Inception330(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(Inception330, self).__init__()
        self.newPool = nn.AdaptiveAvgPool2d((224,224))
        self.reference = models.inception_v3(pretrained=True)
    def forward(self, x):
        features = self.newPool(x)
        out = self.reference.forward(features)
        return out


# In[13]:


inception330 = Inception330()


# In[15]:


task3_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(380),
        transforms.FiveCrop(330), 
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])(norm) for norm in norms]))
    ])
transformed_dataset = ImageNetData(SYNSET_WORDS,IMAGE_DIRECTORY,XML_DIRECTORY, task3_transform)
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
print("Input Size 330 with Inception v3")
test(inception330, device, dataloader)


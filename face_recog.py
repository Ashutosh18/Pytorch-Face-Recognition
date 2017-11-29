import torch
import torchvision
import torchvision.datasets as dsets
from PIL import Image
import torchvision.transforms as transform
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.functional as F
import PIL.ImageOps

#function to show images
def im_show(img,text,save=False):
    npimg= img.numpy()
    plt.axis('off')
    if text:
        plt.text(75,8,text)

    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()


def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()



class Config():
    training_dir= "./orl_faces/Training/"
    testing_dir= "./orl_faces/Testing/"
    train_batch_size = 16
    train_number_epochs = 100

class SiameseNetworkDataset(Dataset):
    
    #any image dataset in pytorch must have __getitem__ and len method in them
    def __init__(self,ImageFolderDataset,transform=None,should_invert=True):
        self.imagefolderdataset= ImageFolderDataset
        self.transform = transform
        self.should_invert= should_invert

    def __getitem__(self,index):
        img0_tuple= random.choice(self.imagefolderdataset.imgs)
        #we want to get same amount of positive and negative image pairs
        should_get_same_class = random.randint(0,1)
        #loop to get the same tuple from images
        if should_get_same_class==True:
            while True:
                img1_tuple= random.choice(self.imagefolderdataset.imgs)
                if(img0_tuple==img1_tuple):
                    break

        else:
            img1_tuple= random.choice(self.imagefolderdataset.imgs)
            
        #load images
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        #convert to black and white
        img0 = img0.convert("L")
        img1 = img1.convert("L")


        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imagefolderdataset.imgs)

#folder where images are present
folder_dataset = dsets.ImageFolder(root= Config.training_dir)
#load_dataset
siamese_dataset = SiameseNetworkDataset(folder_dataset, transform=transform.Compose([transform.Scale((100,100)),transform.ToTensor()]),should_invert=False)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn= nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,4,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8)
        )

        self.fc1= nn.Sequential(
            nn.Linear(100*100*8,500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(500,250),
            nn.ReLU(inplace= True),
            nn.Dropout(0.5),

            nn.Linear(500,5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss,self).__init__()
        self.margin= margin

    def forward(self, output1, output2,label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=1,
                                  batch_size=Config.train_batch_size)
train_dataloader

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)

# counter for ------
counter = []
loss_history = []
iteration_number= 0

for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1, label= data
        img0, img1, label= Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
        output1, output2 = net(img0, img1)
        optimizer.zero_grad()
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.data[0]))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data[0])
    show_plot(counter, loss_history)









        

        
    
    
    

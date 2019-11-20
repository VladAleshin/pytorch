from torchvision import transforms, datasets
from torch import nn, optim
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import random
import datetime
from PIL import ImageFile
from discriminator import Discriminator
from generator import Generator

ImageFile.LOAD_TRUNCATED_IMAGES = True
RESUME_TRAINING = False
seed = 100
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dataroot = '/content/data'
filepath = '/content/drive/My Drive/model_weights/model_weights_new.pth' # path to file with weights
num_seconds = 28800  ## training duration
beta1 = 0.5
device = torch.device("cuda:0" 
                      if torch.cuda.is_available() else "cpu")

batch_size = 128
dataset = datasets.ImageFolder(root=dataroot,
               transform=transforms.Compose([               
               transforms.Resize((256, 256)),                   
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=1)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0) 
  
def cross_entropy_uniform(pred, data_len, n_classes):
    logsoftmax = nn.LogSoftmax(dim=1)
    unif = torch.full((data_len, n_classes), 1 / n_classes).to(device)
    return torch.mean(-torch.sum(unif * logsoftmax(pred), 1)).to(device)   


criterion = nn.BCELoss()
cross_entropy_loss = nn.CrossEntropyLoss()

n_classes = 10
discriminator = Discriminator(n_classes).to(device)
generator = Generator().to(device)

optimizerD = optim.Adam(discriminator.parameters(), lr=0.0001,
                            betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=0.0001,
                            betas=(beta1, 0.999))

if RESUME_TRAINING:    
    state = torch.load(filepath)
    start_epoch = state['epoch'] + 1
    discriminator.load_state_dict(state['state_dict_discriminator'])
    generator.load_state_dict(state['state_dict_generator'])
    optimizerD.load_state_dict(state['optimizerD'])
    optimizerG.load_state_dict(state['optimizerG'])    
else:
    start_epoch = 0
    generator.apply(weights_init)
    discriminator.apply(weights_init)                                          

#training
start_time = datetime.datetime.now()

for epoch in range(start_epoch, 225, 1):
    for i, (data, y) in enumerate(dataloader):
      
        discriminator.zero_grad()
        data = data.to(device)              
        y = y.to(device)        
        data_len = len(data)        
        
        label = torch.full((data_len, ), 1, device=device)      
        output, real_classes = discriminator(data)
        output = output.view(-1)
        error_real_D = criterion(output, label) \
                          + cross_entropy_loss(real_classes, y)
        
        error_real_D.backward()      

        noise = torch.randn(data_len, generator.z_size, device=device)
        fake = generator(noise)
        label = torch.full((data_len, ), 0.0, device=device)
        output, fake_classes = discriminator(fake.detach())
        output = output.view(-1)        
        error_fake_D = criterion(output, label)      
        error_fake_D.backward(retain_graph=True)
        error_D = error_fake_D + error_real_D            
        optimizerD.step()            
        
        
        generator.zero_grad()
        label = torch.full((data_len, ), 1, device=device)      
        output, _ = discriminator(fake)        
        
        fake_error = cross_entropy_uniform(fake_classes, 
                                            data_len, fake_classes.size(1)) 
        
        error_G = criterion(output, label)
        error_G = error_G + fake_error 
        error_G.backward()      
        optimizerG.step()  

    state = {    
        'epoch': epoch,
        'state_dict_discriminator': discriminator.state_dict(),
        'state_dict_generator': generator.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'optimizerG': optimizerG.state_dict(),    
    }
    torch.save(state, filepath)   
      
    if epoch % 5 == 0:
        print('Error D = {}, error G = {}, epoch = {}'.format(error_D, error_G,
                                    epoch))
        noise = torch.randn(5, 100, device=device)
        fake = generator(noise).detach().cpu()        
        plt.figure(figsize=(20, 20))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(make_grid(fake, 
                    padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()
        
    end_time = datetime.datetime.now()
    if (end_time - start_time).seconds > num_seconds:
        print('Time is over. Epoch - {}'.format(epoch))
        break      
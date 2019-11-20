import torch.nn as nn

class Discriminator(nn.Module):  
  
    def __init__(self, n_classes=2):
      
        super(Discriminator, self).__init__()        
        self.n_classes = n_classes
        self.main = nn.Sequential(
        
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1,
                     bias=False),            
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1,
                     bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1,
                     bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1,
                     bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1,
                     bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1,
                     bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)          
            
        )
        
        self.lin = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()       
        )    
        
        self.classify = nn.Sequential(
        
            nn.Linear(512 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, self.n_classes),                              
        )        
    
    def forward(self, x):      
        x = self.main(x) 
        x = x.view(-1, 512 * 4 * 4)       
        probabilities =  self.lin(x)          
        class_out = self.classify(x)    
        
        return probabilities, class_out
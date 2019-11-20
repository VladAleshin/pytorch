import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self, z_size=100):
        
        super(Generator, self).__init__()
        self.z_size = z_size
        self.linear = nn.Linear(self.z_size, 1024 * 4 * 4)
        self.main = nn.Sequential(            
            nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1,
                              bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1,
                               bias=False),           
            nn.Tanh()            
        )
        
    def forward(self, z):
        out = self.linear(z)
        out = out.view(-1, 1024, 4, 4)
        return self.main(out)       
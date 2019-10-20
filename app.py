import os
import uuid

import torch
from flask import Flask, redirect, render_template
from flask_wtf import FlaskForm
from torch import nn
from torchvision.utils import save_image
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class MyForm(FlaskForm):
    generate_images = SubmitField('Generate image...',)    

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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config.update(dict(    
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))
device = torch.device("cuda:0" 
                      if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
state = torch.load(os.path.join(app.config['UPLOAD_FOLDER'], 'model_weights.pth'), map_location='cpu')
model.load_state_dict(state['state_dict_generator'])

@app.route("/",  methods=('GET', 'POST'))
def home():
    form = MyForm()
    if form.validate_on_submit():
        if form.generate_images.data:
            noise = torch.randn(5, 100, device=device)
            fake = model(noise).detach().cpu() 
            path_to_folder = 'static/'       
            image1, path_to_file1 = fake[0], path_to_folder + str(uuid.uuid4()) + '.jpg'  
            image2, path_to_file2 = fake[1], path_to_folder + str(uuid.uuid4()) + '.jpg'
            image3, path_to_file3 = fake[2], path_to_folder + str(uuid.uuid4()) + '.jpg'                       
            save_image(image1, path_to_file1)
            save_image(image2, path_to_file2)
            save_image(image3, path_to_file3)                        
            return render_template('submit.html', form=form, user_image1=path_to_file1, 
                                        user_image2=path_to_file2, user_image3=path_to_file3) 
    
    return render_template('submit.html', form=form, user_image1='', 
                                        user_image2='', user_image3='')    

if __name__ == "__main__":
    app.run(debug=False)  

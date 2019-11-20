import os
import uuid

import torch
from flask import Flask, render_template
from flask_wtf import FlaskForm
from torch import nn
from torchvision.utils import save_image
from wtforms import SubmitField
from generator import Generator

class MyForm(FlaskForm):
    generate_images = SubmitField('Generate images...')   

app = Flask(__name__)
app.config.update(dict(    
    SECRET_KEY=uuid.uuid4().hex,
    WTF_CSRF_SECRET_KEY=uuid.uuid4().hex
))

device = torch.device("cuda:0" 
                      if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
state = torch.load('static/generator_weights.pth', map_location='cpu')
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
                                        user_image2=path_to_file2, user_image3=path_to_file3,
                                        is_visible='visible')     
    return render_template('submit.html', form=form, user_image1='', 
                                        user_image2='', user_image3='', is_visible='not-visible')    


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)  

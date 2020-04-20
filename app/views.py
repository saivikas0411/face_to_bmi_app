from app import app
from flask import render_template
from flask import request
import numpy as np
from flask import  jsonify
import pickle
import os
import pandas as pd
import tensorflow

import scipy
import flask
import keras
import sklearn
import PIL
from flask import Flask
from werkzeug.utils import secure_filename
from PIL import Image
from keras.applications.resnet50 import ResNet50 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
app_root = os.path.dirname(os.path.abspath(__file__))
svrmodel = pickle.load(open('svrmodel.pkl', 'rb'))
print('flask',flask.__version__)
print('keras',keras.__version__)
print('tensroflow',tensorflow.__version__)
print('pandas',pd.__version__)
print('numpy',np.__version__)
print('pil',PIL.__version__)
print('scipy',scipy.__version__)
print('verison printed')
width=224
height=224  
global model
global graph


#importing resnet50
base_model= ResNet50(weights='imagenet')
model=Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
graph = tensorflow.get_default_graph()

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict',methods=["GET","POST"])
def predict():
    target = os.path.join(app_root, 'uploads')
    if not os.path.isdir(target):
        os.makedirs(target)
    if request.method == 'POST':
        print('entered pramod post')
        imgfile=request.files['file']
        gen=request.form["gen"]
        file_name = imgfile.filename or ''
        destination = '/'.join([target, file_name])
        imgfile.save(destination)
        feed = Image.open(target+'/'+file_name)
        out=feed.resize((width,height),Image.ANTIALIAS)
        x=np.array(out)
       #inorder to create batch of images we need an additional dimension(size1,size2,channels) to (samples, size1,size2,channels)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)#its a mandatory statement to check if input satisfied dimensions needed
        print(x.shape)
        with graph.as_default():
            features=model.predict(x)
        print(features.shape)
        fea_gen=np.array([gen])
        fea_gen=fea_gen.reshape(1,1)
        features=np.append(fea_gen,features,axis=1)
        output = svrmodel.predict(features)
        print(output)
        if os.path.exists(target+'/'+file_name):
            os.remove(target+'/'+file_name)
        else:
            print("The file does not exist")
    return render_template('index.html', predicted_bmi='your BMI is  $ {}'.format(output))

app.run(debug=True)




	




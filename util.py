import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
saved_model = load_model("VGG_model.h5")


def check(input_img):
   

    img = image.load_img(input_img , target_size = (224 , 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = saved_model.predict(img)    
    return output
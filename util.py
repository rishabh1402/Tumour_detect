import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

saved_model = load_model("model_f.h5")

# def names(number):
#     if number==0:
#         return 'Its a Tumor'
#     else:
#         return 'No, Its not a tumor'

def check(input_img):
    img = image.load_img(input_img , target_size = (128, 128))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = saved_model.predict(img)
    return output

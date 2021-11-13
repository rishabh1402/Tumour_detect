import os
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request 
from tensorflow.keras.preprocessing.image import load_img 
from util import check

app = Flask(__name__,template_folder='templates')
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
model = load_model(os.path.join(BASE_PATH , 'model_f.h5'))

def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'

def predict(filename , model):

    # res = check(filename)
    img = image.load_img(filename , target_size = (128, 128))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    res = model.predict(img)
    classification = np.where(res == np.amax(res))[1][0]
    txt = str(res[0][classification]*100) + ' % ' + names(classification)
    return txt

@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    if request.method == 'POST':
            upload_file = request.files['file']
            filename = upload_file.filename
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            text = predict(path_save, model)
            # text = "helo"

            return render_template('success.html',img=filename,text_h=text)
    return render_template('index.html',upload=False)

if __name__ == "__main__":
    app.run(debug = True)
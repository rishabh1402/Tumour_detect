import os
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request 
from tensorflow.keras.preprocessing.image import load_img 
app = Flask(__name__,template_folder='templates')
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
model = load_model(os.path.join(BASE_PATH , 'D:/Ingenium/model.hdf5'))


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'


def predict(filename , model):
    img = load_img(filename , target_size = (224 , 224))
    res = model.predict(img)

    classification = np.where(res == np.amax(res))[1][0]
    return str(res[0][classification]*100) + '% Confidence This Is ' + names(classification)




@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
            upload_file = request.files['file']
            if upload_file and allowed_file(upload_file.filename):                
                filename = upload_file.filename
                path_save = os.path.join(UPLOAD_PATH,filename)
                upload_file.save(path_save)

                text = predict(path_save , model)

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = filename , predictions = text)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)

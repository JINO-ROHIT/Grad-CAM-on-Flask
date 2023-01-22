from logging import debug
from flask import Flask, render_template, request
import os 
DISPLAY_FOLDER = os.path.join('static', 'model_photos')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = DISPLAY_FOLDER

from commons import get_tensor
from inference import get_heatmap

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

@app.route('/', methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        print(request.files)
        # if 'image' not in request.files:
        #     print('file not uploaded')
        #     return
        file = request.files['image']
        image = file.read()
        category = get_heatmap(image_bytes = image)

        # tensor = get_tensor(image_bytes = image)
        # print(get_tensor(image_bytes = image))
        original = os.path.join(app.config['UPLOAD_FOLDER'], 'original.jpg')
        predicted = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted.jpg')

        return render_template('result.html', label = category, original = original, predicted = predicted)

if __name__ == '__main__':
    app.run(debug = True)
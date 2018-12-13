from ObjectDetector import Detector
import io
import cv2
import numpy as np
import os

from flask import Flask, render_template, request

from flask import send_file

app = Flask(__name__)

detector = Detector()

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def upload():
	if request.method == 'POST':
		#file = Image.open(request.files['file'])
		#read image file string data
		filestr = request.files['file'].read()
		#convert string data to numpy array
		npimg = np.fromstring(filestr, np.uint8)
		# convert numpy array to image
		file = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
		
		#f = os.path.join(app.config['UPLOAD_FOLDER'], 'testfile')
		#file.save(f)
		#image = image.load_img(file, target_size=(200, 200))
		img = detector.detectObject(file)
		#file = cv2.imencode('.jpg', file)[1].tobytes()
		return send_file(io.BytesIO(img),attachment_filename='image.jpg',mimetype='image/jpg')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)

import cv2
import numpy as np
import pickle
import tensorflow as tf
import json
from flask import Flask, render_template, Response

app = Flask(__name__, template_folder="template")


# how to call in route main
@app.route('/')
def anyname():
    """Video streaming home page."""
    return render_template("index.html")

z = 0
# file video
dir_vid = 'video/parking_1.mp4'
# this is koordinat to draw/detected object
loc_spot1 = 'spot_space/spot1.pickle'

# this to load data from format pickle,
with(open(loc_spot1, 'rb')) as loc:
    x = pickle.load(loc)

# this use to accommodate available spots
spot_available = np.zeros(len(x))

boxes = []
for box in x:
    a, b, c, d = box
    boxes += [[int(a / 2), int(b / 2), int(c / 2), int(d / 2)]]

# this to read/load result for training, process training in the file model train.py
train_model = 'train1.h5'
model = tf.keras.models.load_model(train_model)


# this use predict data based on image parking and model
def prediction(input_model):
    generate_id = []

    for i in input_model:
        predict_model = model.predict(i)
        generate_id += [np.argmax(predict_model[0])]

    return generate_id


# use to convert data become json
def convert(a, b):
    zipped = zip(a, b)
    op = dict(zipped)
    return op


# this function to load video and predict video used model training
def park1():
    """Video streaming generator function."""
    global z, spot_available, db

    status = None
    cap = cv2.VideoCapture(dir_vid)
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, img = cap.read()
        if not ret:
            # if the data/video does not recover, the image will always be repeated because this prototype used video
            # not real camera cctv
            cap = cv2.VideoCapture(dir_vid)
            continue
        if ret:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            z += 1
            # this use to skip frame (10 frame) to predict so that the video runs more normally
            if z % 10 == 0:
                status = True

            if status:
                img_crop = []
                for i, box in enumerate(x):
                    proses1 = img[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]]
                    proses2 = cv2.resize(proses1, (50, 50)) / 255.0
                    img_crop += [np.expand_dims(proses2, axis=0)]

                spot_available = prediction(img_crop)
                status = False

            spot = (len(spot_available) - sum(spot_available))
            # to convert for data spot_available become json
            json_objs = []
            str_s = ['id', 'value']
            for i in range(len(spot_available)):
                conv = convert(str_s, [i + 1, bool(spot_available[i])])
                json_obj = json.dumps(conv)
                json_objs += [json_obj]

            # db.child('parking').child('parking_1').update({'spot_available': int(spot), 'layout': json_objs})
            print('spot avail', spot_available)

            for i in range(len(spot_available)):
                if spot_available[i] == 0:
                    # use to create rectangle if
                    cv2.rectangle(img, (boxes[i][0], boxes[i][1]),
                                  (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (0, 255, 0), 3)
                cv2.putText(img, str(i + 1), (boxes[i][0] + 15, boxes[i][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 3)
            # this add spot available in the video based on spot available that showed
            cv2.putText(img, 'available spot: {}'.format(len(spot_available) - sum(spot_available)), (50, 510),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_parking1')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(park1(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
    app.run(debug=True)
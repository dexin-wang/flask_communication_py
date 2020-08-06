# -*- coding: UTF-8 -*-

from flask import request, Flask
import os
import cv2
from traffic.TrafficNet import TrafficNet

app = Flask(__name__)

def trafficPredict(img_path):
    pred_ret = trafficNet.predict(img_path)
    return pred_ret

@app.route("/", methods=['POST'])
def get_frame():
    upload_file = request.files['file']
    file_name = upload_file.filename
    file_path = os.path.join('/home/wangdx/research/mir_robot/server/getImgs', file_name)

    if upload_file:
        upload_file.save(file_path)
        result = trafficPredict(file_path)

        toClient = str({'cls': int(result)})
        print("success")
        return toClient
    else:
        return 'failed'


if __name__ == "__main__":
    trafficNet = TrafficNet()
    app.run("0.0.0.0", port=1212)
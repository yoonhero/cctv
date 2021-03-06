import cv2
from flask import Flask, render_template, render_template_string, Response
import json
import numpy as np
import requests
import os
from dotenv import load_dotenv


load_dotenv(verbose=True)

KAKAO_TOKEN = os.getenv("KAKAO_TOKEN")
url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

# 사용자 토큰
headers = {
    "Authorization": "Bearer " + KAKAO_TOKEN
}

data = {
    "template_object": json.dumps({"object_type": "text",
                                   "text": "사람이 감지 되었습니다.!",
                                   "link": {
                                        "web_url": "",
                                        "mobile_web_url": ""
                                   },
                                   "button_title": "cctv확인"

})
}

app = Flask(__name__)
video_capture = cv2.VideoCapture(1)
cascade_filename = 'haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)



def gen():
    while True:
        ret, image = video_capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for box in results:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
            cv2.putText(image, "detected", (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
            response = requests.post(url, headers=headers, data=data)
            print(response.status_code)

            if response.json().get('result_code') == 0:
                print('메시지를 성공적으로 보냈습니다.')
            else:
                print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))

        cv2.imwrite('../t.jpg', image)
        yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + open('../t.jpg', 'rb').read() + b'\r\n')
    video_capture.release()


@app.route('/')
def index():
    """Video streaming"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000)
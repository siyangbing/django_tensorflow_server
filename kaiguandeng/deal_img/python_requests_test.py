# -*-coding:utf-8-*-
import requests
import time
import cv2
import base64

img = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/kaiguandeng.jpg")
aa=base64.b64encode(cv2.imencode('.jpg',img)[1]).decode()


data = {
    'result': "11",
    'work_id': "work_id",
    'title1_id': "title1_id",
    'step_id': "step_id",
    'ip': "ip"
}

data_64 = {
    'username': "db",
    'filename': "img.png",
    'image': aa,
}
url_kaiguandeng = 'http://192.168.3.174:8000/kaiguandeng/t'
url_shiziluoding = 'http://192.168.3.174:8000/shiziluoding/szld'
url= 'http://192.168.3.174:8000/kaiguandeng/base64/'
index = 0

# base64_text = aa
# print(r.content.decode("utf-8"))

while True:
    r = requests.post(url, data=data_64)
    # print(r.content)
    print(r.content.decode("utf-8"))
    # requests.get(url_kaiguandeng,params=data)
    # time.sleep(1)
    index =index+1
    # requests.get(url_shiziluoding, params=data)
    print(index)

    break
    # time.sleep(1)



# -*-coding:utf-8-*-

import os
import time

import requests
import cv2
import base64
from django_tensorflow_server.settings import BASE_DIR

# img_kaiguandeng = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/kaiguandeng.jpg")
# img_shiziluoding = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/shiziluoding.jpg")
# img_yiziluoding = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/yiziluoding.jpg")
# img_kougai = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/kougai.jpg")

img_fangfeizuocang = cv2.imread(os.path.join(BASE_DIR, "test_img/zuocangtongyong.jpg"))
# img_fangfeizuocang = cv2.imread(os.path.join(BASE_DIR, "test_img/fangfeizuocang2.jpg"))
# img_fangfeizuocang = cv2.imread(os.path.join(BASE_DIR, "test_img/fangfeizuocang3.jpg"))
# img_fangfeizuocang = cv2.imread(os.path.join(BASE_DIR, "test_img/fangfeizuocang4.jpg"))
# img_tongdianzuocang = cv2.imread(os.path.join(BASE_DIR, "test_img/tongdianzuocang2.jpg"))
# img_tongdianzuocang = cv2.imread(os.path.join(BASE_DIR, "test_img/tongdianzuocang3.jpg"))




# data_kaiguandeng = base64.b64encode(cv2.imencode('.jpg', img_kaiguandeng)[1]).decode()
# data_shiziluoding = base64.b64encode(cv2.imencode('.jpg', img_shiziluoding)[1]).decode()
# data_yiziluoding = base64.b64encode(cv2.imencode('.jpg', img_yiziluoding)[1]).decode()
# data_kougai = base64.b64encode(cv2.imencode('.jpg', img_kougai)[1]).decode()

data_fangfeizuocang = base64.b64encode(cv2.imencode('.jpg', img_fangfeizuocang)[1]).decode()
# data_tongdianzuocang = base64.b64encode(cv2.imencode('.jpg', img_tongdianzuocang)[1]).decode()




# url_kaiguandeng= 'http://192.168.3.174:8080/kaiguandeng/kgd/'
# url_fangfeiduoge = 'http://192.168.3.174:8080/fangfeiduoge/ffdg/'
url_fangfeizuocang = 'http://192.168.9.63:8080/fangfeizuocang/ffzc/'
url_tongdianzuocang = 'http://192.168.9.63:8080/tongdianzuocang/tdzc/'
index = 0

# base64_text = aa
# print(r.content.decode("utf-8"))

while True:
    # r_kaiguandeng = requests.post(url_kaiguandeng, data={'image': data_kaiguandeng})
    # r_shiziluoding = requests.post(url_fangfeiduoge, data={'image': data_shiziluoding})
    # r_yiziluoding = requests.post(url_fangfeiduoge, data={'image': data_yiziluoding})
    # r_kougai = requests.post(url_fangfeiduoge, data={'image': data_kougai})

    r_fangfeizuocang = requests.post(url_fangfeizuocang, data={'image': data_fangfeizuocang})
    # r_tongdianzuocang = requests.post(url_tongdianzuocang, data={'image': data_tongdianzuocang})

    # print(r.content)
    print(r_fangfeizuocang.content.decode("utf-8"))
    # print(r_tongdianzuocang.content.decode("utf-8"))
    # requests.get(url_kaiguandeng,params=data)
    # time.sleep(1)
    index = index + 1
    # requests.post(url_shiziluoding, data=data_642)
    print(index)

    break
    # time.sleep(1)

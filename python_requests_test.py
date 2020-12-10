# -*-coding:utf-8-*-
import requests
import time
import cv2
import base64

img_kaiguandeng = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/kaiguandeng.jpg")
img_shiziluoding = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/shiziluoding.jpg")
img_yiziluoding = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/yiziluoding.jpg")
img_kougai = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/kougai.jpg")

data_kaiguandeng = base64.b64encode(cv2.imencode('.jpg', img_kaiguandeng)[1]).decode()
data_shiziluoding = base64.b64encode(cv2.imencode('.jpg', img_shiziluoding)[1]).decode()
data_yiziluoding = base64.b64encode(cv2.imencode('.jpg', img_yiziluoding)[1]).decode()
data_kougai = base64.b64encode(cv2.imencode('.jpg', img_kougai)[1]).decode()


url_kaiguandeng= 'http://192.168.3.174:8080/kaiguandeng/base64/'
url_fangfeiduoge = 'http://192.168.3.174:8080/fangfeiduoge/ffdg/'
index = 0

# base64_text = aa
# print(r.content.decode("utf-8"))

while True:
    r_kaiguandeng = requests.post(url_kaiguandeng, data={'image': data_kaiguandeng})
    r_shiziluoding = requests.post(url_fangfeiduoge, data={'image': data_shiziluoding})
    r_yiziluoding = requests.post(url_fangfeiduoge, data={'image': data_yiziluoding})
    r_kougai = requests.post(url_fangfeiduoge, data={'image': data_kougai})
    # print(r.content)
    # print(r.content.decode("utf-8"))
    # requests.get(url_kaiguandeng,params=data)
    # time.sleep(1)
    index = index + 1
    # requests.post(url_shiziluoding, data=data_642)
    print(index)

    break
    # time.sleep(1)

# -*-coding:utf-8-*-
import requests
import time
import cv2
import base64

img1 = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/shiziluoding.jpg")
# img2 = cv2.imread("/home/db/bing/django_tensorflow_server/test_img/shiziluoding.jpg")
aa1=base64.b64encode(cv2.imencode('.jpg',img1)[1]).decode()
# aa2=base64.b64encode(cv2.imencode('.jpg',img2)[1]).dcode()
b=""
data_641 = {
    'username': "db",
    'filename': "img.png",
    'image': aa1,
}

data_642 = {
    'username': "db",
    'filename': "img.png",
    'image': aa1,
}
# url_kaiguandeng = 'http://192.168.3.174:8000/kaiguandeng/t'
# url_shiziluoding = 'http://192.168.3.174:8000/shiziluoding/szld/'
# url= 'http://10.30.2.101:8080/kaiguandeng/base64/'
url= 'http://192.168.3.174:8080/fangfeiduoge/ffdg/'
index = 0

# base64_text = aa
# print(r.content.decode("utf-8"))

while True:
    r = requests.post(url, data=data_642)
    # print(r.content)
    print(r.content.decode("utf-8"))
    # requests.get(url_kaiguandeng,params=data)
    # time.sleep(1)
    index =index+1
    # requests.post(url_shiziluoding, data=data_642)
    print(index)

    break
    # time.sleep(1)



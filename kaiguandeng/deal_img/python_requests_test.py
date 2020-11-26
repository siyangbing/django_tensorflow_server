# -*-coding:utf-8-*-
import requests
import time

data = {
    'result': "11",
    'work_id': "work_id",
    'title1_id': "title1_id",
    'step_id': "step_id",
    'ip': "ip"
}
url_kaiguandeng = 'http://192.168.43.134:8000/kaiguandeng/t'
url_shiziluoding = 'http://192.168.43.134:8000/shiziluoding/szld'
index = 0
while True:
    requests.get(url_kaiguandeng,params=data)
    # time.sleep(1)
    index =index+1
    requests.get(url_shiziluoding, params=data)
    print(index)
    # time.sleep(1)



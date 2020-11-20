import time
import os
import json
import requests
from json import dumps

from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import cv2
import numpy as np
from  django.http import JsonResponse

from kaiguandeng.deal_img.location_map import Map_location

import kaiguandeng.imagenet

ftp_dir = "/home/db/myftp/tensorflow"
model_img_input_size = (640, 480)
saved_model_dir = '/home/db/bing/django_test/kaiguandeng/pb_model/saved_model'
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

label_dict = {'1.0': 'ku', '2.0': 'kd', '3.0': 'km', '4.0': 'u', '5.0': 'd', '6.0': 'l', '7.0': 'r', '8.0': 'm',
              '9.0': 'k', '10.0': 'g'}
# class dict
join_label_dict = {"yskg": ["1.0", "2.0", "3.0"], "hskg": ["4.0", "5.0", "6.0", "7.0", "8.0"], "deng": ["9.0", "10.0"]}

deng_location = [12, 8, 8]
yskg_location = [8, 1]
hskg = [1]

treshold = 0.4


def echoRuntime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        endTime = time.time()
        msecs = (endTime - startTime)
        print(func.__name__ + " running time is %.2f s" % msecs)
        return result

    return wrapper


@echoRuntime
def terminal(request):
    print("进入terminal页面")
    # 获取传入数据，为图片相对路径，需要加上ftp的路径
    imgPath = request.GET.get('imgPath')
    print(imgPath)
    work_id = request.GET.get('work_id')
    print(work_id)
    title1_id = request.GET.get('title1_id')
    print(title1_id)
    step_id = request.GET.get('step_id')
    print(step_id)
    ip = request.GET.get('ip')
    print(ip)

    if kaiguandeng.imagenet.flag:
        t0 = time.time()
        # flag=True,执行并返回结果
        print("执行处理")
        # 处理
        kaiguandeng.imagenet.flag = False
        img_dir = os.path.join(ftp_dir, imgPath)
        print("img_path:-------{}".format(imgPath))
        if not os.path.exists(img_dir):
            kaiguandeng.imagenet.flag = True
            return HttpResponse("找不到图片")


        img = cv2.imread(img_dir)  # 读取图片
        img = cv2.resize(img, model_img_input_size)  # 缩放到480*480
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.array(img, dtype=float)  # 改变数据类型为float
        img_array = img_array[np.newaxis, :, :, :]  # 增加一个维度

        input_data = np.array(img_array, dtype=np.float32)

        input = sess.graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        detection_score = sess.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
        num_detections = sess.graph.get_tensor_by_name('num_detections:0')
        # input_data2 = np.vstack((input_data, input_data, input_data, input_data, input_data, input_data, input_data,
        #                      input_data, input_data, input_data, input_data, input_data))
        feed_dict = {input: input_data, }
        result_list = []
        y = sess.run([detection_boxes, detection_score, detection_classes, num_detections], feed_dict=feed_dict)
        try:
            map_location = Map_location(y, treshold, label_dict, join_label_dict)
            # map_location.map_label()
            result_list = map_location.get_location()
            # print(str(result_list))
        except:
            print("detected failed")
            result_list = "g detected failed!"
        print(result_list)
        # url = "http://192.168.3.188:8080/WLZ/back/"
        data = {
            'result': str(result_list),
            'work_id': work_id,
            'title1_id': title1_id,
            'step_id': step_id,
            'ip': ip
        }
        cv2.imshow('resutlt_123', img)
        cv2.waitKey(100)

        # try:
        #     for a in result_list:
        #         for b in a:
        #             for c in b:
        #                 point_1 = (int(c[3] * model_img_input_size[0]), int(c[2] * model_img_input_size[1]))
        #                 point_2 = (int(c[5] * model_img_input_size[0]), int(c[4] * model_img_input_size[1]))
        #                 # print(point_1,point_2)
        #                 cv2.rectangle(img, point_1, point_2, (0, 255, 0), 1)
        #                 str_txt = str(c[0])
        #                 cv2.putText(img, str_txt, point_1, cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 255), 1)
        #     cv2.imshow('resutlt_123', img)
        #     cv2.waitey(100)
        #     print("opopopopopopopopopopop")
        # except:
        #     print("draw failed")
        #     pass


        # for box in result_list:
        #     point_1 = (int(box[1] * 480), int(box[0] * 480))
        #     point_2 = (int(box[3] * 480), int(box[2] * 480))
        #     # print(point_1,point_2)
        #     cv2.rectangle(img, point_1, point_2, (0, 255, 0), 1)
        #     str_txt = str(result_list[5])
        #     cv2.putText(img, str_txt, point_1, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
        # cv2.imshow('resutlt',img_show)
        # cv2.waitKey(10)

        # response = requests.post(url=url, data=dumps(result_list))
        print("time-----------{}".format(time.time() - t0))
        # return JsonResponse(response.text, safe=False)
        # print("ssssttttttrrr---------{}".format(str(data)))

        kaiguandeng.imagenet.flag = True
        # cv2.imshow('resutlt_123', img_show)
        # cv2.waitKey(50)
        return HttpResponse(str(data))
        # kaiguandeng.imagenet.flag = True
        # return HttpResponse([1,2,3])
        # return HttpResponse(result)
    else:
        # flag=False,未处理完成
        print("处理未完成")
        return HttpResponse("处理未完成")



@echoRuntime
def backstage(request):
    result_list = [12,2,2,2,2,2,2,2,2,2]
    return HttpResponse(result_list)

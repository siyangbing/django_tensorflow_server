import time
import os
import json
import requests
from json import dumps

from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import cv2
import traceback
import numpy as np
from django.http import JsonResponse
import base64

from kaiguandeng.deal_img.location_map import Map_location, model_img_input_size, label_dict, config,saved_model_dir,\
    join_label_dict, treshold

import kaiguandeng.imagenet

ftp_dir = "/home/db/myftp/tensorflow"

# sess_kaiguandeng = tf.Session(config=config)
# meta_graph_def = tf.saved_model.loader.load(sess_kaiguandeng, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

def echoRuntime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        endTime = time.time()
        msecs = (endTime - startTime)
        print(func.__name__ + " running time is %.2f s" % msecs)
        return result

    return wrapper

# @echoRuntime
def base64_test(request):
    if (request.method == 'POST'):
        t0 = time.time()

        img_data = request.POST.get('image')  # 本质就是解码字符串
        # print(test_image)
        img_byte = base64.b64decode(img_data)
        img_np_arr = np.fromstring(img_byte, np.uint8)
        image = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        # cv2.imwrite("./pppp.jpg",image)
        code = 200
        try:
            map_location = Map_location(treshold, label_dict, join_label_dict, model_img_input_size)
            img_data_list = map_location.read_img(image)
            y_list = map_location.eval_img_list(img_data_list)
            result_list = map_location.get_location(y_list)
        except:
            code = 200
            result_list = []
        data = {
            'code': code,
            'result': result_list,
        }
        print("处理一张图片需要{}秒".format(time.time()-t0))
    return JsonResponse(data)
    # return HttpResponse("success!!!")


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
    # base64 = request.GET.get('base64')
    # print(base64)

    if kaiguandeng.imagenet.flag:
        t0 = time.time()
        # flag=True,执行并返回结果
        print("执行处理")
        # 处理
        kaiguandeng.imagenet.flag = False
        # img_dir = os.path.join(ftp_dir, imgPath)
        img_dir = "/home/db/bing/django_tensorflow_server/test_img/kaiguandeng.jpg"
        print("img_path:-------{}".format(imgPath))
        if not os.path.exists(img_dir):
            kaiguandeng.imagenet.flag = True
            return HttpResponse("找不到图片")
        try:
            map_location = Map_location(treshold, label_dict, join_label_dict, model_img_input_size)
        except:
            result_list = " map_location failed!"
            traceback.print_exc()
            print("map_location error!")
        try:
            img_data_list = map_location.read_img(img_dir)
        except:
            result_list = " img_data_list failed!"
            traceback.print_exc()
            print("img_data_list error!")
        # print("img_data_list------------------------{}".format(str(img_data_list)))
        try:
            y_list = map_location.eval_img_list(img_data_list)
            # print("ylist_______________{}".format(y_list))
        except:
            result_list = " y_list failed!"
            traceback.print_exc()
            print("y_list error!")

        try:
            result_list = map_location.get_location(y_list)
            print("result_list________{}".format(result_list))
            print("++++++++++++++++++++++++++++++++++++++++++++")
        except:
            result_list = " result_list failed!"
            traceback.print_exc()
            print("result_list error!")

        # print("y_list------------------------{}".format(str(y_list)))
        # try:
        #     result_list = map_location.get_location(y_list)
        # except:
        #     result_list = " result_list failed!"
        #     print("result_list error!")

        data = {
            'result': str(result_list),
            'work_id': work_id,
            'title1_id': title1_id,
            'step_id': step_id,
            'ip': ip
        }

        img1 = cv2.imread(img_dir)

        try:
            for a in result_list:
                for b in a:
                    for c in b:
                        # point_1 = (int(c[3] * model_img_input_size[0]), int(c[2] * model_img_input_size[1]))
                        # point_2 = (int(c[5] * model_img_input_size[0]), int(c[4] * model_img_input_size[1]))
                        point_1 = (int(c[3] * 1920), int(c[2] * 1080))
                        point_2 = (int(c[5] * 1920), int(c[4] * 1080))
                        # print(point_1,point_2)
                        cv2.rectangle(img1, point_1, point_2, (0, 255, 0), 1)
                        str_txt = label_dict[str(c[0])]
                        cv2.putText(img1, str_txt, point_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            # cv2.imshow('resutlt_123', img1)
            cv2.imwrite('/home/db/myftp/tensorflow/resutlt_123.jpg', img1)
        except:
            cv2.imwrite('/home/db/myftp/tensorflow/resutlt_124.jpg', img1)


        print("time-----------{}".format(time.time() - t0))
        print(str(data))
        kaiguandeng.imagenet.flag = True
        # cv2.imshow('resutlt_123', img_show)
        # cv2.waitKey(50)
        # return JsonResponse(data, safe=False)
        # print(str(data))
        return HttpResponse(str(data))

    else:
        print("处理未完成")
        return HttpResponse("处理未完成")


@echoRuntime
def backstage(request):
    result_list = [12, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    return HttpResponse(result_list)

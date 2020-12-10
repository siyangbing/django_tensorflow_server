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

from kaiguandeng.deal_img.location_map import Map_location, model_img_input_size, label_dict, config, saved_model_dir, \
    join_label_dict, treshold


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
        tt = time.time()
        print("接收一张图片需要{}秒".format(tt - t0))
        # print(test_image)
        img_byte = base64.b64decode(img_data)
        img_np_arr = np.fromstring(img_byte, np.uint8)
        image = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        t1 = time.time()
        print("解码张图片需要{}秒".format(t1 - tt))
        cv2.imwrite("./pppp.png", image)
        t2 = time.time()
        print("保存一张图片需要{}秒".format(t2 - t1))
        code = 200
        try:
            map_location = Map_location(treshold, label_dict, join_label_dict, model_img_input_size)
            img_data_list = map_location.read_img(image)
            y_list = map_location.eval_img_list(img_data_list)
            result_list = map_location.get_location(y_list)
        except:
            code = 0
            result_list = []
        num = 0
        for x in result_list:
            for y in x:
                for z in y:
                    num += 1

        if len(result_list) == 3:
            deng, yskg, hskg = result_list
            if len(deng) == 3 and len(yskg) == 2 and len(hskg) == 1:
                one, two, three = deng
                yskg_one,yskg_two = yskg
                one_hskg = hskg[0]
                if len(one)==12 and len(two)==8 and len(three)==8 and len(yskg_one)==8 and len(yskg_two)==1 and len(one_hskg)==1:
                    code = 200
                else:
                    code = 0
            else:
                code = 0
        else:
            code = 0

        t3 = time.time()
        print("code--------------------{}".format(num))
        print("num--------------------{}".format(num))
        print("计算一张图片需要{}秒".format(t3 - t2))
        data = {
            'co de': code,
            'num': num,
            'result': result_list,
        }
        t4 = time.time()
        print("处理一张图片需要{}秒".format(t4 - t0))
    return JsonResponse(data)
    # return HttpResponse("success!!!")


def test(request):
    if (request.method == 'POST'):
        t0 = time.time()

        img_data = request.POST.get('image')  # 本质就是解码字符串
        tt = time.time()
        print("接收一张图片需要{}秒".format(tt - t0))
        # print(test_image)
        img_byte = base64.b64decode(img_data)
        img_np_arr = np.fromstring(img_byte, np.uint8)
        image = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        t1 = time.time()
        print("解码张图片需要{}秒".format(t1 - tt))
        cv2.imwrite("./p1.jpg", image)
        t2 = time.time()
        print("保存一张图片需要{}秒".format(t2 - t1))
        # code = 200
        # try:
        #     map_location = Map_location(treshold, label_dict, join_label_dict, model_img_input_size)
        #     img_data_list = map_location.read_img(image)
        #     y_list = map_location.eval_img_list(img_data_list)
        #     result_list = map_location.get_location(y_list)
        # except:
        #     code = 0
        #     result_list = []
        # num = 0
        # for x in result_list:
        #     for y in x:
        #         for z in y:
        #             num += 1
        # t3 = time.time()
        # print("num--------------------{}".format(num))
        # print("计算一张图片需要{}秒".format(t3 - t2))
        # data = {
        #     'code': code,
        #     'num': num,
        #     'result': result_list,
        # }
        # t4 = time.time()
        # print("处理一张图片需要{}秒".format(t4 - t0))
    return HttpResponse("success!!!")

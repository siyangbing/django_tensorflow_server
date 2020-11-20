from django.shortcuts import render

# Create your views here.
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
from shiziluoding.pingjie_class import CJPJ,resize_shape,crop_size,border,show_rate,sess,meta_graph_def

import kaiguandeng.imagenet

ftp_dir = "/home/db/myftp/tensorflow"
# model_img_input_size = (640, 480)
# saved_model_dir = '/home/db/bing/django_test/kaiguandeng/pb_model/saved_model'
# config = tf.ConfigProto(allow_soft_placement=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
#
# resize_shape = (1920, 1080)
# # 裁剪的大小
# crop_size = (640, 640)
# # 重合的边界
# border = 110
# show_rate = 0.52
# label_dict = {'1.0': 'k', '2.0': 't',}
# # label_dict = {'1.0': 'ku', '2.0': 'kd', '3.0': 'km', '4.0': 'u', '5.0': 'd', '6.0': 'l', '7.0': 'r', '8.0': 'm',
# #               '9.0': 'k', '10.0': 'g'}
# # # class dict
# join_label_dict = {"yskg": ["1.0", "2.0", "3.0"], "hskg": ["4.0", "5.0", "6.0", "7.0", "8.0"], "deng": ["9.0", "10.0"]}
# #
# # deng_location = [12, 8, 8]
# # yskg_location = [8, 1]
# # hskg = [1]
#
# treshold = 0.4


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
def shiziluoding(request):
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

        cjpj = CJPJ(crop_size, border, show_rate)
        croped_img_list = cjpj.crop_img(img)
        y_list = cjpj.eval_img_list(croped_img_list)
        result_list = cjpj.pj(y_list, show_rate)

        # cjpj = CJPJ(crop_size, border, show_rate)
        # croped_img_list = cjpj.crop_img(img)
        # y_list = cjpj.eval_img_list(croped_img_list)
        # result_list_points = cjpj.pj(y_list, show_rate)
        data = {
            'result': str(result_list),
            'work_id': work_id,
            'title1_id': title1_id,
            'step_id': step_id,
            'ip': ip
        }
        print("time-----------{}".format(time.time() - t0))
        kaiguandeng.imagenet.flag = True
        return HttpResponse(str(data))
    else:
        print("处理未完成")
        return HttpResponse("处理未完成")




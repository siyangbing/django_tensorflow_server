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
from shiziluoding.pingjie_class_shizi import CJPJ,resize_shape,crop_size,border,show_rate,sess,meta_graph_def

import kaiguandeng.imagenet

ftp_dir = "/home/db/myftp/tensorflow"

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




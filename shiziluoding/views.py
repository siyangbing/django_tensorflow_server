from django.shortcuts import render

# Create your views here.
import time
import os
import json
import requests
from json import dumps
import traceback

from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import cv2
import numpy as np
from  django.http import JsonResponse
from shiziluoding.pingjie_class_shizi import CJPJ,crop_size,border,show_rate,resize_shape

import kaiguandeng.imagenet
import base64

ftp_dir = "/home/db/myftp/tensorflow"

# sess_shiziluoding = tf.Session(config=config)
# meta_graph_def = tf.saved_model.loader.load(sess_shiziluoding, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

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
        cjpj = CJPJ(crop_size, border, show_rate)

        # try:
        croped_img_list = cjpj.crop_img(image)
        y_list = cjpj.eval_img_list(croped_img_list)
        result_list = cjpj.pj(y_list, show_rate)
        print("pppppppppppppppppppppppppppppppppppppppppp"+str(result_list))
        # img = cv2.resize(image, resize_shape)
        # img_result = cjpj.draw_boxes(result_list, img)
        #     cv2.imwrite('/home/db/myftp/tensorflow/resutlt_szld1.jpg', img_result)
        # except:
        #     traceback.print_exc()
        #     cv2.imwrite('/home/db/myftp/tensorflow/resutlt_sdld2.jpg', img_result)
        t3 = time.time()
        print("计算一张图片需要{}秒".format(t3 - t2))
        data = {
            'code': code,
            'result': result_list,
        }
        t4 = time.time()
        print("处理一张图片需要{}秒".format(t4 - t0))
    return JsonResponse(data)

@echoRuntime
def shiziluoding_bak(request):
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
        img_dir = "/home/db/bing/django_tensorflow_server/test_img/shiziluoding.jpg"
        print("img_path:-------{}".format(imgPath))
        print("img_dir:-------{}".format(img_dir))
        if not os.path.exists(img_dir):
            kaiguandeng.imagenet.flag = True
            return HttpResponse("找不到图片")
        img = cv2.imread(img_dir)  # 读取图片

        cjpj = CJPJ(crop_size, border, show_rate)
        try:
            croped_img_list = cjpj.crop_img(img)
            y_list = cjpj.eval_img_list(croped_img_list)
            result_list = cjpj.pj(y_list, show_rate)
            print(result_list)
            img = cv2.resize(img, resize_shape)
            img_result = cjpj.draw_boxes(result_list, img)
            cv2.imwrite('/home/db/myftp/tensorflow/resutlt_szld1.jpg', img_result)
        except:
            traceback.print_exc()
            cv2.imwrite('/home/db/myftp/tensorflow/resutlt_sdld2.jpg', img_result)

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




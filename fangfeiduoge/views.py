import time
import base64

from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import cv2
import numpy as np
from  django.http import JsonResponse

from deal_one_model.shiziluoding.deal_one_img import ShiZiLuoDingEval
from deal_one_model.yiziluoding.deal_one_img import YhiZiLuoDingEval

# Create your views here.

def fangfeiduoge(request):
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
        cv2.imwrite("./ffdg.png", image)
        t2 = time.time()
        print("保存一张图片需要{}秒".format(t2 - t1))
        code = 200

        # # load_pb_model_szld = ShiZiLuoDingEval()
        # # img_list_szld = load_pb_model_szld.get_detect_result(image)
        # # print("img_list_szld {}".format(img_list_szld))
        # # load_pb_model_yzld = YhiZiLuoDingEval()
        # # img_list_yzld = load_pb_model_yzld.get_detect_result(image)
        # # print("img_list_yzld {}".format(img_list_yzld))
        #
        # a = 3
        #
        # # img_result = load_pb_model.draw_boxes(img_list, img_path)


        try:
            load_pb_model_szld = ShiZiLuoDingEval()
            img_list_szld = load_pb_model_szld.get_detect_result(image)
            print("img_list_szld {}".format(img_list_szld))
            load_pb_model_yzld = YhiZiLuoDingEval()
            img_list_yzld = load_pb_model_yzld.get_detect_result(image)
            print("img_list_yzld {}".format(img_list_yzld))
            result_list = img_list_szld+img_list_yzld
            print("all_result  {}".format(img_list_yzld))
        except:
            code = 0
            result_list = []
        num = 0
        if not result_list:
            for x in result_list:
                for y in x:
                    for z in y:
                        num += 1

        else:
            num = 0

        t3 = time.time()
        print("num--------------------{}".format(num))
        print("计算一张图片需要{}秒".format(t3 - t2))
        data = {
            'code': code,
            'num': num,
            'result': result_list,
        }
        t4 = time.time()
        print("处理一张图片需要{}秒".format(t4 - t0))
    # return JsonResponse(data)
    return HttpResponse("success!!!")

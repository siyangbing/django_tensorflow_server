import os
import time

import base64
import numpy as np
import cv2
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from deal_one_model.fangfeizuocang.deal_one_img import FangFeiZuoCangEval


# Create your views here.
def fangfeizuocang(request):
    if (request.method == 'POST'):
        t0 = time.time()
        img_data = request.POST.get('image')  # 本质就是解码字符串
        part_index = request.POST.get('part')
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
        code = 0

        try:
            load_pb_model_ffzc = FangFeiZuoCangEval()
            result_list, new_code,part = load_pb_model_ffzc.get_detect_result(image,part_index)
            code = new_code
            print("img_list_ffzc {}".format(result_list))
        except:
            code = 0
            result_list = []
        num = len(result_list)
        t3 = time.time()
        print("num--------------------{}".format(num))
        print("计算一张图片需要{}秒".format(t3 - t2))
        data = {
            'code': code,
            'num': num,
            'total': 3,
            'part': part,
            'result': result_list,
        }
        t4 = time.time()
        print("处理一张图片需要{}秒".format(t4 - t0))
    return JsonResponse(data)
    # return HttpResponse("success!!!")

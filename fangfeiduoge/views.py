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
        while True:

            load_pb_model_szld = ShiZiLuoDingEval()
            img_list_szld = load_pb_model_szld.get_detect_result(image)

            load_pb_model_yzld = ShiZiLuoDingEval()
            img_list_yzld = load_pb_model_yzld.get_detect_result(image)

        a = 3

        # img_result = load_pb_model.draw_boxes(img_list, img_path)


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
    # return JsonResponse(data)
    return HttpResponse("success!!!")

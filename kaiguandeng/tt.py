# -*-coding:utf-8-*-
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
from django.http import JsonResponse

from kaiguandeng.deal_img.location_map import Map_location, model_img_input_size, sess, meta_graph_def, label_dict, \
    join_label_dict, treshold


img_dir = "/home/db/dbing/django_test/kaiguandeng/deal_img/1.jpg"

map_location = Map_location(treshold, label_dict, join_label_dict, model_img_input_size)
img_data_list = map_location.read_img(img_dir)
y_list = map_location.eval_img_list(img_data_list)
result_list = map_location.get_location(y_list)


PPP = 4

print(result_list)

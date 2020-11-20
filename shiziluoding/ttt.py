from shiziluoding.pingjie_class import CJPJ,resize_shape,crop_size,border,show_rate,saved_model_dir,sess,meta_graph_def
import tensorflow as tf
import cv2


img_path = "/home/db/bing/django_test/shiziluoding/1.jpg"
img = cv2.imread(img_path)
cjpj = CJPJ(crop_size, border, show_rate)
croped_img_list = cjpj.crop_img(img)
y_list = cjpj.eval_img_list(croped_img_list)
result_list_points = cjpj.pj(y_list, show_rate)
aa = 23423
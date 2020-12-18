import os

import tensorflow as tf
import cv2
from django_tensorflow_server.settings import BASE_DIR

from eval_img_class.load_pb_model import LoadPbModel

img_path = os.path.join(BASE_DIR, "test_img/kaiguandeng.jpg")

model_path = saved_model_dir = os.path.join(BASE_DIR, "pb_model/tongdian/kaiguandeng/saved_model")
resize_shape = (640, 480)
repeat_iou = 0.2
show_rate = 0.4
label_dict = {'1.0': 'ku', '2.0': 'kd', '3.0': 'km', '4.0': 'u', '5.0': 'd', '6.0': 'l', '7.0': 'r', '8.0': 'm',
              '9.0': 'k', '10.0': 'g'}
# class dict
join_label_dict = {"deng": ["9.0", "10.0"], "yskg": ["1.0", "2.0", "3.0"], "hskg": ["4.0", "5.0", "6.0", "7.0", "8.0"]}
hang_dict = {"deng": 3, "yskg": 2, "hskg": 1}

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True
g1 = tf.Graph()
sess = tf.Session(config=config, graph=g1)
meta_graph_def_sig = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)


class KaiGuanDengEval():
    def __init__(self, sess=sess):
        self.load_pb_model = LoadPbModel(sess)

    def get_detect_result(self, img_path, resize_shape=resize_shape):
        img_list = self.load_pb_model.read_img(img_path, resize_shape)
        y = self.load_pb_model.eval_img_data_list(img_list)
        result_list = self.load_pb_model.get_map_location(y, label_dict=label_dict, join_label_dict=join_label_dict, threshold=show_rate, hang_dict=hang_dict)

        # a = 3
        img_result = self.load_pb_model.draw_boxes(result_list,img_list[0])
        cv2.imwrite("123.jpg",img_result)
        # cv2.imshow("img_result", img_result)
        # cv2.waitKey(0)
        return result_list


if __name__ == "__main__":
    load_pb_model = KaiGuanDengEval()
    img_list = load_pb_model.get_detect_result(img_path)
    # img_result = load_pb_model.draw_boxes(img_list, img_path)
    # cv2.imshow("img_result", img_result)
    # cv2.waitKey(0)
    a = 222

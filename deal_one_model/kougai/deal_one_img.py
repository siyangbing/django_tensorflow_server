import os

import tensorflow as tf
import cv2
from django_tensorflow_server.settings import BASE_DIR

from eval_img_class.load_pb_model import LoadPbModel

img_path = os.path.join(BASE_DIR, "test_img/kougai.jpg")

model_path = saved_model_dir = os.path.join(BASE_DIR, "pb_model/fangfei/kougai/saved_model")
resize_shape = (640, 480)
repeat_iou = 0.2
show_rate = 0.5

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True
g1 = tf.Graph()
sess = tf.Session(config=config, graph=g1)
meta_graph_def_sig = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)


class KouGaiEval():
    def __init__(self, sess=sess):
        self.load_pb_model = LoadPbModel(sess)

    def get_detect_result(self, img_path, resize_shape=resize_shape):
        img_list = self.load_pb_model.read_img(img_path, resize_shape)
        y = self.load_pb_model.eval_img_data_list(img_list)
        result_list = self.load_pb_model.get_img_result_list(y, repeat_iou=0.3, show_rate=0.5)
        # img_result = self.load_pb_model.draw_boxes(result_list,img_list[0])
        # cv2.imshow("img_result", img_result)
        # cv2.waitKey(0)
        return result_list


if __name__ == "__main__":
    load_pb_model = KouGai()
    img_list = load_pb_model.get_detect_result(img_path)
    # img_result = load_pb_model.draw_boxes(img_list, img_path)
    # cv2.imshow("img_result", img_result)
    # cv2.waitKey(0)
    a = 222

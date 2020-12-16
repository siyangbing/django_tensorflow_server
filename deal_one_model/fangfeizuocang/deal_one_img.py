import os

import tensorflow as tf
import cv2
from django_tensorflow_server.settings import BASE_DIR

from eval_img_class.load_pb_model import LoadPbModel

# img_path = os.path.join(BASE_DIR, "test_img/zuocangtongyong.jpg")
# img_path = os.path.join(BASE_DIR, "test_img/fangfeizuocang2.jpg")
# img_path = os.path.join(BASE_DIR, "test_img/fangfeizuocang3.jpg")
img_path = os.path.join(BASE_DIR, "test_img/fangfeizuocang4.jpg")
model_path = saved_model_dir = os.path.join(BASE_DIR, "pb_model/fangfei/fangfeizuocang/saved_model")
resize_shape = (640, 480)
repeat_iou = 0.2
show_rate = 0.1

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True
g1 = tf.Graph()
sess = tf.Session(config=config, graph=g1)
meta_graph_def_sig = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

step1_dict = {12.0: 12, 13.0: 1, 14.0: 1}
step2_dict = {8.0: 1}
step3_dict = {2.0: 1, 3.0: 1, 6.0: 1}
step4_dict = {10.0: 2}
# num_dict = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0, 6.0: 0, 7.0: 0, 8.0: 0, 9.0: 0, 10.0: 0,
#             11.0: 0, 12.0: 0, 13.0: 0, 14.0: 0, 15.0: 0}

step_list = [step1_dict, step2_dict, step3_dict, step4_dict]


class FangFeiZuoCangEval():
    def __init__(self, sess=sess):
        self.load_pb_model = LoadPbModel(sess)

    def count(self, result_list):
        num_dict = {}

        for index, box in enumerate(result_list):
            class_label = float(box[4])
            if class_label not in num_dict:
                num_dict[box[4]] = 1
            else:
                num_dict[box[4]] += 1
        return num_dict

    def get_detect_result(self, img_path,part_index, resize_shape=resize_shape):
        img_list = self.load_pb_model.read_img(img_path, resize_shape)
        y = self.load_pb_model.eval_img_data_list(img_list)
        result_list = self.load_pb_model.get_img_result_list(y, repeat_iou=repeat_iou, show_rate=show_rate)
        num_dict = self.count(result_list)
        print(num_dict)

        quyu_index = 0
        for key in step_list[part_index-1]:
            code = 0
            if key in num_dict:
                if num_dict[key]==step_list[part_index-1][key]:
                    quyu_index = part_index
                    code =200
                else:
                    code = 0
                    break
            else:
                code = 0
                break
        # for index, step_dict in enumerate(step_list):
        #     for key in step_dict:
        #         code = 0
        #         if key in num_dict:
        #             if num_dict[key] == step_dict[key]:
        #                 code = 200
        #             else:
        #                 code = 0
        #                 break
        #         else:
        #             code = 0
        #             break
        #     if code == 200:
        #         quyu_index = index + 1
        #         break
        #     else:
        #         pass

        # a = 3
        # img_result = self.load_pb_model.draw_boxes(result_list, img_list[0])
        # cv2.imshow("img_result", img_result)
        # cv2.waitKey(0)
        return result_list, code, quyu_index


if __name__ == "__main__":
    load_pb_model = FangFeiZuoCangEval()
    img_list, code = load_pb_model.get_detect_result(img_path)
    # img_result = load_pb_model.draw_boxes(img_list, img_path)
    # cv2.imshow("img_result", img_result)
    # cv2.waitKey(0)
    a = 222

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import time
from sklearn.cluster import KMeans
import os

from django_tensorflow_server.settings import BASE_DIR


class LoadPbModel():
    def __init__(self,saved_model_dir):
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.config.gpu_options.allow_growth = True
        self.g1 = tf.Graph()
        self.sess = tf.Session(config=self.config, graph=self.g1)
        self.meta_graph_def_sig = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

    def eval_img_data_list(self,img_data_list):
        img_data_list = []
        for img in img_data_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = np.array(img, dtype=float)  # 改变数据类型为float
            img_array = img_array[np.newaxis, :, :, :]  # 增加一个维度
            input_data = np.array(img_array, dtype=np.float32)

            img_data_list.append(input_data)
        img_data = np.vstack((x for x in img_data_list))
        input = sess.graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        detection_score = sess.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
        num_detections = sess.graph.get_tensor_by_name('num_detections:0')
        feed_dict = {input: img_data, }
        y = sess.run([detection_boxes, detection_score, detection_classes, num_detections], feed_dict=feed_dict)
        return y

    def read_img(self, img_path,resize_shape):
        try:
            img = cv2.imread(img_path)  # 读取图片
        except:
            img = img_path
        img = cv2.resize(img, resize_shape)  # 缩放到resize_shape

        return [img]

    def crop_img(self,img):
        pass

if __name__ == "__main__":
    saved_model_dir = saved_model_dir = os.path.join(BASE_DIR,"pb_model/tongdian/kaiguandeng/saved_model")
    img_path = "/home/db/bing/django_tensorflow_server/test_img/shiziluoding.jpg"
    resize_shape = (1920,1080)
    load_pb_model = LoadPbModel(saved_model_dir)
    img_list = load_pb_model.read_img(img_path,resize_shape)
    result_y = load_pb_model.eval_img_data_list(img_list)
    a = 222






img_path = "/home/db/bing/django_tensorflow_server/test_img/kaiguandeng.jpg"
img_resize_shape=(1920, 1080)
model_img_input_size = (640, 480)
saved_model_dir = os.path.join(BASE_DIR,"pb_model/tongdian/kaiguandeng/saved_model")

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True
g1 = tf.Graph()
sess = tf.Session(config=config,graph=g1)
meta_graph_def_sig = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)


label_dict = {'1.0': 'ku', '2.0': 'kd', '3.0': 'km', '4.0': 'u', '5.0': 'd', '6.0': 'l', '7.0': 'r', '8.0': 'm',
              '9.0': 'k', '10.0': 'g'}
# class dict
join_label_dict = {"yskg": ["1.0", "2.0", "3.0"], "hskg": ["4.0", "5.0", "6.0", "7.0", "8.0"], "deng": ["9.0", "10.0"]}
treshold = 0.2
final_label = ("d000","d001","d002","d003","d004","d005","d006","d007","d008","d009","d010","d011",
               "d100","d101","d102","d103","d104","d105","d106","d107",
               "d200","d201","d202","d203","d204","d205","d206","d207",
               "y000","y001","y002","y003","y004","y005","y006","y007",
               "y100",
               "h000")


def echoRuntime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        endTime = time.time()
        msecs = (endTime - startTime)
        print(func.__name__ + " running time is %.2f s" % msecs)
        return result

    return wrapper

class Map_location():
    # @echoRuntime
    def __init__(self, threshold, label_dict, join_label_dict,model_img_input_size):
        # self.location_list = y[0][0]
        # self.score_list = y[1][0]
        # self.class_list = y[2][0]
        # self.num = y[3]
        self.threshold = threshold
        self.label_dict = label_dict
        self.join_label_dict = join_label_dict
        self.model_img_input_size = model_img_input_size

    # @echoRuntime
    def takeSecond(sellf, elem):
        # print(elem)
        return elem[0][2]

    # @echoRuntime
    def julei(self, class_num, point_list, key):
        if key == "lie":
            key_index = 3
            index2 = 4
        else:
            key_index = 4
            index2 = 3
        # t0 =time.time()
        x1 = np.zeros(len(point_list))
        x2 = np.array([v[key_index] for v in point_list])
        x = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
        kmeans = KMeans(n_clusters=class_num)  # n_clusters:number of cluster
        kmeans.fit(x)
        result = list(kmeans.labels_)
        hang_list = [[] for i in range(class_num)]
        for index, value in enumerate(result):
            hang_list[value].append(tuple(point_list[index]))

        for index, value in enumerate(hang_list):
            # print(value)
            temp_tuple = sorted(value, key=lambda x: x[index2])
            hang_list[index] = tuple(temp_tuple)
        hang_list = sorted(hang_list, key=lambda x: x[0][key_index])
        return hang_list

    # @echoRuntime
    def map_label(self):
        point_dict = {k: [] for k in self.label_dict}
        class_obj_dict = {k: [] for k in self.join_label_dict}
        for index, socre in enumerate(self.score_list):
            if socre >= self.threshold:
                point_info_one = [self.class_list[index], self.score_list[index], list(self.location_list[index])]
                point_dict[str(self.class_list[index])].append(point_info_one)

        for obj_key in point_dict:
            for class_label in self.join_label_dict:
                if obj_key in self.join_label_dict[class_label]:
                    class_obj_dict[class_label].extend(point_dict[obj_key])
        # print(class_obj_dict)
        return class_obj_dict

    @echoRuntime
    def read_img(self,img_path):
        try:
            img = cv2.imread(img_path)  # 读取图片
        except:
            img = img_path
        img = cv2.resize(img, self.model_img_input_size )  # 缩放到480*480
        return [img]

    @echoRuntime
    def eval_img_list(self, croped_img_list):
        # meta_graph_def = meta_graph_def
        # while True:
        # sess = tf.Session(config=config)
        # # meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        # with sess as sess_kaiguandneg:
        #     meta_graph_def = tf.saved_model.loader.load(sess_kaiguandneg, [tf.saved_model.tag_constants.SERVING],
        #                                                 saved_model_dir)
        t0 = time.time()
        img_data_list = []
        for img in croped_img_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = np.array(img, dtype=float)  # 改变数据类型为float
            img_array = img_array[np.newaxis, :, :, :]  # 增加一个维度
            input_data = np.array(img_array, dtype=np.float32)

            img_data_list.append(input_data)
        img_data = np.vstack((x for x in img_data_list))
        input = sess.graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        detection_score = sess.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
        num_detections = sess.graph.get_tensor_by_name('num_detections:0')
        feed_dict = {input: img_data, }

        # result_list = [[] for x in range(self.w_num * self.h_num)]

        y = sess.run([detection_boxes, detection_score, detection_classes, num_detections], feed_dict=feed_dict)
        return y

    def mat_inter(self, box1, box2):
        # 判断两个矩形是否相交
        # box=(xA,yA,xB,yB)
        # a = box1
        # b = box2
        # print(a[0], a[1], a[2], a[3], a[4])
        # print(b)
        x01 = box1[0]
        y01 = box1[1]
        x02 = box1[2]
        y02 = box1[3]
        score_1 = box1[4]

        x11 = box2[0]
        y11 = box2[1]
        x12 = box2[2]
        y12 = box2[3]
        score_2 = box2[4]
        # x11, y11, x12, y12, score = box2

        lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
        ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
        sax = abs(x01 - x02)
        sbx = abs(x11 - x12)
        say = abs(y01 - y02)
        sby = abs(y11 - y12)
        if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
            return True
        else:
            return False

    def solve_coincide(self, box1, box2):
        # box=(xA,yA,xB,yB)
        # 计算两个矩形框的重合度
        if self.mat_inter(box1, box2) == True:
            x01 = box1[0]
            y01 = box1[1]
            x02 = box1[2]
            y02 = box1[3]
            score_1 = box1[4]

            x11 = box2[0]
            y11 = box2[1]
            x12 = box2[2]
            y12 = box2[3]
            score_2 = box2[4]

            # x01, y01, x02, y02 = box1
            # x11, y11, x12, y12 = box2
            col = min(x02, x12) - max(x01, x11)
            row = min(y02, y12) - max(y01, y11)
            intersection = col * row
            area1 = (x02 - x01) * (y02 - y01)
            area2 = (x12 - x11) * (y12 - y11)
            coincide = intersection / (area1 + area2 - intersection)
            return coincide
        else:
            return False

    # @echoRuntime
    def del_iou_boxes(self, pingjie_points_list):
        # 删除多余的重合边框
        result_list = pingjie_points_list
        while 1:
            pingjie_points_list = result_list
            if_ok = 1
            for point1_1 in pingjie_points_list:
                for point2_2 in pingjie_points_list:
                    if point1_1 != point2_2:
                        if self.solve_coincide(point1_1, point2_2) > 0.2:
                            # 如果重合面大于0.2就只保留得分高的检测框，删除得分低的检测框
                            if_ok = 0
                            if point1_1[4] > point2_2[4]:
                                result_list.remove(point2_2)
                            else:
                                if point1_1 in result_list:
                                    result_list.remove(point1_1)
                                else:
                                    pass

                                    # print(point1_1)
                                    # print(result_list)
            if if_ok:
                return result_list
            else:
                continue
        return result_list


    # @echoRuntime
    def get_average_xy(self, value_one):
        return [[v[0], v[1], v[2], (v[2][1] + v[2][3]) / 2, (v[2][0] + v[2][2]) / 2] for v in value_one]

    def daw_zxt(self, point_list):  # x = range(len(point_list))
        x = [v[2] for v in point_list]
        # print(x)
        y = [v[3] for v in point_list]
        plt.scatter(x, y)
        # plt.plot(x, y)
        plt.show()

    @echoRuntime
    def get_location(self,y):
        self.location_list = y[0][0]
        self.score_list = y[1][0]
        self.class_list = y[2][0]
        self.num = y[3]

        class_obj_dict = {k: self.get_average_xy(v) for k, v in self.map_label().items()}

        # result_list_points = self.del_iou_boxes(final_list)
        # result = self.julei(3,class_obj_dict["yskg"])
        result_deng = self.julei(3, class_obj_dict["deng"], "hang")
        result_yskg = self.julei(2, class_obj_dict["yskg"], "hang")
        result_hskg = self.julei(1, class_obj_dict["hskg"], "hang")

        all_list = [result_deng, result_yskg, result_hskg]

        # result_llist = [[[[c[0], c[1], c[2][0], c[2][1], c[2][2], c[2][3]] for c in b] for b in a] for a in all_list]
        result_llist = [[[[c[2][0], c[2][1], c[2][2], c[2][3],c[0], c[1]] for c in b] for b in a] for a in all_list]


        for index_a,a in enumerate(result_llist):
            for index_b,b in enumerate(a):
                # temp_list = [[x[2], x[3], x[4], x[5], x[0], x[1]] for x in b]
                result_llist[index_a][index_b] = self.del_iou_boxes(b)
                # ppp = 3
                # print(b)


        result_findal_list = [[[np.array(c, dtype='float64').tolist() for c in b] for b in a] for a in result_llist]

        # flatten_list = self.list_flatten(result_llist)
        # result_list1 = [[x[2], x[3], x[4], x[5], x[0], x[1]] for x in flatten_list]
        # result_list_points1 = self.del_iou_boxes(result_list1)
        #

        # result_findal_list = [[ c in b for b in a] for a in result_llist]
        # result_findal_list = [
        #     [[np.array([c], dtype='float64').tolist() for c in b] for b in a] for a in
        #     result_llist]
        # result_dict = dict(zip(final_label,self.list_flatten(result_findal_list)))

        print("result_findal_list{}".format(result_findal_list))
        # return result_dict

        return result_findal_list

    def list_flatten(self,point_list):
        for each in point_list:
            if not isinstance(each,list):
                yield point_list
            else:
                yield from self.list_flatten(each)

    def draw_boxes(self,result_findal_list,img):
        for a in result_findal_list:
            for b in a:
                for c in b:
                    point_1 = (int(c[3] * img.shape[1]), int(c[2] * img.shape[0]))
                    point_2 = (int(c[5] * img.shape[1]), int(c[4] * img.shape[0]))
                    # print(point_1,point_2)
                    cv2.rectangle(img, point_1, point_2, (0, 255, 0), 1)
                    str_txt = str(c[0])
                    # print(str_txt)
                    str_txt = label_dict[str(c[0])]
                    # print("qqq================={}".format(str_txt))
                    cv2.putText(img, str_txt, point_1, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1)
        return img


# if __name__ == "__main__":
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, img_resize_shape)
#     # cv2.imwrite("src.jpg",img2)
#     # img3 = cv2.resize(img1,(640,480))
#     map_location = Map_location(treshold, label_dict, join_label_dict,model_img_input_size)
#     img_data_list = map_location.read_img(img_path)
#     y_list = map_location.eval_img_list(img_data_list)
#     print("wwwwwwwwwwwwwww")
#     result_list = map_location.get_location(y_list)
#     print(result_list)
#     map_location.draw_boxes(result_list,img)
#     cv2.imshow("ppp",img)
#     # cv2.imwrite("result.jpg",img)
#     cv2.waitKey(0)


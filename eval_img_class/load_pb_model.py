import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import time
from sklearn.cluster import KMeans
import math
import os

from django_tensorflow_server.settings import BASE_DIR


# 自定义通用的识别类
class LoadPbModel():
    # 初始化传入sess对象
    def __init__(self, sess):
        self.sess = sess

    # 读取图片
    def read_img(self, img_path, resize_shape):
        # 读取图片，支持文件路径和frame
        try:
            img = cv2.imread(img_path)  # 读取图片
        except:
            img = img_path  # 如果读取图片失败代表传入的是frame对象
        # 缩放到resize_shape
        img = cv2.resize(img, resize_shape)
        return [img]

    # 裁剪图片
    def crop_img(self, need_crop_img_list, crop_size, border):
        self.crop_size = crop_size  # 裁剪图片的大小
        self.border = border  # 裁剪图片的重合区域
        for img in need_crop_img_list:
            h, w = img.shape[:2]
            # 计算大图片需要裁剪成多少行多少列
            self.h_num = math.floor((h - self.crop_size[0]) / (self.crop_size[0] - self.border)) + 2
            self.w_num = math.floor((w - self.crop_size[1]) / (self.crop_size[1] - self.border)) + 2

            # 补齐边界白像素的宽度的单位
            b_h = self.crop_size[0] - ((h - self.crop_size[0]) % (self.crop_size[0] - border)) - border
            b_w = self.crop_size[1] - ((w - self.crop_size[1]) % (self.crop_size[1] - border)) - border
            # 为了防止裁剪后边缘图像大小不一，需要用白色像素补齐边缘
            img = cv2.copyMakeBorder(img, 0, b_h, 0,
                                     b_w, cv2.BORDER_CONSTANT,
                                     value=[255, 255, 255])
            # 补齐边缘后的h,w
            h, w = img.shape[:2]

            # 裁剪后的图片列表
            croped_img_list = []
            # 裁剪图片
            for x_l in range(self.h_num):
                for y_l in range(self.w_num):
                    # print(x_l, y_l)
                    x_min = x_l * (self.crop_size[0] - self.border)
                    x_max = x_min + self.crop_size[0]
                    y_min = y_l * (self.crop_size[1] - self.border)
                    y_max = y_min + self.crop_size[1]

                    if x_max >= h:
                        x_max = h
                    if y_max >= w:
                        y_max = w

                    img_result = img[x_min:x_max, y_min:y_max]
                    croped_img_list.append(img_result)
            a = 33
            return croped_img_list

    def eval_img_data_list(self, img_data_list):
        # 模型推断图片
        img_data_list_convert = []
        for img in img_data_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = np.array(img, dtype=float)  # 改变数据类型为float
            img_array = img_array[np.newaxis, :, :, :]  # 增加一个维度
            input_data = np.array(img_array, dtype=np.float32)

            img_data_list_convert.append(input_data)
        img_data = np.vstack((x for x in img_data_list_convert))
        # img_data = np.vstack([x for x in img_data_list_convert])
        input = self.sess.graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.sess.graph.get_tensor_by_name('detection_boxes:0')
        detection_score = self.sess.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.sess.graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.sess.graph.get_tensor_by_name('num_detections:0')
        feed_dict = {input: img_data, }
        y = self.sess.run([detection_boxes, detection_score, detection_classes, num_detections], feed_dict=feed_dict)
        return y

    def get_img_result_list(self, y, repeat_iou=0.3, show_rate=0.5):
        self.location_list = y[0][0]
        self.score_list = y[1][0]
        self.class_list = y[2][0]
        self.num = y[3]

        result_list = []

        for index, point in enumerate(self.score_list):
            if point >= show_rate:
                one_boxes = [self.location_list[index][1].astype(np.float64),
                             self.location_list[index][0].astype(np.float64),
                             self.location_list[index][3].astype(np.float64),
                             self.location_list[index][2].astype(np.float64), self.class_list[index].astype(np.float64),
                             point.astype(np.float64)]
                result_list.append(one_boxes)

        result_list = self.del_repeat_boxes(result_list, repeat_iou)
        return result_list

    def pingjie_img(self, y, img, repeat_iou=0.2, show_rate=0.5):
        # 拼接图片
        result_list = [[] for x in range(self.w_num * self.h_num)]

        location_list = y[0]
        score_list = y[1]
        class_list = y[2]
        num_class = y[3]

        for index1, a in enumerate(score_list):
            for index2, b in enumerate(a):
                if b >= show_rate:
                    point = location_list[index1][index2] * self.crop_size[1]
                    # point = location_list[index1][index2]
                    score = score_list[index1][index2]
                    result_list[index1].append(
                        [point[1], point[0], point[3], point[2], score, class_list[index1][index2]])
        crop_img_index = 0
        pj_result_list_points = []
        for x_l in range(self.h_num):
            for y_l in range(self.w_num):
                if result_list[crop_img_index] != []:
                    for point in result_list[crop_img_index]:
                        px_min = (point[0] + y_l * (self.crop_size[0] - self.border)) / img.shape[1]
                        py_min = (point[1] + x_l * (self.crop_size[1] - self.border)) / img.shape[0]
                        px_max = (point[2] + y_l * (self.crop_size[0] - self.border)) / img.shape[1]
                        py_max = (point[3] + x_l * (self.crop_size[1] - self.border)) / img.shape[0]

                        # pj_result_list_points.append([px_min, py_min, px_max, py_max, point[4], point[5]])
                        pj_result_list_points.append(
                            [px_min, py_min, px_max, py_max, point[5].astype(np.float64), point[4].astype(np.float64)])
                crop_img_index += 1

        pj_result_list_points = self.del_repeat_boxes(pj_result_list_points, repeat_iou)
        return pj_result_list_points

    def del_repeat_boxes(self, pj_result_list_points, repeat_iou):
        # 删除多余的矩形框
        result_list = pj_result_list_points
        while 1:
            # 如果删除多余框参数为False则不删除多余框，跳出循环
            if repeat_iou == False:
                break
            pingjie_points_list = result_list
            if_ok = 1
            for point1_1 in pingjie_points_list:
                for point2_2 in pingjie_points_list:
                    if point1_1 != point2_2:
                        if self.solve_coincide(point1_1, point2_2) > repeat_iou:
                            # 如果重合面大于0.2就只保留得分高的检测框，删除得分低的检测框
                            if_ok = 0
                            if point1_1[5] > point2_2[5]:
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
        pp = 10
        return result_list

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

    def mat_inter(self, box1, box2):
        # 判断两个矩形是否相交
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

    def draw_boxes(self, point_list, img):
        # point_list = [[[1],[1],[1]],
        #               [[2],[2]],
        #               [3],
        #               [[[[4],[4]]],[4],[4],[4]],]
        yield_point_list = []
        for x in self.yield_points_from_list(point_list):
            yield_point_list.append(x)
        # c =1
        for last_point in yield_point_list:
            # 绘制最终拼接的检测结果
            cv2.rectangle(img, (int(last_point[0] * img.shape[1]), int(last_point[1] * img.shape[0])),
                          (int(last_point[2] * img.shape[1]), int(last_point[3] * img.shape[0])),
                          (0, 255, 0), 1, 8)
            cv2.putText(img, str(last_point[4])[:6],
                        (int(last_point[0] * img.shape[1]), int(last_point[1] * img.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(img, str(last_point[5]),
                        (int(last_point[0] * img.shape[1]), int(last_point[1] * img.shape[0] + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        return img

    def yield_points_from_list(self, point_list):
        # 从point_list中迭代出每一个boxes
        for each in point_list:
            # 如果子元素不列表，当前元素就是boxes元素
            if not isinstance(each, list):
                # yield需要的boxe值
                yield point_list
                # 判断完需要跳出循环，否则或多次yield同一个boxes
                break
            else:
                # 如果当前元素不是子元素，那么继续进入下一层判断
                yield from self.yield_points_from_list(each)

    def get_map_location(self, y, label_dict, join_label_dict, threshold, hang_dict):
        self.location_list = y[0][0]
        self.score_list = y[1][0]
        self.class_list = y[2][0]
        self.num = y[3]

        class_obj_dict = {k: self.get_average_xy(v) for k, v in
                          self.map_label(label_dict, join_label_dict, threshold).items()}

        all_list = []
        for x in join_label_dict:
            one_class_list = self.julei(hang_dict[x], class_obj_dict[x], "hang")
            all_list.append(one_class_list)

        # result_llist = [[[[c[2][0], c[2][1], c[2][2], c[2][3], c[0], c[1]] for c in b] for b in a] for a in all_list]
        result_llist = [[[[c[2][1], c[2][0], c[2][3], c[2][2], c[0], c[1]] for c in b] for b in a] for a in all_list]

        for index_a, a in enumerate(result_llist):
            for index_b, b in enumerate(a):
                result_llist[index_a][index_b] = self.del_repeat_boxes(b,threshold)
                # ppp = 3

        result_findal_list = [[[np.array(c, dtype='float64').tolist() for c in b] for b in a] for a in result_llist]

        print("result_findal_list{}".format(result_findal_list))

        return result_findal_list

    def get_average_xy(self, value_one):
        return [[v[0], v[1], v[2], (v[2][1] + v[2][3]) / 2, (v[2][0] + v[2][2]) / 2] for v in value_one]

    def map_label(self, label_dict, join_label_dict, threshold):
        point_dict = {k: [] for k in label_dict}
        class_obj_dict = {k: [] for k in join_label_dict}
        for index, socre in enumerate(self.score_list):
            if socre >= threshold:
                point_info_one = [self.class_list[index], self.score_list[index], list(self.location_list[index])]
                point_dict[str(self.class_list[index])].append(point_info_one)

        for obj_key in point_dict:
            for class_label in join_label_dict:
                if obj_key in join_label_dict[class_label]:
                    class_obj_dict[class_label].extend(point_dict[obj_key])
        # print(class_obj_dict)
        return class_obj_dict

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


if __name__ == "__main__":
    saved_model_dir = os.path.join(BASE_DIR, "pb_model/fangfei/shiziluoding/saved_model")
    img_path = os.path.join(BASE_DIR, "test_img/shiziluoding.jpg")
    resize_shape = (1920, 1080)
    crop_size = (640, 640)
    border = 110
    show_rate = 0.5
    repeat_iou = 0.2
    load_pb_model = LoadPbModel(saved_model_dir)
    img_list = load_pb_model.read_img(img_path, resize_shape)
    croped_img_list = load_pb_model.crop_img(img_list, crop_size, border)
    y = load_pb_model.eval_img_data_list(croped_img_list)
    result_list = load_pb_model.pingjie_img(y, img_list[0], repeat_iou, show_rate)
    img_resuit = load_pb_model.draw_boxes(result_list, img=img_list[0])
    cv2.imshow("img", img_resuit)
    cv2.waitKey(0)
    a = 222

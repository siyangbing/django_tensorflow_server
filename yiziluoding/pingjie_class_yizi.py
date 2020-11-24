import math
import cv2
import tensorflow as tf
import numpy as np
import time

# 原始图片的缩放比例
resize_shape = 0.5
# 原始img文件夹路径
img_path = "../test_img/yiziluoding.jpg"
saved_img_path = "/home/db/图片/img"

resize_shape = (1920, 1080)
# 裁剪的大小
crop_size = (640, 640)
# 重合的边界
border = 110
show_rate = 0.52

saved_model_dir_szld = '/home/db/PycharmProjects/django_tensorflow_server/yiziluoding/saved_model'
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir_szld)

def echoRuntime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        endTime = time.time()
        msecs = (endTime - startTime)
        print(func.__name__ + " running time is %.2f s" % msecs)
        return result

    return wrapper


class CJPJ():
    def __init__(self,crop_size, border, show_rate):
        # self.img = cv2.resize(img, resize_shape)
        self.crop_size = crop_size
        self.border = border
        self.show_rate = show_rate

    # 拼接图片列表
    def crop_img(self, img):
        img = cv2.resize(img, resize_shape)
        # cv2.imshow("sss", img)
        # cv2.imwrite("2.jpg", img)
        # cv2.waitKey(0)
        h, w = img.shape[:2]
        self.h_num = math.floor((h - self.crop_size[1]) / (self.crop_size[1] - self.border)) + 2
        self.w_num = math.floor((w - self.crop_size[1]) / (self.crop_size[1] - self.border)) + 2

        img = cv2.copyMakeBorder(img, 0, self.h_num * self.crop_size[0] + self.border - h, 0,
                                 self.w_num * self.crop_size[1] + self.border - w, cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])

        h, w = img.shape[:2]

        # pingjie_points_list = []
        croped_img_list = []
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
                # cv2.imshow("{}{}".format(x_l,y_l),img_result)
        a = 33

        return croped_img_list

    # 预测图片列表
    def eval_img_list(self, croped_img_list):
        sess = tf.Session(config=config)
        # meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        with sess as sess_yiziluoding:
            meta_graph_def = tf.saved_model.loader.load(sess_yiziluoding, [tf.saved_model.tag_constants.SERVING],
                                                        saved_model_dir_szld)
        # saved_model_dir = '/home/db/bing/django_test/shiziluoding/saved_model'
        # config = tf.ConfigProto(allow_soft_placement=True)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        # meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

        # img = cv2.imread(img_dir)  # 读取图片
        # img = cv2.resize(img, model_img_input_size)  # 缩放到480*480
        # while True:
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

    # 拼接图片列表
    def pj(self, y, show_rate):

        result_list = [[] for x in range(self.w_num * self.h_num)]

        location_list = y[0]
        score_list = y[1]
        class_list = y[2]
        num_class = y[3]

        for index1, a in enumerate(score_list):
            for index2, b in enumerate(a):
                if b >= show_rate:
                    point = location_list[index1][index2] * self.crop_size[1]
                    score = score_list[index1][index2]
                    result_list[index1].append(
                        [point[1], point[0], point[3], point[2], score, class_list[index1][index2]])
        crop_img_index = 0
        final_list = []
        for x_l in range(self.h_num):
            for y_l in range(self.w_num):
                if result_list[crop_img_index] != []:
                    for point in result_list[crop_img_index]:
                        px_min = point[0] + y_l * (self.crop_size[0] - self.border)
                        py_min = point[1] + x_l * (self.crop_size[1] - self.border)
                        px_max = point[2] + y_l * (self.crop_size[0] - self.border)
                        py_max = point[3] + x_l * (self.crop_size[1] - self.border)

                    final_list.append([px_min, py_min, px_max, py_max, point[4], point[5]])
                crop_img_index += 1

        result_list_points = self.del_iou_boxes(final_list)
        return result_list_points

    def draw_boxes(self,result_list_points,img):
        for last_point in result_list_points:
            # 绘制最终拼接的检测结果
            cv2.rectangle(img, (int(last_point[0]), int(last_point[1])), (int(last_point[2]), int(last_point[3])),
                          (0, 255, 0), 1, 8)
            cv2.putText(img, str(last_point[4])[:6], (int(last_point[0]), int(last_point[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            cv2.putText(img, str(last_point[5]), (int(last_point[0]), int(last_point[1]+10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        # print("time------------{}".format(time.time() - t0))
        # a = 111

        return img

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



if __name__ == '__main__':
    img_data = cv2.imread(img_path)
    img = cv2.resize(img_data, resize_shape)
    cjpj = CJPJ(crop_size, border, show_rate)
    croped_img_list = cjpj.crop_img(img_data)
    y_list = cjpj.eval_img_list(croped_img_list)
    result = cjpj.pj(y_list, show_rate)
    img_result = cjpj.draw_boxes(result,img)
    cv2.imshow("ppp",img_result)
    cv2.waitKey(0)

    print(result)

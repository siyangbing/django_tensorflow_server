import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import time
from sklearn.cluster import KMeans

img_path = "/home/db/bing/django_tensorflow_server/test_img/kaiguandeng.jpg"
img_resize_shape=(1920, 1080)
model_img_input_size = (640, 480)
saved_model_dir = '/home/db/bing/django_tensorflow_server/kaiguandeng/pb_model/saved_model'
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
# final_label = {"d000":1,"d001":1,"d002":1,"d003":1,"d004":1,"d005":1,"d006":1,"d007":1,"d008":1,"d009":1,"d010":1,"d011":1,
#                "d100":1,"d101":1,"d102":1,"d103":1,"d104":1,"d105":1,"d106":1,"d107":1,
#                "d200":1,"d201":1,"d202":1,"d203":1,"d204":1,"d205":1,"d206":1,"d207":1,
#                "y000":1,"y001":1,"y002":1,"y003":1,"y004":1,"y005":1,"y006":1,"y007":1,
#                "y100":1,
#                "h000":1,}
final_label = ("d000","d001","d002","d003","d004","d005","d006","d007","d008","d009","d010","d011",
               "d100","d101","d102","d103","d104","d105","d106","d107",
               "d200","d201","d202","d203","d204","d205","d206","d207",
               "y000","y001","y002","y003","y004","y005","y006","y007",
               "y100",
               "h000")




y = [np.array([[[0.6194794, 0.7747532, 0.63610065, 0.78724253],
                [0.75446314, 0.7821034, 0.7726782, 0.79639196],
                [0.6225289, 0.7051878, 0.6393924, 0.7178557],
                [0.62376326, 0.66926455, 0.6404876, 0.6820011],
                [0.62932354, 0.5068463, 0.64614326, 0.5184878],
                [0.6209915, 0.7393299, 0.63758, 0.7521612],
                [0.6277768, 0.5705021, 0.64406306, 0.58180124],
                [0.6333259, 0.44383854, 0.6491694, 0.45605463],
                [0.7584651, 0.573682, 0.77535504, 0.5858876],
                [0.69700813, 0.7126549, 0.72138524, 0.73183495],
                [0.6955177, 0.75318414, 0.7201827, 0.7716575],
                [0.7561329, 0.63306785, 0.77311784, 0.6455209],
                [0.6262094, 0.60321474, 0.6426854, 0.6156635],
                [0.6323058, 0.47433788, 0.64830536, 0.4857555],
                [0.7978807, 0.6927414, 0.81497926, 0.70583534],
                [0.75640863, 0.7514579, 0.7734619, 0.7648319],
                [0.75834966, 0.60270774, 0.7748383, 0.6150377],
                [0.62433887, 0.63596445, 0.64114153, 0.64866656],
                [0.7965493, 0.784477, 0.813239, 0.7985259],
                [0.7975741, 0.75398564, 0.8139371, 0.76740885],
                [0.7978055, 0.7229569, 0.8143485, 0.73626924],
                [0.7036697, 0.46560892, 0.72762686, 0.4830245],
                [0.7560162, 0.69016105, 0.77366537, 0.70255953],
                [0.79881287, 0.5750036, 0.8159846, 0.58745086],
                [0.7558391, 0.66138184, 0.77243483, 0.67371273],
                [0.7559418, 0.72062635, 0.7729656, 0.73462737],
                [0.79698414, 0.63365626, 0.8141981, 0.64702225],
                [0.70132864, 0.5408178, 0.7238765, 0.5567388],
                [0.7979158, 0.6044591, 0.81489813, 0.6160956],
                [0.6973103, 0.67403954, 0.7219268, 0.69268125],
                [0.7026188, 0.5030955, 0.7266883, 0.5198343],
                [0.7049396, 0.4292523, 0.72791386, 0.44551215],
                [0.6988867, 0.6338371, 0.7229858, 0.6514548],
                [0.7788705, 0.43009838, 0.8011518, 0.44729158],
                [0.77134275, 0.49426022, 0.8116703, 0.52629876],
                [0.7980961, 0.6622715, 0.8148678, 0.6759963],
                [0.6341178, 0.41364446, 0.6514998, 0.42476234],
                [0.6276424, 0.5360415, 0.6447185, 0.5483799],
                [0.75096744, 0.66153806, 0.7680456, 0.67552394],
                [0.6860716, 0.63078123, 0.7103208, 0.64629596],
                [0.61886734, 0.37351218, 0.6408604, 0.3884819],
                [0.6988867, 0.6338371, 0.7229858, 0.6514548],
                [0.75437677, 0.50351965, 0.7892026, 0.53032863],
                [0.7026188, 0.5030955, 0.7266883, 0.5198343],
                [0.6793002, 0.4012164, 0.7013277, 0.41424435],
                [0.5479996, 0.36405593, 0.5788003, 0.38660812],
                [0.7094424, 0.42860565, 0.7350111, 0.44572237],
                [0.6262094, 0.60321474, 0.6426854, 0.6156635],
                [0.72674936, 0.49717575, 0.75722796, 0.5184079],
                [0.6341178, 0.41364446, 0.6514998, 0.42476234],
                [0.67638063, 0.5653974, 0.7011348, 0.57820547],
                [0.2619016, 0.06361915, 0.32913944, 0.10141398],
                [0.83128536, 0.5472278, 0.86877906, 0.5645208],
                [0.7559418, 0.72062635, 0.7729656, 0.73462737],
                [0.6988867, 0.6338371, 0.7229858, 0.6514548],
                [0.6973103, 0.67403954, 0.7219268, 0.69268125],
                [0.838866, 0.48656562, 0.87923265, 0.51289624],
                [0.6988867, 0.6338371, 0.7229858, 0.6514548],
                [0.72630936, 0.3238294, 0.84051234, 0.515389],
                [0.7558391, 0.66138184, 0.77243483, 0.67371273],
                [0.8196077, 0.74413776, 0.8413612, 0.76282096],
                [0.70132864, 0.5408178, 0.7238765, 0.5567388],
                [0.70132864, 0.5408178, 0.7238765, 0.5567388],
                [0., 0., 1., 0.5019046],
                [0., 0.04433483, 1., 0.45567125],
                [0., 0.04433483, 1., 0.45567125],
                [0., 0.04433483, 1., 0.45567125],
                [0.1641061, 0., 0.8358674, 0.753788],
                [0., 0., 1., 0.5019046],
                [0.1641061, 0., 0.8358674, 0.753788],
                [0.01268527, 0., 0.9873581, 0.615518],
                [0., 0.04433483, 1., 0.45567125],
                [0., 0., 1., 0.5019046],
                [0.01268527, 0., 0.9873581, 0.615518],
                [0.0250414, 0., 0.97499543, 0.6062429],
                [0.1641061, 0., 0.8358674, 0.753788],
                [0.0250414, 0., 0.97499543, 0.6062429],
                [0., 0., 1., 0.5019046],
                [0.0250414, 0., 0.97499543, 0.6062429],
                [0.01268527, 0., 0.9873581, 0.615518],
                [0.01268527, 0., 0.9873581, 0.615518],
                [0.01268527, 0., 0.9873581, 0.615518],
                [0.225764, 0., 0.774227, 0.8671057],
                [0.01268527, 0., 0.9873581, 0.615518],
                [0.225764, 0., 0.774227, 0.8671057],
                [0.225764, 0., 0.774227, 0.8671057],
                [0.1641061, 0., 0.8358674, 0.753788],
                [0.01268527, 0., 0.9873581, 0.615518],
                [0.225764, 0., 0.774227, 0.8671057],
                [0.1641061, 0., 0.8358674, 0.753788],
                [0., 0.49809337, 1., 1.],
                [0., 0., 1., 0.5019046],
                [0., 0.04433483, 1., 0.45567125],
                [0.225764, 0., 0.774227, 0.8671057],
                [0., 0.54432803, 1., 0.95567185],
                [0., 0.54432803, 1., 0.95567185],
                [0., 0.54432803, 1., 0.95567185],
                [0.16412231, 0.24618763, 0.8358841, 1.],
                [0.16412231, 0.24618763, 0.8358841, 1.],
                [0., 0.49809337, 1., 1.]]], dtype=np.float32), np.array(
    [[0.96996194, 0.96767664, 0.9668672, 0.96615267, 0.96584946,
      0.96458817, 0.9615091, 0.96133476, 0.96043164, 0.95929414,
      0.95906353, 0.9585671, 0.95677507, 0.9555607, 0.9550693,
      0.95419854, 0.9537331, 0.9533138, 0.9525844, 0.9499291,
      0.9480826, 0.9479706, 0.94775116, 0.94441867, 0.9421833,
      0.9418723, 0.9398862, 0.9396988, 0.93949556, 0.9351788,
      0.93232924, 0.91843563, 0.90940917, 0.8129661, 0.7507911,
      0.72411376, 0.7090574, 0.64846504, 0.11407492, 0.0940429,
      0.08925384, 0.07076231, 0.0688307, 0.06220767, 0.05306014,
      0.05243781, 0.05021262, 0.04843739, 0.04592046, 0.04573974,
      0.04522941, 0.04488924, 0.04399627, 0.04296693, 0.04150933,
      0.0381147, 0.03775302, 0.03761727, 0.0367628, 0.03650409,
      0.03590038, 0.03482801, 0.03443551, 0.03428447, 0.03424364,
      0.03423801, 0.0342316, 0.03423053, 0.03422925, 0.03422773,
      0.03422558, 0.03422311, 0.0342229, 0.03422242, 0.03421915,
      0.03421718, 0.03421658, 0.03421524, 0.0342133, 0.03421208,
      0.03421107, 0.03420892, 0.03420708, 0.03420439, 0.03420147,
      0.03420034, 0.03419873, 0.03419599, 0.03419462, 0.03419381,
      0.03419337, 0.03418565, 0.03418463, 0.03417832, 0.03415805,
      0.03414771, 0.03414515, 0.03414422, 0.03414401, 0.03414172]],
    dtype=np.float32), np.array([[10., 10., 10., 10., 10., 10., 10., 10., 10., 2., 2., 10., 10.,
                                  10., 10., 10., 10., 9., 10., 10., 10., 2., 10., 10., 10., 10.,
                                  10., 3., 10., 2., 2., 2., 1., 2., 8., 10., 10., 10., 10.,
                                  1., 9., 2., 8., 3., 2., 9., 2., 9., 8., 1., 3., 1.,
                                  8., 2., 10., 1., 8., 3., 2., 1., 2., 2., 9., 4., 7.,
                                  10., 2., 10., 3., 2., 10., 8., 9., 7., 6., 7., 1., 5.,
                                  4., 5., 9., 8., 8., 2., 6., 5., 3., 3., 1., 4., 4.,
                                  1., 6., 9., 7., 10., 2., 2., 10., 3.]], dtype=np.float32),
     np.array([100.], dtype=np.float32)]


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


        # result_findal_list = [[[[c[4], c[5], c[0], c[1], c[2], c[3]] for c in b] for b in a] for a in result_llist]

        # flatten_list = self.list_flatten(result_llist)
        # result_list1 = [[x[2], x[3], x[4], x[5], x[0], x[1]] for x in flatten_list]
        # result_list_points1 = self.del_iou_boxes(result_list1)

        result_findal_list = [[[np.array([c],dtype='float64').tolist() for c in b] for b in a] for a in result_llist]
        # result_findal_list = [
        #     [[np.array([c], dtype='float64').tolist() for c in b] for b in a] for a in
        #     result_llist]
        # result_dict = dict(zip(final_label,self.list_flatten(result_findal_list)))

        print("result_dict_0000000000{}".format(result_findal_list))
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


if __name__ == "__main__":
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_resize_shape)
    # cv2.imwrite("src.jpg",img2)
    # img3 = cv2.resize(img1,(640,480))
    map_location = Map_location(treshold, label_dict, join_label_dict,model_img_input_size)
    img_data_list = map_location.read_img(img_path)
    y_list = map_location.eval_img_list(img_data_list)
    print("wwwwwwwwwwwwwww")
    result_list = map_location.get_location(y_list)
    print(result_list)
    map_location.draw_boxes(result_list,img)
    cv2.imshow("ppp",img)
    # cv2.imwrite("result.jpg",img)
    cv2.waitKey(0)

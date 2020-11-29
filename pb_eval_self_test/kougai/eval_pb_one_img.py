#-*-coding:utf-8-*-
# -*-coding:utf-8-*-
import tensorflow as tf
import cv2
import numpy as np

# from del_cf import Del_Cf

saved_model_dir = "/home/db/dbing/models-master/research/object_detection/ztuiduan/new_models/eval_img/saved_model"
# saved_model_dir = "/home/sucom/Documents/led_lamp_0610/pb_dir/saved_model"
image = cv2.imread("/home/db/文档/口盖/19710514_210826.jpg")
result_img_path = "./kougai.png"

label_dict = {'1.0':'g','2.0':'k'}
color_dict = {'1.0':(255,0,0),'2.0':(0,255,0)}
# label_dict = {'1.0': 'ku', '2.0': 'kd', '3.0': 'km', '4.0': 'u', '5.0': 'd', '6.0': 'l', '7.0': 'r', '8.0': 'm',
#               '9.0': 'k', '10.0': 'g'}
# color_dict = {'1.0': (255, 0, 0), '2.0': (0, 255, 0), '3.0': (0, 0, 255), '4.0': (0, 255, 255), '5.0': (255, 255, 0),
#               '6.0': (155, 155, 0), '7.0': (155, 0, 155), '8.0': (155, 0, 255), '9.0': (155, 255, 255), '10.0': (255, 255, 155)}

src_shape = (2400, 3200)
frame = cv2.resize(image, (640, 480))
# frame = cv2.resize(image,(480,360))
img_shape = frame.shape[0:2]

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess1 = tf.Session(config=config)
with sess1:
    # load pb
    meta_graph_def = tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

    # def deal_one_img(frame):

    img = frame[..., ::-1]
    # img = cv2.resize(img, (480, 480))
    img_array = np.array(img, dtype=float)
    img_array = img_array[np.newaxis, :, :, :]
    input_data = np.array(img_array, dtype=np.float32)
    # get 输入输出
    input = sess1.graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = sess1.graph.get_tensor_by_name('detection_boxes:0')
    detection_score = sess1.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess1.graph.get_tensor_by_name('detection_classes:0')
    num_detections = sess1.graph.get_tensor_by_name('num_detections:0')

    feed_dict = {input: input_data, }
    # for x in range(30):
    #   t0 = time.time()

    y = sess1.run([detection_boxes, detection_score, detection_classes, num_detections], feed_dict=feed_dict)
    location_list = y[0][0]
    score_list = y[1][0]
    class_list = y[2][0]
    num = y[3]
    # print(location_list)
    # print(score_list)
    # print(class_list)
    # print(num)
    k = 0

    result_list = []
    for x in range(len(score_list)):
        if score_list[x] > 0.1:
            box_one = [location_list[x][1], location_list[x][0], location_list[x][3], location_list[x][2],
                       score_list[x], class_list[x]]
            result_list.append(box_one)

    # deal_cf = Del_Cf()
    # result_list = deal_cf.del_iou_boxes(result_list)

    point_list = [[point[1], point[0], point[3], point[2], point[4], point[5]] for point in result_list]

    for point_box in point_list:
        # print()
        # point_1 = (int(point_box[1] * img_shape[1]*5), int(point_box[0] * img_shape[0]*5))
        # point_2 = (int(point_box[3] * img_shape[1]*5), int(point_box[2] * img_shape[0]*5))
        point_1 = (int(point_box[1] * src_shape[1]), int(point_box[0] * src_shape[0]))
        point_2 = (int(point_box[3] * src_shape[1]), int(point_box[2] * src_shape[0]))
        # print(location_list[x])
        # print(score_list[x])
        # print(point_2)
        # location =
        # score = str(score_list[x])
        class_num = str(point_box[5])
        # class_name = str(map_dict[int(class_num)-1])
        cv2.rectangle(image, point_1, point_2, color_dict[str(class_num)], 1, 1)
        # cv2.putText(frame, class_name, point_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(image, label_dict[str(class_num)], point_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color_dict[str(class_num)], 1)
        k += 1

        # return frame

        # result = deal_one_img(frame)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image", image)
    cv2.imwrite(result_img_path, image)
    cv2.waitKey(0)


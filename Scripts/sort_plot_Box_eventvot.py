import cv2
import math
import os
import numpy as np
import glob


video_name = '/recording_2022-10-14_11-00-53'
root = r"D:\Compare_Trackers" + video_name
img_root = root + '/img'
save_root = root + '/vis'
img_list = os.listdir(img_root)
img_list = sorted(img_list)

results_root = r"D:\Compare_Trackers\eventvot_tracking_results"

for i in range(0, 7):
    if i == 0:
        file_name = results_root + '/OSTrack_tracking_result' + video_name + '.txt' # pink
    elif i == 1:
        file_name = results_root + '/HDETrack_tracking_result' + video_name + '.txt'  # yellow
    elif i == 2:
        file_name = results_root + '/MixFormer_tracking_result' + video_name + '.txt'  # blue
    elif i == 3:
        file_name = results_root + '/STARK_tracking_result' + video_name + '.txt'  # Cyan1
    elif i == 4:
        file_name = results_root + '/ATOM_tracking_result' + video_name + '.txt'  # dark
    elif i == 5:
        file_name = root + '/groundtruth.txt'  # green
    elif i == 6:
        file_name = results_root + '/Ours_tracking_result' + video_name + '.txt'  # red


    x = []
    y = []
    w = []
    h = []
    with open(file_name, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 读取整行数据
            if not lines:
                break
                pass

            if i == 5:
                x_1, y_1, w_1, h_1 = [float(i) for i in
                                  lines.split(',')]
            else : 
                x_1, y_1, w_1, h_1 = [float(i) for i in
                                  lines.split('	')]  # 将整行数据分割处理，如果分割符是\t.传入	，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            x.append(x_1)
            y.append(y_1)
            w.append(w_1)
            h.append(h_1)
            pass
        x = np.array(x)
        y = np.array(y)
        w = np.array(w)
        h = np.array(h)
    length = len(open(file_name, 'r').readlines())

    # 1-length帧  k=0 ,k+1帧
    for k, img_name in enumerate(img_list):
        image_path = img_root + '/' + img_name
        image = cv2.imread(image_path)
        first_point = (math.ceil(x[k]), math.ceil(y[k]))
        last_point = (math.ceil(x[k] + w[k]), math.ceil(y[k] + h[k]))
        text_num = '#%04d' % (k+1)
        cv2.putText(image, text_num, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 8)
        if i == 0:
            cv2.rectangle(image, first_point, last_point, (147, 20, 255), 4)  #  pink
        elif i == 1:
            cv2.rectangle(image, first_point, last_point, (0, 255, 255), 4)  #  yellow
        elif i == 2:
            cv2.rectangle(image, first_point, last_point, (255, 0, 0), 4)  #  blue
        elif i == 3:
            cv2.rectangle(image, first_point, last_point, (255, 255, 0), 4)  #  Cyan1
        elif i == 4:
            cv2.rectangle(image, first_point, last_point, (0, 0, 0), 4)  # dark SiamR-CNN
        elif i == 5:
            cv2.rectangle(image, first_point, last_point, (0, 255, 0), 4)  # gt green
        elif i == 6:
            cv2.rectangle(image, first_point, last_point, (0, 0, 255), 4)  # ours red
        
        cv2.imwrite(image_path, image)

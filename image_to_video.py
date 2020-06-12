import cv2
import os
import numpy as np
from PIL import Image

import yolo


#raw_image_dir='D:\MachineLearning11\\UA-DETRAC\\DETRAC-train-data\Insight-MVT_Annotation_Train\\MVI_20012'
raw_image_dir='D:\MachineLearning11\\UA-DETRAC\\DETRAC-test-data\Insight-MVT_Annotation_Test\\MVI_40891'
image_list = os.listdir(raw_image_dir)#列出文件夹下所有的目录与文件

img_size=(416, 416)
vedio_writer = cv2.VideoWriter('D:\\黄迪和\\黄迪和2\\大四上\\智能信息处理导论\\yolo3\\test_all\\MVI_40891-iou0.5-score0.3.mp4', fourcc=cv2.VideoWriter_fourcc(*'avc1'), fps=40, frameSize=img_size)

yolo1=yolo.YOLO()

for image_path in image_list:
    #image = cv2.imread(raw_image_dir +'/'+image_path)
    image = Image.open(raw_image_dir +'/'+image_path)

    r_image = yolo1.detect_image(image)
    r_image = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
    r_image= cv2.resize(r_image, img_size)

    vedio_writer.write(r_image)
    #cv2.imshow('frame',r_image)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

vedio_writer.release()
#cv2.waitKey(0)
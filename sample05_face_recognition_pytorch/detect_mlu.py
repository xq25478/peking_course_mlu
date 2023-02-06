"""
使用说明
对存放于images下的每一张图片进行逐一人脸检测和人脸识别对比

1.人脸检测功能 
通过mtcnn模型进行人脸检测,将人脸在检测图片上进行标注
人脸检测图片存放于output/detect_img下
人脸分割

2.人脸对比功能
通过facenet模型对人脸特征向量进行提取,与预先数据库中具有权限的人脸特征向量进行对比，
输出数值(0-1)表明最小特征向量距离,值越低表明与数据库的人脸特征越相似，具有权限的概率越大
"""

from face import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 是否需要重新生成预定人脸特征数据库 
# 存放权限的人脸照片路径 datas/faces_lib
# 存放权限的人脸特征路径 datas/faces_feature

GENGRATE_FACE_LIB = True
faces_lib_path = 'datas/faces_lib'
faces_feature_path = 'datas/faces_feature'

if __name__ == "__main__":
    fs = FaceSystem()

    if GENGRATE_FACE_LIB :
        features = []
        feature_images_paths = []
        for face_path in os.listdir(faces_lib_path):
            print(face_path)
            #获取图片
            image = Image.open(os.path.join(faces_lib_path,face_path))
            #检测人脸
            result = fs.face_detect(image)
            #默认数据库图片只有一个人脸 因此只提取第一个box
            image = np.array(image)
            box = result[0]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face = image[y1: y2, x1: x2, :]
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            feature = fs.get_face_feature(face)
            features.append(feature)
            feature_images_paths.append(face_path)
            
        np.save(os.path.join(faces_feature_path,"feature"),features)
        np.save(os.path.join(faces_feature_path,"image_path"),feature_images_paths)
            
    # load faces_feature database
    features = np.load(os.path.join(faces_feature_path,"feature.npy"))
    feature_images_paths = np.load(os.path.join(faces_feature_path,"image_path.npy"))
    
    detect_images_path = 'images'
    save_face_path = 'output/faces_img'
    save_detect_path = 'output/detect_img'
    save_recognition_path = 'output/recognition_img'
    
    for detect_img in os.listdir(detect_images_path):
        
        # 人脸检测 当前示例仅支持1个人脸
        print("Face Detecting and Recognizing: ",detect_img)
        image = Image.open(os.path.join(detect_images_path,detect_img))
        result = fs.face_detect(image)
        fs.save_faces(image, result,save_path=os.path.join(save_face_path,detect_img))
        fs.save_face_boxes(image,result,save_path=os.path.join(save_detect_path,detect_img))

        # 人脸对比
        # 1.提取特征 对一个box进行匹配
        for i in range(len(result)):
            image = np.array(image)
            box = result[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face = image[y1: y2, x1: x2, :]
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            feature = fs.get_face_feature(face)
            min_value = 1.0
            min_value_image_path = ''
            # 2.对比特征 找到与数据库当中所有图片最小的特征向量距离图片
            for idx in range(len(features)):
                dist = fs.feature_compare(feature, features[idx])
                if dist < min_value:
                    min_value = dist
                    min_value_image_path = feature_images_paths[idx]
            cv2.putText(image,str(min_value_image_path)+" "+str(min_value),(x2,y1),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255, 0), 1)
        
        mask = cv2.imread(os.path.join(faces_lib_path,min_value_image_path))
        ratio = 3
        r,c,ch = mask.shape
        mask = cv2.resize(mask,(c//ratio,r//ratio))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image[0:r//ratio,0:c//ratio,:] = mask
        cv2.imwrite(os.path.join(save_recognition_path,detect_img),image)

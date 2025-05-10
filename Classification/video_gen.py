import cv2
import os

# 图像文件夹路径和视频保存路径
image_folder = './SCA_boundary_vis/'
video_name = './link0-link1.avi'

# 获取图像列表并按文件名排序
images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
images.sort()

# 读取第一张图像，获取图像尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 创建视频编写器对象
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))

# 逐帧写入视频
for loop in range(400):
    video.write(cv2.imread(os.path.join(image_folder, f'1_01_{loop+1}.png')))
    print(loop)

# 释放视频编写器和关闭视频文件
video.release()
cv2.destroyAllWindows()

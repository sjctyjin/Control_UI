# import time
# import cv2
# import numpy as np
# import pyrealsense2 as rs
# from PIL import Image
# import serial
# from ultralytics import YOLO  # 将YOLOv8导入到该py文件中
# # import torch
# #
# # print(torcu.cuda.is_available())
# ''' 深度相机 '''
# pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
# config = rs.config()  # 定义配置config
#
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流
#
# pipe_profile = pipeline.start(config)  # streaming流开始
# align = rs.align(rs.stream.color)
# #
# Auto_Mode_switch = False
# terminate_thread = False
# image_queue = []
# image_queue_buffer = []
# labels_api = []
# """
# Arduino
# """
# # port = 'COM7'  # 根据您的情况更改端口号，例如在Windows上可能是 'COM3'
# # baudrate = 115200  # 波特率需要与Arduino上的设置一致
# # Arduino = serial.Serial(port, baudrate, timeout=0.2)
#
# def process_images():#OAK-D
#     global image_queue
#     global image_queue_buffer
#     global terminate_thread
#     global labels_api
#     global threshold_value
#     global Auto_Mode_switch
#
#     model = YOLO(r"D:\技術文件\Python專案\ultralytics\yolov8n.pt")  # 加载权重文件，如需要更换为自己训练好的权重best.pt即可
#     set_move = -1  # 初始位移状态
#     # 设置计时器
#     start_time = time.time()
#     interval = 0.01  # 间隔时间（秒）
#     counter = 0
#     fps = 0
#     color = (255, 255, 255)
#     imOut = []
#     try:
#         while True:
#             if terminate_thread:
#                 if Auto_Mode_switch:
#                     depth_intrin, img_color, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数
#                     # 检查是否达到间隔时间
#                     if time.time() - start_time >= interval:
#                         start_time = time.time()  # 重置计时器
#                         # source = [img_color]
#                         source = img_color
#                         # 调用YOLOv8中的推理，还是相当于把d435i中某一帧的图片进行detect推理
#                         results = model.predict(source, save=False, show_conf=False)
#                         height = source.shape[0]
#                         width = source.shape[1]
#                         a = 0
#                         labels_api = []
#                         for result in results:  # 相当于都预测完了才进行的打印目标框，这样就慢了
#                             boxes = result.boxes.xywh.tolist()
#                             tags = result.names  # 获取标签
#                             # im_array = result.plot()  # plot a BGR numpy array of predictions
#                             im_array = source  # plot a BGR numpy array of predictions
#
#                             for i in range(len(boxes)):
#                                 # if tags[result.boxes[i].cls[0].item()] == 'cup':  # 筛选出杯子
#                                 x1, y1, x2, y2 = result.boxes[i].data[0][0].item(), result.boxes[i].data[0][
#                                     1].item(),result.boxes[i].data[0][2].item(), result.boxes[i].data[0][
#                                                      3].item()
#                                 confidence = int(result.boxes[i].conf[0].item() * 100)  # 置信度
#                                 # category = result.boxes[i].cls[0].item()  # 类别
#
#                                 w = int(x2)
#                                 h = int(y2)
#                                 x = int(x1)
#                                 y = int(y1)
#                                 # cv2.rectangle(im_array, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 画出目标框
#                                 cv2.rectangle(im_array, (x, y), (w, h), (122, 61, 0), 5)
#                                 cv2.rectangle(colorizer_depth, (x, y), (w, h), (255, 255, 255), 3)
#
#                                 # 設定透明度
#                                 alpha = 0.25
#                                 overlay = im_array.copy()
#                                 cv2.rectangle(overlay, (x, y), (w, h), (0, 200, 0), -1)  # 區內上色
#                                 cv2.addWeighted(overlay, alpha, im_array, 1 - alpha, 0, im_array)
#                                 """
#                                     =============== 背景 ===============
#                                 """
#                                 x_offset = 10
#                                 y_offset = 20
#                                 fontFace = cv2.FONT_HERSHEY_SIMPLEX
#                                 fontScale = 0.4
#                                 thickness = 1
#                                 labelSize = cv2.getTextSize(tags[result.boxes[i].cls[0].item()], fontFace,
#                                                             fontScale, thickness)
#                                 _x2 = x + labelSize[0][0]  # topright x of text
#                                 _y2 = y - labelSize[0][1]  # topright y of text
#                                 """
#                                     Lable
#                                 """
#
#                                 cv2.rectangle(im_array, (x - 3, y - y_offset - 30), (_x2 + 10, _y2 - y_offset),
#                                               (97, 48, 0),
#                                               cv2.FILLED)  # 畫text內框
#                                 cv2.rectangle(im_array, (x - 3, y - y_offset + 20), (_x2 + 10, _y2 - y_offset),
#                                               (122, 61, 0),
#                                               -1)  # 畫全部外框
#                                 cv2.circle(im_array, (int((w + x) / 2), int((y + h) / 2)), 2, (255, 0, 0), -1)
#
#                                 ux, uy = int(boxes[i][0]), int(boxes[i][1])  # 计算像素坐标系的x
#                                 dis = aligned_depth_frame.get_distance(ux, uy)
#                                 camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
#                                 camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
#                                 camera_xyz = np.array(list(camera_xyz)) * 1000
#                                 camera_xyz = list(camera_xyz)
#
#                                 cv2.circle(im_array, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
#                                 cv2.putText(im_array, str(camera_xyz), (ux + 20, uy + 10), 0, 0.5,
#                                             [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标
#                                 labels_api.append(
#                                     {0: [f"{i} - {str(tags[result.boxes[i].cls[0].item()])}", f" {confidence}% , "
#                                                                                               f"X: {int(camera_xyz[0])} mm,"
#                                                                                               f"Y: {int(camera_xyz[1])} mm,"
#                                                                                               f"Z: {int(camera_xyz[2])} mm"]})
#
#                         cv2.circle(im_array, (im_array.shape[1] // 2, im_array.shape[0] // 2), 5, (0, 0, 255), 5)  # 标出中心点
#                         cv2.circle(im_array, (im_array.shape[1] // 2 + 66, im_array.shape[0] // 2 + 140), 1, (0, 0, 255),
#                                    1)  # 标出中心点
#                         cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
#                                                            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
#                         colorizer_depth = cv2.applyColorMap(colorizer_depth, cv2.COLORMAP_JET)
#                         colorizer_depth = cv2.resize(colorizer_depth, (640, 480))
#                         imOut = np.hstack((im_array, colorizer_depth))
#                         cv2.resizeWindow('detection', 640, 480)
#                         cv2.imshow('detection', im_array)
#                         cv2.imshow('detectio2', imOut)
#                         # cv2.waitKey(0)
#
#                     key = cv2.waitKey(1)
#                     # Press esc or 'q' to close the image window
#                     if key == 27 or key == ord('k'):
#                         set_move = 1
#                     if key & 0xFF == ord('q') or key == 27:
#                         cv2.destroyAllWindows()
#                         pipeline.stop()
#                         break
#             else:
#                 break
#             image_queue.append(imOut)
#             image_queue_buffer = imOut
#     finally:
#         # Stop streaming
#         pipeline.stop()
#
#
#
# def generate_frame():
#     global image_queue
#     global terminate_thread
#     global image_queue_buffer
#     try:
#         while True:
#             if not image_queue:
#                 continue
#             # Get the latest processed image from the queue
#             if terminate_thread:
#                 if image_queue != []:
#                     try:
#                         frame = image_queue.pop()
#                         # Encode the image to JPEG format
#                         _, img_encoded = cv2.imencode('.jpg', frame)
#                         image_queue = []
#                     except:
#                         print("wrong")
#             else:
#                 if image_queue != []:
#                     try:
#                         frame = image_queue[0]
#                         # Encode the image to JPEG format
#                         _, img_encoded = cv2.imencode('.jpg', frame)
#                     except:
#                         print("wrong")
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')
#     except:
#         return "Empty"
from flask import Flask, Response, render_template
import cv2
import time
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from ultralytics import YOLO  # 将YOLOv8导入到该py文件中
app = Flask(__name__)

# 打開攝像頭（默認攝像頭為 0）



def get_aligned_images(pipeline,align):
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

    depth_colormap = cv2.applyColorMap \
        (cv2.convertScaleAbs(img_depth, alpha=0.008)
         , cv2.COLORMAP_JET)

    return depth_intrin, img_color, aligned_depth_frame

def generate_frame():
    ''' 深度相机 '''
    pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
    config = rs.config()  # 定义配置config

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流

    pipe_profile = pipeline.start(config)  # streaming流开始
    align = rs.align(rs.stream.color)
    model = YOLO(r"D:\技術文件\Python專案\ultralytics\yolov8n.pt")  # 加载权重文件，如需要更换为自己训练好的权重best.pt即可

    # 设置计时器
    start_time = time.time()
    interval = 0.01  # 间隔时间（秒）
    try:
        while True:
            depth_intrin, img_color, aligned_depth_frame = get_aligned_images(pipeline,align)  # 获取对齐图像与相机参数
            # 检查是否达到间隔时间
            if time.time() - start_time >= interval:
                start_time = time.time()  # 重置计时器
                source = [img_color]

                # 调用YOLOv8中的推理，还是相当于把d435i中某一帧的图片进行detect推理
                results = model.predict(source, save=False, show_conf=False)

                for result in results:  # 相当于都预测完了才进行的打印目标框，这样就慢了
                    boxes = result.boxes.xywh.tolist()
                    im_array = result.plot()  # plot a BGR numpy array of predictions

                    for i in range(len(boxes)):
                        ux, uy = int(boxes[i][0]), int(boxes[i][1])  # 计算像素坐标系的x
                        dis = aligned_depth_frame.get_distance(ux, uy)
                        camera_xyz = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
                        camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                        camera_xyz = np.array(list(camera_xyz)) * 1000
                        camera_xyz = list(camera_xyz)

                        cv2.circle(im_array, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
                        cv2.putText(im_array, str(camera_xyz), (ux + 20, uy + 10), 0, 0.5,
                                    [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标

                # cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                #                                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                # cv2.resizeWindow('detection', 640, 480)
                # cv2.imshow('detection', im_array)
                cv2.waitKey(1)
                ret, buffer = cv2.imencode('.jpg', im_array)
                frame = buffer.tobytes()

                # 使用多部分數據來傳輸圖像
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # key = cv2.waitKey(1)
            # # Press esc or 'q' to close the image window
            # if key & 0xFF == ord('q') or key == 27:
            #     cv2.destroyAllWindows()
            #     pipeline.stop()
            #     break
    finally:
        # Stop streaming
        pipeline.stop()

        # # 讀取攝像頭畫面
        # success, frame = camera.read()
        # if not success:
        #     break
        # else:
        #     # 將幀轉換為 JPEG 格式
        #     ret, buffer = cv2.imencode('.jpg', frame)
        #     frame = buffer.tobytes()
        #
        #     # 使用多部分數據來傳輸圖像
        #     yield (b'--frame\r\n'
        #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 影像畫面路由
@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 主頁面路由
@app.route('/')
def index():
    return render_template('index.html')  # 簡單的 HTML 來顯示視頻流


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

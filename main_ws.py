from flask import Flask, request, abort,jsonify,render_template, Response,session
import json
import datetime
import time
from flask_cors import CORS
import struct
import os
import threading
import time
import serial  # 用於讀取串口
import pyrealsense2 as rs
from PIL import Image
from ultralytics import YOLO  # 将YOLOv8导入到该py文件中
import numpy as np
import cv2
from flask_socketio import SocketIO
import base64
import eventlet
import secrets
# from camera import process_images,generate_frame  # 假設 generate_frame 已經定義在 camera.py 中

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 這個密鑰應該是隨機且保密的

socketio = SocketIO(app, async_mode='eventlet')
# 定義文件路徑
STATIC_FOLDER = os.path.join(app.root_path, 'static')
JSON_FILE_PATH = os.path.join(STATIC_FOLDER, 'Argument.json')

# 模擬機器狀態字典
Arduino = None
Emm42 = None
MKS_Aspina = None

machine_status = {
    "X": 0,
    "Y": 0,
    "Z": 0,
    "W": 0,
    "mode": 1,
    "max_value" : "False",
    "zeroing": "False",
    "stop" : "False",
    "cam_Ready" : "False",
    "moving_status": "False",
}
Auto_Mode_switch = True
terminate_thread = False
image_queue = []
image_queue_buffer = []
labels_api = []
set_pos = -1      #旋轉姿態 4 種
height_limit = 1 #高度限制開關
threshold_value = 50  # 閾值
modelname = "Fruit"
# modelname = "yolov8n"
step_move_state = 0
camera_running = False
current_user = None  # 用來記錄當前使用攝像頭的使用者
baudrate = 115200  # 波特率需要与Arduino上的设置一致
"""
Arduino
"""
# port = '/dev/ttyACM0'
port = 'COM16'  # 根据您的情况更改端口号，例如在Windows上可能是 'COM3'
# Arduino = serial.Serial(port, baudrate, timeout=0.1)
"""
Emm42
"""
# port = '/dev/ttyUSB0'  # 根据您的情况更改端口号，例如在Windows上可能是 'COM3'
port = 'COM11'  # 根据您的情况更改端口号，例如在Windows上可能是 'COM3'
# Emm42 = serial.Serial(port, baudrate, timeout=0.2)
"""
MKS 57Servo : Aspina
"""
# port = 'COM19'  # 根据您的情况更改端口号，例如在Windows上可能是 'COM3'
# MKS_Aspina = serial.Serial(port, baudrate, timeout=0.2)


@app.route('/', methods=['GET'])
def home():
    if 'user_id' not in session:
        session['user_id'] = request.remote_addr
    print(session['user_id'])
    return render_template(f'WS_index.html')
"""
/////////////////////////////////////////////////   EMM42 旋轉控制     /////////////////////////////////////////////////////////////////////////////////////
"""
def calculate_crc(data):
    # 簡單的累加和取反
    crc = sum(data) & 0xFF  # 累加和取低 8 位
    return crc
def zero_angle(ser):
    ser.write(b'\x01\x32\x6B')#讀取脈衝
    data = ser.readall()
    current_angel = struct.unpack('>I', data[3:7])[0]
    degree = current_angel // 44.4
    if data[2] == 1:
        degree = degree * -1

    # print((struct.unpack('>I', data[1:-1])[0]))
    # print((struct.unpack('>I', data[1:-1])[0]) // 44.4)

    print("脈衝數值 : " , current_angel)
    print("角度 : " , degree)

    if degree < 0:
    # 发送指令
        send_pos = [0x01, 0xfd, 0x00, 0x00, 0x64, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6B]
        # send_pos = [0x01, 0xfd, 0x10, 0x1C, 0x00, 0x00, 0x00, 0x00, 0x6B]
        send_pos[6:10] = struct.pack('>I', abs(current_angel))
        print("小於0 : ",send_pos)
        print("小於0 : ",bytes(send_pos))
        ser.write(bytes(send_pos))
    else:
        send_pos = [0x01, 0xfd, 0x01, 0x00, 0x64, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6B]
        send_pos[6:10] = struct.pack('>I', abs(current_angel))
        print("大於0 : ",send_pos)
        ser.write(bytes(send_pos))
    ser.readall()
    time.sleep(2)  # 等待Arduino响应
#設定旋轉四位置
def set_degress(ser,mode):
    global height_limit
    global set_pos
    global machine_status
    # 0: 原點 1: 右邊 2: 後面 3: 左邊
    ser.readall()
    degree_list = [0,90,180,-90]
    ser.write(b'\x01\x32\x6B')  # 讀取脈衝
    data = ser.readall()
    current_angel = struct.unpack('>I', data[3:7])[0]
    degree = current_angel // 44.4
    if data[2] == 1:
        degree = degree * -1
        current_angel = current_angel * -1
    print("歸零 : ", (current_angel * 360) / 65535)
    print("歸零 : ", current_angel)
    print("角度 : ", degree)

    time.sleep(0.1)


    if data[2] == 1:
        # print(decode_degree - 65536 ** 2)
        # current_degree = ((decode_degree - 65536 ** 2) * 360) / 65536
        print("角度負值 : ", degree)
    else:
        # current_degree = ((decode_degree * 360)) / 65536
        print("角度 : ", degree )
    #高度限制確認
    if height_limit == 1:
        print("高度限制，無法設定")
        return
    if set_pos == 2 or mode == 0:
        if height_limit == 1:
            print("高度限制，無法設定")
            return
    elif mode == 2:
        if height_limit == 1:
            print("高度限制，無法設定")
            return
    #模式選擇
    speed = 0x78#速度設定
    acceleration = 0x50#速度設定
    Foward = 0x10   # 正反轉(順時針)
    Backward = 0x00 # 正反轉(逆時針)
    cudeg = degree
    machine_status["mode"] = mode
    if mode == 0:
        print("歸零----------")
        zero_angle(ser)
    elif mode == 1:
        # -90度以下 情況
        if cudeg < 0:
            print("角度脈衝值 : ",current_angel)
            send_pos = [0x01, 0xfd, Backward,0x00, speed,acceleration, 0x00, 0x00, 0x00, 0x00,0x00, 0x00, 0x6B]
            send_pos[6:10] = struct.pack('>I', abs(current_angel  + int(degree_list[mode]*44.4)))
            print("移動到右邊-小於0 : ", send_pos)
            print("小於0 : ", bytes(send_pos))
            print("小於0 : ", abs(current_angel  + int(degree_list[mode]*44.4)))
            ser.write(bytes(send_pos))
        # 90度以上 情況
        elif cudeg > 90:
            print("角度脈衝值 : ", current_angel)
            send_pos = [0x01, 0xfd, Foward,0x00, speed,acceleration, 0x00, 0x00, 0x00, 0x00,0x00, 0x00, 0x6B]
            print("移動到右邊-大於90 : ",current_angel + (degree_list[mode]*44.4))
            send_pos[6:10] = struct.pack('>I', abs(current_angel + int(degree_list[mode]*44.4)))
            ser.write(bytes(send_pos))
            print("移動到右邊-大於90 : ", send_pos)
        # 0~90度 情況
        else:
            print("角度脈衝值 : ",current_angel)
            send_pos = [0x01, 0xfd, Foward,0x00, speed,acceleration, 0x00, 0x00, 0x00, 0x00,0x00, 0x00, 0x6B]
            print("移動到右邊-0~90 : ", (degree_list[mode] ))
            send_pos[6:10] = struct.pack('>I', abs(current_angel+int(degree_list[mode]*44.4)))
            print("移動到右邊-0~90 : ", send_pos)
            ser.write(bytes(send_pos))
    elif mode == 2:
        if cudeg < 0:
            send_pos = [0x01, 0xfd, Foward, 0x00, speed, acceleration, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6B]
            send_pos[6:10] = struct.pack('>I', abs(current_angel + int(degree_list[mode] * 44.4)))
            print("移動到後面-小於0 : ", current_angel + (degree_list[mode] * 44.4))
            ser.write(bytes(send_pos))
        # 90度以上 情況
        elif cudeg > 90:
            print("角度脈衝值 : ", current_angel)
            send_pos = [0x01, 0xfd, Backward, speed, acceleration, 0x00, 0x00, 0x00, 0x6B]
            send_pos[6:8] = struct.pack('>h', abs(current_angel + int(degree_list[mode] * 44.4)))
            print("移動到後面-等於180 : ", send_pos)
        # 0~90度 情況
        else:
            send_pos = [0x01, 0xfd, Foward, 0x00, speed, acceleration, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6B]
            send_pos[6:10] = struct.pack('>I', abs(current_angel + int(degree_list[mode] * 44.4)))
            print("移動到後面-0~90 : ", current_angel + (degree_list[mode] * 44.4))
            ser.write(bytes(send_pos))
    elif mode == 3:
        print("當前脈衝數 : ",current_angel)
        print("當前脈角度 : ",cudeg)
        if cudeg < 0:
            print("脈衝設定值 : ",abs(current_angel + int(degree_list[mode] * 44.4)))
            send_pos = [0x01, 0xfd, Backward, 0x00, speed, acceleration, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6B]
            send_pos[6:10] = struct.pack('>I', abs(current_angel + int(degree_list[mode] * 44.4)))
            print("移動到左邊-小於0 : ", send_pos)
            ser.write(bytes(send_pos))
        else:
            send_pos = [0x01, 0xfd, Backward, 0x00, speed, acceleration, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6B]
            send_pos[6:10] = struct.pack('>I', abs(current_angel + int(degree_list[mode] * 44.4)))
            print("移動到左邊-0~180 : ", send_pos)
            ser.write(bytes(send_pos))
    elif mode == 4:#不使能
        ser.write(b'\x01\xf3\xab\00\00\x6B')
        ser.readall()
        print("不使能")
    elif mode == 5:#使能
        ser.write(b'\x01\xf3\xab\01\00\x6B')
        ser.readall()
        print("使能")
    elif mode == 6:#角度清零
        ser.write(b'\x01\x0A\x6D\x6B')
        ser.readall()
        print("角度清零")
    ser.readall()
"""
////////////////////////////////////////////////    三軸移動    //////////////////////////////////////////////////////////////////////////////////////
"""
def Axis_move(ser2,sx=0,sy=0,sz=0,sw=0):
    """
    maxX = 210.04
    maxY = 189.65
    maxZ = 277.24
    maxW = 359.99
    :return:
    """
    global machine_status
    global step_move_state
    # time.sleep(1)  # 等待Arduino响应
    # data = ser2.readall().decode()  #
    if sx != 0 or sy != 0 or sz != 0 or sw != 0:
        print("發送移動指令")
        ser2.write(bytes(f"{sx},{sy},{sz},{sw}\n".encode()))  # 发送'hello'指令
        time.sleep(1)
    previous_line = ""
    count_time = 0
    while True:
        if count_time >= 50:
            step_move_state = 0
            machine_status["moving_status"] = 0
            break
        if ser2.in_waiting > 0:
            response = ser2.readline().decode().strip()
            print("Arduino:", response)
            if response == "DONE":
                x = float(previous_line.split(',')[0].split(':')[1])
                y = float(previous_line.split(',')[1].split(':')[1])
                z = float(previous_line.split(',')[2].split(':')[1])
                w = float(previous_line.split(',')[3].split(':')[1])
                machine_status["X"] = x
                machine_status["Y"] = y
                machine_status["Z"] = z
                machine_status["W"] = w
                step_move_state = 0
                machine_status["moving_status"] = 0
                break
            previous_line = response
        count_time += 1
        time.sleep(0.1)

def GET_POS(ser2):
    global height_limit
    global step_move_state
    previous_line = ""
    time.sleep(1)  # 等待Arduino响应
    ser2.write(b'GETPOS\n')  # 发送'hello'指令
    time.sleep(0.2)  # 等待Arduino响应
    while True:
        # 從串口讀取一行數據
        current_line = ser2.readline().decode().strip()

        if current_line:
            # print(f"Current Line: {current_line}")
            # 如果讀取到 "Done"，則回溯上一段字串
            if current_line == "DONE":
                # print(f"Previous Line before 'Done': {previous_line}")
                step_move_state = 0
                machine_status["moving_status"] = 0
                break
            # 更新 previous_line 為當前讀取的數據
            previous_line = current_line

        else:
            break
    return previous_line


def ZMAX(ser2):
    global height_limit
    ser2.write(b'StepperZ+ 10000\n')  # 发送'hello'指令

def EMSTOP(ser2):
    ser2.write(b'stop\n')  # 发送'hello'指令

# 計算反饋手臂座標
@app.route('/robotcoord', methods=['POST'])
def robotcoord():
    global labels_api
    global Arduino
    global step_move_state

    data_res = request.get_json()
    print(data_res['classname'])
    print(int(data_res['camx']))
    print(int(data_res['camy']))
    print(int(data_res['camz']))
    camx = int(data_res['camx'])
    camy = int(data_res['camy'])
    camz = int(data_res['camz'])
    x = float(data_res['coordx'])
    y = float(data_res['coordy'])
    z = float(data_res['coordz'])
    w = float(data_res['coordw'])

    # data = GET_POS(Arduino)
    # print(data)
    # x = float(data.split(',')[0].split(':')[1])
    # y = float(data.split(',')[1].split(':')[1])
    # z = float(data.split(',')[2].split(':')[1])
    current_base_pos = (x, y, z)
    if camz > 300:
        print("超出手臂範圍")
    target_camera_coords = (camx, camy, camz)
    new_base_pos = calculate_new_base_position(current_base_pos, target_camera_coords)

    print(f"原始座標: {current_base_pos}")
    print(f"相機座標: {target_camera_coords}")
    print(f"相機座標: {target_camera_coords}")
    print(f"新的基座目標位置 {new_base_pos}")
    print(f"新的基座目標位置X: {new_base_pos[0]}")
    print(f"新的基座目標位置Z: {new_base_pos[2]}")
    # set_move = 0
    if x > 250 or y > 200 or z > 300:
        print("超過範圍")
        return jsonify({"move": f"fail"}), 200
    elif step_move_state == 1:
        return jsonify({"move": f"busy"}), 200
    else:
        print("開始移動")

    # Axis_move(Arduino, new_base_pos[0], y, new_base_pos[2], 0)
    # ser2.write(b'GETPOS\n')  # 发送'hello'指令
    # Arduino.write(bytes(f"SETYPOS {new_base_pos[2]}\n".encode()))  # 发送'hello'指令
    return jsonify({"X":new_base_pos[0],"Y":new_base_pos[1],"Z":new_base_pos[2]}), 200

"""
////////////////////////////////////////////////    深度相机     //////////////////////////////////////////////////////////////////////////////////////

"""


def get_aligned_images(pipeline,align):
    colorizer  = rs.colorizer()

    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
    colorizer_depth = np.asanyarray(colorizer.colorize(frames.get_depth_frame()).get_data())
    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

    depth_colormap = cv2.applyColorMap \
        (cv2.convertScaleAbs(img_depth, alpha=0.008)
         , cv2.COLORMAP_JET)

    return depth_intrin, img_color, aligned_depth_frame,colorizer_depth

def process_images():
    global image_queue
    global image_queue_buffer
    global terminate_thread
    global labels_api
    global threshold_value
    global modelname
    global threshold_value

    print(modelname)

    model = YOLO(r"static\models\{}.pt".format(modelname))  # 加载权重文件，如需要更换为自己训练好的权重best.pt即可
    # 设置计时器
    start_time = time.time()
    interval = 0.01  # 间隔时间（秒）
    print("啟動")
    imOut = []
    try:
        pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
        config = rs.config()  # 定义配置config
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流
        pipe_profile = pipeline.start(config)  # streaming流开始
        align = rs.align(rs.stream.color)
        print("啟動中。。。")
        check_start = 0
        while True:
            # print("start")
            # print("terminate_thread 狀態 : ",terminate_thread)
            # print("Auto_Mode_switch 狀態 : ",Auto_Mode_switch)
            if terminate_thread:
                if check_start == 0:
                    print("偵測啟動")
                depth_intrin, img_color, aligned_depth_frame, colorizer_depth = get_aligned_images(pipeline,
                                                                                                   align)  # 获取对齐图像与相机参数
                if check_start == 0:
                    print("讀取串流",img_color)
                # 检查是否达到间隔时间
                if time.time() - start_time >= interval:

                    start_time = time.time()  # 重置计时器

                    # source = [img_color]
                    # source = img_color

                    # 调用YOLOv8中的推理，还是相当于把d435i中某一帧的图片进行detect推理
                    results = model.predict(img_color, save=False, show_conf=False, verbose=False)
                    if check_start == 0:
                        print("讀取模型完成 : ",results)
                    labels_api = []
                    for result in results:  # 相当于都预测完了才进行的打印目标框，这样就慢了
                        boxes = result.boxes.xywh.tolist()
                        tags = result.names  # 获取标签
                        # im_array = result.plot()  # plot a BGR numpy array of predictions
                        im_array = img_color  # plot a BGR numpy array of predictions

                        for i in range(len(boxes)):
                            # if tags[result.boxes[i].cls[0].item()] == 'cup':  # 筛选出杯子
                            x1, y1, x2, y2 = result.boxes[i].data[0][0].item(), result.boxes[i].data[0][
                                1].item(), result.boxes[i].data[0][2].item(), result.boxes[i].data[0][
                                                 3].item()
                            confidence = int(result.boxes[i].conf[0].item() * 100)  # 置信度
                            # category = result.boxes[i].cls[0].item()  # 类别
                            if confidence >= threshold_value:
                                w = int(x2)
                                h = int(y2)
                                x = int(x1)
                                y = int(y1)
                                # cv2.rectangle(im_array, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 画出目标框
                                # cv2.rectangle(im_array, (x, y), (w, h), (122, 61, 0), 5)
                                cv2.rectangle(colorizer_depth, (x, y), (w, h), (255, 255, 0), 3)

                                # 設定透明度
                                alpha = 0.4
                                overlay = im_array.copy()
                                if result.boxes[i].cls[0].item() == 2:
                                    cv2.rectangle(overlay, (x, y), (w, h), (0, 250, 0), -1)  # 區內上色
                                    cv2.rectangle(im_array, (x, y), (w, h), (0, 250, 0), 5)
                                else:
                                    cv2.rectangle(overlay, (x, y), (w, h), (0, 0, 250), -1)  # 區內上色
                                    cv2.rectangle(im_array, (x, y), (w, h), (0, 0, 250), 5)

                                cv2.addWeighted(overlay, alpha, im_array, 1 - alpha, 0, im_array)
                                """
                                    =============== 背景 =============== 
                                """
                                x_offset = 10
                                y_offset = 20
                                fontFace = cv2.FONT_HERSHEY_SIMPLEX
                                fontScale = 0.4
                                thickness = 1
                                labelSize = cv2.getTextSize(tags[result.boxes[i].cls[0].item()], fontFace,
                                                            fontScale, thickness)
                                _x2 = x + labelSize[0][0]  # topright x of text
                                _y2 = y - labelSize[0][1]  # topright y of text
                                """
                                    Lable
                                """
                                if result.boxes[i].cls[0].item() == 2:

                                    cv2.rectangle(im_array, (x - 3, y - y_offset - 30), (_x2 + 10, _y2 - y_offset),
                                                  (104, 178, 1),
                                                  cv2.FILLED)  # 畫text內框
                                    cv2.rectangle(im_array, (x - 3, y - y_offset + 20), (_x2 + 10, _y2 - y_offset),
                                                  (178, 254, 77),
                                                  -1)  # 畫全部外框
                                else:
                                    cv2.rectangle(im_array, (x - 3, y - y_offset - 30), (_x2 + 10, _y2 - y_offset),
                                                  (0, 0, 173),
                                                  cv2.FILLED)  # 畫text內框
                                    cv2.rectangle(im_array, (x - 3, y - y_offset + 20), (_x2 + 10, _y2 - y_offset),
                                                  (82, 82, 255),
                                                  -1)  # 畫全部外框
                                cv2.circle(im_array, (int((w + x) / 2), int((y + h) / 2)), 2, (255, 0, 0), -1)

                                ux, uy = int(boxes[i][0]), int(boxes[i][1])  # 计算像素坐标系的x
                                dis = aligned_depth_frame.get_distance(ux, uy)
                                camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
                                camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                                camera_xyz = np.array(list(camera_xyz)) * 1000
                                camera_xyz = list(camera_xyz)

                                cv2.circle(im_array, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
                                cv2.putText(im_array, str(camera_xyz), (ux + 20, uy + 10), 0, 0.5,
                                            [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标
                                cv2.putText(im_array, f'{i}-{tags[result.boxes[i].cls[0].item()]}', (x + 5, y - 35),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                                            cv2.LINE_AA)
                                cv2.putText(im_array, f'{confidence}%', (x + 5, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                                            cv2.LINE_AA)
                                labels_api.append(
                                    {0: [f"{i} - {str(tags[result.boxes[i].cls[0].item()])}", f" {confidence}% , "
                                                                                          f"X: {int(camera_xyz[0])} mm,"
                                                                                          f"Y: {int(camera_xyz[1])} mm,"
                                                                                          f"Z: {int(camera_xyz[2])} mm"]})

                    cv2.circle(im_array, (im_array.shape[1] // 2, im_array.shape[0] // 2), 5, (0, 0, 255),
                               5)  # 标出中心点
                    cv2.circle(im_array, (im_array.shape[1] // 2 + 66, im_array.shape[0] // 2 + 140), 1,
                               (0, 0, 255),
                               1)  # 标出中心点
                    # cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                    #                                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    colorizer_depth = cv2.applyColorMap(colorizer_depth, cv2.COLORMAP_JET)
                    colorizer_depth = cv2.resize(colorizer_depth, (640, 480))
                    imOut = im_array

                    # imOut = np.hstack((im_array, colorizer_depth))
                    # image_queue_buffer = imOut
                    _, buffer = cv2.imencode('.jpg', imOut)
                    frame_encoded = base64.b64encode(buffer).decode('utf-8')
                    # 發送至前端
                    socketio.emit('camera_frame', {'frame': frame_encoded})
                    eventlet.sleep(0.02)  # 每幀間隔時間（50幀/秒，大約20ms）
                    # cv2.resizeWindow('detection', 640, 480)
                    # cv2.imshow('detection', im_array)
                    # cv2.imshow('detectio2', imOut)
                    # cv2.waitKey(0)
                # _, img_encoded = cv2.imencode('.jpg', imOut)
                key = cv2.waitKey(1)
                if check_start == 0:
                    print("啟動成功")
                    check_start = 1
                # image_queue = img_encoded
                # Press esc or 'q' to close the image window
                if key == 27 or key == ord('k'):
                    set_move = 1
                if terminate_thread == False:
                    pipeline.stop()
                    break
                # image_queue = imOut
                # print("程式尾巴")
                # yield (b'--frame\r\n'
                #        b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')
    finally:

        # Stop streaming
        try:
            _, img_encoded = cv2.imencode('.jpg', image_queue_buffer)
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')
            pipeline.stop()
        except:
            print("already dying")
"""






    ============     ==============    ============    ==         ==    ==============   ==============
    ==               ==          ==    ==              ==        ==     ==                     ==
    ==               ==          ==    ==              ==       ==      ==                     ==
    ==               ==          ==    ==              ======          ==                     ==
    ============     ==          ==    ==              ==     ==        ============           ==
              ==     ==          ==    ==              ==       ==      ==                     ==
              ==     ==          ==    ==              ==         ==    ==                     ==
    ============     ==============    ============    ==           ==  ==============         ==
    
    
    
    
    
    
    

"""
# 接收前端開關命令，控制攝像頭
@socketio.on('toggle_camera')
def handle_toggle_camera(data):
    global terminate_thread,current_user
    # 檢查當前的 session ID
    print("影像控制者 :  ",current_user)
    user_id = request.remote_addr
    print("當前使用者 : ",user_id)
    if data['status'] == 'start':
        if not terminate_thread:
            print("啟動")
            current_user = user_id
            terminate_thread = True
            socketio.start_background_task(process_images)
        elif current_user != user_id:
            socketio.emit('camera_control_error', {'message': 'Camera is already in use by another user.'})

    elif data['status'] == 'stop':
        if terminate_thread and current_user == user_id:
            print("停止")
            terminate_thread = False
            current_user = None

# @socketio.on('connect')
# def on_connect():
#     socketio.start_background_task(process_images)
# 處理移動指令和座標查詢的 WebSocket 事件
@socketio.on('move_command')
def handle_move_command(data):
    global step_move_state, machine_status
    # 移動指令處理
    if data.get('moveaxis'):
        if step_move_state == 0:
            step_move_state = 1
            machine_status["moving_status"] = 1
            axis = data.get('moveaxis')  # 獲取移動指令
            if axis in ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]:
                Arduino.write(f'Stepper{axis} 10\n'.encode())
                socketio.emit('status_update', machine_status)  # 通知前端開始移動

                # 開始讀取狀態
                count_time = 0
                previous_line = ""
                while True:
                    if count_time >= 30:
                        machine_status["moving_status"] = 0
                        break

                    if Arduino.in_waiting > 0:
                        response = Arduino.readline().decode().strip()
                        print("Arduino:", response)
                        if response == "DONE":
                            # 假設更新機械臂當前座標
                            machine_status["X"] = float(previous_line.split(',')[0].split(':')[1])
                            machine_status["Y"] = float(previous_line.split(',')[1].split(':')[1])
                            machine_status["Z"] = float(previous_line.split(',')[2].split(':')[1])
                            machine_status["moving_status"] = 0
                            step_move_state = 0
                            socketio.emit('status_update', machine_status)  # 回傳更新的狀態
                            break
                        previous_line = response

                    count_time += 1
                    socketio.emit('status_update', machine_status)  # 實時回報當前狀態
                    time.sleep(0.1)
            else:
                socketio.emit('status_error', {"error": "Invalid axis"})
        else:
            socketio.emit('status_error', {"error": "Machine is busy"})

    # 讀取座標的功能
    if data.get('getpos'):
        try:
            print("讀取座標")
            pos_data = GET_POS(Arduino)
            print(pos_data)
            x = float(pos_data.split(',')[0].split(':')[1])
            y = float(pos_data.split(',')[1].split(':')[1])
            z = float(pos_data.split(',')[2].split(':')[1])
            machine_status["X"] = x
            machine_status["Y"] = y
            machine_status["Z"] = z
            machine_status["moving_status"] = step_move_state
            socketio.emit('status_update', machine_status)  # 返回座標
        except Exception as e:
            print("Failed to get coordinates")
            socketio.emit('status_error', {"error": str(e)})
    # 設置座標功能
    if data.get('set_coord'):
        x = float(data['set_coord']['coordx'])
        y = float(data['set_coord']['coordy'])
        z = float(data['set_coord']['coordz'])
        w = float(data['set_coord']['coordw'])
        print(x,y,z,w)
        # 檢查是否超出範圍
        if x > 250 or y > 200 or z > 300:
            print("超過範圍")
            socketio.emit('status_error', {"move": "fail"})
        elif step_move_state == 1:
            print("移動中...")
            socketio.emit('status_error', {"move": "busy"})
        else:
            step_move_state = 1
            machine_status["moving_status"] = 1
            Axis_move(Arduino, x, y, z, w)
            print("移動完成")
            step_move_state = 0
            machine_status["moving_status"] = 0
            socketio.emit('status_update', machine_status)  # 回傳狀態
"""







"""
def calculate_new_base_position(current_base_pos, target_camera_coords):
    x_base, y_base, z_base = current_base_pos
    x_cam, y_cam, z_cam = target_camera_coords
    # 計算新基座位置
    print("當前POS : ", machine_status['mode'])
    x_new = x_base - (x_cam) + 5
    y_new = y_base - z_cam + 120
    if machine_status['mode'] == 3:
        x_new = x_base + (x_cam) - 5
        y_new = y_base + z_cam - 120
    z_new = z_base - (y_cam) + 50

    return (x_new, y_new, z_new)


def write_to_file(machine_status):
    try:
        # 從 POST 請求中獲取 JSON 數據
        # 將數據寫入 static/Argument.json 文件
        with open(JSON_FILE_PATH, 'w') as json_file:
            json.dump(machine_status, json_file, indent=4)
        print("保存")
    except Exception as e:
        print(e)



# API：接收點動指令
@app.route('/move', methods=['POST'])
def move_axis():
    global Arduino
    global machine_status
    global step_move_state
    def read_status():
        global step_move_state
        # Arduino.readall()
        count_time = 0
        while True:
            if count_time >= 30:

                step_move_state = 0
                machine_status["moving_status"] = 0
                break
            if Arduino.in_waiting > 0:
                response = Arduino.readline().decode().strip()
                print("Arduino:", response)
                if response == "DONE":
                    x = float(previous_line.split(',')[0].split(':')[1])
                    y = float(previous_line.split(',')[1].split(':')[1])
                    z = float(previous_line.split(',')[2].split(':')[1])
                    machine_status["X"] = x
                    machine_status["Y"] = y
                    machine_status["Z"] = z
                    step_move_state = 0
                    machine_status["moving_status"] = 0
                    break
                previous_line = response

            count_time += 1
            print("計算 : ",count_time)
            time.sleep(0.1)
    try:
        if step_move_state == 0:
            step_move_state = 1
            machine_status["moving_status"] = 1
            data = request.get_json()
            axis = data.get('moveaxis')  # X, Y, or Z
            if axis == "X+":
                Arduino.write(f'StepperX+ 10\n'.encode())
                read_status()
                return jsonify(machine_status), 200
            elif axis == "X-":
                Arduino.write(f'StepperX- 10\n'.encode())
                read_status()
                return jsonify(machine_status), 200
            elif axis == "Y+":
                Arduino.write(f'StepperY+ 10\n'.encode())
                read_status()
                return jsonify(machine_status), 200
            elif axis == "Y-":
                Arduino.write(f'StepperY- 10\n'.encode())
                read_status()
                return jsonify(machine_status), 200
            elif axis == "Z+":
                Arduino.write(f'StepperZ+ 10\n'.encode())
                read_status()
                return jsonify(machine_status), 200
            elif axis == "Z-":
                Arduino.write(f'StepperZ- 10\n'.encode())
                read_status()
                return jsonify(machine_status), 200
            else:
                return jsonify({"error": "Invalid axis"}), 400
        else:
            print("忙碌中>>>")
            # data = GET_POS(Arduino)
            # print(data)
            # x = float(data.split(',')[0].split(':')[1])
            # y = float(data.split(',')[1].split(':')[1])
            # z = float(data.split(',')[2].split(':')[1])
            # machine_status["X"] = x
            # machine_status["Y"] = y
            # machine_status["Z"] = z
            # write_to_file(machine_status)
            return jsonify({"move": "busy"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API：接收歸零指令
@app.route('/zeroing', methods=['GET'])
def zeroing():
    global Arduino
    global machine_status
    try:
        print("歸零")
        if machine_status['mode'] in [0,2]:
            return jsonify({"move": "fail"}), 200
        else:
            # 更新機器狀態
            machine_status['zeroing'] = True
            # 發送歸零指令到串口
            Arduino.write(b'zero\n')

            return jsonify({"message": "Zeroing initiated"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# API：接收最大值指令
@app.route('/max_value', methods=['POST'])
def max_value():
    global Arduino
    try:
        # 更新機器狀態
        machine_status['max_value'] = True

        # 發送最大值指令到串口
        Arduino.write(b'max\n')

        return jsonify({"message": "Max value command sent"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API：每秒讀取機器狀態
@app.route('/status', methods=['GET'])
def get_machine_status():
    try:
        return jsonify(machine_status), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API：提供實時 webcam 影像
@app.route('/video_feed')
def video_feed():
    def generate_frame():
        while True:
            if terminate_thread:
                if image_queue is not None and image_queue != []:
                    try:
                        # print("讀取 : ", image_queue)
                        frame = image_queue
                        # Encode the image to JPEG format
                        _, img_encoded = cv2.imencode('.jpg', frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')
                    # except GeneratorExit:
                    #     # 捕獲 GeneratorExit 異常，客戶端中斷連接時退出生成器
                    #     print("Client disconnected, stopping video stream.")
                    except Exception as e:
                        # 捕獲其他異常
                        print(f"An error occurred: {e}")
            else:
                if image_queue_buffer is not None and image_queue_buffer != []:
                    try:
                        # print("讀取停止 : ", image_queue_buffer)
                        frame = image_queue_buffer
                        # Encode the image to JPEG format
                        _, img_encoded = cv2.imencode('.jpg', frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')


                    # except GeneratorExit:
                    #
                    #     # 捕獲 GeneratorExit 異常，客戶端中斷連接時退出生成器
                    #
                    #     print("Client disconnected, stopping video stream.")

                    except Exception as e:

                        # 捕獲其他異常

                        print(f"An error occurred: {e}")

    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 打開偵測
@app.route('/open', methods=['GET'])
def open_screen():
    global terminate_thread
    global machine_status
    if not terminate_thread:
        machine_status["cam_Ready"] = "True"
        write_to_file(machine_status)# 保存機器狀態
        print("開啟")
        return jsonify({"status":"done"})
    else:
        return jsonify({"status": "busy"})

# 調整閾值
@app.route('/threshold', methods=['POST'])
def threshold():
    global threshold_value
    data_res = request.get_json()
    if data_res != "":
        threshold_value = int(data_res['threshold'])
    print(threshold_value)
    return jsonify({f"setoff": 1})

# 關閉相機並保存最後一幀
@app.route('/close', methods=['GET'])
def close_screen():
    global terminate_thread
    global machine_status
    global current_user
    user_id = request.remote_addr
    if terminate_thread and user_id == current_user:
        terminate_thread = False
        write_to_file(machine_status)  # 保存機器狀態
        print("開啟")
        return jsonify({"status": "close"})
    else:
        return jsonify({"status": "busy"})

# 輸出偵測資料
@app.route('/label_list', methods=['GET'])
def label_list():
    global labels_api
    return jsonify(labels_api)

# 緊急停止
@app.route('/stop_emergency', methods=['GET'])
def stop_emergency():
    global Arduino
    EMSTOP(Arduino)
    return jsonify({"status":"stop"})

# 輸出座標資料
@app.route('/ReadCoords', methods=['GET'])
def ReadCoords():
    global machine_status
    global set_pos
    global Arduino
    global Emm42
    global step_move_state
    try:
        data = GET_POS(Arduino)
        print(data)
        x = float(data.split(',')[0].split(':')[1])
        y = float(data.split(',')[1].split(':')[1])
        z = float(data.split(',')[2].split(':')[1])
        machine_status["X"] = x
        machine_status["Y"] = y
        machine_status["Z"] = z
        machine_status["moving_status"] = step_move_state
        write_to_file(machine_status)
    except:
        print("fail")
    return jsonify(machine_status)

@app.route('/initial_parameter', methods=['GET'])
def initial_parameter():
    global machine_status
    global set_pos
    try:
        # 檢查 JSON 文件是否存在
        if not os.path.exists(JSON_FILE_PATH):
            return jsonify({"error": "JSON file not found"}), 404
        # 讀取 JSON 文件內容
        with open(JSON_FILE_PATH, 'r') as json_file:
            data = json.load(json_file)
            machine_status = data
            set_pos = machine_status["mode"]
        # 返回 JSON 數據
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# 列出模型
@app.route('/modellist', methods=['GET', 'POST'])
def modellist():
    global modelname
    if request.method == 'POST':
        # 这里处理 POST 请求，例如获取表单数据
        data = request.get_json()['data']
        print(data)
        modelname = data
        # 处理数据...
        return 'Data received: ' + data
    else:
        modlist = []
        for mod in os.listdir("static/models"):
            modlist.append(mod[:-3])
            print(mod)
        return jsonify({f"model": modlist, "curmod": modelname})

@app.route('/mode_check', methods=['POST'])
def mode_check():
    global set_pos
    global Emm42
    global machine_status
    global height_limit

    data_res = request.get_json()
    print(data_res["data"])
    # 更新機器狀態
    data = GET_POS(Arduino)
    print(data)
    # x = float(data.split(',')[0].split(':')[1])
    # y = float(data.split(',')[1].split(':')[1])
    z = float(data.split(',')[2].split(':')[1])
    if z > 265:
        height_limit = 0
        machine_status["mode"] = data_res["data"]
        write_to_file(machine_status)
    else:
        height_limit = 1

    set_degress(Emm42,data_res["data"])

    if height_limit == 1:
        return jsonify({"move": f"fail"}), 200
    else:
        return jsonify({"move": f"success"}), 200
    # return jsonify({f"move": "done"})

# 合爪
@app.route('/close_grab', methods=['GET'])
def close_grab():
    global Emm42
    #不始能
    Emm42.write(b'\x02\xf3\xab\00\00\x6B')
    time.sleep(0.2)
    #始能
    Emm42.write(b'\x02\xf3\xab\01\00\x6B')
    time.sleep(0.2)
    # 01 9A 00 00 6B
    # 合爪
    Emm42.write(b'\x02\xfd\x01\x00\x64\x1E\x00\x00\x0B\x80\x00\x00\x6B')#180度
    return jsonify({f"aspina": "close"})

# 開爪
@app.route('/open_grab', methods=['GET'])
def open_grab():
    global Emm42
    # 不始能
    Emm42.write(b'\x02\xf3\xab\00\00\x6B')
    time.sleep(0.5)
    # 始能
    Emm42.write(b'\x02\xf3\xab\01\00\x6B')
    time.sleep(0.5)
    # 01 9A 00 00 6B
    # 觸發回零
    Emm42.write(b'\x02\x9a\x01\x00\x6B')
    return jsonify({f"aspina": "close"})


@app.route('/spin', methods=['POST'])
def spin():
    """   MKS板子
    :return:
    global MKS_Aspina
    print("TEST")
    data = request.get_json()
    spin = data.get('move')  # X, Y, or Z
    speed = 0x32  # 100
    if spin == "W+":
        command = [0xFA, 0x01, 0xFD, 0x00, speed, 0x10, 0x00, 0x00, 0x0C, 0x80]  # 送衝1600脈衝
    else:
        command = [0xFA, 0x01, 0xFD, 0x80, speed, 0x10, 0x00, 0x00, 0x0C, 0x80]  # 送衝1600脈衝
    crc = calculate_crc(command)

    command.append(crc)
    print(command)
    # 發送指令
    MKS_Aspina.write(bytes(command))
    time.sleep(0.5)

    MKS_Aspina.readall()
    """

    global Arduino
    data = request.get_json()
    spin = data.get('move')  # X, Y, or Z
    print(spin)
    if spin == "W+":
        Arduino.write(b'GRAP\n')
        print("扭轉")
    else:
        Arduino.write(b'Release\n')
    return jsonify({f"MKS": "close"})

@app.route('/set_coord', methods=['POST'])
def set_coord():
    global Arduino
    global step_move_state

    data_res = request.get_json()
    x = float(data_res['coordx'])
    y = float(data_res['coordy'])
    z = float(data_res['coordz'])
    w = float(data_res['coordw'])
    if x > 250 or y > 200 or z > 300:
        print("超過範圍")
        return jsonify({"move": f"fail"}), 200
    elif step_move_state == 1:
        print("移動中。。。")
        return jsonify({"move": f"busy"}), 200
    else:
        # set_move = 0
        step_move_state = 1
        machine_status["moving_status"] = 1
        Axis_move(Arduino,x, y, z,w)
        print("移動完成")
        return jsonify(machine_status)


@app.route('/grab_coord', methods=['POST'])
def grab_coord():
    global Arduino
    global step_move_state
    global machine_status

    data_res = request.get_json()

    x = float(data_res['coordx'])
    y = float(data_res['coordy'])
    z = float(data_res['coordz'])
    # w = float(data_res['coordw'])
    if x > 250 or y > 200 or z > 300:
        print("超過範圍")
        return jsonify({"move": f"fail"}), 200
    elif step_move_state == 1:
        print("移動中。。。")
        return jsonify({"move": f"busy"}), 200
    else:
        # set_move = 0
        # Axis_move(Arduino, x, y, z, w)
        # step_move_state = 1
        step_move_state = 1
        machine_status["moving_status"] = 1
        print("移動 X ",x)
        print("移動 Z ",z)
        Axis_move(Arduino, x, machine_status["Y"], z)
        while True:
            print("移動中。。。")
            if step_move_state == 0:
                break
        Arduino.write(bytes(f"SETYPOS {y}\n".encode()))
        # count_time = 0
        #
        # while True:
        #     if count_time >= 30:
        #         step_move_state = 0
        #         break
        #     if Arduino.in_waiting > 0:
        #         response = Arduino.readline().decode().strip()
        #         print("Arduino:", response)
        #         if response == "DONE":
        #             break
        #     count_time += 1
        #     print("計算Y軸座標移動時間 : ", count_time)
        #     time.sleep(0.1)
        # Emm42.write(b'\x02\xf3\xab\00\00\x6B')
        # time.sleep(0.2)
        # # 始能
        # Emm42.write(b'\x02\xf3\xab\01\00\x6B')
        # time.sleep(0.2)
        # # 01 9A 00 00 6B
        # # 合爪
        # Emm42.write(b'\x02\xfd\x01\x00\x64\x1E\x00\x00\x0B\x80\x00\x00\x6B')  # 180度
        # time.sleep(1)
        # Arduino.write(b'GRAB\n')
        # time.sleep(1)
        # Arduino.write(b'Release\n')
        # Arduino.write(b'StepperZ+ 10000\n')  # 发送'hello'指令
        # time.sleep(3)
        # set_degress(Emm42, 2)
        # time.sleep(1)
        # # 不始能
        # Emm42.write(b'\x02\xf3\xab\00\00\x6B')
        # time.sleep(0.5)
        # # 始能
        # Emm42.write(b'\x02\xf3\xab\01\00\x6B')
        # time.sleep(0.5)
        # # 01 9A 00 00 6B
        # # 觸發回零
        # Emm42.write(b'\x02\x9a\x01\x00\x6B')#開爪
        # set_degress(Emm42, machine_status["mode"])

        return jsonify({f"move": "close"})



if __name__ == '__main__':

    Auto_Mode_switch = True
    terminate_thread = False
    image_queue = []
    image_queue_buffer = []
    labels_api = []

    # t2 = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'debug': True, 'use_reloader': False})
    # t2.start()
    # 如果 static 文件夾不存在，創建它
    if not os.path.exists(STATIC_FOLDER):
        os.makedirs(STATIC_FOLDER)
    with open(JSON_FILE_PATH, 'r') as json_file:
        data = json.load(json_file)
        machine_status = data
    # if machine_status["cam_Ready"] == "True":
    #     print("開始")
    #     terminate_thread = True
    #     processing_thread = threading.Thread(target=process_images)
    #     # processing_thread.daemon = True
    #     processing_thread.start()
    #     print("結束")
    # else:
    #     terminate_thread = False
    # 啟動 Flask 應用
    # t2 = threading.Thread(target=app.run, args=('0.0.0.0',True))  #

    # app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(host='192.168.68.200', ssl_context=('ssl.csr', 'ssl.key'), threaded=True)

    socketio.run(app, host='0.0.0.0', port=5000)
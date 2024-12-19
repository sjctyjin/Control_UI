# from flask import Flask, render_template
# from flask_socketio import SocketIO, emit
# import random
# import time
# import eventlet
#
# # 初始化 Flask 和 SocketIO
# app = Flask(__name__)
# socketio = SocketIO(app, async_mode='eventlet')
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# # 每20ms發送隨機數到前端
# def send_random_data():
#     while True:
#         random_number = random.random()  # 產生0到1之間的隨機數
#         socketio.emit('update_number', {'number': random_number})  # 發送數據到所有連接的客戶端
#         eventlet.sleep(0.02)  # 20ms間隔
#
# # 啟動背景任務
# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')
#     socketio.start_background_task(target=send_random_data)
#
# if __name__ == '__main__':
#     socketio.run(app, debug=True)
import traceback

from flask import Flask, render_template,session,request
from flask_socketio import SocketIO
import cv2
import base64
import eventlet
import time
import secrets
import serial
import struct
import traceback
import threading

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 這個密鑰應該是隨機且保密的

socketio = SocketIO(app, async_mode='eventlet')
CLK = 0
ERR = 0
MKS_degree = 0
X42_ERR = 0
X42_CLK = 0
X42_degree = 0
MKS57 = 0

# 創建一個鎖
serial_lock = threading.Lock()

def calculate_crc(data):
    # 簡單的累加和取反
    crc = sum(data) & 0xFF  # 累加和取低 8 位
    return crc
ser = serial.Serial('COM19', 38400, timeout=0.005)
X42 = serial.Serial('COM7', 115200, timeout=0.005)
move_flag = False
@app.route('/')
def index():
    user_ip = request.remote_addr
    print(user_ip)
    if 'user_id' not in session:
        session['user_id'] = str(id(session))
    print(session['user_id'])
    return render_template('index.html')

def capture_camera_frames():
    cap = cv2.VideoCapture(0)  # 開啟設備端的webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # 將幀數據編碼為JPEG格式
            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')
            # 發送至前端
            socketio.emit('camera_frame', {'frame': frame_encoded})
            eventlet.sleep(0.02)  # 每幀間隔時間（50幀/秒，大約20ms）
        else:
            break
    cap.release()

# @socketio.on('connect')
# def on_connect():
#     socketio.start_background_task(capture_camera_frames)

# 接收前端開關命令，控制攝像頭
@socketio.on('start_rotation')
def handle_toggle_camera(data):
    global move_flag
    if data['status'] == 'start':
        move_flag = True
        speed = struct.pack('H',100)[0]  # 100
        plause = 6400
        accleration = struct.pack('H',20)[0]
        # 馬達反轉含轉速及pluse，脈衝值為正
        command = [0xFA, 0x01, 0xFD, 0x83, speed, 0x08, 0x00, 0x00, 0x00, 0x00]  # 送衝6400脈衝
        # 頭幀,從機地址,位置指令0xFD,第一位(正反轉 0x0 or 0x8) 第二位~第五位 0x02,0x80(速度)
        # command = [0xFA, 0x01, 0xFD, 0x02, 0x80, 0x05,0x00, 0x09, 0xC4, 0x00]  # 送衝6400脈衝
        command[6:10] = struct.pack('>I',plause)
        crc = calculate_crc(command)
        command.append(crc)
        # 發送指令
        ser.write(bytes(command))
        # 清空接收緩衝區
        ser.readall()
    elif data['status'] == 'stop':
        move_flag = True
        speed = struct.pack('H', 100)[0]  # 100
        plause = 6400
        accleration = struct.pack('H', 20)[0]
        # 馬達反轉含轉速及pluse，脈衝值為正

        command = [0xFA, 0x01, 0xFD, 0x03, speed, 0x08, 0x00, 0x00, 0x00, 0x00]  # 送衝6400脈衝
        # command = [0xFA, 0x01, 0xFD, 0x82, 0x80, 0x08, 0x00, 0x30, 0xD4,0x00]  # 送衝6400脈衝
        # #馬達正轉含轉速及pluse，脈衝值為負
        command[6:10] = struct.pack('>I', plause)
        crc = calculate_crc(command)
        command.append(crc)
        # 發送指令
        ser.write(bytes(command))
        # 清空接收緩衝區
        ser.readall()
    time.sleep(0.8)
    move_flag = False

@socketio.on('slider_test')
def handle_degree_control(data):
    print(data.get('value'))

# 接收前端Bar拖曳控制
@socketio.on('degree_control')
def handle_degree_control(data):
    global CLK, ERR,X42_CLK,X42_ERR,MKS57,move_flag
    print(data.get('obj'))
    with serial_lock:  # 加入互斥鎖，保護串口訪問
        if data.get('obj') == "X57":
            if data.get('svalue') is not None:
                move_direction = CLK -  int(data.get('svalue'))
                print("網頁值 : ", int(data.get('svalue')))
                print("位移植 : ", move_direction)
                move_flag = True
                # if data.get('dirs') == "forward":
                if move_direction < 0:
                    speed = struct.pack('H', 100)[0]  # 100
                    # plause = abs(data.get('svalue'))
                    plause = abs(move_direction)
                    accleration = struct.pack('H', 4)[0]
                    # 馬達反轉含轉速及pluse，脈衝值為正
                    command = [0xFA, 0x01, 0xFD, 0x83, speed, 0x08, 0x00, 0x00, 0x00, 0x00]  # 送衝6400脈衝
                    command[6:10] = struct.pack('>I', plause)
                    crc = calculate_crc(command)
                    command.append(crc)
                    # 發送指令
                    ser.write(bytes(command))
                    # 清空接收緩衝區
                    print(int(data.get('svalue')))
                else:
                    speed = struct.pack('H', 100)[0]  # 100
                    # plause = abs(data.get('svalue'))
                    plause = abs(move_direction)
                    accleration = struct.pack('H', 4)[0]
                    # 馬達反轉含轉速及pluse，脈衝值為正
                    command = [0xFA, 0x01, 0xFD, 0x03, speed, 0x08, 0x00, 0x00, 0x00, 0x00]  # 送衝6400脈衝
                    command[6:10] = struct.pack('>I', plause)
                    crc = calculate_crc(command)
                    command.append(crc)
                    # # 發送指令
                    ser.write(bytes(command))
                    # 清空接收緩衝區
                    print(int(data.get('svalue')))
        else:
            if data.get('svalue') is not None:
                move_flag = True
                move_direction = X42_CLK -  int(data.get('svalue'))
                print("網頁值 : ", int(data.get('svalue')))
                print("位移植 : ", move_direction)
                if move_direction < 0:
                    print(move_direction)
                    speed = struct.pack('H', 100)  # 100
                    plause = abs(move_direction)
                    accleration = struct.pack('H', 0)[0]
                    # 馬達反轉含轉速及pluse，脈衝值為正
                    # 位置控制
                    """
                    命令格式：地址 + 0xFD + 方向 + 速度+ 加速度 + 脉冲数 + 相对/绝对模式标志 + 多机同步标志 + 校验字节
    
                            命令返回：地址 + 0xFD + 命令状态 + 校验字节
                    """
                    # 01 FD 01 05 DC 00 00 00 7D 00 00 00 6B
                    # ser.write(b'\x01\xfd\x01\x00\x64\x00\x00\x00\x0C\x80\x00\x00\x6B')
                    # Emm42.write(b'\x02\xfd\x01\x00\x64\x1E\x00\x00\x0B\x80\x00\x00\x6B')  # 180度
                    # b'\x02\xfd\x01\x00\x64\x1E\x00\x00\x0B\x80\x00\x00\x6B'
                    command = [0x01, 0xfd, 0x00, speed[1],speed[0] ,accleration, 0x00, 0x00, 0x0B, 0x80, 0x00, 0x00, 0x6B]  # 180度]  # 送衝6400脈衝
                    command[6:10] = struct.pack('>I', plause)
                    # crc = calculate_crc(command)
                    # command.append(crc)
                    # 發送指令
                    X42.write(bytes(command))
                    # 清空接收緩衝區
                    X42.readall()
                else:
                    print(move_direction)
                    speed = struct.pack('H',100)  # 100
                    plause = abs(move_direction)
                    accleration = struct.pack('H', 0)[0]
                    # 馬達反轉含轉速及pluse，脈衝值為正
                    command = [0x01, 0xfd, 0x01, speed[1],speed[0] ,accleration, 0x00, 0x00, 0x0B, 0x80, 0x00, 0x00, 0x6B]  # 180度]  # 送衝6400脈衝
                    command[6:10] = struct.pack('>I', plause)
                    # 發送指令
                    X42.write(bytes(command))
                    # 清空接收緩衝區
                    X42.readall()
                    # print(CLK - int(data.get('svalue')))
        time.sleep(0.8)
        move_flag = False


# # 每20ms發送隨機數到前端
def send_random_data():
    global CLK, ERR,X42_CLK,X42_ERR,move_flag
    curr_ERR = 0
    while True:
        try:
            with serial_lock:  # 使用互斥鎖，避免競爭
                X42.write(b'\x01\x32\x6B')  # 讀取脈衝
                X42data = X42.readall()
                X42_CLK = struct.unpack('>I', X42data[3:7])[0]
                if X42data[2] == 1:
                    X42_CLK = X42_CLK * -1
                # degree = current_angel // 44.4

                X42.write(b'\x01\x37\x6B')  # 讀取脈衝
                X42data = X42.readall()

                Err_angel = struct.unpack('>I', X42data[3:7])[0]
                if X42data[2] == 1:
                    Err_angel = Err_angel * -1
                X42_ERR = Err_angel * 360 / 65536

                # print(X42_CLK,X42_ERR)
                # 讀取馬達角度

                command = [0xFA, 0x01, 0x30]
                # 計算CRC並附加到指令後面
                crc = calculate_crc(command)
                command.append(crc)
                # 發送指令
                ser.write(bytes(command))
                data = ser.readall()
                MKS_degree = struct.unpack('>i', data[3:7])[0]
                #讀取馬達脈衝
                current_CLK = CLK
                command = [0xFA, 0x01, 0x33]
                # 計算CRC並附加到指令後面
                crc = calculate_crc(command)
                command.append(crc)
                # 發送指令
                ser.write(bytes(command))
                data = ser.readall()
                CLK  = struct.unpack('>i', data[3:7])[0]
                if abs(CLK) < 100000:
                    current_CLK = CLK
                if abs(CLK) > 100000:
                    CLK = current_CLK
                #角度誤差
                command = [0xFA, 0x01, 0x39]
                # 計算CRC並附加到指令後面
                crc = calculate_crc(command)
                command.append(crc)
                # 發送指令
                ser.write(bytes(command))
                data = ser.readall()
                ERR = struct.unpack('>h', data[3:5])[0] / 182.444
                # if abs(ERR - curr_ERR) > 20:
                # move_direction = ERR - curr_ERR
                # print("角度誤差 : ",curr_ERR)
                # print("換算脈衝 : ",curr_ERR // 0.1125)
                # if abs(move_direction) > 20:
                #     if move_direction > 0:
                #         speed = struct.pack('H', 500)  # 100
                #         plause = abs(int(move_direction // 0.1125))
                #         accleration = struct.pack('H', 100)[0]
                #         # 馬達反轉含轉速及pluse，脈衝值為正
                #         command = [0x01, 0xfd, 0x01, speed[1], speed[0], accleration, 0x00, 0x00, 0x0B, 0x80, 0x00,
                #                    0x00, 0x6B]  # 180度]  # 送衝6400脈衝
                #         command[6:10] = struct.pack('>I', plause)
                #         # 發送指令
                #         X42.write(bytes(command))
                #         # 清空接收緩衝區
                #         X42.readall()
                #     else:
                #         speed = struct.pack('H', 500)  # 100
                #         plause = abs(int(move_direction // 0.1125))
                #         accleration = struct.pack('H', 100)[0]
                #         # 馬達反轉含轉速及pluse，脈衝值為正
                #         command = [0x01, 0xfd, 0x00, speed[1], speed[0], accleration, 0x00, 0x00, 0x0B, 0x80, 0x00,
                #                    0x00, 0x6B]  # 180度]  # 送衝6400脈衝
                #         command[6:10] = struct.pack('>I', plause)
                #         # 發送指令
                #         X42.write(bytes(command))
                #         # 清空接收緩衝區
                #         X42.readall()
                #
                # curr_ERR = ERR

                # 計算 PID 控制器輸出
                # if ERR > 0.5 and move_flag == False:
                #     roll_control()

        except:
            print("慢了")
            print(traceback.print_exc())

        socketio.emit('update_number', {'CLK': CLK, 'ERR': ERR,'X42_CLK':X42_CLK,'X42_ERR':X42_ERR})
        eventlet.sleep(0.01)  # 50ms間隔

def roll_control():
    # with serial_lock:  # 使用互斥鎖，避免競爭
    speed = struct.pack('H', 100)[0]  # 100
    plause = 800
    accleration = struct.pack('H', 50)[0]
    # 馬達反轉含轉速及pluse，脈衝值為正
    command = [0xFA, 0x01, 0xFD, 0x82, speed, accleration, 0x00, 0x00, 0x00, 0x00]  # 送衝6400脈衝
    command[6:10] = struct.pack('>I', plause)
    crc = calculate_crc(command)
    command.append(crc)
    # 發送指令
    ser.write(bytes(command))
    # 清空接收緩衝區
    ser.readall()
# # 啟動背景任務
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.start_background_task(target=send_random_data)
    # socketio.start_background_task(capture_camera_frames)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)

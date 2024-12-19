import serial
import struct
import time
baudrate = 115200

port = 'COM7'  # 根据您的情况更改端口号，例如在Windows上可能是 'COM3'
ser = serial.Serial(port, baudrate, timeout=0.1)

ser.write(b'\x01\xfd\x01\x00\x64\x00\x00\x00\x0C\x80\x00\x00\x6B')#180度
# while True:
#     # ser.write(b'\x02\x32\x6B')  # 讀取脈衝
#     # data = ser.readall()
#     # current_angel = struct.unpack('>I', data[3:7])[0]
#     # degree = current_angel // 44.4
#     # if data[2] == 1:
#     #     degree = degree * -1
#     #
#     # # print((struct.unpack('>I', data[1:-1])[0]))
#     # # print((struct.unpack('>I', data[1:-1])[0]) // 44.4)
#     #
#     # print("脈衝數值 : ", current_angel)
#     # print("角度 : ", degree)
#     ser.write(bytes('GETPOS'.encode()))  # 讀取脈衝
#     data = ser.readall()
#     print("脈衝數值 : ", data)
#
#     time.sleep(0.5)
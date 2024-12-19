"""
  日期 : 2024/10/31
  版本 : 1.0.0
  功能 : 建立與 SignalR 的連線，並接收數據
  說明 : 連線失敗，無法透過python與SignalR溝通，可能是網路問題或SignalR服務器問題
  作者 : Jim
"""
import json
from signalr import Connection
from requests import Session
import urllib3
import time
from flask import jsonify
import json
# 忽略不安全的請求警告

# 建立與 SignalR 的連線
base_url = "http://127.0.0.1:8082/signalr"
session = Session()
session.verify = False  # 忽略 SSL 憑證

connection = Connection(f"{base_url}/hubs", session)
# 獲取 hub 參考
chat_hub = connection.register_hub('chatHub')
# notification_hub = connection.register_hub('notificationHub')
# robot_hub = connection.register_hub('robotRead')

# 定義接收到的數據處理方式
def on_message(**message):
    print("Message received:", message)

connection.start()

# 註冊回調函數
chat_hub.client.on('broadcastMessage', on_message)
# notification_hub.client.on('receiveNotification', on_message)
# robot_hub.client.on('receiveCoordinates', on_message)

# 啟動連線

# 發送數據


chat_hub.server.invoke('sends', ["SSS", "Hello from Python!"])
chat_hub.server.invoke('TEST', ["SSS", "Hello from Python!"])
# notification_hub.server.invoke('SendNotification', ["Python Notification", "This is a test message."])
time.sleep(1)
print("傳送")
# robot_hub.server.invoke('SendJointData', json.dumps({
#     "j1": 100, "j2": 150, "j3": 200, "j4": -50,
#     "j5": 90, "j6": -100, "x": 50, "y": -30,
#     "z": 75, "w": 0, "p": 0, "r": 90
# }))
connection.wait(1)
connection.close()
# 保持連線開啟
# try:
#     while True:
#         pass
# except KeyboardInterrupt:
#     connection.close()

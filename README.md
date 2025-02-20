
在VM中 Ubuntu_Mecharm_pi_270_ROS 開啟內容


** 1 .專案下啟動scout_2.0

```bash
cd ~/catkin_ws && source devel/setup.bash

roslaunch scout_description display_combined_robot.launch
```
** 2 .啟動Rosbridge
```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```
** 3 .啟動座標轉換
```bash
rosrun tf2_web_republisher tf2_web_republisher
```
** 4 .啟動跨資源取用，讓網頁端獲取urdf的dae文件

```bash
mecharm@ubuntu:~/catkin_ws/src$ python3 cors_pro.py
```
** 發布圖形化關節GUI
```bash
python3 /opt/ros/melodic/lib/joint_state_publisher_gui/joint_state_publisher_gui
```

在此專案下啟動python main.py

確認 websocket的IP位置是否正確

網頁文件 : RoboVisionHub.html

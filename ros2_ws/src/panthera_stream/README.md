# panthera_stream

ROS2 独立推流节点，订阅原始图和标注图并提供 MJPEG 页面。它不拥有机械臂，不应包含任何机器人动作调用。

## 输入

- `/vision/image_raw`
- `/vision/annotated`

图像通过 `sensor_msgs/Image` 解码成 BGR，并缓存最新帧供 HTTP 线程读取。

## 注意

- 该页面是早期 ROS2 推流实现，不等同于当前单进程 `VisionStreamer` 的 Cookie、控制状态机、深度页和 FULL_LOAD 放置 UI。
- 不要把 ROS2 页面视为当前安全控制页面。
- 运行会占用配置的 HTTP 端口；避免与根目录主程序的 8080 冲突。

# panthera_grasp_brain（当前禁止实机使用）

该节点原计划作为唯一机械臂拥有者：订阅视觉快照、可选消费语音命令，并复用 `GraspPlanner` 执行抓取。

## 当前阻塞问题

主规划器已升级为“抓取成功 → FULL_LOAD HOME → 显式放置请求 → PUT2/PUT1”，而该节点仍按旧调用方式启动 `run_grasp_loop()`，没有提供 `wait_for_place` 控制接口，也没有网页 `FULL_LOAD` 状态机。

因此当前代码仅供消息同步和生命周期参考，**不得用于真实机械臂**。否则成功夹持后无法进入现行正常放置闭环，并可能转入统一停机路径。

## 已保留能力

- `SynchronizedFrameBuffer` 严格按时间戳组合 RGB、depth 和 detections。
- CameraInfo 等待与 late-joiner 兼容。
- robot worker 的 shutdown/join 生命周期。
- 可选语音命令和 `/arm/status` 状态输出。

## 恢复条件

必须新增明确的放置 service/action、互斥状态、主线程消费机制，以及 FULL_LOAD 中止测试；随后完成 fake robot 与真机低速验收，才能解除本警告。

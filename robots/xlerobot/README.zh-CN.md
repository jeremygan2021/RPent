# Codex 直接控制 XLeRobot

[English](README.md)

RPent 让 Codex 通过选定的**真机**或**仿真**后端完成观察、动作和验证。
真机主链路不需要启动 Isaac、Pi0.5 或 PPO。Codex 负责决策，工具负责计算与验证，
已有唯一电机 owner 负责实时执行。
网络控制盒也可提供真实状态和相机；自主动作继续受控制盒验收门槛约束。
[网络配置模板](control.network.example.json)登记明确的 owner 地址、profile SHA 和相机来源。

## 直接真机任务

`http://localhost:8899/#location` 的 RPent 面板提供明确的后端选择；旧 API 请求未填写
`backend` 时仍默认为 `simulation`。任务启动后不能切换后端。界面分别显示执行阶段、
任务结果、停止确认及各项能力的阻塞原因；真机页面不要求模型服务或仿真映射。

软件链路为 8899 界面 → 8900 鉴权转发 → 8919 RPent → 8920 标定技能 → 已有 8916
电机 owner。8095 模型服务为可选诊断。技能服务需要独立配置、安装和现场验收；
源码修改不会自动更新正在运行的服务。

工具包含 `observe_robot`、`robot_status`、底盘/单关节短动作、`locate_object`、
`navigate_to_workspace`、`grasp_object`、`verify_grasp`、`task_status`、`stop_robot`
和 `finish`。仿真额外保留 USD 关节动作与诊断 `pi05_predict`。相机工具直接返回图片内容。

两种后端的托管 planner 都在 bubblewrap 中运行，只挂载任务工作目录、只读运行时和
任务令牌；Landlock 将 TCP 限制到 RPent API、任务 MCP 端口及配置的模型端口。
机器人控制凭据和串口设备不会挂载。可沿用已有 Codex 登录，不强制新增 API key。

`request_id` 单次使用。回执未知时通过 `GET /v1/rpent/tasks/{id}` 和 `/events` 查询
原任务，不重新下发运动。`run_status`、`stage`、`task_outcome`、`stop_state` 独立表达。
进程正常退出不代表拿到杯子；成功需要受信任的离桌至少 5 cm、持稳 3 秒、视觉随动和
夹爪反馈证据。STOP 撤销后续动作，独立触达电机 owner，并等待新鲜停止反馈；未确认
停止时拒绝新任务，重启不自动续跑。

首版针对登记过的空拉花杯、单右臂和登记桌位。缺少原始深度、实测相机/机器人变换、
私有目标模板、控制器身份证明、机械臂运动学/限位/碰撞标定或执行器明确解锁时，相应
技能保持阻塞。移动取杯还需要现场地图、定位及障碍输入。RPent 不自动开启扭矩。

配置与部署步骤见[真机实现与验收指南](DIRECT_HARDWARE.md)及
[私有配置模板](control.config.example.json)。当前是经过离线回归的软件实现，尚未
进行真机抓杯及 20 次至少成功 18 次的验收。已有仿真映射继续作为互斥的可选控制模式，
STOP 后关闭，断线和重启不恢复。不要把服务管理员令牌交给非托管 planner。

以下内容是**可选的仿真推理诊断**及历史记录，不能作为真机抓取能力证明。

## 独立推理验证

RPent、`env_pi05`、`models/openpi` 和 `xlerobot_isaac` 位于同一工作区。
RPent 使用自己的 `.venv`，模型使用 `env_pi05`，已有仿真启动器使用
`env_isaaclab`。接入不升级原有模型和仿真环境。

在工作区根目录运行：

```bash
./start_rpent_xlerobot.sh
./start_rpent_xlerobot.sh status
./start_rpent_xlerobot.sh infer --prompt "Keep both arms still." --repeats 2
```

默认启动或复用 Isaac Sim 窗口和模型服务，执行两轮真实推理后保持服务运行。
`start --no-verify` 只启动服务，不推理。
`stop-model` 只关闭模型，保留 Isaac Sim 窗口。
HTTP 服务只监听 `127.0.0.1:8095`。

也可以在 RPent 目录直接调用验证入口：

```bash
.venv/bin/python -m robots.xlerobot.diagnostic \
  --endpoint http://127.0.0.1:8095 \
  --output logs/xlerobot/my-verification \
  --prompt "Keep both arms still." --repeats 2
```

输出目录必须是新目录；`--runtime` 可指定另一份本机 `codex_runtime`。
这不是实机硬件配置入口。

## 输入与输出

- 头部、左腕、右腕真实 RGB 画面分别对应 `base_0_rgb`、
  `left_wrist_0_rgb`、`right_wrist_0_rgb`，OpenPI 缩放至 224×224。
- 状态顺序先左臂后右臂，各含 Rotation、Pitch、Elbow、Wrist_Pitch、
  Wrist_Roll、Jaw。十二个绝对关节角按重建 USD 限位映射至 [-1, 1]，
  与指令一起编码，再将状态补齐至模型的 32 维表示。
- 限位归一化仅用于接口验证，**不等于训练数据的统计量**；不包含头部与车轮。
- 使用完整的 3,353,433,872 参数 Pi0.5，BF16 精度，十步流匹配采样。
  每次返回原始归一化 `float32[50,32]` 动作张量。
- 输出只保存，不下发电机。当前没有 XLeRobot 专用微调权重、已标定动作映射，
  也不以联通成功代表抓取或导航任务成功。

独立验证客户端在采样前与清理阶段停车，仿真封装只接受 `status`、`observe`、
`stop`。模型服务不直接连接仿真控制接口。

## 验证记录

每次运行生成 `report.json`；各样本目录包含三路相机 PNG、源
`observation.json`、`model_input.npz`、`actions.npy` 和 `result.json`。
`server_seconds` 包括 GPU 同步等待，首轮还包含 JAX 编译；
`rpc_seconds` 额外包含通信耗时。`PASS` 表示真实观测得到了有效数值输出，
不代表完成物理操作任务。

首次验证在 `logs/xlerobot/verification-first/report.json`：
首轮含编译 25.68 秒，第二轮 0.574 秒，使用 RTX 3060 12GB 与 Isaac Sim
同时运行。启动和客户端日志在 `logs/xlerobot/`；仿真日志仍在工作区同级
`xlerobot_isaac/codex_runtime/`。

GPU1 部署还通过 8899 面板完成了真实 Codex 会话：观测 → Pi0.5 推理 →
观测 → Head_Yaw 动作 → 观测 → 停止 → finish。有效 50×32 张量推理耗时
0.378 秒；请求增加 0.03 rad 时，Head_Yaw 实测增加 0.03066 rad。
此前底盘短动作实测移动约 0.0135 m 后停止。这些结果来自仿真反馈，
没有验证抓取；头部动作期间观察到肩部漂移。证据在本机
`/home/ai/AI_dev/work_logs/rpent-control-20261011/` 目录。

离线检查：

```bash
.venv/bin/pytest tests/unit_tests/robots/xlerobot -q
.venv/bin/ruff check --preview robots/xlerobot tests/unit_tests/robots/xlerobot
```

将 Pi0.5 输出接入闭环执行需要匹配的微调权重、数据集归一化统计、明确的关节/动作约定，
并验证重建机器人的碰撞与相机标定。

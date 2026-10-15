# RPent 真机取空拉花杯

实现范围：直接真机任务、相机图片、原始深度定位、单右臂几何抓取、登记桌位导航、
独立成功验证和可追踪停止。Isaac、Pi0.5、PPO 不构成真机依赖。
源码与离线测试不等于完成现场标定、部署或真机验收。

## 服务与控制权

| 组件 | 接口 | 职责 |
| --- | --- | --- |
| 8899 管理界面 | `/api/admin/rpent/*` | 后端选择、任务阶段/结果/停止状态 |
| 8900 主机 API | `/v1/rpent/*` | 管理鉴权后的严格路径转发 |
| 8919 RPent | `/v1/rpent/*` | 隔离 Codex、任务令牌、事件及动作撤销 |
| 8920 技能服务 | `/v1/status`, `/v1/tasks`, `/v1/locate`, `/v1/verify` | 受信任感知、IK/碰撞、分段接近和成功验证 |
| 8916 唯一电机 owner | `/v1/position-control/*`, trajectory 扩展 | 原有队列、坐标保持、限位、互斥租约、反馈与停止 |
| 本地感知 | `/v1/sensor-sources/{id}/depth`, `/calibration`, `/locate` | 原始毫米深度、内外参、实测 FK 和目标跟踪 |

真机任务固定 `backend=hardware`，不经过仿真映射。旧请求缺少 backend 时仍按仿真处理。
手动控制、仿真映射、技能执行必须争用同一个 owner；扩展不创建第二个串口驱动。

## 配置与安装

当前现场使用网络控制盒 `whole_body16`。RPent 的 `native_endpoint`、
`controller_contract=whole_body` 和 `motor_profile_sha256` 必须绑定该控制盒，
不能沿用本机无 USB 设备的旧 owner。网络状态同时核验 native 兼容状态、
whole-body state/profile；两套 generation 分别用于兼容动作和 STOP，不混用。
现有 `fresh/held` 仅表示当前反馈和保持，不能代替自主执行证明。

网络相机配置 `hardware_cameras` 显式登记 head/left_wrist/right_wrist 三个来源、
完整 JPEG URL 和私有 token 文件。工具核对来源标头及真实采集时间，动作后等待
晚于动作完成的图像。跨主机的米制定位仍需要时钟与坐标标定；观察图片可先使用。

1. 从 [control.config.example.json](control.config.example.json) 建立私有配置。令牌文件
   设为 `0600`，任务目录设为 `0700`。保留既有 native token 和 console key 的路径。
   `controller_id` 填写已核对的当前控制器身份；留空或与 owner 不一致时拒绝真机操作。
2. 使用机器人源码中的 `official_control/native_trajectory.py` 接入现有 owner 队列。
   按其安装说明在已有 mapping API 后安装；不要将新进程作为第二个电机 owner。
3. 按机器人源码的 `docs/CALIBRATED_PERCEPTION.md` 建立私有标定、相机提供器、
   模板及机械臂/头部实测变换。预览灰度图不能替代毫米深度。
4. 配置并启动 `grasp_runtime.server:create_app`，仅监听 loopback，使用独立私有令牌。
   其状态必须给出定位、抓取、导航各自的 readiness。服务启动默认为停止锁定，
   在现场确认反馈、当前保持和停止后，才从操作员入口显式 resume；RPent 不自动 resume。
5. 8919 保持原有启动入口，`RPENT_CONTROL_CONFIG` 指向私有配置。部署 8900 转发与
   8899 静态资源后，界面才能看到新后端。运行中的旧进程不会自动加载这些修改。

Planner 需要 Linux bubblewrap、Landlock ABI ≥4、RPent 虚拟环境和 `/usr` 下的 Codex
二进制。默认将已有 `~/.codex/auth.json` 复制进私有 planner home；也可配置
`planner.auth_file`。可选自定义模型使用 `planner.base_url`、`planner.api_key_file`、
`planner.model`。API key 不写进 JSON 或日志。Planner 仅看到任务工作目录和只读运行时，
不可读取电机凭据、设备、其他任务记录或宿主配置；TCP 仅开放 RPent、该任务 MCP 和
配置的模型端口。这是本机任务进程的隔离，并非对同一宿主管理员的隔离。

## API 与任务语义

所有接口经过 Bearer 鉴权。管理员令牌留在管理服务；planner 使用任务限定令牌及
`X-RPent-Session`。任务令牌不能启动其他任务、切换后端、修改映射或解锁执行器。

提交 `POST /v1/rpent/tasks`：

```json
{"request_id":"pickup-session-0001","backend":"hardware","instruction":"观察并抓起登记的空拉花杯；验证持稳后结束。"}
```

不要把这个示例当作自动验收脚本；启动需要现场就绪条件。
同一 ID、同一参数返回原任务，不重新执行；同一 ID 换参数拒绝。
结果未知时查询 `GET /v1/rpent/tasks/pickup-session-0001` 与 `/events`。

- `run_status`：运行、完成、中断等进程生命周期。
- `stage`：当前观察、定位、导航、抓取或验证阶段。
- `task_outcome`：`pending`、`succeeded`、`failed`、`unknown`。
- `stop_state`：`stopping`、`stopped`、`unknown`。只有反馈确认才能进入 stopped。

异步技能返回稳定 `executor_id`。查询原任务能找到对应 skill request 与实际结果。
任务重启、超时或连接中断后不重发运动。STOP 先改变控制代次并撤销任务，再独立触达
电机 owner 和技能服务；即使技能服务离线也会尝试电机停止。停止未确认时不接受新任务。
晚到的提交会再次撤销，旧反馈不能恢复运行状态。正常技能完成只释放技能租约；显式
STOP 或异常才锁住执行器。携杯停止维持 owner 的保持状态，不打开夹爪。

启动被 readiness 拒绝的任务也保留在任务记录中，状态为 `rejected`、阶段为
`preflight`、`motion_submitted=false`。查询原 ID 可以看到具体门槛；重复提交同 ID
不会在条件改变后偷偷执行。完成验收后应以新的任务 ID 提交新的执行意图。

## 实际执行与验收

顺序为观察 → 找杯 → 到登记桌位 → 停稳 → 重新定位把手 → 预抓取 → 接近 → 闭爪 →
抬升 → 独立验证。固定桌前可以省略导航。反光、遮挡、标定不匹配、过期反馈、歧义
目标或不可达/碰撞路径均拒绝接触，重新观察或结束任务。

首版目标只能来自服务器登记的空杯与受信任模板；planner 不提供任意三维坐标来绕过
感知。深度、RGB、关节反馈有时间戳、序号、坐标系和标定版本约束。抓取使用右臂
数值 IK、分段轨迹和碰撞检查；导航复用已有 Nav2 协议，需要实测地图、定位和障碍输入。

`finish`、正常退出、轨迹受理均不能单独判定取杯成功。独立验证需要离桌 ≥5 cm、
持续持稳 ≥3 秒，以及视觉随动、夹爪状态与无明显滑移证据。验证后再运动会使原成功
证据失效。辅助吸附仿真和模型推理结果不能替代这些证据。

先验证真机服务在 Isaac/Pi0.5 关闭时可用，再做固定底盘空载接近、空杯抓取、已建图
移动抓取，最后执行故障注入。保留每次任务、图像、标定版本、技能结果和停止结果。
完成 20 次受控试验且至少 18 次成功、无碰撞/掉杯/停止失效后，才移除实验能力标识。
目前现场试验未执行，软件应保持实验状态。

## 离线验证

```bash
.venv/bin/pytest tests/unit_tests/robots/xlerobot tests/unit_tests/rpent/planner/test_codex_contracts.py tests/unit_tests/rpent/planner/test_http_mcp_server.py -q
.venv/bin/ruff check --preview robots/xlerobot tests/unit_tests/robots/xlerobot
```

机器人源码另有原始深度、提供器、IK、碰撞、trajectory owner、技能协议和停止故障
测试。测试使用合成图像/深度、伪 owner 和临时 HTTP 服务，不构成真机成功声明。

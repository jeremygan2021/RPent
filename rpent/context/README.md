# Context image plugin

This optional plugin intercepts full-history Codex HTTP Responses requests.
It implements visual scheme B (metadata cosine similarity + 64-bit dHash novelty,
mean score, optional-history score mass NC >= 0.5). Image-only mode preserves text, tool calls/results,
instructions and opaque reasoning. Optional public Text selection is described below. No simulator or policy code changes.

## Install and run

```bash
pip install -e '.[context]'
python -m rpent.context.proxy --mode image \
  --cameras agentview_policy,agentview_high,wrist_high \
  --log logs/context-selection.jsonl
```

In the process launching RPent:

```bash
export RPENT_CONTEXT_PROXY_URL=http://127.0.0.1:18971
rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
  --planner codex --model gpt-5.6
```

Keep the proxy running for that episode, then stop it with Ctrl-C. Use a separate
port/process for each concurrent episode. The proxy binds loopback only. When the
simulator runs remotely, a loopback SSH reverse tunnel can reach the local proxy.
The proxy uses existing Codex ChatGPT authentication, never logs authorization
headers, and forwards to `https://chatgpt.com` by default. This login route is
separate from `CODEX_BASE_URL`; setting both is rejected. Unset the plugin variable
to restore the original planner behavior. `--mode full` forwards original request
body bytes, including original compression, without selection.

Camera slots must match the tool's image order, not a model guess. The three
slots above match the tested LIBERO ToolResult. Unidentified images/cameras are
preserved. The current adapter recognizes snapshot JSON with `step`, `state`,
`task_language` and `log.command` followed by images inside a Codex custom tool
output. Other native messages remain untouched; add an adapter for other schemas.

## Selection rules

- Protect images in the latest two complete native tool interactions and the
  latest recognized observation (even after subsequent read-only calls).
- Exclude step-0 snapshots from scoring, NC and novelty comparisons. Send them
  while protected (initial/latest observation or recent interaction), then omit them.
- Candidates are previously retained images plus newly observed occurrences.
  Previously omitted occurrences do not re-enter or affect novelty. Explicit native
  protection takes priority, including after a history rollback.
- Compute novelty against the preceding eight eligible non-initial images from the
  same camera, including earlier new images in the current batch. Scores are frozen
  during the current NC selection; comparisons are not iteratively recomputed.
- Cache dHash by image-content SHA256 and embeddings by text; distinct image
  occurrences retain distinct IDs. A repeated identical request reuses the previous selection, avoiding repeated pruning.
- Image scores use a fixed denominator: `(R + N) / 2` with a comparison,
  `R / 2` without one. Missing novelty stays null in the log and contributes
  no bonus; measured zero novelty remains distinct from missing evidence.
- Metadata uses existing snapshot `log.command`. Missing metadata scores zero.
- The default query uses the existing task text. Stage is `Unknown`; `--stage`
  can supply a fixed known stage description. Automatic stage inference is not implemented. Text and Action selection
  are described below.
- NC covers only the current eligible optional pool, not the original archive.
- Select optional images by descending average score; ties prefer newer records.
  Zero-total-score candidates are all retained and NC is null.
- Replace omitted image blocks with an explicit text placeholder. Preserve every
  tool pair, native item ID, non-image field and the original request object.
- Select from each current full native history. Never inject records across a
  native compaction, rollback or branch change. There is no persistent history DB.
- Requests with `previous_response_id` bypass selection. WebSocket/incremental
  transports and Claude/API adapters are not implemented in this version.
- Processing errors fall back to the original body with an explicit log entry.
  Counts cover recognized snapshot images; unrecognized images are preserved.

## Offline replay

```bash
python -m rpent.context.replay logs/codex_libero_context_20260922 \
  --cameras agentview_policy,agentview_high,wrist_high \
  --output logs/codex_libero_context_20260922/image-selection.jsonl
```

The directory must contain sanitized `requests/*.json` captures and original PNG
files. Images are restored only after matching recorded SHA256 and byte count;
hashes alone cannot provide dHash. The encoder is loaded lazily on CPU. For fully
offline/reproducible execution, pass `--encoder /path/to/a/pinned/local/snapshot`.
Selection logs contain IDs, scores and byte counts, not image bodies or credentials.

Offline replay verifies selection and request structure on the original trajectory.
It does not simulate how a model would act after receiving compressed context.
A real closed-loop run is needed to check continued model/tool execution. Task
success and resource savings need separate repeated evaluations.

## Extension points

`ContextSession.process(request)` composes an adapter (`extract`, `render`), cached
features and `select_by_nc`. The adapter owns native message locations and protocol
constraints. The NC function is independent of modality and backend. Future API
integration should use its existing `ProcessHistory` instead of a network proxy;
Claude requires a separately verified native adapter. Robot-specific camera/state
metadata belongs in configuration/adaptation, never in NC selection.

## Public Text selection

`--mode text` selects only public assistant text; `--mode image_text` enables both
modalities. `full` and `image` preserve their existing behavior. Text has independent
incremental state and NC; the image algorithm is unchanged.

The runtime adapter accepts complete assistant messages with stable IDs and plain
`output_text` blocks, completed non-robot query code, and ordinary paired tool
result text. A short label immediately followed by a JSON object/list is selected
as one complete candidate. Candidate scoring follows native item/block order;
equal scores prefer the later candidate. Rendering preserves the native order
and call/result identifiers. The latest two public messages, plans preceding
recent/outstanding calls, recent results, and active errors are protected.

Tool interfaces are rules rather than Text candidates. Recognition is shared by
selection, fixed-reference deduplication, and cost accounting: named JSON
interfaces require a structured `inputSchema`/`parameters` object or a declaration
of that same tool in their description. TS declarations and line-based
`Tool name:` plus `Parameters:`/`Arguments:`/`Input schema:` descriptions are also
recognized. A result's `name` and `description` alone do not make it a tool rule.
The fixed reference contains the complete configured interfaces; deduplication
only replaces matching definitions and rejects changed definitions.

System/developer/user content, annotated or mixed assistant messages, unknown
formats, and opaque reasoning remain outside Text selection. Public messages
shorter than the omission notice are retained. The chronological tool-rule
revision has its own feature/scoring versions; earlier checkpoints/replay cannot
be resumed under that revision without explicit revalidation.

Text scores combine task relevance, relevance-weighted novelty and an event indicator with a fixed denominator of three. Novelty is
one minus the maximum cosine similarity to the preceding eight eligible public
texts. With missing novelty the score is `(R + E) / 3`; otherwise
`(R + R*N + E) / 3`. Novelty alone cannot compensate for zero task relevance;
event credit remains independent. Missing comparison evidence earns no bonus.
This is a heuristic score, not a performance guarantee or a calibrated probability.
Changing the score changes NC coverage even at the same threshold.
The query uses observed task language and subsequent user updates,
or the user prompt if no task observation exists. Before such an observation, long
prompts are subject to the encoder's own sequence limit.

The event score is deliberately conservative: a message must explicitly contain a
native call ID whose matching visible tool result reports a boolean `success` or
an error. Mere words like "success"/"failure", or temporal adjacency, do not count.
The existing captured episode has no such explicit references, so all E values are
zero. This is not a semantic event-linking model or a guarantee that plans remain
useful. Tool-reported success is not independent physical ground truth.

Omitted public text messages are removed. Retained message IDs, tool pairs and
opaque reasoning remain unchanged.
Raw novelty, its relevance-weighted contribution, scores, original public text,
event references and selection flags are
logged under `text`; image metrics keep their original fields. Logs can therefore
contain task-related text and should be treated as local experiment artifacts.
Counts are characters, not tokenizer usage or a performance guarantee.

```bash
python -m rpent.context.replay logs/codex_libero_context_20260922 \
  --mode image_text --cameras agentview_policy,agentview_high,wrist_high \
  --output logs/context_plugin_validation/image-text-selection.jsonl
```

Public statements may contain coordinates, constraints or failure explanations.
The current scorer does not reliably recognize these semantics: inspect the offline
omission report and run closed-loop evaluations before claiming decision quality.

## Action/State 与三模态联合选择

`--mode all` 启用 Image、Text、Action；`--mode action` 只启用 Action。
统一入口为 `session.py`。Action 在图片和 Text 之后处理，所以先前的评分仍使用完整工具结果。

`action.py` 复用 `selection.py` 的 NC>=0.5。比较池为上轮保留、新记录、当前保护记录；按顺序与池内上一条记录比较，不与已省略历史比较。无变化比较对象时 Delta=0，并记录缺失。相同请求重试不重复筛选。

- 动作类型变化为 1；同类动作比较共有参数：指令用 1-cosine，类别用是否变化，连续参数仅使用已声明合法范围。
- 数值状态变化为 min(1, abs(new-old)/(upper-lower))，对有效维度平均。状态描述字符串不参与；布尔状态使用 0/1 差异。
- Delta 为动作变化、状态变化中有效项的平均值，两项缺失时为 0。
- E=1：明确 success=false、error、terminated/truncated=true、可比较的 success 布尔切换或 holding/gripper_closed 布尔切换。终止不等同于任务成功。不推断抓取、碰撞或重试。
- score=(Delta+E)/2。保护记录不进入 NC 分母；总分全零时保留全部。

通过 `--action-ranges robots/libero/context_ranges.json` 显式加载范围，或用
`ContextSession(..., action_ranges={"state.position.0": (lower, upper)})` 传入。
JSON 必须把 `state.*` 或 `command.*` 字段映射到两个有限数值，且上界大于下界；
无效配置在代理启动时直接报错，不等到请求中回退。省略该参数时不推断数值范围。
实验 worker 同样支持 `--action-ranges`，并将它传给代理。

LIBERO 配置适用于本实验的 Panda/PandaGripper：

- `state.robot0_gripper_qpos.0`：`[0, 0.04]` 米。
- `state.robot0_gripper_qpos.1`：`[-0.04, 0]` 米。
- `command.gripper`：`[-1, 1]`，开启/闭合控制输入。

前两项来自已安装 robosuite 的 `models/assets/grippers/panda_gripper.xml`
中 `finger_joint1`、`finger_joint2` 的 joint range；关节顺序对应 XML 中的两指关节，PandaGripper 的
`init_qpos` 也采用一正一负的相同顺序。控制输入范围对应 `robots/libero/tools.py` 的 gripper 约定。
这些范围不是从某次成功轨迹的最小值/最大值拟合出来的。

位置没有跨场景通用的已确认范围，四元数也不应逐分量当线性位移比较，因此配置
不包含它们。日志中的 `numeric_ranges` 记录实际配置，`unscaled_numeric_fields`
列出当前有效候选中未配置范围的数值字段。缺失的变化不是观测到变化为零。
后续接入其他机器人时应提供对应配置，不复用 Panda 的物理范围。

```bash
python -m rpent.context.proxy --mode all \
  --rho-image 0.2 --rho-text 0.1 --rho-action 0.1 \
  --action-ranges robots/libero/context_ranges.json \
  --cameras agentview_policy,agentview_high,wrist_high \
  --log /path/to/selection.jsonl
```

保护最近两次工具返回的快照、最新快照、仍有图片的快照。只处理 Codex 的已配对 custom_tool_call_output 或 function_call_output 中已识别的 RPent snapshot。省略时保留原生调用、ID、命令、任务、step、artifacts 和明确结果标志；state 和 result 细节改为 context_omitted 标记。不是删除整个调用，也不是压缩底层控制动作。action-only 若所有图片原样保留，图片依赖保护可能使 Action 无法缩减。

离线报告：`logs/context_plugin_validation/ACTION-STEPS.zh-CN.md`。三模态联合回放使用 `--mode all`；仍不能代替压缩后的闭环任务评估。

## Threshold controller and decision records

Existing CLI thresholds create a `FixedController`. A custom controller can be
passed as `ContextSession(..., controller=...)`; it returns a `Decision`
containing validated image, text and action thresholds. Scoring, protection,
incremental eligibility and selector order are unchanged.

Observations contain current counts and pre-image-selection protection ratios,
plus explicitly named **previous-selection** score summaries. They do not
compute current scores a second time. Missing values have separate indicators.
Action protection counts precede image selection and are not the final Action
candidate set. No raw images, text or model reasoning enter the feature vector.

Each selection log includes a decision ID, thresholds, feature/scoring versions,
and the ordered feature vector. Exact consecutive request retries reuse the
decision and rendered result; changed bodies count as new decisions. One proxy
still serves one sequential episode. Reusing a proxy for a new episode is not
supported.

With `--audit-dir`, the existing benchmark summarizer also writes
`trajectory.json`. It associates HTTP attempts, reported usage, timings and
selected IDs with decisions. Missing usage, transport failures, duplicate
response IDs and compression fallback remain explicit. The existing transcript
now records `environment_success` and `agent_error`; an agent's finish message
cannot supply an absent environment verdict. Legacy transcripts without this
field remain unknown.

`audit_complete` describes logging completeness, not PPO eligibility or task
success. Fixed decisions have no sampling probabilities. The MLP and single-batch PPO update can be enabled as described below;
automatic online collection and repeated training are not implemented. Runner elapsed time and benchmark task timing retain their
original, distinct boundaries.

## MLP threshold inference

Install the optional `context-rl` extra for PyTorch. The fixed controller and
proxy help do not import it. MLP inference uses CPU and a frozen checkpoint.

Create an **untrained** checkpoint initialized near Image/Text/Action =
0.2/0.1/0.1:

```bash
python -m rpent.context.policy --output /path/to/initial.pt
```

Add `--controller-checkpoint /path/to/initial.pt` to the existing proxy or
benchmark worker command. The checkpoint overrides the fixed threshold decision;
existing rho arguments still supply the previous-rho features on the first
request. Full mode rejects a learned controller. Without the checkpoint flag,
the existing fixed mode is unchanged.

Evaluation uses `sigmoid(mu)`. For online data collection, add
`--controller-sample --controller-seed 1`; choose different seeds for different
workers. Sampling uses a private CPU generator. Exact consecutive retries reuse
the original choice. Deterministic evaluation records no sampling probability.

The network has two shared 64-unit Tanh layers, three latent Normal actions and
one value head. Normalization mean and scale are checkpoint buffers; initial
values are zero and one. Feature names/order, feature/scoring versions and
weights are validated on load. Each worker copies its checkpoint before starting;
decision logs identify the file by SHA256.

Sampled decisions store z and its joint Normal log probability. PPO evaluates
the Normal density at this stored z, rather than at the sigmoid threshold. All three components are included even when a selector is disabled.
Sigmoid output is clamped to [1e-12, 1-1e-12] only for numerical saturation.

The initial actor outputs constant thresholds; it has not learned to adapt.
The value head initially outputs zero and does not judge task success.

## Reward calculation and one PPO update

`rpent.context.training` consumes completed `trajectory.json` files collected
with `--controller-sample`. It does not launch a robot, call a remote model or
automatically collect another batch.

For each decision, input token counts from its HTTP attempts are summed once.
Cached tokens are already part of input tokens and are not added again.
The reward is `-beta * input_tokens / token_scale`, with an additional 1 on the
last decision if the environment reports success. Failure adds zero.
`token_scale` is a positive reference scale fixed before training, not a value
estimated from future evaluation episodes. Cost currently covers all requests
in the completed episode, including any finalization requests in the log.

Returns sum subsequent rewards within each episode (`gamma=1`). Advantages
are returns minus the recorded old value estimate, without batch normalization.
The updater evaluates the stored latent Gaussian action, uses the PPO clipped
objective plus value MSE, clips gradients, and stops further epochs if approximate
KL exceeds the configured target. The KL check is not a hard bound on a single
update. There is no entropy bonus or GAE in this first implementation.

```bash
python -m rpent.context.training \
  --checkpoint /path/to/collection-policy.pt \
  --trajectories /path/to/run1/trajectory.json /path/to/run2/trajectory.json \
  --output /path/to/next-policy.pt \
  --token-scale 1000000 \
  --beta 0.1
```

The numeric scale above is an example, not an experimentally selected optimum.
Defaults are four full-batch epochs, learning rate 0.0003, ratio clipping 0.2,
value weight 0.5, gradient norm limit 0.5, and KL stopping target 0.02.
The command prints update metrics and saves a new deployable checkpoint. The
parent policy remains frozen. Adam is initialized for this single invocation;
optimizer state and repeated online-training orchestration are not persisted.

The batch is rejected on incomplete audits, unknown environment verdicts,
agent errors requiring review, missing costs, duplicate episodes/responses,
schema/policy mismatches or absent sampling metadata. Stored probabilities,
values and thresholds are checked against the collection checkpoint before
updating. A valid environment failure remains a training sample. Error cases
are rejected explicitly, not silently removed or relabeled as task failures.
Fixed-policy and deterministic evaluation logs cannot be used for PPO.

Controlled test trajectories establish accounting and update correctness only;
they do not establish improved robot task success or compression efficiency.


## Legacy logs and offline IQL

The offline adapter accepts only audited fixed-scale incremental All-mode logs.
It verifies the source selector version, completed environment success, response
usage and actual before/after selections. Unsupported, incomplete or fallback
runs are rejected. Original logs are never edited. This legacy adapter supports
verified successes only; it is not a general failure-label converter.

Run from the repository root:

```bash
python -m scripts.context_benchmark.convert_legacy \
  --episodes /path/to/run1/output /path/to/run2/output \
  --output /path/to/legacy_libero.json \
  --token-scale 1000000 --beta 0.1

python -m rpent.context.iql \
  --dataset /path/to/legacy_libero.json \
  --checkpoint /path/to/initial-policy.pt \
  --output /path/to/iql-policy.pt \
  --updates 100 --batch-size 64 --seed 0
```

The 56-feature observation and its version are unchanged. Reconstruction uses
current pre-selection records and previously logged selection scores, not future
outcomes. Each transition stores state, the three actual rho values, reward,
next_state and done. Consecutive identical retries share a decision and sum their
attempt costs. The last transition has done=true and zero next_state; it never
bootstraps. Reward accounting matches PPO: all requests in the completed Agent
session, including finalization. Attempts after environment success are marked
separately. Cached tokens are not added again.

IQL follows [Kostrikov et al.](https://arxiv.org/abs/2110.06169): independent twin
Q networks estimate return for (state, rho); expectile regression fits V to the
smaller target Q; Q fits reward + (1-done)*V(next_state); the Actor fits recorded
actions weighted by exp(3*(Q-V)), capped at 100. Defaults use expectile 0.7,
target update rate 0.005 and learning rate 0.0003. These are validation defaults,
not tuned robot-task settings. The Actor reuses the existing controller and
checkpoint schema. Its PPO value head is reset to zero: the independent IQL V is
not exported as that head. Subsequent PPO must collect fresh sampled rollouts
with the exported checkpoint.

This command performs bounded CPU updates and exports a deployable Actor.
Q/V weights and optimizer state are not saved for resuming training.
The current 11-episode dataset contains only successful runs at fixed rho
(0.2, 0.1, 0.1). It validates conversion, updates and checkpoint loading; it cannot
establish the benefits of adaptive thresholds or distinguish alternative actions.
More action diversity and verified failure trajectories are needed for that.

Online PPO needs no state-format change. Continue recording sampled decisions
(observation, latent action, log probability, value and policy version), attempt
usage and the environment verdict through the existing trajectory writer.
Legacy fixed-action logs have no valid old sampling probabilities and remain
ineligible for PPO.

# 首期集成测试（ST）

本目录使用独立 pytest 配置，调用真实产品入口。当前适配仓库中的
`hcu-envcheck/bin/hcu-cluster-run`；不依赖未提交的 `hcu-envchck-new`。

当前训练脚本为 GLM53 结构的缩小预训练负载：随机初始化、mock 数据、NullTokenizer，无 HF 资源依赖。
GLM53 简化流程已于 2026-11-06 在超节点完成 2×4＋1备真实验证，当时使用 hcu-envchck-new。
当前 hcu-envcheck 后端已于 2026-11-06 在超节点 2×4＋1备通过 ENV-001、REL-010 真实验证：
单次 SIGKILL 后由备用节点替换故障节点，加载第290步 torch_dist checkpoint 并续训至第300步。
local checkpoint 按配置关闭，未验收；三项明确配置的 platform WARN 保留原始结果并按原因码豁免。
此前使用镜像内 nvidia_resiliency_ext，不代表仓内 hcu_resiliency_ext 后续适配已经验证。
`list` / `plan` 只读本地配置，不连接服务器、不调用 Docker 或 release。

## 用例与执行流程

| ID | 验证内容 |
| --- | --- |
| ENV-001 | 训练前 IDLE、必要时 release、节点池容器状态、platform、所有训练节点 BUSY/备用节点 IDLE |
| REL-010 | 同样的环境检查，然后验证 torch_dist 保存、按配置启用 local 保存、worker 退出、备用节点替换后加载并续训 |

两条用例独立创建资源。**ENV-001 也会通过 reliability 启动自有训练**，以验证 BUSY；
它不是只查询已有容器的命令。选 `all` 会顺序执行两个独立用例。

1. 生成独立工作目录、容器名、日志和黑名单路径。
2. 宿主 `idle-check`：池内所有节点最终都必须 IDLE，且各节点实际卡数不少于 `training.gpus`。BUSY 时仅对 `release_nodes` 中的节点调用
   产品已有 `idle-check --release`；保护名单命中、归属不完整、UNKNOWN 或复检失败均阻断。
3. reliability 沿用现有环境配置，创建节点池全部节点的专用容器；按 `training.nodes × training.gpus` 分配训练，其余节点备用。
4. 核验容器完整 ID、镜像和相同绝对路径的 case 挂载。等待完整训练分配、有效迭代及 manager RUNNING；每节点 local rank 完整、全局 rank 覆盖全部 worker，且属于同一 attempt。
5. 查询 `container-status` 和 `platform`，再只读检查所有训练节点 BUSY、所有备用节点 IDLE。
   platform/status 可以在训练已启动后执行，不为它们修改产品启动流程。
   platform 默认要求 PASS；现场已确认可忽略的 WARN 可在配置中用
   `[platform] ignored_reason_codes = ["RDMA_USERSPACE_DEVICE_OPEN_FAILED", "IB_ENDPOINT_EVIDENCE_MISSING", "IB_CONGESTION_SIGNAL_GROWTH"]`
   明确列出。只有报告中的所有 WARN 均在该列表、节点采集及传输成功且问题表完整时继续；
   FAIL、未列出的告警、缺失或损坏的证据仍阻断。原始 WARN 不改写，豁免项写入日志和结果 JSON。
6. REL-010 等待 **torch_dist 真实完成保存**，并登记对应数据文件；`local_interval>0` 时同时要求 local 完成保存。
   当前站点模板暂设 `local_interval=0`，关闭 local 保存及等待，结果标记 local 为 `NOT_EXECUTED`。改回5即可恢复原双 checkpoint 流程。
7. 在实际分配 hostfile 的首个训练节点、本次容器 PID 空间中重新核验 `LOCAL_RANK=1` 的 PID、启动时间、父进程和 attempt，
   单次 SIGKILL。实际 worker 的 `TORCHELASTIC_MAX_RESTARTS` 必须为 0。
8. 只观察 reliability 重启完整训练分配，并用至少一个原备用节点替换故障节点；健康原节点允许复用。
   新 attempt 加载有效 checkpoint，加载迭代 > 0，之后产生更大的有效迭代；local 关闭时，加载路径必须属于本 case 的 `checkpoints/global`。
   旧 attempt 的进程必须在所有池节点退出；新 attempt 的全局 rank 和每节点 worker 数须完整。
9. 先停止自有 manager 进程组，收集日志，再按登记的完整容器 ID 回收本用例容器。保留运行证据目录。

正常空闲不代表 release 分支已实测；结果会记录当轮是否真的执行了释放。
release 会停止整个占卡容器，不能只停某个 rank。`release_nodes` 表示这些节点已交付给本轮释放，
`protected_containers` 可额外列出必须保留的名字或完整 ID。当前实现没有产品侧容器白名单，
因此实际站点必须确保这些节点没有其他需要保留的业务。训练中和恢复过程中不调用 release。

首期不验证恢复来源优先级、严格注入窗口、超过故障前进度或恢复后再次完成两类保存。
主动通信、hang 和异步 checkpoint 留待后续。

## 配置与运行

在 Linux 控制端使用 Python 3.11+、pytest 8+。产品所需 ClusterShell、reliability 依赖、SSH 和 Docker
沿用原运行环境；pytest 不加入产品安装包。Windows 可进行计划、收集及纯离线验证。

### 测试前拉取 Bridge

镜像已有训练运行时、编译扩展和 Bridge 的 Python 依赖，准备入口只拉取源码，不执行 pip、环境脚本或训练。
`config/dependencies.toml` 已固定用户指定的 Bridge main 提交 `4430df5388a84b0ae88c8633ebc8fe5488b8385c`。
分支名、Tag、短 SHA 仍会在调用 Git 前被拒绝。GLM53 示例需要 MCore dev，`bridge.mcore="dev"`
在独立副本中选取该上游 Bridge 的 `.dev.commit`，当前为 `43e45e13074f36da68b34c7a189cd997a01d1b41`；
准备入口只执行 Git，不执行下载的 switch 脚本，并在 manifest 中同时记录 gitlink 基线和实际 dev SHA。

```bash
# 单独准备依赖，不要求 training.ready=true，也不会开始集群操作。
python3 -B tests/st/run_st.py prepare \
  --site /path/to/site.toml \
  --dependencies tests/st/config/dependencies.toml

# 输出 JSON 中的 site 是新生成的配置路径，原 site.toml 不会被修改。
python3 -B tests/st/run_st.py plan --site /absolute/path/to/site.prepared.toml
# 仅当训练脚本、其他资源与超时都就绪后执行：
python3 -B tests/st/run_st.py run --site /absolute/path/to/site.prepared.toml

# 也可以由真实 run 在执行前自动拉取；其余运行条件先通过检查。
python3 -B tests/st/run_st.py run --site /path/to/ready-site.toml \
  --dependencies tests/st/config/dependencies.toml
```

默认每次创建独立的 `tests/st/.deps/<时间-随机ID>/Megatron-bridge-das`，不复用或覆盖已有工作副本。
ST 自己的 `.gitignore` 忽略 `.deps/`，第三方源码和生成的配置/证据不提交到工程仓库。
可通过 `--deps-root /path/to/isolated/deps` 指定其他父目录；建议自定义目录也放在源码外。
独立准备脚本 `prepare_dependencies.py` 接受相同准备参数，便于在有网络的控制端执行。

准备过程按固定 SHA fetch、detached checkout，再递归初始化提交中记录的子模块，不执行远端源码。
核验 `run_recipe.py`、`env/env.sh`、上游 Bridge 和嵌套 Megatron-LM 入口。
每轮保留 `prepare.log`、`manifest.json` 和 `site.prepared.toml`；失败时保留 FAILED 记录和已有目录，不自动删除。
manifest 记录 URL、完整 commit、递归子模块版本和源配置校验值；生成配置记录 manifest 的 SHA256，
只替换 `paths.bridge_root` 并添加依赖记录引用，不会把 ready 改为 true、填写超时或猜测其他训练参数。
生成 TOML 保留配置值但不保留原文件注释；Windows 生成的路径需在实际 Linux 部署位置重新准备，不能直接拿去远端运行。
真实 case 会将依赖记录复制到证据目录，并写入 result.json 的 configuration.bridge_dependency。

READY 仅表示源码准备完成，不表示训练可用。若 Bridge 环境脚本根据 `$0` 定位或复制注册文件，manifest 保留提示。
内置 ST 脚本不 source 此文件，也不写入上游注册文件；`st_run_recipe.py` 在进程内注册 ST 配方再调用 Bridge 原 runner。
容器必须能在相同绝对路径访问独立依赖副本；不使用运行时切分支、pip 安装或网络资源下载。
若超节点无法访问 GitHub，可在有网端准备后连同副本和证据部署到相同绝对路径；路径不同应在目标端重新准备或明确重新生成配置和版本记录。
原先手工准备 Bridge、直接填写 `paths.bridge_root` 的方式仍可使用；不传 `--dependencies` 时不会自动执行 Git。

从仓库根目录执行：

```bash
python3 -B tests/st/run_st.py list
python3 -B tests/st/run_st.py plan --site /path/to/site.toml
python3 -B tests/st/run_st.py run --site /path/to/site.toml --case ENV-001
python3 -B tests/st/run_st.py run --site /path/to/site.toml --case REL-010
# 默认 --case all，顺序执行本阶段两条用例。
```

配置目录只保留实际入口需要的 `site.ci.toml`、`dependencies.toml` 和 `container.ci.env.sh`。
通过 `ci_entry.py render` 生成源码目录之外的实际 site，再按站点需要填写以下内容；模板不能直接传给 `run_st.py`：

| 配置 | 说明 |
| --- | --- |
| `site.nodes/image` | 实际节点池及已有训练镜像；训练/备用身份依据 reliability 实际分配确定 |
| `site.release_nodes` | 明确允许释放当前占卡容器的节点；空列表时 BUSY 会阻断 |
| `site.protected_containers` | 必须保留的容器名字或完整 ID |
| `paths.repo_root` | 当前仓库绝对路径，必须包含新 envcheck 和 reliability 入口 |
| `paths.shared_root` | 独立共享 ST 目录，在控制端、宿主和容器中使用相同绝对路径；位于源码目录之外 |
| `paths.control_python` | 已具备产品控制端依赖的可执行 Python |
| `paths.bridge_root` | prepare 生成的独立 Bridge 副本；内置预训练不需要 hf_path |
| `paths.env_script` | 容器内 platform 使用的已有环境脚本 |
| `reliability.env_script/env` | 已有环境配置；只 source 环境文件，不能填写会启动业务任务的 start.sh |
| `reliability.train_config` | 内置配方使用 `auto:glm53-st`，从同一配方设置生成并冻结 JSON；外部脚本仍可填写已有配置文件 |
| `training.script` | 用户完善后的原训练脚本，运行时复制冻结，不修改原文件 |
| `training.nodes/gpus` | 同时训练的节点数、每节点卡数；总 worker 数为两者乘积，备用数为池大小减去训练节点数 |
| `training.ready` | 确认训练脚本及依赖就绪后才设 true |
| `training.train_iters` | 足以完成初始保存、故障及恢复后有效迭代的总步数 |
| `training.env` | recipe、并行和扩展参数等；交由用户脚本解释 |
| `timeouts.*` | 所有数值需实测后填写正数；0 表示未校准，禁止真实 run |

例如：2 节点池、`training.nodes=1`、`gpus=8`，自动得到 1 个备用节点；
超节点场景使用 4 节点池、`training.nodes=2`、`gpus=4`，自动得到 2 个备用节点。
也支持总卡数不为 8，例如 4 节点池、`training.nodes=3`、`gpus=4`，共 12 个 worker、1 个备用节点。
不单独配置备用数量；节点名须唯一，训练节点数和每节点卡数须为正整数，分配不能超出池大小。
ENV-001 允许没有备用节点和每节点 1 卡；REL-010 需要至少 1 个备用节点，且每节点至少 2 卡以注入 `LOCAL_RANK=1`。
这些约束在创建资源前检查。已有站点文件须显式补充 `training.nodes`。
模型并行参数仍由用户配置，并须适配目标训练规模；配置可解析不代表模型一定可运行。

reliability 的 SSH 和账号配置保持原行为。ST 为 case 隔离 `WORK_DIR`、`LOG_DIR`、容器名和黑名单路径，
使用现有 `docker` 模式、`NONE` 节点池、初次创建不复用、故障停止时保留容器。
现有挂载配置必须让 case 与 Bridge 目录在容器中同路径可见；ST 不改宿主认证或 hostname。
站点 SSH 端口须与已有 reliability 环境一致，所有调用沿用同一现有 OpenSSH 配置。
首期拒绝仅对 ST 设置额外 SSH 身份/配置选项。Docker 参数必须已解析为静态参数，
不允许覆盖用例容器名、共享宿主/其他容器 PID 空间或将宿主 `/proc` 覆盖到容器中。

## 训练脚本契约

`payloads/train_st.sh` 基于指定 commit 的 GLM53 示例，使用 ST 自有配方和内存注册入口；不修改 Bridge 业务源码。
默认 2 层、hidden=512、8 个 attention heads、8 experts、seq=128、vocab=4096，保留 MLA/DSA/MoE。
采用原生 PyTorch DSA (`dsa_kernel_backend=none`) 和普通 MoE all-to-all，关闭 DeepEP dispatcher 和路由融合。
这是功能验证负载，不用于生产 GLM53 精度或性能结论；该缩小配方已在指定超节点镜像实测；其他站点仍需核验兼容性。
默认 TP/PP/CP/EP/ETP=1，micro batch=1，global batch=数据并行数×micro batch（默认 1×8 时为 8），总步数2000。
可通过 training.env 的 NUM_LAYERS、SEQ_LENGTH、GLOBAL_BATCH_SIZE、MICRO_BATCH_SIZE 和并行参数调整；
检查层数、heads、experts、world 和 batch 整除关系。NRE 适配仍依赖目标环境提供。
包装层只负责路径、attempt 和日志，不补写成功标记、不搬运 checkpoint、不替 reliability 重启。

- `$1` 是 reliability 提供的**当前分配** slots hostfile；包含 `training.nodes` 个训练节点，不是整个节点池。
- ST 显式提供 `GPUS=training.gpus`、`NNODES=training.nodes`、`TRAIN_ITERS`、配置的 `SAVE_INTERVAL`、`LOCAL_SAVE_INTERVAL`、
  `MEGATRON_BRIDGE_PATH`；内置脚本不使用 LOCAL_HF_PATH，也不读取 HF 配置/权重/tokenizer。
- `CKPT_DIR=<case>/checkpoints/global`；`LOCAL_CKPT_DIR=<case>/checkpoints/local`。
  两者都在共享存储，所有训练/备用节点使用同一内容；用户脚本必须将它们接到真实框架参数。
- torch_dist 始终启用；local 由 `training.local_interval` 控制，0表示暂时关闭，正数表示保存步数间隔。首期同步，关闭内层自动重试。
  local 启用时，global 间隔必须是大于 local 的整数倍。环境变量本身不代表功能已启用，
  ST 仍检查真实完成日志、数据文件、实际 worker 的重试设置和恢复后迭代。
- torch_dist 使用 checkpoint.save/load 同一共享根目录、间隔10；local 使用 non_persistent_ckpt_type=local、
  non_persistent_local_ckpt_dir、间隔5、atomic，均为同步，不启用 checkpoint 复制。local 关闭时，配方将 non_persistent_save_interval、
  non_persistent_ckpt_type 和 non_persistent_local_ckpt_dir 设为 None；无需修改 train_st.sh。保留最近3份全局 checkpoint。
  首次从空目录随机初始化，恢复实际加载已有 checkpoint；不使用该版本未定义的 auto_resume 字段。
- torchrun 显式 --max-restarts=0，配方不启用 inprocess_restart；容器 SSH 端口默认36000，与 reliability 默认一致，
  如站点另有端口，通过 training.env.PORT 配合 reliability.env.DOCKER_CONTAINER_SSH_PORT 一起修改。
- TRAIN_ENV_SCRIPT 默认 /opt/dtk/env.sh，由每个节点 source；网络设置沿用站点已有环境，不能假定某个网卡名。
- 使用 MODE=dryrun 可调用 Bridge 的 --dry-run 解析配置，仍需要真实框架依赖；本次后端适配的本地验证使用假启动器，CI dry-run 仍依赖真实镜像。
- 包装层通过既有 MPI 的 OMPI/PRTE 环境列表传递运行 ID 和 attempt ID，保留站点已有条目，
  用于关联另一节点的 worker；不修改原训练脚本或 SSH 配置。
- 保留框架真实的保存完成、加载完成和迭代输出。包装层捕获全部 stdout/stderr 并保留真实退出码。

## 日志与判定

每个 case 保留三类原始日志及一个 `result.json`：

- `st.log`：配置摘要、检查、命令结果、目标身份、注入和收尾。
- `manager.log`：manager 的原始 stdout/stderr。
- `attempts/<id>/train.log`：每次启动独立日志；metadata 冻结当次 hostfile，防止后续分配覆盖旧身份。

`monitor/` 只保留当前 attempt 日志的硬链接。旧日志仍保留在 attempts 中，兼容产品按分钟选择日志的现有行为，
不修改产品 offset。共享文件系统需支持同目录树硬链接和 flock。

保存必须同时具备完成日志和对应非空数据文件。torch_dist 日志为 `successfully saved checkpoint ...`，
并核对该迭代实际打印的保存格式；local 启用时还要求 `successfully saved local checkpoint ...` 和 local 数据文件。
`saving`、`scheduled`、空目录和仅有 tracker/config 均不能通过。
文件清单在确认保存时登记，避免保留策略清理旧文件造成事后误判。不反序列化或全量哈希权重。

恢复判据是包含原备用节点的新分配日志出现 `successfully loaded checkpoint ... at iteration N`，`N>0`，
加载路径属于本 case 的 checkpoints，之后有有效 iteration `>N`，
并且 manager 仍存活、新分配有完整的真实 worker、旧 attempt 的进程已全部退出、故障节点已被替换。
结果记录初始训练节点、备用节点、新训练节点、故障节点、总 worker 数和完整 attempt 分配，训练日志每 attempt 只归档一次。

结果区分工程缺陷、依赖/镜像、环境、测试框架、待归因和未执行。platform 默认要求健康 PASS；
只有明确配置的 WARN 原因码可以豁免，UNKNOWN 和未列出的问题继续阻断；版本差异信息本身不判失败。训练中预期的 BUSY 非零退出不按命令失败处理。
任一必需检查或收尾失败都不能整体 PASS。

所有等待受阶段 deadline 与总 deadline 约束，普通日志不会续期。收集和收尾有独立预算。
缺少 `--site` 时两条真实用例明确 skip，不视为已验证集群功能。

本地只收集用例可运行：

```bash
python3 -B -m pytest -c tests/st/pytest.ini tests/st/testcases --collect-only
python3 -B -m pytest -c tests/st/pytest.ini tests/st/testcases
bash -n tests/st/payloads/run_train.sh
```

这些命令不提供站点，因此不会执行 SSH、release、Docker 或训练。真实验证需在脚本和配置就绪后显式 run。

Bridge 准备功能的离线回归可显式执行，不需要训练环境：

```bash
python3 -B -m pytest -c tests/st/pytest.ini tests/st/unit -q
```

这些回归仅在临时本地 Git 夹具仓库中验证固定提交、递归子模块、独立目录和失败证据。
控制端需要 Git；`bridge.command_timeout` 是每条 Git 命令预算，与训练阶段超时独立。


## Kubernetes CI：跨节点 checkpoint 恢复

`st-cluster.yml` 改用 Kubernetes，相关路径的 PR 自动执行，也支持手工触发；没有 push 或 nightly 测试。
原有宿主 SSH/Docker CLI 保留；ENV-001 尚未迁移到新工作流。

训练资源读取 `job-template-hygon-bw1100-8card`，申请 8 卡，沿用该池 CPU、内存、共享内存和 Volcano 队列。
控制任务使用不占卡的 `hygon-cpu` runner，避免重复占住训练资源。
两台节点必须明确分配：第一台模拟故障，第二台备用；本工作流不解除 cordon。
控制容器和训练 Pod 均需挂载同一 `ci-public` RWX PVC 到 `/ci_public`。

### 首次配置

| 配置项 | 类型 | 说明 |
| --- | --- | --- |
| `ST_KUBECONFIG` | secret | 测试集群账号 kubeconfig；不要上传生产集群管理员凭据 |
| `ST_K8S_NODES` | variable | 两个分配节点，空格分隔，例如 `test-node-a test-node-b` |
| `ST_TRAINING_IMAGE` | variable | 已验证训练镜像 `registry/image@sha256:<64位摘要>` |
| `ST_SHARED_ROOT` | variable | 默认 `/ci_public/cluster-manager-st`，须在 checkout 外 |
| `ST_K8S_NAMESPACE` | variable | 默认 `arc-runners`，须存在共享 PVC 和卡池 ConfigMap |
| `ST_ALLOW_DEGRADED` | variable | 默认 false；经授权使用异常节点时显式设置 true，只放宽本轮 Pod 的健康标签过滤 |

测试账号需要该 namespace 内 ConfigMap、Volcano Job、Pod（含日志、exec）、Event、FaultEvent 及其 status 权限；
读取两个测试节点、patch 故障节点，以及创建/读取/删除 FaultEvent CRD 的权限。权限应限制到测试环境和明确的故障节点。
入口不安装生产 FT 控制面、Webhook 或 NodeHealth DaemonSet；发现已有 FaultEvent CRD 时停止，避免重复处理事件。
[`rbac.example.yaml`](../../.github/ci/k8s/rbac.example.yaml) 提供专用账号权限示例，应用前替换节点名和 namespace；
RBAC 无法按名称限制 CRD 的 create 权限，因此此账号仅用于经过授权的测试集群。

Fork PR 继续自动执行。控制脚本、Bridge 依赖 pin、FaultEvent schema 和临时 FT Operator 都取自受信任的 base commit；
PR 训练源码只在不挂载 ServiceAccount token、不接收 kubeconfig 或 GitHub 凭据的训练 Pod 内执行。
因此 PR 中的 Operator 改动须在合入 main 后手工触发验证，不能把 PR 阶段解释为已测试该版本的 Operator。
新工作流及脚本先合入 main，再通过新的 PR 事件生效。

### 验收流程

1. 准备固定版本 Bridge，在共享存储生成独立源码、配置和运行记录。Energon 7.4.0 与固定 Bridge lockfile 对齐，只安装到本轮私有前缀。
2. 首台节点完成真实 torch_dist 保存，确认数据文件和完整 worker 集合。
3. 注入一个模拟 `NodeHealthCheckFailed` FaultEvent。原生 FT Operator 添加本轮唯一 NoSchedule taint，仅删除指定测试 Pod。
4. Volcano 依据 PodFailed/PodEvicted 自动 RestartJob；注入前后不修改调度模板。必须确认新 Pod UID、备用节点、完整 worker、非零 checkpoint 加载及更大的后续迭代。
5. 训练跑完 1000 步且退出码为 0。删除训练 Pod 后，由独立控制容器读取持久化 checkpoint 文件。
6. 停止临时 Operator，按登记 UID 删除本轮对象，只撤销本轮 taint。日志、版本、文件清单和结果上传 artifact，权重留在共享盘。

NoSchedule 不驱逐其他现有 Pod，但测试期间其他新任务也会避开模拟故障节点，需事先协调测试池。
此场景覆盖事件处理、节点隔离、跨机调度和 checkpoint 恢复；真实断电、NHC 硬件探测和 local checkpoint 未覆盖。
固定 concurrency group 串行运行，cancel-in-progress 为 false。取消时正常收尾，再由 always 步骤补偿清理。
控制 runner 整体失联后，恢复访问并对原始记录执行：

```bash
export ST_K8S_ARTIFACTS=/path/to/original/artifacts
export KUBECONFIG=/path/to/test-cluster.kubeconfig
python3 -B .github/ci/k8s/cleanup.py
```

配置缺失、资源不足、恢复或清理失败均不能标记 PASS。

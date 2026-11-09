<!--
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
SPDX-License-Identifier: Apache-2.0
-->

# hcu_resiliency_ext 安装与验证

## 组件边界

`hcu_resiliency_ext` 不是完整的 `nvidia-resiliency-ext` 分支，也不是当前可独立 `pip install` 的 Python 包。它通过 import hook，把：

```text
nvidia_resiliency_ext.shared_utils.profiling
```

重定向到本仓库的 HCU 适配实现。该操作会修改当前 Python 进程的模块导入行为，因此只能在经过验证的第三方源码版本上启用。

## 1. 初始化第三方子模块

在仓库根目录执行：

```bash
git submodule update --init nvidia_resiliency_ext
```

该 submodule 配置为跟踪上游 `main`。获取本仓库后同步最新代码：

```bash
git submodule sync -- nvidia_resiliency_ext
git submodule update --init --remote --merge nvidia_resiliency_ext
```

每次同步后都需要重新运行适配测试和实际训练验证；构建产物应记录本次 submodule 的 commit，便于回溯。

## 2. 自动构建

构建脚本会在当前工程目录的 `tmp/` 临时目录中复制 submodule，将 `hcu_resiliency_ext` 一并放入
`nvidia-resiliency-ext` 的 wheel，并自动注入导入入口。原始 submodule 不会被修改：

```bash
python -m hcu_resiliency_ext.build_nvidia_resiliency_ext --outdir dist-nvidia
python -m pip install dist-nvidia/nvidia_resiliency_ext-*.whl
```

构建版本从 submodule 当前 commit 推导，便于回溯；默认设置
`STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1`，如需编译 CUPTI 扩展可显式设置为 `0`。

## 3. 环境和依赖

禁用第三方库中不适用于当前 HCU 环境的 CUPTI 扩展构建：

```bash
export STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1
```

One Logger telemetry 现在是可选功能。默认情况下不会重定向 profiling，也不会要求安装
`nv-one-logger-core` 和 `nv-one-logger-training-telemetry`，容错流程使用 NVIDIA
上游 profiling 实现。

需要启用 structured telemetry 时，先安装可选依赖并设置开关：

```bash
pip install 'dist-nvidia/<wheel-file>[telemetry]'
export NVRX_ENABLE_TELEMETRY=1
export NVRX_TELEMETRY_EXPORTER=file
export NVRX_TELEMETRY_FILE=/path/to/nvrx_telemetry.json
```

其中 `<wheel-file>` 替换为实际生成的 `nvidia_resiliency_ext-*.whl` 文件名；也可以直接
单独安装两个可选依赖：

```bash
pip install nv-one-logger-core nv-one-logger-training-telemetry
```

如果只设置 `NVRX_TELEMETRY_EXPORTER` 而没有设置 `NVRX_ENABLE_TELEMETRY`，也会
为了兼容旧配置而启用 telemetry。显式设置 `NVRX_ENABLE_TELEMETRY=0` 可以关闭它。

启用 telemetry 但依赖未安装时，适配层会告警并回退到上游 profiling，不应阻断容错流程。

生产环境建议记录 Python、PyTorch、DTK 和构建时 submodule commit 的确切版本。

## 4. 验证重定向

默认配置不会重定向 profiling。可以先验证默认回退行为：

```bash
NVRX_ENABLE_TELEMETRY=0 python - <<'PY'
import nvidia_resiliency_ext.shared_utils.profiling as profiling

print(profiling.__file__)
assert "hcu_resiliency_ext" not in profiling.__file__
print("Upstream profiling is active by default")
PY
```

验证 HCU telemetry 重定向前，必须先按第 3 节安装
`nv-one-logger-core` 和 `nv-one-logger-training-telemetry`，并在启动新的 Python 进程前
显式启用 telemetry：

```bash
export NVRX_ENABLE_TELEMETRY=1

python - <<'PY'
import nvidia_resiliency_ext.shared_utils.profiling as profiling

print(profiling.__file__)
assert "hcu_resiliency_ext" in profiling.__file__
print("HCU profiling adaptor is active")
PY
```

然后在仓库根目录运行适配单元测试：

```bash
python -m pytest nvidia_resiliency_ext/tests -q
```

单元测试只验证重定向和补丁逻辑。正式使用前还需要在目标训练环境验证：

- 首次导入和重复导入不会产生重复 hook。
- profiling 事件可以正常初始化、记录和关闭。
- Slurm、MPI 和本地环境的 session metadata 正确。
- 未启用 WandB 时不会因可选依赖缺失而失败。
- 训练异常恢复路径能够完整执行。

## 卸载与回退

1. 卸载或切回原始 `nvidia-resiliency-ext` 安装。
2. 不需要回滚或修改 submodule 中的文件；重新安装未注入适配器的官方 wheel。
3. 启动新的 Python 进程。import hook 已进入当前进程的 `sys.meta_path`，不能仅通过重新导入可靠回退。

## 当前限制

- 适配层尚未提供独立的包元数据和稳定公共 API。
- 启用逻辑在导入 `nre_adaptor` 时立即执行。
- 上游版本兼容性需要逐 commit 验证。
- 构建脚本只修改临时副本，submodule 工作树保持干净。

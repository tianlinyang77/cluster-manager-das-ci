<!--
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
SPDX-License-Identifier: Apache-2.0
-->

# HCU Resiliency Extension

这是 NVIDIA [Resiliency Extension](https://github.com/NVIDIA/nvidia-resiliency-ext)
的 HCU 适配层和自动构建工程。

项目目标是在不直接修改 NVIDIA 上游 submodule 的前提下，将 HCU 环境所需的
profiling 和运行时适配逻辑集成到 `nvidia-resiliency-ext` wheel 中。安装生成的
wheel 后会自动加载适配入口；默认仍使用 NVIDIA 上游 profiling，只有显式启用
telemetry 且可选依赖完整时才会重定向到 HCU 实现。


## 适配原理

安装构建后的 wheel 后，导入上游包时会自动加载适配入口：

```text
nvidia_resiliency_ext
        │
        └── hcu_resiliency_ext.adaptor.nre_adaptor
                │
                ├── 默认：保留上游 shared_utils.profiling
                └── telemetry 启用且依赖完整：重定向到 HCU profiling
```

当前主要适配 `nvidia_resiliency_ext.shared_utils.profiling`，包括 profiling 事件、
训练周期以及 HCU 环境中的会话元数据处理。适配器使用 import hook 工作，因此不会
覆盖上游其它模块，也不会修改源代码仓库中的文件。

## 同步上游代码

submodule 跟踪 NVIDIA 上游 `main` 分支：

```bash
git submodule sync -- nvidia_resiliency_ext
git submodule update --init --remote --merge nvidia_resiliency_ext
```


## 自动构建 wheel

在项目根目录执行：

```bash
python -m hcu_resiliency_ext.build_nvidia_resiliency_ext \
    --outdir dist-nvidia
```

生成结果位于：

```text
dist-nvidia/nvidia_resiliency_ext-*.whl
```

默认跳过 CUPTI 扩展编译，适合没有 CUDA/CUPTI 开发环境的 HCU 构建机：

```bash
export STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1
```

如需编译 CUPTI 扩展，可以设置为 `0`，并确保 CUDA/CUPTI 头文件和库可用。

## 安装和使用

```bash
python -m pip install dist-nvidia/nvidia_resiliency_ext-*.whl
```

默认安装后可以直接使用上游 API，此时 profiling 保持 NVIDIA 上游实现：

```python
import nvidia_resiliency_ext
import nvidia_resiliency_ext.shared_utils.profiling as profiling

print(nvidia_resiliency_ext.__version__)
print(profiling.__file__)
```

如需启用 HCU telemetry profiling，先安装 `nv-one-logger-core` 和
`nv-one-logger-training-telemetry`，再在启动 Python 进程前设置：

```bash
export NVRX_ENABLE_TELEMETRY=1
```

此时 `profiling.__file__` 才应指向 wheel 中的
`hcu_resiliency_ext/shared_utils/profiling.py`。如果依赖缺失，适配层会告警并安全回退到
NVIDIA 上游 profiling。


## 许可证

HCU 适配代码使用 Apache License 2.0。NVIDIA 上游代码及其许可证声明随 submodule
保留，具体请参阅上游仓库中的 `LICENSE.txt` 和本项目的许可证文件。

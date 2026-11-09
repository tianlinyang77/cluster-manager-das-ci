#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Register the ST recipe in memory, then use the pinned Bridge runner."""
import os
from pathlib import Path
import runpy
import sys


def main():
    root = Path(os.environ["MEGATRON_BRIDGE_PATH"]).resolve()
    upstream = root / "3rdparty/Megatron-Bridge"
    paths = [root, root / "hcu_megatron_bridge/src", upstream / "src", upstream / "3rdparty/Megatron-LM"]
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import megatron.bridge.recipes as recipes
    from st_glm53_recipe import RECIPE_NAME, glm53_st_pretrain_config
    setattr(recipes, RECIPE_NAME, glm53_st_pretrain_config)
    sys.argv[0] = str(root / "run_recipe.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()

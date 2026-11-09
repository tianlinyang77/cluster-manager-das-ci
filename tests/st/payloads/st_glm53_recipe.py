# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Small GLM MLA/DSA/MoE pretraining workload; no HF config or weights.

Based on Bridge 4430df5388a84b0ae88c8633ebc8fe5488b8385c. This is a
reliability workload, not the production GLM53 model or a performance recipe.
"""
import os

RECIPE_NAME = "glm53_st_pretrain_config"


def settings(environ=None):
    environ = os.environ if environ is None else environ
    def number(name, default):
        raw = environ.get(name, default)
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be an integer, not a boolean")
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    values = {name: number(name, default) for name, default in (
        ("NNODES", 1), ("GPUS", 4), ("NUM_LAYERS", 2), ("SEQ_LENGTH", 128),
        ("TP", 1), ("PP", 1), ("CP", 1), ("EP", 1), ("ETP", 1),
        ("MICRO_BATCH_SIZE", 1), ("TRAIN_ITERS", 2000),
    )}
    world = values["NNODES"] * values["GPUS"]
    mp = values["TP"] * values["PP"] * values["CP"]
    if world % mp or values["NUM_LAYERS"] < 2 or values["NUM_LAYERS"] % values["PP"]:
        raise ValueError("world size and layers must fit TP/PP/CP, with at least two layers")
    if (4 % values["TP"] or 8 % values["EP"] or values["TP"] % values["ETP"]
            or world % (values["PP"] * values["EP"] * values["ETP"])):
        raise ValueError("parallel sizes do not fit the small model's heads or eight experts")
    if values["SEQ_LENGTH"] < 64 or values["SEQ_LENGTH"] % values["CP"]:
        raise ValueError("SEQ_LENGTH must be >=64 and divisible by CP")
    dp = world // mp
    values["GLOBAL_BATCH_SIZE"] = number("GLOBAL_BATCH_SIZE", dp * values["MICRO_BATCH_SIZE"])
    if values["GLOBAL_BATCH_SIZE"] % (dp * values["MICRO_BATCH_SIZE"]):
        raise ValueError("GLOBAL_BATCH_SIZE must be divisible by DP * micro batch size")
    return values


def manager_config(environ):
    """Use the same topology/batch values for reliability's flat parameter record."""
    v = settings(environ)
    return {"tensor-model-parallel-size": v["TP"], "pipeline-model-parallel-size": v["PP"],
            "context-parallel-size": v["CP"], "expert-model-parallel-size": v["EP"],
            "expert-tensor-parallel-size": v["ETP"], "num-layers": v["NUM_LAYERS"],
            "hidden-size": 512, "ffn-hidden-size": 1024, "num-attention-heads": 8,
            "num-experts": 8, "moe-ffn-hidden-size": 256, "seq-length": v["SEQ_LENGTH"],
            "micro-batch-size": v["MICRO_BATCH_SIZE"], "global-batch-size": v["GLOBAL_BATCH_SIZE"],
            "world-size": v["NNODES"] * v["GPUS"], "save-interval": int(environ.get("SAVE_INTERVAL", 10))}


def glm53_st_pretrain_config():
    import torch.nn.functional as F
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.bridge.models.mla_provider import MLAModelProvider
    from megatron.bridge.recipes.common import _pretrain_common
    from megatron.bridge.training.tokenizers.config import TokenizerConfig

    values = settings()
    cfg = _pretrain_common()
    cfg.model = MLAModelProvider(
        num_layers=values["NUM_LAYERS"], hidden_size=512, ffn_hidden_size=1024,
        num_attention_heads=8, num_query_groups=8, vocab_size=4096, seq_length=values["SEQ_LENGTH"],
        normalization="RMSNorm", activation_func=F.silu, gated_linear_unit=True,
        add_bias_linear=False, share_embeddings_and_output_weights=False,
        position_embedding_type="rope", rotary_base=10000, qk_layernorm=True,
        multi_latent_attention=True, q_lora_rank=256, kv_lora_rank=128,
        qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=128,
        transformer_layer_spec=get_transformer_block_with_experimental_attention_variant_spec,
        experimental_attention_variant="dsa", dsa_kernel_backend="none",
        dsa_indexer_n_heads=4, dsa_indexer_head_dim=128, dsa_indexer_topk=64,
        dsa_indexer_rotate_activation=False, dsa_indexer_k_norm_epsilon=1e-6,
        dsa_indexer_loss_coeff=0.001, dsa_indexer_use_sparse_loss=True,
        num_moe_experts=8, moe_ffn_hidden_size=256, moe_shared_expert_intermediate_size=256,
        moe_layer_freq=[0] + [1] * (values["NUM_LAYERS"] - 1),
        moe_router_topk=2, moe_router_score_function="sigmoid", moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True, moe_router_pre_softmax=True,
        moe_router_load_balancing_type="seq_aux_loss", moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False, moe_shared_expert_overlap=False,
        moe_router_fusion=False, moe_permute_fusion=False,
        tensor_model_parallel_size=values["TP"], pipeline_model_parallel_size=values["PP"],
        context_parallel_size=values["CP"], expert_model_parallel_size=values["EP"],
        expert_tensor_parallel_size=values["ETP"], sequence_parallel=values["TP"] > 1,
        pipeline_model_parallel_layout=None, mtp_num_layers=None,
        attention_dropout=0.0, hidden_dropout=0.0, calculate_per_token_loss=True,
        cross_entropy_loss_fusion=False, gradient_accumulation_fusion=False,
    )
    cfg.tokenizer = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=4096,
                                    use_tokenizer_vocab_size=True)
    cfg.dataset.seq_length = values["SEQ_LENGTH"]
    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.num_workers = 0
    cfg.train.train_iters = values["TRAIN_ITERS"]
    cfg.train.micro_batch_size = values["MICRO_BATCH_SIZE"]
    cfg.train.global_batch_size = values["GLOBAL_BATCH_SIZE"]
    cfg.train.exit_interval = None
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.scheduler.lr_decay_iters = values["TRAIN_ITERS"]
    cfg.scheduler.lr_warmup_iters = min(2, values["TRAIN_ITERS"] - 1)
    cfg.checkpoint.save = os.environ["CKPT_DIR"]
    cfg.checkpoint.load = cfg.checkpoint.save
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.save_interval = int(os.environ.get("SAVE_INTERVAL", "10"))
    local_interval = int(os.environ.get("LOCAL_SAVE_INTERVAL", "5"))
    if (local_interval < 0 or cfg.checkpoint.save_interval <= 0
            or local_interval > 0 and (cfg.checkpoint.save_interval <= local_interval
                                     or cfg.checkpoint.save_interval % local_interval)):
        raise ValueError("local interval must be 0 (disabled) or positive; global must be positive and, when local is enabled, an integer multiple greater than local")
    cfg.checkpoint.non_persistent_save_interval = local_interval or None
    cfg.checkpoint.non_persistent_ckpt_type = "local" if local_interval else None
    cfg.checkpoint.non_persistent_local_ckpt_dir = os.environ["LOCAL_CKPT_DIR"] if local_interval else None
    cfg.checkpoint.non_persistent_local_ckpt_algo = "atomic"
    cfg.checkpoint.async_save = False
    cfg.checkpoint.replication = False  # Both roots use shared storage.
    cfg.checkpoint.most_recent_k = 3
    cfg.checkpoint.also_save_hf_checkpoint = False
    cfg.inprocess_restart = None
    cfg.logger.log_interval = 1
    cfg.logger.logging_level = 20
    cfg.logger.log_throughput = True
    cfg.logger.tensorboard_dir = None
    cfg.logger.save_config_filepath = os.path.join(os.environ["LOG_DIR"], "resolved-config.yaml")
    return cfg

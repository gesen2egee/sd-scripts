"""Build runtime config files and commands for ANIMA UI jobs."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, List

from .dataset_inspector import detect_image_dirs, normalize_path

ROOT_DIR = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT_DIR / "ui" / "runtime"


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _not_none(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def _is_list_of_tables(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(item, dict) for item in value)


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return _toml_scalar(str(value))


def _toml_value(value: Any) -> str:
    if isinstance(value, list):
        if _is_list_of_tables(value):
            raise ValueError("list of tables must be emitted by table writer")
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    return _toml_scalar(value)


def _write_table(lines: List[str], prefix: str | None, data: Dict[str, Any]) -> None:
    if prefix:
        lines.append(f"[{prefix}]")

    nested_dicts: List[tuple[str, Dict[str, Any]]] = []
    nested_tables: List[tuple[str, List[Dict[str, Any]]]] = []

    for key, value in data.items():
        if isinstance(value, dict):
            nested_dicts.append((key, value))
        elif _is_list_of_tables(value):
            nested_tables.append((key, value))
        elif value is not None:
            lines.append(f"{key} = {_toml_value(value)}")

    for key, value in nested_dicts:
        lines.append("")
        child = f"{prefix}.{key}" if prefix else key
        _write_table(lines, child, value)

    for key, table_list in nested_tables:
        child = f"{prefix}.{key}" if prefix else key
        for item in table_list:
            lines.append("")
            lines.append(f"[[{child}]]")

            child_dicts: List[tuple[str, Dict[str, Any]]] = []
            child_tables: List[tuple[str, List[Dict[str, Any]]]] = []
            for item_key, item_value in item.items():
                if isinstance(item_value, dict):
                    child_dicts.append((item_key, item_value))
                elif _is_list_of_tables(item_value):
                    child_tables.append((item_key, item_value))
                elif item_value is not None:
                    lines.append(f"{item_key} = {_toml_value(item_value)}")

            for item_key, item_dict in child_dicts:
                lines.append("")
                _write_table(lines, f"{child}.{item_key}", item_dict)

            for item_key, item_tables in child_tables:
                nested_name = f"{child}.{item_key}"
                for t in item_tables:
                    lines.append("")
                    lines.append(f"[[{nested_name}]]")
                    for t_key, t_val in t.items():
                        if t_val is not None and not isinstance(t_val, (dict, list)):
                            lines.append(f"{t_key} = {_toml_value(t_val)}")
                        elif isinstance(t_val, list) and not _is_list_of_tables(t_val):
                            lines.append(f"{t_key} = {_toml_value(t_val)}")


def _write_toml(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    _write_table(lines, None, payload)
    content = "\n".join(lines).strip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _default_resolution_profiles(train_batch_size: int) -> List[Dict[str, Any]]:
    return [
        {
            "resolution": [1024, 1024],
            "batch_size": train_batch_size,
            "enable_bucket": True,
            "min_bucket_reso": 512,
            "max_bucket_reso": 1280,
            "bucket_reso_steps": 16,
            "bucket_no_upscale": True,
        },
        {
            "resolution": [640, 640],
            "batch_size": train_batch_size,
            "enable_bucket": True,
            "min_bucket_reso": 256,
            "max_bucket_reso": 1024,
            "bucket_reso_steps": 16,
            "bucket_no_upscale": True,
        },
        {
            "resolution": [768, 768],
            "batch_size": train_batch_size,
            "enable_bucket": True,
            "min_bucket_reso": 384,
            "max_bucket_reso": 1152,
            "bucket_reso_steps": 16,
            "bucket_no_upscale": True,
        },
    ]


def _make_subset(image_dir: str, repeats: int, is_reg: bool, dataset: Dict[str, Any]) -> Dict[str, Any]:
    subset = {
        "image_dir": image_dir,
        "num_repeats": repeats,
        "caption_extension": dataset.get("caption_extension", ".txt"),
        "color_aug": bool(dataset.get("color_aug", False)),
        "flip_aug": bool(dataset.get("flip_aug", False)),
        "random_crop": bool(dataset.get("random_crop", False)),
        "shuffle_caption": bool(dataset.get("shuffle_caption", False)),
        "caption_dropout_rate": float(dataset.get("caption_dropout_rate", 0.0) or 0.0),
        "caption_tag_dropout_rate": float(dataset.get("caption_tag_dropout_rate", 0.0) or 0.0),
        "token_warmup_step": float(dataset.get("token_warmup_step", 0.0) or 0.0),
    }
    if is_reg:
        subset["is_reg"] = True
    return subset


def build_dataset_config(profile: Dict[str, Any], runtime_dir: Path) -> Path:
    dataset = profile.get("dataset", {})
    trainer = profile.get("trainer", {})

    dataset_path = normalize_path(dataset.get("dataset_path", ""))
    reg_dataset_path = normalize_path(dataset.get("reg_dataset_path", ""))
    repeats = _as_int(dataset.get("repeats")) or 1
    train_batch_size = _as_int(trainer.get("train_batch_size")) or 1

    train_dirs = detect_image_dirs(dataset_path) or ([dataset_path] if dataset_path else [])
    reg_dirs = detect_image_dirs(reg_dataset_path) if reg_dataset_path else []

    profiles = dataset.get("resolution_profiles")
    if not isinstance(profiles, list) or not profiles:
        profiles = _default_resolution_profiles(train_batch_size)

    datasets = []
    for item in profiles:
        res = item.get("resolution", [1024, 1024])
        if not isinstance(res, list) or len(res) != 2:
            res = [1024, 1024]

        ds = {
            "resolution": [int(res[0]), int(res[1])],
            "batch_size": _as_int(item.get("batch_size")) or train_batch_size,
            "enable_bucket": bool(item.get("enable_bucket", True)),
            "min_bucket_reso": _as_int(item.get("min_bucket_reso")) or 256,
            "max_bucket_reso": _as_int(item.get("max_bucket_reso")) or 1024,
            "bucket_reso_steps": _as_int(item.get("bucket_reso_steps")) or 16,
            "bucket_no_upscale": bool(item.get("bucket_no_upscale", True)),
            "subsets": [],
        }

        for image_dir in train_dirs:
            ds["subsets"].append(_make_subset(image_dir, repeats, False, dataset))

        for image_dir in reg_dirs:
            ds["subsets"].append(_make_subset(image_dir, repeats, True, dataset))

        datasets.append(ds)

    payload = {
        "general": {
            "shuffle_caption": bool(dataset.get("shuffle_caption", False)),
            "caption_extension": dataset.get("caption_extension", ".txt"),
        },
        "datasets": datasets,
    }

    dataset_config_path = runtime_dir / "dataset_config.toml"
    _write_toml(dataset_config_path, payload)
    return dataset_config_path


def build_sample_prompts(profile: Dict[str, Any], runtime_dir: Path) -> Path | None:
    sample = profile.get("sample", {})
    if not sample.get("enabled"):
        return None

    prompt = (sample.get("prompt") or "").strip()
    if not prompt:
        return None

    parts = [prompt]
    if sample.get("negative_prompt"):
        parts.append(f"--n {sample['negative_prompt']}")
    if sample.get("width"):
        parts.append(f"--w {int(sample['width'])}")
    if sample.get("height"):
        parts.append(f"--h {int(sample['height'])}")
    if sample.get("seed") not in (None, ""):
        parts.append(f"--d {int(sample['seed'])}")
    if sample.get("sample_steps"):
        parts.append(f"--s {int(sample['sample_steps'])}")
    if sample.get("cfg_scale") not in (None, ""):
        parts.append(f"--g {float(sample['cfg_scale'])}")
    if sample.get("sample_sampler"):
        parts.append(f"--ss {sample['sample_sampler']}")

    prompt_path = runtime_dir / "sample_prompts.txt"
    prompt_path.write_text(" ".join(parts) + "\n", encoding="utf-8")
    return prompt_path


def _resolve_network_module(profile: Dict[str, Any]) -> str:
    fmt = profile.get("format", {})
    module = (fmt.get("network_module") or "").strip()
    if module:
        return module

    model_type = (fmt.get("model_type") or "LyCORIS").strip().lower()
    if model_type == "lora":
        return "networks.lora_anima"
    return "lycoris.kohya"


def build_train_config(profile: Dict[str, Any], runtime_dir: Path, dataset_config_path: Path, sample_prompt_path: Path | None) -> Path:
    model = profile.get("model", {})
    precision = profile.get("precision_opt", {})
    trainer = profile.get("trainer", {})
    advanced = profile.get("advanced", {})
    fmt = profile.get("format", {})
    saving = profile.get("saving", {})
    resume = profile.get("resume", {})
    sample = profile.get("sample", {})

    network_args = fmt.get("network_args") if isinstance(fmt.get("network_args"), list) else []
    optimizer_args = trainer.get("optimizer_args") if isinstance(trainer.get("optimizer_args"), list) else []

    max_train_epochs = _as_int(fmt.get("accus_epochs")) or _as_int(trainer.get("max_train_epochs"))
    train_batch_size = _as_int(fmt.get("batch_size")) or _as_int(trainer.get("train_batch_size")) or 1

    payload: Dict[str, Any] = {
        "model": _not_none(
            {
                "pretrained_model_name_or_path": model.get("pretrained_model_name_or_path"),
                "qwen3": model.get("qwen3"),
                "vae": model.get("vae"),
                "llm_adapter_path": model.get("llm_adapter_path") or None,
                "t5_tokenizer_path": model.get("t5_tokenizer_path") or None,
            }
        ),
        "dataset": {"dataset_config": str(dataset_config_path)},
        "output": _not_none(
            {
                "output_dir": saving.get("output_dir"),
                "logging_dir": saving.get("logging_dir"),
                "output_name": saving.get("output_name"),
                "save_model_as": saving.get("save_model_as"),
                "save_precision": precision.get("save_precision"),
                "save_every_n_steps": _as_int(saving.get("save_every_n_steps")),
                "save_every_n_epochs": _as_int(saving.get("save_every_n_epochs")),
                "save_last_n_steps": _as_int(saving.get("save_last_n_steps")),
                "save_last_n_epochs": _as_int(saving.get("save_last_n_epochs")),
            }
        ),
        "network": _not_none(
            {
                "network_module": _resolve_network_module(profile),
                "network_dim": _as_int(fmt.get("network_dim")),
                "network_alpha": _as_float(fmt.get("network_alpha")),
                "network_args": network_args or None,
                "network_train_unet_only": bool(fmt.get("network_train_unet_only", True)),
            }
        ),
        "train": _not_none(
            {
                "learning_rate": _as_float(trainer.get("learning_rate")),
                "text_encoder_lr": _as_float(trainer.get("text_encoder_lr")),
                "llm_adapter_lr": _as_float(trainer.get("llm_adapter_lr")),
                "optimizer_type": trainer.get("optimizer_type"),
                "optimizer_args": optimizer_args or None,
                "lr_scheduler": trainer.get("lr_scheduler"),
                "lr_warmup_steps": _as_int(trainer.get("lr_warmup_steps")),
                "train_batch_size": train_batch_size,
                "max_train_steps": _as_int(trainer.get("max_train_steps")),
                "max_train_epochs": max_train_epochs,
                "max_grad_norm": _as_float(trainer.get("max_grad_norm")),
                "gradient_checkpointing": bool(precision.get("gradient_checkpointing", False)),
                "resume": resume.get("path") if resume.get("enabled") else None,
            }
        ),
        "memory": _not_none(
            {
                "mixed_precision": precision.get("mixed_precision"),
                "cache_latents": bool(precision.get("cache_latents", False)),
                "cache_latents_to_disk": bool(precision.get("cache_latents_to_disk", False)),
                "cache_text_encoder_outputs": bool(precision.get("cache_text_encoder_outputs", False)),
                "cache_text_encoder_outputs_to_disk": bool(precision.get("cache_text_encoder_outputs_to_disk", False)),
                "blocks_to_swap": _as_int(precision.get("blocks_to_swap")),
                "xformers": bool(precision.get("xformers", False)),
                "split_attn": bool(precision.get("split_attn", False)),
                "attn_mode": precision.get("attn_mode") or None,
            }
        ),
        "anima": _not_none(
            {
                "timestep_sampling": advanced.get("timestep_sampling"),
                "discrete_flow_shift": _as_float(advanced.get("discrete_flow_shift")),
                "sigmoid_scale": _as_float(advanced.get("sigmoid_scale")),
                "noise_offset": _as_float(advanced.get("noise_offset")),
                "ip_noise_gamma": _as_float(advanced.get("ip_noise_gamma")),
                "min_timestep": _as_int(advanced.get("min_timestep")),
                "max_timestep": _as_int(advanced.get("max_timestep")),
            }
        ),
    }

    if sample_prompt_path:
        payload.setdefault("sample", {})
        payload["sample"].update(
            _not_none(
                {
                    "sample_prompts": str(sample_prompt_path),
                    "sample_every_n_steps": _as_int(sample.get("sample_every_n_steps")),
                    "sample_every_n_epochs": _as_int(sample.get("sample_every_n_epochs")),
                    "sample_sampler": sample.get("sample_sampler"),
                }
            )
        )

    train_config_path = runtime_dir / "train_config.toml"
    _write_toml(train_config_path, payload)
    return train_config_path


def _flatten_extra_args(raw_args: List[str] | None) -> List[str]:
    if not raw_args:
        return []
    tokens: List[str] = []
    for arg in raw_args:
        if not arg or not str(arg).strip():
            continue
        tokens.extend(shlex.split(str(arg), posix=False))
    return tokens


def build_job(runtime_job_dir: Path, profile: Dict[str, Any]) -> Dict[str, Any]:
    runtime_job_dir.mkdir(parents=True, exist_ok=True)
    (runtime_job_dir / "profile_snapshot.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_config_path = build_dataset_config(profile, runtime_job_dir)
    sample_prompt_path = build_sample_prompts(profile, runtime_job_dir)
    train_config_path = build_train_config(profile, runtime_job_dir, dataset_config_path, sample_prompt_path)

    precision = profile.get("precision_opt", {})
    resume = profile.get("resume", {})
    raw_args = _flatten_extra_args(profile.get("raw_args"))

    accelerate_exe = ROOT_DIR / "venv" / "Scripts" / "accelerate.exe"
    accelerate_cmd = str(accelerate_exe if accelerate_exe.exists() else "accelerate")

    cmd = [
        accelerate_cmd,
        "launch",
        "--dynamo_backend",
        "no",
        "--dynamo_mode",
        "default",
        "--mixed_precision",
        str(precision.get("mixed_precision") or "bf16"),
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--num_cpu_threads_per_process",
        "2",
        str(ROOT_DIR / "anima_train_network.py"),
        "--config_file",
        str(train_config_path),
        "--dataset_config",
        str(dataset_config_path),
    ]

    if resume.get("enabled") and resume.get("path"):
        cmd.extend(["--resume", str(resume.get("path"))])

    if precision.get("xformers"):
        cmd.append("--xformers")
    if precision.get("split_attn"):
        cmd.append("--split_attn")
    if precision.get("attn_mode"):
        cmd.extend(["--attn_mode", str(precision.get("attn_mode"))])
    if precision.get("blocks_to_swap") not in (None, ""):
        cmd.extend(["--blocks_to_swap", str(precision.get("blocks_to_swap"))])

    cmd.extend(raw_args)

    return {
        "train_config_path": str(train_config_path),
        "dataset_config_path": str(dataset_config_path),
        "sample_prompts_path": str(sample_prompt_path) if sample_prompt_path else None,
        "command": cmd,
    }


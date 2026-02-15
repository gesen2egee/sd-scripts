"""Stdlib HTTP backend for ANIMA UI."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

from .dataset_inspector import inspect_dataset, normalize_path
from .defaults_provider import load_defaults
from .runner import BatchRunner

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "ui" / "frontend"
RUNTIME_DIR = ROOT_DIR / "ui" / "runtime"

runner = BatchRunner()
DEFAULTS, DEFAULTS_WARNING = load_defaults()

PROFILE_SCHEMA_V2: Dict[str, Any] = {
    "version": 2,
    "steps": [
        {
            "id": "model",
            "sections": ["model", "precision_opt", "format"],
            "required": ["model.pretrained_model_name_or_path", "model.qwen3", "model.vae"],
        },
        {
            "id": "dataset",
            "sections": ["dataset", "advanced"],
            "required": ["dataset.dataset_path"],
        },
        {
            "id": "trainer",
            "sections": ["trainer", "saving", "resume", "sample"],
            "required": ["saving.output_dir", "saving.output_name"],
        },
        {
            "id": "execute",
            "sections": ["execute"],
            "required": [],
        },
    ],
    "sections": [
        "model",
        "precision_opt",
        "dataset",
        "trainer",
        "advanced",
        "format",
        "saving",
        "resume",
        "sample",
        "execute",
    ],
}


def _bool(value: Any) -> bool:
    return bool(value)


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _append_issue(index: Dict[str, List[str]], path: str, message: str) -> None:
    index.setdefault(path, []).append(message)


def _make_profile_template(index: int = 1) -> Dict[str, Any]:
    now = time.time()
    return {
        "id": uuid.uuid4().hex,
        "name": f"Task {index}",
        "run_enabled": True,
        "model": {
            "pretrained_model_name_or_path": "",
            "qwen3": "",
            "vae": "",
            "llm_adapter_path": "",
            "t5_tokenizer_path": "",
        },
        "precision_opt": {
            "mixed_precision": "bf16",
            "save_precision": "bf16",
            "xformers": False,
            "attn_mode": "torch",
            "split_attn": False,
            "gradient_checkpointing": True,
            "blocks_to_swap": "",
            "cache_latents": False,
            "cache_latents_to_disk": False,
            "cache_text_encoder_outputs": False,
            "cache_text_encoder_outputs_to_disk": False,
        },
        "dataset": {
            "dataset_path": "",
            "reg_dataset_path": "",
            "repeats": 1,
            "caption_extension": ".txt",
            "color_aug": False,
            "flip_aug": False,
            "random_crop": False,
            "shuffle_caption": False,
            "caption_dropout_rate": 0,
            "caption_tag_dropout_rate": 0,
            "token_warmup_step": 0,
        },
        "trainer": {
            "learning_rate": 1,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 300,
            "optimizer_type": "AdamW",
            "optimizer_args": [],
            "text_encoder_lr": 0,
            "llm_adapter_lr": 0,
            "train_batch_size": 1,
            "max_train_steps": 3000,
            "max_train_epochs": "",
            "max_grad_norm": 1,
        },
        "advanced": {
            "timestep_sampling": "sigmoid",
            "discrete_flow_shift": 1,
            "sigmoid_scale": 1,
            "noise_offset": "",
            "ip_noise_gamma": "",
            "min_timestep": "",
            "max_timestep": "",
        },
        "format": {
            "model_type": "LyCORIS",
            "network_module": "",
            "network_dim": 32,
            "network_alpha": 1,
            "accus_epochs": "",
            "batch_size": 1,
            "network_train_unet_only": True,
            "network_args": [],
        },
        "saving": {
            "output_dir": "",
            "logging_dir": "",
            "output_name": "",
            "save_model_as": "safetensors",
            "save_every_n_steps": 250,
            "save_every_n_epochs": "",
            "save_last_n_steps": "",
            "save_last_n_epochs": "",
        },
        "resume": {"enabled": False, "path": ""},
        "sample": {
            "enabled": False,
            "sample_every_n_steps": "",
            "sample_every_n_epochs": "",
            "sample_sampler": "ddim",
            "sample_steps": 20,
            "cfg_scale": 7,
            "prompt": "",
            "negative_prompt": "",
            "width": 1024,
            "height": 1024,
            "seed": "",
        },
        "raw_args": [],
        "execute": {"tb_logdir": "", "tb_port": 6006, "validation": None},
        "meta": {
            "created_at": now,
            "updated_at": now,
            "validation_state": {
                "errors": [],
                "warnings": [],
                "field_errors": {},
                "field_warnings": {},
                "valid": False,
            },
            "last_run_snapshot": None,
        },
    }


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _normalize_profile(profile: Any, index: int) -> Dict[str, Any]:
    template = _make_profile_template(index)
    if not isinstance(profile, dict):
        return template

    merged = _deep_merge(template, profile)
    merged["id"] = str(merged.get("id") or uuid.uuid4().hex)
    merged["name"] = str(merged.get("name") or f"Task {index}")
    merged["run_enabled"] = bool(merged.get("run_enabled", True))
    merged["meta"]["updated_at"] = time.time()
    merged["meta"]["created_at"] = float(merged["meta"].get("created_at") or time.time())
    return merged


def _migrate_state(candidate: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    source = candidate.get("ui_state") if isinstance(candidate.get("ui_state"), dict) else candidate
    version = int(source.get("version") or candidate.get("version") or 1)
    if version == 1:
        warnings.append("migrated_from_v1")
    elif version != 2:
        warnings.append(f"unsupported_version_{version}_coerced_to_v2")

    profiles = source.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles is required and must be a non-empty array")

    normalized_profiles = [_normalize_profile(item, idx + 1) for idx, item in enumerate(profiles)]
    profile_ids = [p["id"] for p in normalized_profiles]

    active_profile_id = str(source.get("active_profile_id") or source.get("activeProfileId") or "")
    if active_profile_id not in profile_ids:
        active_profile_id = profile_ids[0]

    active_step = str(source.get("active_step") or source.get("activeStep") or "model")
    known_steps = {step["id"] for step in PROFILE_SCHEMA_V2["steps"]}
    if active_step not in known_steps:
        active_step = "model"

    migrated = {
        "version": 2,
        "lang": str(source.get("lang") or "zh-TW"),
        "layout_mode": str(source.get("layout_mode") or "dashboard"),
        "active_profile_id": active_profile_id,
        "active_step": active_step,
        "profiles": normalized_profiles,
        "recent_paths": source.get("recent_paths") if isinstance(source.get("recent_paths"), list) else [],
    }
    return migrated, warnings


def _profile_required_fields() -> List[str]:
    result: List[str] = []
    for step in PROFILE_SCHEMA_V2["steps"]:
        result.extend(step.get("required", []))
    return result


def validate_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    field_errors: Dict[str, List[str]] = {}
    field_warnings: Dict[str, List[str]] = {}

    model = profile.get("model", {})
    dataset = profile.get("dataset", {})
    precision = profile.get("precision_opt", {})
    trainer = profile.get("trainer", {})
    saving = profile.get("saving", {})
    sample = profile.get("sample", {})
    resume = profile.get("resume", {})

    for key in ("pretrained_model_name_or_path", "qwen3", "vae"):
        value = str(model.get(key) or "").strip()
        if not value:
            path = f"model.{key}"
            errors.append(f"{path} is required")
            _append_issue(field_errors, path, "Required field")

    dataset_path = normalize_path(dataset.get("dataset_path", ""))
    if not dataset_path:
        errors.append("dataset.dataset_path is required")
        _append_issue(field_errors, "dataset.dataset_path", "Required field")
    else:
        stats = inspect_dataset(dataset_path)
        if not stats.get("exists"):
            errors.append("dataset.dataset_path not found")
            _append_issue(field_errors, "dataset.dataset_path", "Path not found")
        elif stats.get("image_count", 0) <= 0:
            errors.append("dataset has no images")
            _append_issue(field_errors, "dataset.dataset_path", "No images found")
        if stats.get("caption_invalid_count", 0) > 0:
            warnings.append("dataset has images with missing/invalid captions")
            _append_issue(field_warnings, "dataset.dataset_path", "Some images have invalid or missing captions")

    output_dir = normalize_path(saving.get("output_dir", ""))
    if not output_dir:
        errors.append("saving.output_dir is required")
        _append_issue(field_errors, "saving.output_dir", "Required field")

    output_name = str(saving.get("output_name") or "").strip()
    if not output_name:
        errors.append("saving.output_name is required")
        _append_issue(field_errors, "saving.output_name", "Required field")

    if resume.get("enabled") and not str(resume.get("path") or "").strip():
        errors.append("resume.path is required when resume.enabled is true")
        _append_issue(field_errors, "resume.path", "Required when resume is enabled")

    cache_enabled = _bool(precision.get("cache_latents")) or _bool(precision.get("cache_text_encoder_outputs"))
    aug_enabled = any(
        [
            _bool(dataset.get("color_aug")),
            _bool(dataset.get("flip_aug")),
            _bool(dataset.get("random_crop")),
            _bool(dataset.get("shuffle_caption")),
            _to_float(dataset.get("caption_dropout_rate")) > 0,
            _to_float(dataset.get("caption_tag_dropout_rate")) > 0,
            _to_float(dataset.get("token_warmup_step")) > 0,
        ]
    )

    if cache_enabled and aug_enabled:
        errors.append("cache_latents/cache_text_encoder_outputs cannot be used with data augmentation")
        _append_issue(field_errors, "precision_opt.cache_latents", "Cannot combine cache with augmentation")
        _append_issue(field_errors, "precision_opt.cache_text_encoder_outputs", "Cannot combine cache with augmentation")
        _append_issue(field_errors, "dataset.color_aug", "Disable augmentation when cache is enabled")
        _append_issue(field_errors, "dataset.flip_aug", "Disable augmentation when cache is enabled")
        _append_issue(field_errors, "dataset.random_crop", "Disable augmentation when cache is enabled")
        _append_issue(field_errors, "dataset.shuffle_caption", "Disable augmentation when cache is enabled")

    attn_mode = str(precision.get("attn_mode") or "").strip().lower()
    if attn_mode == "xformers" and not _bool(precision.get("split_attn")):
        warnings.append("attn_mode=xformers typically requires split_attn=true")
        _append_issue(field_warnings, "precision_opt.split_attn", "xformers usually pairs with split_attn")

    if sample.get("enabled"):
        if not str(sample.get("prompt") or "").strip():
            errors.append("sample.prompt is required when sample.enabled is true")
            _append_issue(field_errors, "sample.prompt", "Required when sample is enabled")
        if not sample.get("sample_every_n_steps") and not sample.get("sample_every_n_epochs"):
            warnings.append("sample enabled but neither sample_every_n_steps nor sample_every_n_epochs is set")
            _append_issue(field_warnings, "sample.sample_every_n_steps", "Set step or epoch cadence")
            _append_issue(field_warnings, "sample.sample_every_n_epochs", "Set step or epoch cadence")

    if trainer.get("max_train_steps") in (None, "") and trainer.get("max_train_epochs") in (None, ""):
        warnings.append("Neither max_train_steps nor max_train_epochs is set")
        _append_issue(field_warnings, "trainer.max_train_steps", "Set max_train_steps or max_train_epochs")
        _append_issue(field_warnings, "trainer.max_train_epochs", "Set max_train_steps or max_train_epochs")

    for required in _profile_required_fields():
        value = profile
        valid = True
        for part in required.split("."):
            if not isinstance(value, dict) or part not in value:
                valid = False
                break
            value = value.get(part)
        if not valid or (isinstance(value, str) and not value.strip()):
            if required not in field_errors:
                _append_issue(field_errors, required, "Required field")

    return {
        "errors": errors,
        "warnings": warnings,
        "field_errors": field_errors,
        "field_warnings": field_warnings,
        "valid": len(errors) == 0,
    }


def build_batch_summary(status: Dict[str, Any]) -> Dict[str, Any]:
    jobs = status.get("jobs", [])
    counts = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0, "cancelled": 0}
    last_error = None
    current_job = None
    for job in jobs:
        state = str(job.get("status") or "queued")
        if state in counts:
            counts[state] += 1
        if state == "failed" and not last_error:
            last_error = {"name": job.get("name"), "message": job.get("message"), "id": job.get("id")}
        if job.get("id") == status.get("current_job_id") or (state == "running" and current_job is None):
            current_job = {
                "id": job.get("id"),
                "name": job.get("name"),
                "status": state,
                "message": job.get("message"),
            }

    if not last_error:
        for job in reversed(jobs):
            if job.get("status") == "failed":
                last_error = {"name": job.get("name"), "message": job.get("message"), "id": job.get("id")}
                break

    return {
        "batch_running": bool(status.get("batch_running")),
        "stop_requested": bool(status.get("stop_requested")),
        "total_jobs": len(jobs),
        "queued": counts["queued"],
        "running": counts["running"],
        "succeeded": counts["succeeded"],
        "failed": counts["failed"],
        "cancelled": counts["cancelled"],
        "queue_remaining": counts["queued"],
        "current_job": current_job,
        "last_error": last_error,
        "tensorboard": status.get("tensorboard", {}),
    }


def _json(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    body = handler.rfile.read(length)
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return {}


def _serve_static(handler: BaseHTTPRequestHandler, relative_path: str) -> None:
    requested = (relative_path or "index.html").lstrip("/")
    file_path = (FRONTEND_DIR / requested).resolve()
    if not str(file_path).startswith(str(FRONTEND_DIR.resolve())) or not file_path.exists() or not file_path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return

    mime, _ = mimetypes.guess_type(file_path.name)
    data = file_path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", (mime or "application/octet-stream") + "; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_sse_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()


def _send_sse_event(handler: BaseHTTPRequestHandler, event: str, payload: Dict[str, Any], event_id: int) -> None:
    data = json.dumps(payload, ensure_ascii=False)
    lines = [f"id: {event_id}", f"event: {event}"]
    for line in data.splitlines() or ["{}"]:
        lines.append(f"data: {line}")
    lines.append("")
    raw = "\n".join(lines).encode("utf-8")
    handler.wfile.write(raw)
    handler.wfile.flush()


class UIHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        global DEFAULTS, DEFAULTS_WARNING
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            _json(self, 200, {"ok": True})
            return

        if path == "/api/defaults":
            params = parse_qs(parsed.query)
            if params.get("refresh", ["0"])[0] == "1":
                DEFAULTS, DEFAULTS_WARNING = load_defaults()
            payload = {"ok": True, "defaults": DEFAULTS}
            if DEFAULTS_WARNING:
                payload["warning"] = DEFAULTS_WARNING
            _json(self, 200, payload)
            return

        if path == "/api/profile/schema":
            _json(self, 200, {"ok": True, "schema": PROFILE_SCHEMA_V2})
            return

        if path == "/api/batch/status":
            _json(self, 200, {"ok": True, **runner.get_status()})
            return

        if path == "/api/batch/summary":
            status = runner.get_status()
            _json(self, 200, {"ok": True, "summary": build_batch_summary(status)})
            return

        if path == "/api/batch/events":
            _send_sse_headers(self)
            event_id = 0
            last_hash = None
            try:
                while True:
                    status = runner.get_status()
                    summary = build_batch_summary(status)
                    snapshot = {"status": status, "summary": summary, "ts": time.time()}
                    digest = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
                    if digest != last_hash:
                        event_id += 1
                        _send_sse_event(self, "snapshot", snapshot, event_id)
                        last_hash = digest
                    else:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                return

        if path in ("/", ""):
            _serve_static(self, "index.html")
            return

        if path.startswith("/"):
            _serve_static(self, path[1:])
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        body = _read_json(self)
        path = urlparse(self.path).path

        if path == "/api/dataset/inspect":
            result = inspect_dataset(body.get("path", ""))
            _json(self, 200, {"ok": True, "result": result})
            return

        if path == "/api/profile/validate":
            profile = body.get("profile") or {}
            checked = validate_profile(profile)
            _json(self, 200, {"ok": True, **checked})
            return

        if path == "/api/profile/import":
            try:
                normalized_state, warnings = _migrate_state(body or {})
            except Exception as exc:
                _json(self, 400, {"ok": False, "error": "invalid_import_payload", "detail": str(exc)})
                return
            _json(self, 200, {"ok": True, "state": normalized_state, "warnings": warnings})
            return

        if path == "/api/batch/start":
            profiles = body.get("profiles") or []
            all_errors: Dict[str, Dict[str, Any]] = {}
            for profile in profiles:
                name = str(profile.get("name") or profile.get("id") or "unknown")
                checked = validate_profile(profile)
                if checked["errors"]:
                    all_errors[name] = {
                        "errors": checked["errors"],
                        "field_errors": checked["field_errors"],
                    }

            if all_errors:
                _json(self, 400, {"ok": False, "error": "validation_failed", "details": all_errors})
                return

            started = runner.start_batch(profiles)
            _json(self, 200 if started.get("ok") else 400, started)
            return

        if path == "/api/batch/stop":
            _json(self, 200, runner.stop_batch())
            return

        if path == "/api/tensorboard/start":
            logdir = normalize_path(body.get("logdir", ""))
            try:
                port = int(body.get("port", 6006))
            except Exception:
                port = 6006
            if not logdir:
                _json(self, 400, {"ok": False, "error": "logdir_required"})
                return
            result = runner.start_tensorboard(logdir, port)
            _json(self, 200 if result.get("ok") else 400, result)
            return

        if path == "/api/tensorboard/stop":
            _json(self, 200, runner.stop_tensorboard())
            return

        _json(self, 404, {"ok": False, "error": "not_found"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # keep backend console concise
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), UIHandler)
    print(f"[INFO] ANIMA UI backend listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

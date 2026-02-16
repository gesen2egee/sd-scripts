from contextlib import nullcontext
import importlib.machinery
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch

try:
    import xformers.ops  # noqa: F401
except Exception:  # pragma: no cover - environment dependent optional deps
    xformers_stub = types.ModuleType("xformers")
    xformers_stub.__spec__ = importlib.machinery.ModuleSpec("xformers", loader=None)
    xformers_stub.__version__ = "0.0.0"
    xformers_ops_stub = types.ModuleType("xformers.ops")
    xformers_ops_stub.__spec__ = importlib.machinery.ModuleSpec("xformers.ops", loader=None)
    xformers_stub.ops = xformers_ops_stub
    sys.modules["xformers"] = xformers_stub
    sys.modules["xformers.ops"] = xformers_ops_stub

import anima_train_network
from library import train_util


class _DummyDatasetGroup:
    def is_text_encoder_output_cacheable(self, cache_supports_dropout=True):
        return True

    def verify_bucket_reso_steps(self, steps):
        return None


class _DummyAccelerator:
    def __init__(self, device: torch.device):
        self.device = device

    def autocast(self):
        return nullcontext()


class _DummyAnima(torch.nn.Module):
    def forward(
        self,
        x,
        timesteps,
        prompt_embeds,
        padding_mask=None,
        target_input_ids=None,
        target_attention_mask=None,
        source_attention_mask=None,
    ):
        return x


def _parse_default_args():
    with patch("sys.argv", [""]):
        parser = anima_train_network.setup_parser()
        args = parser.parse_args()
    return parser, args


def _run_and_capture_noise(monkeypatch, args, latents, trainer=None, is_train=True):
    if trainer is None:
        trainer = anima_train_network.AnimaNetworkTrainer()
    captured = {}

    def fake_get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype):
        captured["noise"] = noise.detach().clone()
        bsz = latents.shape[0]
        noisy_model_input = ((1.0 - 0.5) * latents + 0.5 * noise).to(dtype)
        timesteps = torch.full((bsz,), 500.0, device=device, dtype=dtype)
        sigmas = torch.full((bsz, 1, 1, 1), 0.5, device=device, dtype=dtype)
        return noisy_model_input, timesteps, sigmas

    monkeypatch.setattr(
        anima_train_network.flux_train_utils, "get_noisy_model_input_and_timesteps", fake_get_noisy_model_input_and_timesteps
    )

    batch_size = latents.shape[0]
    text_encoder_conds = (
        torch.zeros(batch_size, 8, 4, dtype=latents.dtype),
        torch.ones(batch_size, 8, dtype=torch.bool),
        torch.zeros(batch_size, 8, dtype=torch.long),
        torch.ones(batch_size, 8, dtype=torch.bool),
    )

    trainer.get_noise_pred_and_target(
        args=args,
        accelerator=_DummyAccelerator(torch.device("cpu")),
        noise_scheduler=object(),
        latents=latents,
        batch={},
        text_encoder_conds=text_encoder_conds,
        unet=_DummyAnima(),
        network=None,
        weight_dtype=latents.dtype,
        train_unet=True,
        is_train=is_train,
    )
    return captured["noise"]


def test_parser_default_values_for_random_noise_args():
    _, args = _parse_default_args()
    assert args.noise_selection == "gaussian"
    assert args.knn_noise_k == 8
    assert args.random_noise_shift == 0.0
    assert args.random_noise_multiplier == 0.0
    assert args.random_noise_shift_random_strength is False
    assert args.random_noise_multiplier_random_strength is False
    assert args.random_noise_shift_decay == 1.0
    assert args.random_noise_multiplier_decay == 1.0


def test_parser_can_parse_random_noise_args_from_cli():
    with patch("sys.argv", [""]):
        parser = anima_train_network.setup_parser()
        args = parser.parse_args(
            [
                "--random_noise_shift",
                "0.1",
                "--random_noise_multiplier",
                "0.2",
                "--noise_selection",
                "knn",
                "--knn_noise_k",
                "16",
                "--random_noise_shift_random_strength",
                "--random_noise_multiplier_random_strength",
                "--random_noise_shift_decay",
                "0.99",
                "--random_noise_multiplier_decay",
                "0.98",
            ]
        )
    assert args.noise_selection == "knn"
    assert args.knn_noise_k == 16
    assert args.random_noise_shift == 0.1
    assert args.random_noise_multiplier == 0.2
    assert args.random_noise_shift_random_strength is True
    assert args.random_noise_multiplier_random_strength is True
    assert args.random_noise_shift_decay == 0.99
    assert args.random_noise_multiplier_decay == 0.98


def test_config_file_can_set_random_noise_args(tmp_path: Path):
    config_path = tmp_path / "anima_noise.toml"
    config_path.write_text(
        "\n".join(
            [
                "[anima]",
                "noise_selection = 'knn'",
                "knn_noise_k = 12",
                "random_noise_shift = 0.1",
                "random_noise_multiplier = 0.2",
                "random_noise_shift_random_strength = true",
                "random_noise_multiplier_random_strength = true",
                "random_noise_shift_decay = 0.99",
                "random_noise_multiplier_decay = 0.98",
            ]
        ),
        encoding="utf-8",
    )

    with patch("sys.argv", [""]):
        parser = anima_train_network.setup_parser()
        args = parser.parse_args(["--config_file", str(config_path)])
    with patch("sys.argv", ["", "--config_file", str(config_path)]):
        args = train_util.read_config_from_file(args, parser)

    assert args.noise_selection == "knn"
    assert args.knn_noise_k == 12
    assert args.random_noise_shift == 0.1
    assert args.random_noise_multiplier == 0.2
    assert args.random_noise_shift_random_strength is True
    assert args.random_noise_multiplier_random_strength is True
    assert args.random_noise_shift_decay == 0.99
    assert args.random_noise_multiplier_decay == 0.98


def test_assert_extra_args_rejects_negative_values():
    _, args = _parse_default_args()
    trainer = anima_train_network.AnimaNetworkTrainer()
    dataset = _DummyDatasetGroup()

    args.random_noise_shift = -0.01
    with pytest.raises(ValueError, match="random_noise_shift"):
        trainer.assert_extra_args(args, dataset, None)

    args.random_noise_shift = 0.0
    args.random_noise_multiplier = -0.01
    with pytest.raises(ValueError, match="random_noise_multiplier"):
        trainer.assert_extra_args(args, dataset, None)

    args.random_noise_multiplier = 0.0
    args.random_noise_shift_decay = 1.1
    with pytest.raises(ValueError, match="random_noise_shift_decay"):
        trainer.assert_extra_args(args, dataset, None)

    args.random_noise_shift_decay = 1.0
    args.random_noise_multiplier_decay = -0.1
    with pytest.raises(ValueError, match="random_noise_multiplier_decay"):
        trainer.assert_extra_args(args, dataset, None)

    args.random_noise_multiplier_decay = 1.0
    args.knn_noise_k = 0
    with pytest.raises(ValueError, match="knn_noise_k"):
        trainer.assert_extra_args(args, dataset, None)


def test_noise_augmentation_changes_noise_and_preserves_shape_dtype_device(monkeypatch):
    _, args = _parse_default_args()
    args.gradient_checkpointing = False
    args.weighting_scheme = "none"
    args.noise_selection = "gaussian"

    latents = torch.randn(2, 4, 8, 8, dtype=torch.float32)

    # Regression baseline: with 0.0 options, noise should match plain torch.randn_like(latents).
    torch.manual_seed(1234)
    expected_plain_noise = torch.randn_like(latents)
    args.random_noise_shift = 0.0
    args.random_noise_multiplier = 0.0
    torch.manual_seed(1234)
    baseline_noise = _run_and_capture_noise(monkeypatch, args, latents)
    assert torch.equal(baseline_noise, expected_plain_noise)

    # Shift-only should alter noise values.
    args.random_noise_shift = 0.15
    args.random_noise_multiplier = 0.0
    torch.manual_seed(1234)
    shifted_noise = _run_and_capture_noise(monkeypatch, args, latents)
    assert not torch.equal(shifted_noise, baseline_noise)

    # Multiplier-only should alter noise values.
    args.random_noise_shift = 0.0
    args.random_noise_multiplier = 0.2
    torch.manual_seed(1234)
    multiplied_noise = _run_and_capture_noise(monkeypatch, args, latents)
    assert not torch.equal(multiplied_noise, baseline_noise)

    for noise in (baseline_noise, shifted_noise, multiplied_noise):
        assert noise.shape == latents.shape
        assert noise.dtype == latents.dtype
        assert noise.device == latents.device


def test_knn_noise_selection_preserves_shape_dtype_device(monkeypatch):
    _, args = _parse_default_args()
    args.gradient_checkpointing = False
    args.weighting_scheme = "none"
    args.noise_selection = "knn"
    args.knn_noise_k = 4
    args.random_noise_shift = 0.0
    args.random_noise_multiplier = 0.0

    latents = torch.randn(2, 4, 8, 8, dtype=torch.float32)
    noise = _run_and_capture_noise(monkeypatch, args, latents)

    assert noise.shape == latents.shape
    assert noise.dtype == latents.dtype
    assert noise.device == latents.device


def test_knn_noise_applies_before_random_noise_augmentation(monkeypatch):
    _, args = _parse_default_args()
    args.gradient_checkpointing = False
    args.weighting_scheme = "none"
    args.noise_selection = "knn"
    args.knn_noise_k = 8
    args.random_noise_shift = 0.5
    args.random_noise_multiplier = 0.0
    args.random_noise_shift_random_strength = False
    args.random_noise_multiplier_random_strength = False

    latents = torch.randn(2, 4, 8, 8, dtype=torch.float32)

    def fake_sample_training_noise(args, latents):
        return torch.zeros_like(latents)

    monkeypatch.setattr(anima_train_network.train_util, "sample_training_noise", fake_sample_training_noise)
    torch.manual_seed(7)
    noise = _run_and_capture_noise(monkeypatch, args, latents)

    # If augmentation were applied before KNN selection, this could be all zeros due to replacement.
    assert not torch.allclose(noise, torch.zeros_like(noise))


def test_random_strength_flags_change_noise(monkeypatch):
    _, args = _parse_default_args()
    args.gradient_checkpointing = False
    args.weighting_scheme = "none"
    latents = torch.randn(2, 4, 8, 8, dtype=torch.float32)

    args.random_noise_shift = 0.2
    args.random_noise_multiplier = 0.3
    args.random_noise_shift_random_strength = False
    args.random_noise_multiplier_random_strength = False
    torch.manual_seed(4321)
    fixed_strength_noise = _run_and_capture_noise(monkeypatch, args, latents)

    args.random_noise_shift_random_strength = True
    args.random_noise_multiplier_random_strength = True
    torch.manual_seed(4321)
    random_strength_noise = _run_and_capture_noise(monkeypatch, args, latents)

    assert not torch.equal(fixed_strength_noise, random_strength_noise)


def test_decay_updates_internal_strength_per_training_step(monkeypatch):
    _, args = _parse_default_args()
    args.gradient_checkpointing = False
    args.weighting_scheme = "none"
    args.random_noise_shift = 0.2
    args.random_noise_multiplier = 0.4
    args.random_noise_shift_decay = 0.5
    args.random_noise_multiplier_decay = 0.25
    args.random_noise_shift_random_strength = False
    args.random_noise_multiplier_random_strength = False
    latents = torch.randn(2, 4, 8, 8, dtype=torch.float32)
    trainer = anima_train_network.AnimaNetworkTrainer()

    _run_and_capture_noise(monkeypatch, args, latents, trainer=trainer, is_train=True)
    assert trainer._random_noise_shift_current == pytest.approx(0.1)
    assert trainer._random_noise_multiplier_current == pytest.approx(0.1)

    _run_and_capture_noise(monkeypatch, args, latents, trainer=trainer, is_train=True)
    assert trainer._random_noise_shift_current == pytest.approx(0.05)
    assert trainer._random_noise_multiplier_current == pytest.approx(0.025)

    shift_before = trainer._random_noise_shift_current
    multiplier_before = trainer._random_noise_multiplier_current
    _run_and_capture_noise(monkeypatch, args, latents, trainer=trainer, is_train=False)
    assert trainer._random_noise_shift_current == pytest.approx(shift_before)
    assert trainer._random_noise_multiplier_current == pytest.approx(multiplier_before)

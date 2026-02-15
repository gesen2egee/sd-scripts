import tempfile
import unittest
from pathlib import Path

from ui.backend.config_builder import build_job


def make_profile(dataset_dir: str):
    return {
        "id": "p1",
        "name": "job1",
        "run_enabled": True,
        "model": {
            "pretrained_model_name_or_path": "m.safetensors",
            "qwen3": "q.safetensors",
            "vae": "v.safetensors",
        },
        "precision_opt": {"mixed_precision": "bf16", "save_precision": "bf16"},
        "dataset": {
            "dataset_path": dataset_dir,
            "reg_dataset_path": "",
            "repeats": 1,
            "caption_extension": ".txt",
        },
        "trainer": {
            "learning_rate": 1,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 300,
            "optimizer_type": "AdamW",
            "optimizer_args": [],
            "train_batch_size": 1,
            "max_train_steps": 100,
            "max_grad_norm": 1,
        },
        "advanced": {"timestep_sampling": "sigmoid", "discrete_flow_shift": 1, "sigmoid_scale": 1},
        "format": {
            "model_type": "LyCORIS",
            "network_dim": 32,
            "network_alpha": 1,
            "network_train_unet_only": True,
            "network_args": [],
            "batch_size": 1,
        },
        "saving": {
            "output_dir": "out",
            "logging_dir": "logs",
            "output_name": "test",
            "save_model_as": "safetensors",
            "save_every_n_steps": 50,
        },
        "resume": {"enabled": False, "path": ""},
        "sample": {"enabled": False},
        "raw_args": [],
    }


class ConfigBuilderTests(unittest.TestCase):
    def test_build_job_creates_runtime_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_dir = tmp_path / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "a.png").write_bytes(b"x")
            (dataset_dir / "a.txt").write_text("caption", encoding="utf-8")

            profile = make_profile(str(dataset_dir))
            runtime_dir = tmp_path / "runtime"
            built = build_job(runtime_dir, profile)

            train_path = Path(built["train_config_path"])
            dataset_path = Path(built["dataset_config_path"])
            self.assertTrue(train_path.exists())
            self.assertTrue(dataset_path.exists())
            self.assertEqual(built["command"][1], "launch")

            text = dataset_path.read_text(encoding="utf-8")
            escaped_path = str(dataset_dir).replace("\\", "\\\\")
            self.assertIn(escaped_path, text)
            self.assertIn("[[datasets.subsets]]", text)


if __name__ == "__main__":
    unittest.main()

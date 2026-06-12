import math
from typing import Tuple

import torch


class NoSparseGradientError(Exception):
    def __init__(self, optimizer_name: str):
        super().__init__(f"{optimizer_name} does not support sparse gradient.")


class NoComplexParameterError(Exception):
    def __init__(self, optimizer_name: str):
        super().__init__(f"{optimizer_name} does not support complex parameter.")


class CAME(torch.optim.Optimizer):
    """Confidence-guided Adaptive Memory Efficient Optimization.

    This standalone version does not import pytorch_optimizer. Optional 8-bit
    state storage is enabled with use_8bit=True and keeps large optimizer state
    tensors quantized between steps.
    """

    _8BIT_STATE_NAMES = (
        "exp_avg",
        "exp_avg_sq",
        "exp_avg_sq_row",
        "exp_avg_sq_col",
        "exp_avg_res_row",
        "exp_avg_res_col",
        "exp_avg_sq_hat",
        "kahan_comp",
    )

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.1,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        clip_threshold: float = 1.0,
        ams_bound: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-16,
        maximize: bool = False,
        allora: bool = False,
        eta: float = 2.0,
        automagic: bool = False,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-7,
        agreement_threshold: float = 0.6,
        cautious_wd: bool = True,
        use_kahan: bool = True,
        use_magma: bool = True,
        mask_p: float = 0.5,
        magma_tau: float = 2.0,
        magma_ema: float = 0.9,
        mask_1d_params: bool = False,
        use_8bit: bool = False,
        min_8bit_size: int = 4096,
        **kwargs,
    ):
        del kwargs

        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps1, "eps1")
        self.validate_non_negative(eps2, "eps2")

        if eta <= 0.0:
            raise ValueError(f"eta must be > 0. Got {eta}")

        if min_lr < 0.0:
            raise ValueError(f"min_lr must be >= 0. Got {min_lr}")

        if max_lr < 0.0:
            raise ValueError(f"max_lr must be >= 0. Got {max_lr}")

        if min_lr > max_lr:
            raise ValueError(f"min_lr must be <= max_lr. Got min_lr={min_lr}, max_lr={max_lr}")

        if lr_bump < 0.0:
            raise ValueError(f"lr_bump must be >= 0. Got {lr_bump}")

        if not (0.0 <= agreement_threshold <= 1.0):
            raise ValueError(f"agreement_threshold must be in [0, 1]. Got {agreement_threshold}")

        if not (0.0 < mask_p <= 1.0):
            raise ValueError(f"mask_p must be in (0, 1]. Got {mask_p}")

        if magma_tau <= 0.0:
            raise ValueError(f"magma_tau must be > 0. Got {magma_tau}")

        if not (0.0 <= magma_ema < 1.0):
            raise ValueError(f"magma_ema must be in [0, 1). Got {magma_ema}")

        if use_magma and betas[0] <= 0.0:
            raise ValueError("Magma requires beta1 > 0 because it uses first-moment momentum.")

        if min_8bit_size < 0:
            raise ValueError(f"min_8bit_size must be >= 0. Got {min_8bit_size}")

        self.clip_threshold = clip_threshold
        self.eps1 = eps1
        self.eps2 = eps2
        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "ams_bound": ams_bound,
            "eps1": eps1,
            "eps2": eps2,
            "allora": allora,
            "eta": eta,
            "automagic": automagic,
            "min_lr": min_lr,
            "max_lr": max_lr,
            "lr_bump": lr_bump,
            "agreement_threshold": agreement_threshold,
            "cautious_wd": cautious_wd,
            "use_kahan": use_kahan,
            "use_magma": use_magma,
            "mask_p": mask_p,
            "magma_tau": magma_tau,
            "magma_ema": magma_ema,
            "mask_1d_params": mask_1d_params,
            "use_8bit": use_8bit,
            "min_8bit_size": int(min_8bit_size),
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "CAME"

    @staticmethod
    def validate_non_negative(x, name: str) -> None:
        if x is not None and x < 0.0:
            raise ValueError(f"{name} must be non-negative")

    @staticmethod
    def validate_learning_rate(learning_rate) -> None:
        if learning_rate is not None and learning_rate < 0.0:
            raise ValueError(f"learning rate must be non-negative. Got {learning_rate}")

    @staticmethod
    def validate_range(x: float, name: str, low: float, high: float, range_type: str = "[)") -> None:
        if range_type == "[)" and not low <= x < high:
            raise ValueError(f"{name} must be in the range [{low}, {high})")
        if range_type == "[]" and not low <= x <= high:
            raise ValueError(f"{name} must be in the range [{low}, {high}]")

    def validate_betas(self, betas: Tuple[float, float, float]) -> None:
        if len(betas) != 3:
            raise ValueError(f"betas must contain 3 values. Got {betas}")

        self.validate_range(betas[0], "beta1", 0.0, 1.0, range_type="[)")
        self.validate_range(betas[1], "beta2", 0.0, 1.0, range_type="[)")
        self.validate_range(betas[2], "beta3", 0.0, 1.0, range_type="[]")

    @staticmethod
    def maximize_gradient(grad: torch.Tensor, maximize: bool = False) -> None:
        if maximize:
            grad.neg_()

    @staticmethod
    def apply_weight_decay(
        p: torch.Tensor,
        grad,
        lr: float,
        weight_decay: float,
        weight_decouple: bool,
        fixed_decay: bool,
    ) -> None:
        if weight_decouple:
            p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr))
        elif weight_decay > 0.0 and grad is not None:
            grad.add_(p, alpha=weight_decay)

    def init_group(self, group: dict, **kwargs) -> None:
        del kwargs

        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad

            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            grad_shape: Tuple[int, ...] = grad.shape
            factored: bool = self.get_options(grad_shape)

            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p)

                if factored:
                    state["exp_avg_sq_row"] = torch.zeros(
                        grad_shape[:-1],
                        dtype=grad.dtype,
                        device=grad.device,
                    )
                    state["exp_avg_sq_col"] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:],
                        dtype=grad.dtype,
                        device=grad.device,
                    )
                    state["exp_avg_res_row"] = torch.zeros(
                        grad_shape[:-1],
                        dtype=grad.dtype,
                        device=grad.device,
                    )
                    state["exp_avg_res_col"] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:],
                        dtype=grad.dtype,
                        device=grad.device,
                    )
                else:
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if group["ams_bound"]:
                    state["exp_avg_sq_hat"] = torch.zeros_like(grad)

                if group["allora"] and p.ndim == 2:
                    row_norm = p.detach().norm(dim=1, keepdim=True)
                    state["row_scaling"] = (
                        1.0 / torch.sqrt(row_norm + 1.0 / (group["eta"] ** 2))
                    ).mean().item()
                else:
                    state["row_scaling"] = 1.0

                if group["automagic"]:
                    initial_lr = max(
                        float(group["min_lr"]),
                        min(float(group["lr"]), float(group["max_lr"])),
                    )

                    state["layer_lr"] = torch.full(
                        (),
                        initial_lr,
                        dtype=torch.float32,
                        device=p.device,
                    )
                    state["last_polarity"] = None
                    state["lr_max_val"] = max(initial_lr, 1e-8)
                    state["avg_lr_no_allora"] = initial_lr
                else:
                    state["lr_max_val"] = max(float(group["lr"]), 1e-8)
                    state["avg_lr_no_allora"] = float(group["lr"])

                if group["use_kahan"]:
                    state["kahan_comp"] = torch.zeros_like(p)

                if group["use_magma"]:
                    state["magma_s"] = torch.tensor(
                        1.0,
                        device=p.device,
                        dtype=torch.float32,
                    )

                state["RMS"] = 0.0

    @staticmethod
    def get_options(shape: Tuple[int, ...]) -> bool:
        return len(shape) >= 2

    @staticmethod
    def get_rms(x: torch.Tensor) -> torch.Tensor:
        return x.norm(2) / math.sqrt(x.numel())

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        r_factor: torch.Tensor = (
            exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)
        ).rsqrt_().unsqueeze(-1)
        c_factor: torch.Tensor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    def _dequantize_state_tensors(self, state: dict) -> None:
        for name in self._8BIT_STATE_NAMES:
            tensor = state.get(name)
            scale = state.get(f"{name}_8bit_scale")

            if not torch.is_tensor(tensor) or tensor.dtype != torch.uint8 or scale is None:
                continue

            original_dtype = state.get(f"{name}_8bit_dtype", torch.float32)
            scale = scale.to(device=tensor.device, dtype=torch.float32)
            state[name] = (tensor.to(torch.float32) - 128.0).mul_(scale).to(original_dtype)

    def _quantize_state_tensors(self, state: dict, group: dict) -> None:
        if not group["use_8bit"]:
            return

        min_size = int(group["min_8bit_size"])

        for name in self._8BIT_STATE_NAMES:
            tensor = state.get(name)

            if (
                not torch.is_tensor(tensor)
                or tensor.dtype == torch.uint8
                or not tensor.is_floating_point()
                or tensor.numel() < min_size
            ):
                continue

            fp32_tensor = tensor.detach().to(torch.float32)
            max_abs = fp32_tensor.abs().max()

            if not bool(torch.isfinite(max_abs).item()) or float(max_abs.item()) == 0.0:
                scale = torch.ones((), device=tensor.device, dtype=torch.float32)
                quantized = torch.full(tensor.shape, 128, dtype=torch.uint8, device=tensor.device)
            else:
                scale = (max_abs / 127.0).to(torch.float32)
                quantized = torch.clamp(torch.round(fp32_tensor / scale) + 128.0, 0, 255).to(torch.uint8)

            state[name] = quantized
            state[f"{name}_8bit_scale"] = scale.detach()
            state[f"{name}_8bit_dtype"] = tensor.dtype

    @torch.no_grad()
    def get_learning_rates(self):
        learning_rates = []

        for group in self.param_groups:
            group_rates = []

            for p in group["params"]:
                state = self.state.get(p)

                if state and "avg_lr_no_allora" in state:
                    group_rates.append(float(state["avg_lr_no_allora"]))

            if group_rates:
                learning_rates.append(sum(group_rates) / len(group_rates))
            else:
                learning_rates.append(float(group["lr"]))

        return learning_rates

    @torch.no_grad()
    def get_avg_learning_rate(self) -> float:
        learning_rates = self.get_learning_rates()
        return sum(learning_rates) / len(learning_rates) if learning_rates else 0.0

    @torch.no_grad()
    def get_lr_to_use(
        self,
        update: torch.Tensor,
        p: torch.Tensor,
        state: dict,
        group: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        allora_scaling = float(state.get("row_scaling", 1.0))

        if group["automagic"]:
            layer_lr = state["layer_lr"]

            current_polarity = update > 0
            last_polarity = state.get("last_polarity", None)

            if last_polarity is None:
                state["last_polarity"] = current_polarity.detach()
            else:
                same_polarity = last_polarity == current_polarity
                state["last_polarity"] = current_polarity.detach()

                agreement = same_polarity.to(torch.float32).mean()

                direction = torch.where(
                    agreement >= float(group["agreement_threshold"]),
                    torch.tensor(1.0, device=p.device, dtype=torch.float32),
                    torch.tensor(-1.0, device=p.device, dtype=torch.float32),
                )

                layer_lr.add_(direction * float(group["lr_bump"])).clamp_(
                    min=float(group["min_lr"]),
                    max=float(group["max_lr"]),
                )

            lr_no_allora = layer_lr
            state["avg_lr_no_allora"] = float(lr_no_allora.detach().cpu().item())

        else:
            lr_no_allora = torch.tensor(
                float(group["lr"]),
                device=p.device,
                dtype=torch.float32,
            )
            state["avg_lr_no_allora"] = float(group["lr"])

        current_lr_no_allora = float(lr_no_allora.detach().cpu().item())

        state["lr_max_val"] = max(
            current_lr_no_allora,
            float(state.get("lr_max_val", float(group["lr"]))),
        )

        lr_to_use = lr_no_allora * allora_scaling

        return lr_to_use, lr_no_allora

    @torch.no_grad()
    def apply_came_weight_decay(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        update: torch.Tensor,
        group: dict,
        state: dict,
        lr_to_use: torch.Tensor,
    ) -> None:
        weight_decay = float(group["weight_decay"])

        if weight_decay == 0.0:
            return

        if not group["weight_decouple"]:
            self.apply_weight_decay(
                p=p,
                grad=grad,
                lr=float(lr_to_use.detach().cpu().item()),
                weight_decay=weight_decay,
                weight_decouple=group["weight_decouple"],
                fixed_decay=group["fixed_decay"],
            )
            return

        if group["fixed_decay"]:
            decay_scale = torch.tensor(weight_decay, device=p.device, dtype=torch.float32)
        else:
            base_lr = float(group["lr"])
            lr_max_val = max(float(state.get("lr_max_val", base_lr)), 1e-8)
            decay_scale = weight_decay * lr_to_use * (base_lr / lr_max_val)

        if group["cautious_wd"]:
            decay_mask = (update * p >= 0).to(p.dtype)
            p.add_(-(decay_scale.to(p.dtype) * p * decay_mask))
        else:
            p.add_(-(decay_scale.to(p.dtype) * p))

    @torch.no_grad()
    def apply_magma(
        self,
        update_tensor: torch.Tensor,
        alignment_tensor: torch.Tensor,
        first_moment: torch.Tensor,
        state: dict,
        group: dict,
        p: torch.Tensor,
    ) -> torch.Tensor:
        if not group["use_magma"]:
            return update_tensor

        if update_tensor.ndim < 2 and not group["mask_1d_params"]:
            return update_tensor

        mask_p = float(group["mask_p"])
        tau = float(group["magma_tau"])
        ema = float(group["magma_ema"])

        momentum_flat = first_moment.reshape(-1).float()
        align_flat = alignment_tensor.reshape(-1).float()

        denom = (momentum_flat.norm(p=2) * align_flat.norm(p=2)).clamp_min(1e-12)
        cosine_similarity = (momentum_flat.dot(align_flat) / denom).clamp(-1.0, 1.0)

        s_hat = torch.sigmoid(cosine_similarity / tau).to(torch.float32)

        prev_score = state.get("magma_s")
        if prev_score is None or not torch.is_tensor(prev_score) or prev_score.device != p.device:
            prev_score = torch.tensor(1.0, device=p.device, dtype=torch.float32)

        score = ema * prev_score + (1.0 - ema) * s_hat
        state["magma_s"] = score.detach()

        mask = (torch.rand((), device=p.device) < mask_p).to(update_tensor.dtype)

        return update_tensor * mask * score.to(update_tensor.dtype)

    @torch.no_grad()
    def apply_kahan_update(
        self,
        p: torch.Tensor,
        update_tensor: torch.Tensor,
        state: dict,
    ) -> None:
        kahan_comp = state["kahan_comp"]

        value_to_add = -update_tensor.to(kahan_comp.dtype)
        compensated_update = value_to_add - kahan_comp

        new_p = p + compensated_update.to(p.dtype)
        new_comp = (new_p - p) - compensated_update.to(p.dtype)

        kahan_comp.copy_(new_comp.to(kahan_comp.dtype))
        p.copy_(new_p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group["step"] += 1

            beta1, beta2, beta3 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]
                self._dequantize_state_tensors(state)

                grad_shape: Tuple[int, ...] = grad.shape
                factored: bool = self.get_options(grad_shape)

                state["RMS"] = self.get_rms(p)

                update = torch.mul(grad, grad).add_(self.eps1)

                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)

                    self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
                    torch.rsqrt(exp_avg_sq, out=update)

                if group["ams_bound"]:
                    exp_avg_sq_hat = state["exp_avg_sq_hat"]
                    torch.max(exp_avg_sq_hat, 1 / update, out=exp_avg_sq_hat)
                    torch.rsqrt(exp_avg_sq_hat / beta2, out=update)

                update.mul_(grad)
                update.div_((self.get_rms(update) / self.clip_threshold).clamp_(min=1.0))

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

                res = update - exp_avg
                res.pow_(2).add_(self.eps2)

                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]

                    exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
                    exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)

                    self.approximate_sq_grad(exp_avg_res_row, exp_avg_res_col, update)
                    update.mul_(exp_avg)
                else:
                    update = exp_avg

                lr_to_use, _ = self.get_lr_to_use(
                    update=update,
                    p=p,
                    state=state,
                    group=group,
                )

                self.apply_came_weight_decay(
                    p=p,
                    grad=grad,
                    update=update,
                    group=group,
                    state=state,
                    lr_to_use=lr_to_use,
                )

                update_tensor = update * lr_to_use.to(update.dtype)

                if group["use_magma"]:
                    update_tensor = self.apply_magma(
                        update_tensor=update_tensor,
                        alignment_tensor=update,
                        first_moment=exp_avg,
                        state=state,
                        group=group,
                        p=p,
                    )

                if group["use_kahan"]:
                    self.apply_kahan_update(
                        p=p,
                        update_tensor=update_tensor,
                        state=state,
                    )
                else:
                    p.add_(-update_tensor)

                self._quantize_state_tensors(state, group)

        return loss

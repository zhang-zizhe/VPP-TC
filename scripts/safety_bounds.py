#!/usr/bin/env python3
# safety_bounds.py
# -------------------------------------------------------------
#  γ 评估与在线搜索安全关节位置区间（含自碰撞和外部碰撞两套模型）
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import torch
from typing import Tuple

# ---------------------------------------------------------------------------
# 1.  γ(self‑collision) – 旧回归模型 (TransformerGamma)
# ---------------------------------------------------------------------------
from Transformer import TransformerGamma

device_sc   = torch.device("cpu")   # 自碰撞模型放 CPU 足够
_sc_model   = TransformerGamma().to(device_sc)
_sc_model.load_state_dict(
    torch.load("../models/network/transformer_gamma.pt", map_location=device_sc)
)
_sc_model.eval()

# @torch.no_grad()
def gamma_model(
    q_batch: torch.Tensor,
    dq_batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """自碰撞 γ：数值越大越安全"""
    x = torch.cat([q_batch, dq_batch], dim=1).to(device_sc)
    x.requires_grad_(True)
    _, gamma = _sc_model(x)
    return gamma.squeeze(), x            # (B,)

# ---------------------------------------------------------------------------
# 2.  γ(external‑collision) – 新二分类模型 (extransformer)
#      Γ = γ₁ − γ₂    (Γ≥0 → safe, Γ<0 → collision)
# ---------------------------------------------------------------------------
# from external_transformer_cls import extransformer_cls

# device_ext  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# _ext_model  = extransformer_cls().to(device_ext)
# _ext_model.load_state_dict(
#     torch.load("../models/network/extransformer_pybullet_cls.pt", map_location=device_ext)
# )
# _ext_model.eval()

# from external_transformer_reg import extransformer_reg

# device_ext  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ext_model_reg  = extransformer_reg().to(device_ext)
# ext_model_reg.load_state_dict(
#     torch.load("../models/network/extransformer_pybullet_9m.pt", map_location=device_ext)
# )
# ext_model_reg.eval()

# A_MAX = torch.tensor([15, 7.5, 10, 12.5, 15, 20, 20], dtype=torch.float32)
# def gamma_external_reg(q_batch: torch.Tensor,
#                        dq_batch: torch.Tensor) -> torch.Tensor:

#     # 1）确定目标 device：优先取外部模型 ext_model_reg 的 device
#     device = ext_model_reg.parameters().__next__().device

#     q_batch  = q_batch.to(device)
#     dq_batch = dq_batch.to(device)
#     A_max = A_MAX.to(device)

#     # 2）计算 q_final
#     q_final = q_batch + 0.5 * dq_batch * dq_batch.abs() / A_max  # (B, 7)

#     # 3）构造模型输入
#     x = torch.cat([q_batch, q_final], dim=1)  # (B, 14)

#     # 4）前向推理
#     d_pred = ext_model_reg(x).squeeze()      # (B,)
#     return d_pred
#     # γ (B,)
# @torch.no_grad()
# def gamma_external(q_batch: torch.Tensor, dq_batch: torch.Tensor) -> torch.Tensor:
#     device_in = q_batch.device                # 输入在哪（CPU/GPU）
#     device_ext = _ext_model.classifier[0].weight.device  # 模型所在设备

#     # 计算 q_final
#     A_max = A_MAX.to(q_batch.device)                          # 搬到输入设备
#     q_final = q_batch + 0.5 * dq_batch * dq_batch.abs() / A_max
#     # 拼接 q 和 q_final
#     x = torch.cat([q_batch, q_final], dim=1).to(device_ext)   # 搬到模型设备
#     _, gamma = _ext_model(x)
#     return gamma.squeeze().to(device_in)                      # 搬回输入设备

# def gamma_external_grad(q_batch: torch.Tensor, dq_batch: torch.Tensor) -> torch.Tensor:
#     device_in = q_batch.device
#     device_ext = _ext_model.classifier[0].weight.device

#     A_max = A_MAX.to(q_batch.device)
#     q_final = q_batch + 0.5 * dq_batch * dq_batch.abs() / A_max
#     x = torch.cat([q_batch, q_final], dim=1).to(device_ext)

#     _, gamma = _ext_model(x)
#     return gamma.squeeze().to(device_in)
# ---------------------------------------------------------------------------
# 3.  辅助函数
# ---------------------------------------------------------------------------

def _same_device(*tensors: torch.Tensor) -> torch.device:
    for t in tensors:
        if t.is_cuda:
            return t.device
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# 4.  自碰撞：并行二分 / 网格扫描
# ---------------------------------------------------------------------------

def online_search(
    q: torch.Tensor,
    qd: torch.Tensor,
    gamma_fn,
    a_max_np: np.ndarray,
    *,
    delta_t: float,
    tol: float,
    threshold: float,
):
    """并行二分搜索安全区间 (Γ≥threshold 认为安全)"""
    q, qd = q.clone().float(), qd.clone().float()
    a_max = torch.tensor(a_max_np, dtype=torch.float32)
    q_base = q + qd * delta_t

    q_min = torch.empty(7)
    q_max = torch.empty(7)

    for direction in ("pos", "neg"):
        deltaq = 0.5 * a_max * delta_t**2
        if direction == "pos":
            lo, hi = q_base.clone(), q_base + deltaq
        else:
            lo, hi = q_base - deltaq, q_base.clone()
        lo, hi = torch.min(lo, hi), torch.max(lo, hi)

        while torch.any((hi - lo) > tol):
            mid = (lo + hi) / 2.0
            q_batch = q_base.unsqueeze(0).repeat(7, 1)
            q_batch[range(7), range(7)] = mid
            qd_batch = qd.unsqueeze(0).repeat(7, 1)
            gamma_mid = gamma_fn(q_batch, qd_batch)

            if direction == "pos":
                lo = torch.where(gamma_mid >= threshold, mid, lo)
                hi = torch.where(gamma_mid <  threshold, mid, hi)
            else:
                hi = torch.where(gamma_mid >= threshold, mid, hi)
                lo = torch.where(gamma_mid <  threshold, mid, lo)

        if direction == "pos":
            q_max = lo.clone()
        else:
            q_min = hi.clone()

    return q_min.numpy(), q_max.numpy()


def online_gridsearch(
    q: torch.Tensor,
    qd: torch.Tensor,
    gamma_fn,
    a_max_np: np.ndarray,
    *,
    delta_t: float,
    step_size: float,
    threshold: float,
):
    """固定步长扫描安全区间 (Γ≥threshold 认为安全)"""
    q, qd = q.clone().float(), qd.clone().float()
    a_max  = torch.tensor(a_max_np, dtype=torch.float32)
    q_base = q + qd * delta_t

    q_batch = q_base.unsqueeze(0).repeat(7, 1)
    qd_batch = qd.unsqueeze(0).repeat(7, 1)
    gamma0 = gamma_fn(q_batch, qd_batch)

    q_min = torch.where(gamma0 < threshold, q, q_base).clone()
    q_max = torch.where(gamma0 < threshold, q, q_base).clone()
    done  = torch.zeros(7, dtype=torch.bool)

    for direction, sign in (("pos", +1), ("neg", -1)):
        done_dir = done.clone()
        current  = q_base.clone()
        alphas   = torch.arange(step_size, 1.0 + 1e-6, step_size)

        for a in alphas:
            q_step = q_base + 0.5 * sign * a * a_max * delta_t**2
            q_batch = q_base.unsqueeze(0).repeat(7, 1)
            q_batch[range(7), range(7)] = q_step
            gamma = gamma_fn(q_batch, qd_batch)

            collided   = (~done_dir) & (gamma < threshold)
            still_safe = (~done_dir) & (gamma >= threshold)
            if collided.any():
                if direction == "pos":
                    q_max[collided] = current[collided]
                else:
                    q_min[collided] = current[collided]
                done_dir |= collided
            current[still_safe] = q_step[still_safe]
            if done_dir.all():
                break

        unfinished = ~done_dir
        if unfinished.any():
            if direction == "pos":
                q_max[unfinished] = current[unfinished]
            else:
                q_min[unfinished] = current[unfinished]

    return q_min.numpy(), q_max.numpy()

# ---------------------------------------------------------------------------
# 5.  外部碰撞：并行二分 / 网格扫描
# ---------------------------------------------------------------------------

def online_search_external(
    q: torch.Tensor,
    qd: torch.Tensor,
    a_max_np: np.ndarray,
    *,
    delta_t: float,
    tol: float,
    threshold: float,
):
    gamma_fn = gamma_external
    q, qd = q.clone().float(), qd.clone().float()
    a_max = torch.tensor(a_max_np, dtype=torch.float32)
    q_base = q + qd * delta_t

    q_min = torch.empty(7)
    q_max = torch.empty(7)

    for direction in ("pos", "neg"):
        deltaq = 0.5 * a_max * delta_t**2
        if direction == "pos":
            lo, hi = q_base.clone(), q_base + deltaq
        else:
            lo, hi = q_base - deltaq, q_base.clone()
        lo, hi = torch.min(lo, hi), torch.max(lo, hi)

        while torch.any((hi - lo) > tol):
            mid = (lo + hi) / 2.0
            q_batch = q_base.unsqueeze(0).repeat(7, 1)
            q_batch[range(7), range(7)] = mid
            qd_batch = qd.unsqueeze(0).repeat(7, 1)
            gamma_mid = gamma_fn(q_batch, qd_batch)

            if direction == "pos":
                lo = torch.where(gamma_mid >= threshold, mid, lo)
                hi = torch.where(gamma_mid <  threshold, mid, hi)
            else:
                hi = torch.where(gamma_mid >= threshold, mid, hi)
                lo = torch.where(gamma_mid <  threshold, mid, lo)

        if direction == "pos":
            q_max = lo.clone()
        else:
            q_min = hi.clone()

    return q_min.numpy(), q_max.numpy()


def online_gridsearch_external(
    q: torch.Tensor,
    qd: torch.Tensor,
    a_max_np: np.ndarray,
    *,
    delta_t: float,
    step_size: float,
    threshold: float,
):
    """固定步长扫描安全区间 (Γ≥threshold 认为安全)"""
    gamma_fn = gamma_external
    q, qd = q.clone().float(), qd.clone().float()
    a_max  = torch.tensor(a_max_np, dtype=torch.float32)
    q_base = q + qd * delta_t

    q_batch = q_base.unsqueeze(0).repeat(7, 1)
    qd_batch = qd.unsqueeze(0).repeat(7, 1)
    gamma0 = gamma_fn(q_batch, qd_batch)

    q_min = torch.where(gamma0 < threshold, q, q_base).clone()
    q_max = torch.where(gamma0 < threshold, q, q_base).clone()
    done  = torch.zeros(7, dtype=torch.bool)

    for direction, sign in (("pos", +1), ("neg", -1)):
        done_dir = done.clone()
        current  = q_base.clone()
        alphas   = torch.arange(step_size, 1.0 + 1e-6, step_size)

        for a in alphas:
            q_step = q_base + 0.5 * sign * a * a_max * delta_t**2
            q_batch = q_base.unsqueeze(0).repeat(7, 1)
            q_batch[range(7), range(7)] = q_step
            gamma = gamma_fn(q_batch, qd_batch)

            collided   = (~done_dir) & (gamma < threshold)
            still_safe = (~done_dir) & (gamma >= threshold)
            if collided.any():
                if direction == "pos":
                    q_max[collided] = current[collided]
                else:
                    q_min[collided] = current[collided]
                done_dir |= collided
            current[still_safe] = q_step[still_safe]
            if done_dir.all():
                break

        unfinished = ~done_dir
        if unfinished.any():
            if direction == "pos":
                q_max[unfinished] = current[unfinished]
            else:
                q_min[unfinished] = current[unfinished]

    return q_min.numpy(), q_max.numpy()

@torch.no_grad()
def online_search_external_reg(
    q: torch.Tensor,
    qd: torch.Tensor,
    a_max_np: np.ndarray,
    *,
    delta_t: float = 0.02,
    tol: float = 1e-5,
    gamma_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """并行二分 (外部碰撞 Γ ≥ gamma_thresh ⇒ safe)"""
    gamma_fn = gamma_external_reg
    q, qd = q.clone().float(), qd.clone().float()
    device = _same_device(q)
    q, qd = q.to(device), qd.to(device)
    a_max = torch.tensor(a_max_np, dtype=torch.float32, device=device)

    q_base = q + qd * delta_t
    q_min = torch.empty(7, device=device)
    q_max = torch.empty(7, device=device)

    for direction in ("pos", "neg"):
        deltaq = 0.5 * a_max * delta_t**2
        lo, hi = (q_base.clone(), q_base + deltaq) if direction == "pos" else (q_base - deltaq, q_base.clone())
        lo, hi = torch.min(lo, hi), torch.max(lo, hi)

        while torch.any((hi - lo) > tol):
            mid = (lo + hi) / 2.0
            q_batch = q_base.unsqueeze(0).repeat(7, 1)
            q_batch[range(7), range(7)] = mid
            qd_batch = qd.unsqueeze(0).repeat(7, 1)
            gamma_mid = gamma_fn(q_batch, qd_batch)

            if direction == "pos":
                lo = torch.where(gamma_mid >= gamma_thresh, mid, lo)
                hi = torch.where(gamma_mid <  gamma_thresh, mid, hi)
            else:
                hi = torch.where(gamma_mid >= gamma_thresh, mid, hi)
                lo = torch.where(gamma_mid <  gamma_thresh, mid, lo)

        if direction == "pos":
            q_max = lo.clone()
        else:
            q_min = hi.clone()

    return q_min.cpu().numpy(), q_max.cpu().numpy()


@torch.no_grad()
def online_gridsearch_external_reg(
    q: torch.Tensor,
    qd: torch.Tensor,
    a_max_np: np.ndarray,
    *,
    delta_t: float = 0.02,
    step_size: float = 0.1,
    gamma_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """固定步长扫描 (外部碰撞 Γ ≥ gamma_thresh ⇒ safe)"""
    gamma_fn = gamma_external_reg
    q, qd = q.clone().float(), qd.clone().float()
    device = _same_device(q)
    q, qd = q.to(device), qd.to(device)
    a_max = torch.tensor(a_max_np, dtype=torch.float32, device=device)

    q_base = q + qd * delta_t
    q_batch = q_base.unsqueeze(0).repeat(7, 1)
    qd_batch = qd.unsqueeze(0).repeat(7, 1)
    gamma0 = gamma_fn(q_batch, qd_batch)

    q_min = torch.where(gamma0 < gamma_thresh, q, q_base).clone()
    q_max = torch.where(gamma0 < gamma_thresh, q, q_base).clone()
    done = torch.zeros(7, dtype=torch.bool, device=device)

    for direction, sign in (("pos", +1), ("neg", -1)):
        done_dir = done.clone()
        current = q_base.clone()
        alphas = torch.arange(step_size, 1.0 + 1e-6, step_size, device=device)
        for a in alphas:
            q_step = q_base + 0.5 * sign * a * a_max * delta_t**2
            q_batch = q_base.unsqueeze(0).repeat(7, 1)
            q_batch[range(7), range(7)] = q_step
            gamma = gamma_fn(q_batch, qd_batch)

            collided = (~done_dir) & (gamma < gamma_thresh)
            still_safe = (~done_dir) & (gamma >= gamma_thresh)

            if collided.any():
                if direction == "pos":
                    q_max[collided] = current[collided]
                else:
                    q_min[collided] = current[collided]
                done_dir |= collided

            current[still_safe] = q_step[still_safe]

            if done_dir.all():
                break

        unfinished = ~done_dir
        if unfinished.any():
            if direction == "pos":
                q_max[unfinished] = current[unfinished]
            else:
                q_min[unfinished] = current[unfinished]

    return q_min.cpu().numpy(), q_max.cpu().numpy()


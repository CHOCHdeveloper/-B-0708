#!/usr/bin/env python3
"""
Q3: Arbitrary Dual-Pattern Compatibility Solver
=================================================
Based on Q3建模.md — existence judgment of dual "meaningful" patterns
under cylindrical mirror anamorphosis via constrained optimization.

Mathematical Core (ref. Q3建模.md):
  Model 1: D₁/D₂ region decomposition — D₁ = Φ_p⁻¹(Θ_vis×[0,H])
  Model 2: Generalized compatibility — S(f|D₁, A*|D₁) ≥ τ_sense ∧ S(M[f], B*) ≥ τ_sense
  Model 3: Compatibility index κ = K(A*,B*) / (|D₁|·ΔI² + |Θ_vis×H|·ΔI²)

Physics engine adapted from q1_solver.py (Q1 verified).
Optimization framework adapted from problem2_dual_pattern_regularization.py (Q2 verified).

Author: MathModel Assistance Skill — 编程手
Date:   2026-07-08
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Fix Windows GBK encoding for Unicode math symbols
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.interpolate import RectBivariateSpline

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# SECTION 0: Configuration & Global Settings
# ═══════════════════════════════════════════════════════════════════

# --- Matplotlib setup (deferred until plotting) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

# Suppress CJK font glyph warnings (harmless substitute messages)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)

# Font configuration
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
# Reduce bold-related warnings: use normal weight as fallback
plt.rcParams["font.weight"] = "normal"

# --- Paths ---
_BASE_OUTPUT_DIR = Path("D:/vscode/output/q3_results")
OUTPUT_DIR = _BASE_OUTPUT_DIR  # overridden inside main() per run

A4_W, A4_L = 210.0, 297.0  # mm


@dataclass(frozen=True)
class Q3Config:
    """Central configuration for Q3 compatibility solver."""

    # --- Cylinder geometry (Q1 optimized) ---
    radius_mm: float = 18.0
    cylinder_height_mm: float = 50.0
    center_mm: Tuple[float, float] = (105.0, 120.0)

    # --- Observer ---
    observer_mm: Tuple[float, float, float] = (0.0, -300.0, 250.0)

    # --- Resolution ---
    paper_dpi: float = 3.0  # pixels per mm → ~630×891
    mirror_n_theta: int = 400
    mirror_n_z: int = 300

    # --- Optimization ---
    iterations: int = 200
    learning_rate: float = 0.15
    lambda_paper: float = 1.0  # λ₁ — paper fidelity weight (D₁ only)
    lambda_mirror: float = 1.0  # λ₂ — mirror fidelity weight
    lambda_reg: float = 0.05  # λ₃ — TV regularization weight

    # --- Semantic scoring weights (Q3 model Sec.2, Eq. for S) ---
    w_ssim: float = 0.55
    w_edge: float = 0.30
    w_entropy: float = 0.15

    # --- Judgment thresholds ---
    tau_sense: float = 0.65  # semantic recognizability threshold
    alpha_tau: float = 0.15  # τ = α·(1 - τ_sense)

    @property
    def paper_nx(self) -> int:
        return int(A4_W * self.paper_dpi)

    @property
    def paper_ny(self) -> int:
        return int(A4_L * self.paper_dpi)

    @property
    def tau(self) -> float:
        """Compatibility threshold τ = α·(1 - τ_sense)."""
        return self.alpha_tau * (1.0 - self.tau_sense)


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: Physics Engine (from Q1 — q1_solver.py)
# ═══════════════════════════════════════════════════════════════════

def ray_to_paper(
    theta: np.ndarray, z: np.ndarray,
    R: float, E: np.ndarray, C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward ray-trace: eye → mirror point (θ,z) → reflect → paper (z=0 plane).

    Args:
        theta: angular coordinate on cylinder [rad], shape (N,)
        z: vertical coordinate on cylinder [mm], shape (N,)
        R: cylinder radius [mm]
        E: eye position (Ex, Ey, Ez) [mm]
        C: cylinder center on paper (Cx, Cy) [mm]

    Returns:
        (px, py): paper intersection coordinates [mm], nan where invalid.
    """
    theta = np.float64(theta)
    z = np.float64(z)

    Mx = C[0] + R * np.cos(theta)
    My = C[1] + R * np.sin(theta)
    Mz = z

    Nx, Ny = np.cos(theta), np.sin(theta)

    # Incident direction (eye → mirror)
    dx, dy, dz = Mx - E[0], My - E[1], Mz - E[2]
    nrm = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-15
    ix, iy, iz = dx / nrm, dy / nrm, dz / nrm

    # Reflection: r = i - 2(i·n)n
    dot = ix * Nx + iy * Ny  # < 0 for front-face visibility
    rx = ix - 2.0 * dot * Nx
    ry = iy - 2.0 * dot * Ny
    rz = iz  # unchanged (cylinder normal has no z-component)

    # Intersection with z=0 plane: Mz + t*rz = 0 → t = -Mz/rz
    t = np.where(np.abs(rz) > 1e-12, -Mz / rz, np.nan)
    px = Mx + t * rx
    py = My + t * ry

    bad = (dot >= 0) | (rz >= -1e-12) | (t <= 0) | ~np.isfinite(t)
    return np.where(bad, np.nan, px), np.where(bad, np.nan, py)


def visible_range(R: float, E: np.ndarray, C: np.ndarray) -> Tuple[float, float]:
    """
    Find the contiguous visible theta range of the cylinder from eye position.

    Returns:
        (theta_start, theta_span) in radians.
    """
    th = np.linspace(0, 2 * np.pi, 2000)
    dot = (C[0] - E[0]) * np.cos(th) + (C[1] - E[1]) * np.sin(th) + R
    front = dot < 0

    diff = np.diff(front.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if len(starts) == 0 or len(ends) == 0:
        return 0.0, 2.0 * np.pi

    lens = [(e - s if e > s else 2000 - s + e) for s, e in zip(starts, ends)]
    bi = np.argmax(lens)
    s, e = starts[bi], ends[bi]
    t0 = float(th[s])
    t1 = float(th[e])
    if t1 <= t0:
        t1 += 2.0 * np.pi

    print(f"  Visible θ: {np.degrees(t0):.0f}° → {np.degrees(t1):.0f}° "
          f"(span {np.degrees(t1 - t0):.0f}°)")
    return t0, t1 - t0


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Inverse Mapping (adapted from problem2)
# ═══════════════════════════════════════════════════════════════════

def compute_inverse_map(
    cfg: Q3Config, th0: float, dth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each mirror pixel (i_z, i_θ), compute which paper pixel it looks at.

    Paper coordinate convention: row 0 = top of paper (y = A4_L),
    consistent with standard image display.

    Returns:
        row: paper row indices [n_z, n_θ], integer
        col: paper column indices [n_z, n_θ], integer
        valid: boolean mask [n_z, n_θ]
    """
    n_th = cfg.mirror_n_theta
    n_z = cfg.mirror_n_z
    R = cfg.radius_mm
    H = cfg.cylinder_height_mm
    C = np.array(cfg.center_mm)
    E = np.array(cfg.observer_mm)

    theta = np.linspace(th0, th0 + dth, n_th, dtype=np.float64)
    zv = np.linspace(0, H, n_z, dtype=np.float64)
    TV, ZV = np.meshgrid(theta, zv)

    px, py = ray_to_paper(TV.ravel(), ZV.ravel(), R, E, C)
    px = px.reshape(n_z, n_th)
    py = py.reshape(n_z, n_th)

    # Convert mm → paper pixel indices (image convention: row 0 = top)
    col = np.rint(px * cfg.paper_dpi).astype(np.int64)
    row = np.rint((A4_L - py) * cfg.paper_dpi).astype(np.int64)

    valid = (
        np.isfinite(px) & np.isfinite(py)
        & (col >= 0) & (col < cfg.paper_nx)
        & (row >= 0) & (row < cfg.paper_ny)
    )

    return row, col, valid


def sample_mirror(
    paper: np.ndarray, row: np.ndarray, col: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    """
    Render mirror image from paper: g = M_p[f].

    Args:
        paper: paper pattern [ny, nx] or [ny, nx, 3]
        row, col: inverse map indices [n_z, n_θ]
        valid: boolean mask [n_z, n_θ]

    Returns:
        mirror: rendered mirror image [n_z, n_θ] or [n_z, n_θ, 3]
    """
    if paper.ndim == 2:
        out = np.ones((row.shape[0], row.shape[1]), dtype=np.float64)
        out[valid] = paper[row[valid], col[valid]]
    else:
        out = np.ones((row.shape[0], row.shape[1], paper.shape[2]), dtype=np.float64)
        out[valid] = paper[row[valid], col[valid]]
    return out


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: D₁ / D₂ Region Decomposition (Q3 Model 1)
# ═══════════════════════════════════════════════════════════════════

def compute_d1_mask(
    cfg: Q3Config, row: np.ndarray, col: np.ndarray, valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute D₁ (constrained) and D₂ (free) paper regions.

    D₁ = {paper pixels hit by ≥1 mirror ray}  — Q3 Eq. for D₁
    D₂ = Ω_paper \\ D₁

    Also computes the D₁/D₂ boundary for visualization.

    Returns:
        d1_mask: boolean [ny, nx], True where pixel is in D₁
        d2_mask: boolean [ny, nx], True where pixel is in D₂
    """
    d1_mask = np.zeros((cfg.paper_ny, cfg.paper_nx), dtype=bool)

    rr = row[valid]
    cc = col[valid]
    # Mark all paper pixels hit by mirror rays
    d1_mask[rr, cc] = True

    # Fill small holes in D₁ (disconnected interior pixels)
    from scipy.ndimage import binary_fill_holes, binary_dilation

    d1_mask = binary_fill_holes(d1_mask)

    d2_mask = ~d1_mask

    d1_area = d1_mask.sum()
    total_area = cfg.paper_ny * cfg.paper_nx
    coverage = d1_area / total_area * 100

    print(f"\n  D₁/D₂ Decomposition:")
    print(f"    D₁ (constrained): {d1_area:,} px ({coverage:.1f}%)")
    print(f"    D₂ (free):        {total_area - d1_area:,} px ({100 - coverage:.1f}%)")
    print(f"    Coverage η ≈ {coverage:.1f}%")

    return d1_mask, d2_mask


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: Semantic Scoring S(I, T) (Q3 Model 2, Eq. for S)
# ═══════════════════════════════════════════════════════════════════

def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Simplified SSIM for grayscale images."""
    if a.ndim == 3:
        ga = a.mean(axis=2)
        gb = b.mean(axis=2)
    else:
        ga, gb = a, b
    mu_a = ga.mean()
    mu_b = gb.mean()
    var_a = ga.var()
    var_b = gb.var()
    cov = ((ga - mu_a) * (gb - mu_b)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * cov + c2))
        / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    )


def edge_map(img: np.ndarray) -> np.ndarray:
    """Sobel-like edge detection via simple gradient magnitude."""
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    mag = np.sqrt(gx * gx + gy * gy)
    threshold = np.percentile(mag, 78)
    return mag > threshold


def edge_f1(a: np.ndarray, b: np.ndarray) -> float:
    """Edge F1 score between two images."""
    ea = edge_map(a)
    eb = edge_map(b)
    tp = np.logical_and(ea, eb).sum()
    fp = np.logical_and(ea, ~eb).sum()
    fn = np.logical_and(~ea, eb).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def image_entropy(img: np.ndarray) -> float:
    """Shannon entropy of grayscale image histogram."""
    if img.ndim == 3:
        gray = np.clip((img.mean(axis=2) * 255).astype(np.uint8), 0, 255)
    else:
        gray = np.clip((img * 255).astype(np.uint8), 0, 255)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def entropy_fit_score(candidate: np.ndarray, target: np.ndarray) -> float:
    """
    Inverted-U entropy fit: penalizes over-smooth AND over-noisy results.
    Q3 Model 2: Entropy_Fit(I) component.
    """
    h0 = image_entropy(target)
    hc = image_entropy(candidate)
    sigma_h = max(0.20 * h0, 0.35)
    return float(math.exp(-((hc - h0) ** 2) / (2.0 * sigma_h * sigma_h)))


def semantic_score(
    candidate: np.ndarray, target: np.ndarray, cfg: Q3Config,
) -> float:
    """
    Semantic recognizability score S(I, T).
    Q3 Model 2, Eq:
      S(I,T) = w₁·SSIM(I,T) + w₂·Edge_Similarity(I,T) + w₃·Entropy_Fit(I)
    """
    ssim_val = max(0.0, min(1.0, ssim_gray(candidate, target)))
    edge_val = edge_f1(candidate, target)
    ent_val = entropy_fit_score(candidate, target)
    return float(cfg.w_ssim * ssim_val + cfg.w_edge * edge_val + cfg.w_entropy * ent_val)


def total_variation(img: np.ndarray) -> float:
    """Total Variation norm (L1 of gradients)."""
    if img.ndim == 3:
        dx = np.abs(np.diff(img, axis=1)).mean()
        dy = np.abs(np.diff(img, axis=0)).mean()
    else:
        dx = np.abs(np.diff(img, axis=1)).mean()
        dy = np.abs(np.diff(img, axis=0)).mean()
    return float(dx + dy)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: Target Pattern Generation (Synthetic Demo Targets)
# ═══════════════════════════════════════════════════════════════════

def generate_paper_target(cfg: Q3Config) -> np.ndarray:
    """
    Generate synthetic paper target A*.
    Creates a recognizable pattern on A4: central emblem + border decorations.
    """
    ny, nx = cfg.paper_ny, cfg.paper_nx
    # White background
    A_star = np.ones((ny, nx), dtype=np.float64)

    # --- Central circle (like an emblem) ---
    yy, xx = np.ogrid[:ny, :nx]
    # Center in paper coords (mm): (105, 148.5)
    cx_px = int(cfg.center_mm[0] * cfg.paper_dpi)
    cy_px = int((A4_L - cfg.center_mm[1]) * cfg.paper_dpi)  # image convention

    # Large central ring
    rr = np.sqrt((xx - cx_px) ** 2 + (yy - cy_px) ** 2)
    ring_r = 40 * cfg.paper_dpi  # 40mm radius
    ring_w = 5 * cfg.paper_dpi
    ring_mask = np.abs(rr - ring_r) < ring_w
    A_star[ring_mask] = 0.15

    # Inner filled circle
    inner_mask = rr < ring_r * 0.6
    A_star[inner_mask] = 0.25

    # Cross pattern inside
    cross_h = np.abs(xx - cx_px) < 3 * cfg.paper_dpi
    cross_v = np.abs(yy - cy_px) < 3 * cfg.paper_dpi
    cross_mask = (cross_h | cross_v) & (rr < ring_r * 1.2)
    A_star[cross_mask] = 0.05

    # --- Corner decorations (in D₂ to demonstrate free region) ---
    corner_size = 25 * cfg.paper_dpi
    # Top-left corner square
    A_star[:int(corner_size), :int(corner_size)] = 0.2
    A_star[int(corner_size * 0.15):int(corner_size * 0.85),
           int(corner_size * 0.15):int(corner_size * 0.85)] = 0.8
    # Bottom-right corner square
    A_star[-int(corner_size):, -int(corner_size):] = 0.2
    A_star[-int(corner_size * 0.85):-int(corner_size * 0.15),
           -int(corner_size * 0.85):-int(corner_size * 0.15)] = 0.8

    # --- Grid pattern (subtle) ---
    grid_spacing = 15 * cfg.paper_dpi
    for i in range(1, int(ny / grid_spacing)):
        y = int(i * grid_spacing)
        A_star[y:y + 2, :] = 0.85
    for j in range(1, int(nx / grid_spacing)):
        x = int(j * grid_spacing)
        A_star[:, x:x + 2] = 0.85

    return np.clip(A_star, 0.0, 1.0)


def generate_mirror_target(cfg: Q3Config) -> np.ndarray:
    """
    Generate synthetic mirror target B*.
    Creates a bullseye (concentric rings) pattern — clearly "meaningful"
    but geometrically different from A*.
    """
    nz, nth = cfg.mirror_n_z, cfg.mirror_n_theta
    zz, tt = np.ogrid[:nz, :nth]

    # Normalized coordinates
    z_norm = zz / nz  # [0, 1)
    t_norm = tt / nth  # [0, 1)

    # Bullseye center
    cz, ct = 0.5, 0.5
    rr = np.sqrt((z_norm - cz) ** 2 + (t_norm - ct) ** 2)

    # Alternating rings
    n_rings = 8
    B_star = np.ones((nz, nth), dtype=np.float64)
    for k in range(n_rings):
        r_inner = k / n_rings * 0.7
        r_outer = (k + 0.6) / n_rings * 0.7
        ring = (rr >= r_inner) & (rr < r_outer)
        if k % 2 == 0:
            B_star[ring] = 0.15  # dark ring
        else:
            B_star[ring] = 0.7  # light ring

    # Center dot
    center_dot = rr < 0.03
    B_star[center_dot] = 0.05

    # Cross-hair through center
    ch_v = np.abs(z_norm - cz) < 0.008
    ch_h = np.abs(t_norm - ct) < 0.008
    B_star[ch_v | ch_h] = 0.1

    return np.clip(B_star, 0.0, 1.0)


def load_or_generate_targets(
    cfg: Q3Config,
    paper_path: Optional[str] = None,
    mirror_path: Optional[str] = None,
    compatible_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load target images or generate synthetic ones.
    Returns (A_star_paper, B_star_mirror).

    If compatible_mode=True, generates a pair where B* is derived from
    A* via the cylindrical mapping, ensuring inherent compatibility.
    """
    if paper_path and os.path.exists(paper_path):
        print(f"  Loading paper target A* from: {paper_path}")
        im = Image.open(paper_path).convert("L")
        im = im.resize((cfg.paper_nx, cfg.paper_ny), Image.LANCZOS)
        A_star = np.array(im, dtype=np.float64) / 255.0
    else:
        print("  Generating synthetic paper target A*...")
        A_star = generate_paper_target(cfg)

    if mirror_path and os.path.exists(mirror_path):
        print(f"  Loading mirror target B* from: {mirror_path}")
        im = Image.open(mirror_path).convert("L")
        im = im.resize((cfg.mirror_n_theta, cfg.mirror_n_z), Image.LANCZOS)
        B_star = np.array(im, dtype=np.float64) / 255.0
    elif compatible_mode:
        print("  Generating COMPATIBLE mirror target B* (derived from A* via mapping)...")
        B_star = generate_compatible_mirror_target(cfg, A_star)
    else:
        print("  Generating synthetic mirror target B*...")
        B_star = generate_mirror_target(cfg)

    return A_star, B_star


def generate_compatible_mirror_target(
    cfg: Q3Config, A_star: np.ndarray,
) -> np.ndarray:
    """
    Generate a mirror target B* that is inherently compatible with A*.

    Strategy: Forward-render A* through the cylindrical mirror to get
    what the mirror WOULD see if A* were printed. Then perturb slightly
    to create a "design intent" that differs from exact physical rendering
    but shares structural features in D1.

    This ensures the compatibility index kappa is naturally low.
    """
    # Get physics parameters
    E = np.array(cfg.observer_mm)
    C = np.array(cfg.center_mm)

    th0, dth = visible_range(cfg.radius_mm, E, C)

    # Forward render A* to mirror
    nz, nth = cfg.mirror_n_z, cfg.mirror_n_theta
    theta = np.linspace(th0, th0 + dth, nth, dtype=np.float64)
    zv = np.linspace(0, cfg.cylinder_height_mm, nz, dtype=np.float64)
    TV, ZV = np.meshgrid(theta, zv)

    px, py = ray_to_paper(TV.ravel(), ZV.ravel(), cfg.radius_mm, E, C)
    px = px.reshape(nz, nth)
    py = py.reshape(nz, nth)

    # Sample A* at the ray hit points
    col = np.clip(np.rint(px * cfg.paper_dpi).astype(np.int64), 0, cfg.paper_nx - 1)
    row = np.clip(np.rint((A4_L - py) * cfg.paper_dpi).astype(np.int64), 0, cfg.paper_ny - 1)
    valid_forward = (
        np.isfinite(px) & np.isfinite(py)
        & (px >= 0) & (px <= A4_W) & (py >= 0) & (py <= A4_L)
    )

    B_star = np.ones((nz, nth), dtype=np.float64)
    B_star[valid_forward] = A_star[row[valid_forward], col[valid_forward]]

    # Add slight structural modification so B* is not identical to M[A*]
    # but retains the same semantic structure
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.02, (nz, nth)).astype(np.float64)
    # Apply noise only in valid regions and blend
    blend_mask = valid_forward & (rng.random((nz, nth)) < 0.15)
    B_star[blend_mask] = np.clip(B_star[blend_mask] + noise[blend_mask], 0.0, 1.0)

    print(f"    Compatible B* generated: {valid_forward.sum():,} / {valid_forward.size:,} "
          f"valid pixels ({valid_forward.sum()/valid_forward.size*100:.1f}%)")
    return np.clip(B_star, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: Optimization Engine (Q3 Model 3)
# ═══════════════════════════════════════════════════════════════════

def tv_gradient(img: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Gradient of Total Variation norm ∂TV(f)/∂f.
    TV(f) = ∫|∇f| dx → ∂TV/∂f = -div(∇f/|∇f|).
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    dx = np.diff(img, axis=1, append=img[:, -1:, :])
    dy = np.diff(img, axis=0, append=img[-1:, :, :])
    mag = np.sqrt(dx * dx + dy * dy + eps * eps)
    px = dx / mag
    py = dy / mag
    div_x = px - np.roll(px, 1, axis=1)
    div_y = py - np.roll(py, 1, axis=0)
    grad = -(div_x + div_y)

    if squeeze:
        grad = grad[:, :, 0]
    return grad


def scatter_gradient(
    residual: np.ndarray, row: np.ndarray, col: np.ndarray, valid: np.ndarray,
    shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Back-propagate mirror residual to paper via inverse mapping.
    Each mirror pixel's error is scattered to its corresponding paper pixel.
    Averaged by hit count.
    """
    grad = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape[:2], dtype=np.float64)

    rr = row[valid]
    cc = col[valid]

    if residual.ndim == 2:
        np.add.at(grad, (rr, cc), residual[valid])
    else:
        for ch in range(residual.shape[2]):
            np.add.at(grad, (rr, cc, ch), residual[valid, ch])
        grad = grad.reshape(shape)

    np.add.at(counts, (rr, cc), 1.0)
    used = counts > 0

    if residual.ndim == 2:
        grad[used] /= counts[used]
    else:
        grad[used] /= counts[used, np.newaxis]

    return grad


def optimize_dual_pattern_q3(
    A_star: np.ndarray,
    B_star: np.ndarray,
    d1_mask: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    valid: np.ndarray,
    cfg: Q3Config,
) -> dict:
    """
    Solve Q3 Model 3:
      min_f λ₁·||f|D₁ - A*|D₁||² + λ₂·||M_p[f] - B*||² + λ₃·TV(f)

    Uses gradient descent with momentum.

    Returns dict with: f_opt, g_opt, loss_history, metrics.
    """
    f = A_star.copy()  # Initialize from paper target
    ny, nx = f.shape
    nz, nth = B_star.shape

    # Momentum
    velocity = np.zeros_like(f)
    momentum = 0.9

    loss_history = {
        "total": [],
        "paper_d1": [],
        "mirror": [],
        "tv_reg": [],
        "iteration": [],
    }

    print(f"\n  Optimization: {cfg.iterations} iterations, lr={cfg.learning_rate}")
    print(f"    λ₁(paper D₁)={cfg.lambda_paper}, λ₂(mirror)={cfg.lambda_mirror}, "
          f"λ₃(TV)={cfg.lambda_reg}")

    d1_mask_f = d1_mask.astype(np.float64)

    for it in range(cfg.iterations):
        # --- Forward: render mirror ---
        g = sample_mirror(f, row, col, valid)

        # --- Compute losses ---
        # 1. Paper fidelity (D₁ only)
        paper_diff_d1 = (f - A_star) * d1_mask_f
        loss_paper = float(np.mean(paper_diff_d1 ** 2))

        # 2. Mirror fidelity
        if B_star.ndim == 2:
            mirror_diff = g - B_star
        else:
            mirror_diff = g - B_star
        loss_mirror = float(np.mean(mirror_diff[valid] ** 2)) if valid.any() else 0.0

        # 3. TV regularization
        tv_val = total_variation(f)
        loss_tv = tv_val

        loss_total = (
            cfg.lambda_paper * loss_paper
            + cfg.lambda_mirror * loss_mirror
            + cfg.lambda_reg * loss_tv
        )

        # --- Compute gradients ---
        # ∂/∂f of λ₁·||f|D₁ - A*|D₁||²
        grad_paper = 2.0 * cfg.lambda_paper * paper_diff_d1

        # ∂/∂f of λ₂·||M[f] - B*||² (backprop through sampling)
        grad_mirror = 2.0 * cfg.lambda_mirror * scatter_gradient(
            mirror_diff, row, col, valid, f.shape,
        )

        # ∂/∂f of λ₃·TV(f)
        grad_tv = cfg.lambda_reg * tv_gradient(f)

        total_grad = grad_paper + grad_mirror + grad_tv

        # --- Momentum update ---
        velocity = momentum * velocity + total_grad
        f = np.clip(f - cfg.learning_rate * velocity, 0.0, 1.0)

        # --- Record ---
        if it % 10 == 0 or it == cfg.iterations - 1:
            loss_history["total"].append(loss_total)
            loss_history["paper_d1"].append(cfg.lambda_paper * loss_paper)
            loss_history["mirror"].append(cfg.lambda_mirror * loss_mirror)
            loss_history["tv_reg"].append(cfg.lambda_reg * loss_tv)
            loss_history["iteration"].append(it)
            if it % 50 == 0 or it == cfg.iterations - 1:
                print(f"    iter {it:4d}: total={loss_total:.6f}, "
                      f"paper_D1={cfg.lambda_paper * loss_paper:.6f}, "
                      f"mirror={cfg.lambda_mirror * loss_mirror:.6f}, "
                      f"TV={cfg.lambda_reg * loss_tv:.6f}")

    # Final render
    g_final = sample_mirror(f, row, col, valid)

    return {
        "f_opt": f,
        "g_opt": g_final,
        "loss_history": loss_history,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: Compatibility Judgment (Q3 Model 3 — κ ≤ τ)
# ═══════════════════════════════════════════════════════════════════

def compute_compatibility(
    f_opt: np.ndarray,
    g_opt: np.ndarray,
    A_star: np.ndarray,
    B_star: np.ndarray,
    d1_mask: np.ndarray,
    valid: np.ndarray,
    cfg: Q3Config,
) -> dict:
    """
    Compute compatibility index κ and judgment.

    Q3 Model 3:
      κ = K(A*,B*) / (|D₁|·ΔI² + |Θ_vis×H|·ΔI²)

    where K is the minimized loss (without regularization), ΔI = 1.
    """
    # Raw mismatch (without regularization)
    paper_err_d1 = ((f_opt - A_star) * d1_mask.astype(np.float64)) ** 2
    raw_paper_loss = float(paper_err_d1.sum())

    mirror_err = (g_opt - B_star) ** 2
    raw_mirror_loss = float(mirror_err[valid].sum()) if valid.any() else 0.0

    K = raw_paper_loss + raw_mirror_loss

    # Normalization
    d1_size = float(d1_mask.sum())
    mirror_size = float(valid.sum())
    delta_I_sq = 1.0  # intensity range [0,1]
    norm_factor = d1_size * delta_I_sq + mirror_size * delta_I_sq

    kappa = K / norm_factor if norm_factor > 0 else float("inf")

    # Semantic scores
    S_paper = semantic_score(f_opt, A_star, cfg)
    S_mirror = semantic_score(g_opt, B_star, cfg)

    # Judgment
    tau = cfg.tau
    is_compatible = kappa <= tau

    print(f"\n  {'='*50}")
    print(f"  COMPATIBILITY JUDGMENT")
    print(f"  {'='*50}")
    print(f"    Raw mismatch K(A*,B*)      = {K:.6f}")
    print(f"    Normalization factor        = {norm_factor:.1f}")
    print(f"    Compatibility index κ       = {kappa:.6f}")
    print(f"    Threshold τ = α·(1-τ_sense) = {tau:.6f}")
    print(f"    κ ≤ τ ?  →  {kappa:.6f} ≤ {tau:.6f}  →  "
          f"{'COMPATIBLE ✓' if is_compatible else 'INCOMPATIBLE ✗'}")
    print(f"    Paper semantic S(f, A*)     = {S_paper:.4f}  "
          f"({'≥' if S_paper >= cfg.tau_sense else '<'} {cfg.tau_sense})")
    print(f"    Mirror semantic S(g, B*)    = {S_mirror:.4f}  "
          f"({'≥' if S_mirror >= cfg.tau_sense else '<'} {cfg.tau_sense})")

    return {
        "K_raw": K,
        "kappa": kappa,
        "tau": tau,
        "is_compatible": is_compatible,
        "S_paper": S_paper,
        "S_mirror": S_mirror,
        "d1_size": d1_size,
        "mirror_size": mirror_size,
        "raw_paper_loss": raw_paper_loss,
        "raw_mirror_loss": raw_mirror_loss,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: Demo Figures — 2×3 Composite (补充要求 6)
# ═══════════════════════════════════════════════════════════════════

def generate_demo_figures(
    A_star: np.ndarray,
    B_star: np.ndarray,
    f_opt: np.ndarray,
    g_opt: np.ndarray,
    d1_mask: np.ndarray,
    loss_history: dict,
    compat: dict,
    cfg: Q3Config,
    th0: float,
    dth: float,
) -> None:
    """
    Generate 2×3 composite demo figure for Q3 paper.

    Layout:
      Row 1: [Target Paper A*]  [Target Mirror B*]  [Optimized Paper f_opt]
      Row 2: [D₁ Red Mask+Green Boundary]  [Rendered Mirror g]  [Convergence]
    """
    print("\n  Generating demo figures...")

    # --- Prepare data ---
    # D1 boundary (for green outline)
    from scipy.ndimage import binary_dilation, binary_erosion

    d1_boundary = d1_mask & ~binary_erosion(d1_mask, iterations=2)

    # Mirror extent for plotting
    t0_deg = np.degrees(th0)
    t1_deg = np.degrees(th0 + dth)
    H_mm = cfg.cylinder_height_mm

    # Convergence data
    iters = np.array(loss_history["iteration"])
    total_loss = np.array(loss_history["total"])
    paper_loss = np.array(loss_history["paper_d1"])
    mirror_loss = np.array(loss_history["mirror"])
    tv_loss = np.array(loss_history["tv_reg"])

    # --- Create figure ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), dpi=300)

    # Color palette
    PAPER_CMAP = "gray"
    MIRROR_CMAP = "gray"
    HEAT_CMAP = "hot"

    # ═══ Subplot 1: Target Paper A* ═══
    ax = axes[0, 0]
    ax.imshow(A_star, cmap=PAPER_CMAP, origin="upper",
              extent=[0, A4_W, 0, A4_L])
    ax.set_title("Target Paper Pattern  A*", fontsize=12, fontweight="bold")
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # ═══ Subplot 2: Target Mirror B* ═══
    ax = axes[0, 1]
    ax.imshow(B_star, cmap=MIRROR_CMAP, aspect="auto",
              extent=[t0_deg, t1_deg, H_mm, 0])
    ax.set_title("Target Mirror Pattern  B*", fontsize=12, fontweight="bold")
    ax.set_xlabel("theta (deg)", fontsize=9)
    ax.set_ylabel("z (mm)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # ═══ Subplot 3: Optimized Paper f_opt ═══
    ax = axes[0, 2]
    ax.imshow(f_opt, cmap=PAPER_CMAP, origin="upper",
              extent=[0, A4_W, 0, A4_L])
    ax.set_title("Optimized Paper  f_opt", fontsize=12, fontweight="bold")
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # ═══ Subplot 4: D1 Red Mask + Green Boundary ═══
    ax = axes[1, 0]
    # Show f_opt as background
    ax.imshow(f_opt, cmap="gray", origin="upper",
              extent=[0, A4_W, 0, A4_L])

    # Semi-transparent red mask for D1
    d1_rgba = np.zeros((cfg.paper_ny, cfg.paper_nx, 4), dtype=np.float32)
    d1_rgba[d1_mask, 0] = 1.0  # Red channel
    d1_rgba[d1_mask, 3] = 0.35  # Alpha (semi-transparent)
    ax.imshow(d1_rgba, origin="upper", extent=[0, A4_W, 0, A4_L],
              interpolation="nearest")

    # Green boundary contour
    from scipy.ndimage import binary_dilation as bdil
    boundary_mask = d1_mask.astype(np.uint8) - binary_erosion(d1_mask, iterations=3).astype(np.uint8)
    boundary_mask = boundary_mask > 0

    # Sample boundary points for contour plot
    by, bx = np.where(boundary_mask)
    if len(by) > 0:
        # Convert to mm
        bx_mm = bx / cfg.paper_dpi
        by_mm = (cfg.paper_ny - 1 - by) / cfg.paper_dpi
        # Plot as scatter (efficient approximation of contour)
        stride = max(1, len(by) // 2000)
        ax.scatter(bx_mm[::stride], by_mm[::stride], c="lime", s=0.5,
                   alpha=0.8, marker=".", linewidths=0)

    ax.set_title("Constrained Region D1 (Red Mask)\n& Boundary (Green)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", alpha=0.35, label="D1 (mirror-visible)"),
        Patch(facecolor="lime", alpha=0.8, label="D1-D2 boundary"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
              framealpha=0.8)

    # ═══ Subplot 5: Rendered Mirror g_render ═══
    ax = axes[1, 1]
    ax.imshow(g_opt, cmap=MIRROR_CMAP, aspect="auto",
              extent=[t0_deg, t1_deg, H_mm, 0])
    ax.set_title("Rendered Mirror  g_render = M_p[f_opt]",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("theta (deg)", fontsize=9)
    ax.set_ylabel("z (mm)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # ═══ Subplot 6: Convergence History ═══
    ax = axes[1, 2]
    ax.plot(iters, total_loss, "k-", linewidth=1.8, label="J_total")
    ax.plot(iters, paper_loss, "#E53935", linewidth=1.2,
            label="lambda1 * ||f|D1 - A*|D1||^2")
    ax.plot(iters, mirror_loss, "#0F4D92", linewidth=1.2,
            label="lambda2 * ||M[f] - B*||^2")
    ax.plot(iters, tv_loss, "#767676", linewidth=1.0, linestyle="--",
            label="lambda3 * TV(f)")
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.set_title("Convergence History", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # --- Suptitle ---
    verdict = "COMPATIBLE [+]" if compat["is_compatible"] else "INCOMPATIBLE [-]"
    fig.suptitle(
        f"Q3: Dual-Pattern Compatibility Analysis\n"
        f"kappa = {compat['kappa']:.4f} <= tau = {compat['tau']:.4f} -> {verdict}  |  "
        f"S(f,A*) = {compat['S_paper']:.3f}  S(g,B*) = {compat['S_mirror']:.3f}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout(pad=3.0)

    # --- Save composite ---
    composite_path = OUTPUT_DIR / "Q3_Demo_Summary.png"
    fig.savefig(composite_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    Composite figure saved: {composite_path}")

    # --- Save individual components ---
    _save_individual_components(
        A_star, B_star, f_opt, g_opt, d1_mask, cfg,
    )

    # --- Save convergence data ---
    conv_data = {
        "iteration": loss_history["iteration"],
        "total_loss": loss_history["total"],
        "paper_d1_loss": loss_history["paper_d1"],
        "mirror_loss": loss_history["mirror"],
        "tv_loss": loss_history["tv_reg"],
    }
    conv_path = OUTPUT_DIR / "Q3_convergence_history.json"
    with open(conv_path, "w") as f:
        json.dump(conv_data, f, indent=2)
    print(f"    Convergence data saved: {conv_path}")


def _save_individual_components(
    A_star: np.ndarray,
    B_star: np.ndarray,
    f_opt: np.ndarray,
    g_opt: np.ndarray,
    d1_mask: np.ndarray,
    cfg: Q3Config,
) -> None:
    """Save individual subplot components as separate images for paper layout adjustments."""

    def _save(arr, name, cmap="gray"):
        p = OUTPUT_DIR / name
        if arr.dtype == bool:
            arr_img = (arr.astype(np.uint8) * 255)
        else:
            arr_img = np.clip(arr * 255, 0, 255).astype(np.uint8)
        Image.fromarray(arr_img).save(p)
        print(f"    Saved: {p}")

    _save(A_star, "Q3_sub1_target_paper.png")
    _save(B_star, "Q3_sub2_target_mirror.png")
    _save(f_opt, "Q3_sub3_optimized_paper.png")
    _save(d1_mask, "Q3_sub4_d1_mask.png")
    _save(g_opt, "Q3_sub5_rendered_mirror.png")

    # Also save D1 mask as RGBA overlay
    d1_rgba = np.zeros((cfg.paper_ny, cfg.paper_nx, 4), dtype=np.uint8)
    d1_rgba[d1_mask, 0] = 255
    d1_rgba[d1_mask, 3] = 89  # ~35% alpha
    Image.fromarray(d1_rgba).save(OUTPUT_DIR / "Q3_sub4_d1_overlay_rgba.png")
    print(f"    Saved: {OUTPUT_DIR / 'Q3_sub4_d1_overlay_rgba.png'}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: Metrics Table Output
# ═══════════════════════════════════════════════════════════════════

def print_metrics_table(compat: dict, cfg: Q3Config) -> str:
    """
    Print formatted metrics table to console.
    Returns the formatted string for saving.
    """
    verdict_str = "COMPATIBLE ✓" if compat["is_compatible"] else "INCOMPATIBLE ✗"

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════╗",
        "║           Q3 COMPATIBILITY ANALYSIS — KEY METRICS           ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  Paper Semantic Score    S(f, A*)    = {compat['S_paper']:>8.4f}                     ║",
        f"║  Mirror Semantic Score   S(g, B*)    = {compat['S_mirror']:>8.4f}                     ║",
        f"║  Compatibility Index     κ           = {compat['kappa']:>8.4f}                     ║",
        f"║  Judgment Threshold      τ           = {compat['tau']:>8.4f}                     ║",
        f"║  Final Verdict                       = {verdict_str:>20s}                    ║",
        f"║  ─────────────────────────────────────────────────────────  ║",
        f"║  Raw Mismatch            K(A*,B*)    = {compat['K_raw']:>8.4f}                     ║",
        f"║  D₁ Area (px)                        = {compat['d1_size']:>8.0f}                     ║",
        f"║  Mirror Valid Pixels                 = {compat['mirror_size']:>8.0f}                     ║",
        f"║  Raw Paper Loss (D₁)                 = {compat['raw_paper_loss']:>8.4f}                     ║",
        f"║  Raw Mirror Loss                     = {compat['raw_mirror_loss']:>8.4f}                     ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]

    table_str = "\n".join(lines)
    print(table_str)

    # Save to file
    metrics_path = OUTPUT_DIR / "Q3_metrics_table.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(table_str)
        f.write("\n")
        # Also write in JSON-serializable format
        json.dump({
            "S_paper": compat["S_paper"],
            "S_mirror": compat["S_mirror"],
            "kappa": compat["kappa"],
            "tau": compat["tau"],
            "is_compatible": compat["is_compatible"],
            "verdict": verdict_str,
            "K_raw": compat["K_raw"],
            "d1_size_px": compat["d1_size"],
            "mirror_valid_px": compat["mirror_size"],
            "raw_paper_loss_D1": compat["raw_paper_loss"],
            "raw_mirror_loss": compat["raw_mirror_loss"],
            "tau_sense": cfg.tau_sense,
            "w_ssim": cfg.w_ssim,
            "w_edge": cfg.w_edge,
            "w_entropy": cfg.w_entropy,
        }, f, indent=2)

    print(f"    Metrics saved: {metrics_path}")
    return table_str


# ═══════════════════════════════════════════════════════════════════
# SECTION 10: Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def main(
    paper_target_path: Optional[str] = None,
    mirror_target_path: Optional[str] = None,
    compatible_mode: bool = False,
    cfg: Optional[Q3Config] = None,
    tag: Optional[str] = None,
) -> None:
    """
    Q3: Dual-Pattern Compatibility Solver — main entry point.

    Args:
        paper_target_path: Optional path to paper target image A*.
        mirror_target_path: Optional path to mirror target image B*.
        compatible_mode: If True, generates compatible B* from A* via
            forward cylindrical mapping (demonstrates COMPATIBLE verdict).
        tag: Optional subdirectory tag to avoid overwriting previous results.
    """
    global OUTPUT_DIR

    t_start = time.time()

    if cfg is None:
        cfg = Q3Config()

    # Set output directory — use tag to avoid overwriting
    if tag:
        OUTPUT_DIR = _BASE_OUTPUT_DIR / tag
    else:
        # Auto-generate tag from filenames or mode
        if paper_target_path and mirror_target_path:
            p_stem = Path(paper_target_path).stem
            m_stem = Path(mirror_target_path).stem
            OUTPUT_DIR = _BASE_OUTPUT_DIR / f"paper_{p_stem}__mirror_{m_stem}"
        elif compatible_mode:
            OUTPUT_DIR = _BASE_OUTPUT_DIR / "compatible_demo"
        else:
            OUTPUT_DIR = _BASE_OUTPUT_DIR / "default_synthetic"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mode_label = "COMPATIBLE DEMO" if compatible_mode else "INDEPENDENT PATTERNS"
    print("=" * 65)
    print(f"  Q3: Arbitrary Dual-Pattern Compatibility Solver")
    print(f"  Mode: {mode_label}")
    print("  Based on Q3建模.md + q1_solver physics + problem2 optimization")
    print("=" * 65)
    print(f"\n  Configuration:")
    print(f"    Cylinder: R={cfg.radius_mm}mm, H={cfg.cylinder_height_mm}mm, "
          f"C=({cfg.center_mm[0]},{cfg.center_mm[1]})")
    print(f"    Observer: E=({cfg.observer_mm[0]},{cfg.observer_mm[1]},"
          f"{cfg.observer_mm[2]})")
    print(f"    Paper: {cfg.paper_nx}×{cfg.paper_ny} px @ {cfg.paper_dpi} px/mm")
    print(f"    Mirror: {cfg.mirror_n_theta}×{cfg.mirror_n_z} px")
    print(f"    λ₁={cfg.lambda_paper}, λ₂={cfg.lambda_mirror}, λ₃={cfg.lambda_reg}")
    print(f"    τ_sense={cfg.tau_sense}, τ={cfg.tau:.4f}")
    print(f"    Output: {OUTPUT_DIR}")

    # ═══════════════════════════════════════════════════
    # Step 1: Physics setup
    # ═══════════════════════════════════════════════════
    print(f"\n{'—'*50}")
    print("  Step 1: Physics Setup")
    print(f"{'—'*50}")

    E = np.array(cfg.observer_mm)
    C = np.array(cfg.center_mm)

    th0, dth = visible_range(cfg.radius_mm, E, C)

    # ═══════════════════════════════════════════════════
    # Step 2: Load / generate targets
    # ═══════════════════════════════════════════════════
    print(f"\n{'—'*50}")
    print("  Step 2: Target Patterns")
    print(f"{'—'*50}")

    A_star, B_star = load_or_generate_targets(
        cfg, paper_target_path, mirror_target_path, compatible_mode,
    )
    print(f"    A* shape: {A_star.shape}")
    print(f"    B* shape: {B_star.shape}")

    # ═══════════════════════════════════════════════════
    # Step 3: Inverse mapping & D₁/D₂ decomposition
    # ═══════════════════════════════════════════════════
    print(f"\n{'—'*50}")
    print("  Step 3: Inverse Mapping & D₁/D₂ Decomposition")
    print(f"{'—'*50}")

    row, col, valid = compute_inverse_map(cfg, th0, dth)
    d1_mask, d2_mask = compute_d1_mask(cfg, row, col, valid)

    # ═══════════════════════════════════════════════════
    # Step 4: Optimization
    # ═══════════════════════════════════════════════════
    print(f"\n{'—'*50}")
    print("  Step 4: Dual-Pattern Optimization (Q3 Model 3)")
    print(f"{'—'*50}")

    opt_result = optimize_dual_pattern_q3(
        A_star, B_star, d1_mask, row, col, valid, cfg,
    )

    f_opt = opt_result["f_opt"]
    g_opt = opt_result["g_opt"]
    loss_history = opt_result["loss_history"]

    # ═══════════════════════════════════════════════════
    # Step 5: Compatibility judgment
    # ═══════════════════════════════════════════════════
    print(f"\n{'—'*50}")
    print("  Step 5: Compatibility Judgment")
    print(f"{'—'*50}")

    compat = compute_compatibility(
        f_opt, g_opt, A_star, B_star, d1_mask, valid, cfg,
    )

    # ═══════════════════════════════════════════════════
    # Step 6: Demo figures & metrics
    # ═══════════════════════════════════════════════════
    print(f"\n{'—'*50}")
    print("  Step 6: Demo Figures & Metrics")
    print(f"{'—'*50}")

    generate_demo_figures(
        A_star, B_star, f_opt, g_opt, d1_mask,
        loss_history, compat, cfg, th0, dth,
    )

    print_metrics_table(compat, cfg)

    # ═══════════════════════════════════════════════════
    # Final summary
    # ═══════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  Q3 SOLVER COMPLETE")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"  Verdict: {'COMPATIBLE ✓' if compat['is_compatible'] else 'INCOMPATIBLE ✗'}")
    print(f"  κ = {compat['kappa']:.4f}  |  τ = {compat['tau']:.4f}")
    print(f"  S(f,A*) = {compat['S_paper']:.4f}  |  S(g,B*) = {compat['S_mirror']:.4f}")
    print(f"  All outputs → {OUTPUT_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Q3: Dual-Pattern Compatibility Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python problem3_compatibility_solver.py
  python problem3_compatibility_solver.py --compatible
  python problem3_compatibility_solver.py --paper path/to/A_star.png --mirror path/to/B_star.png
  python problem3_compatibility_solver.py --paper path/to/A_star.png --compatible
        """,
    )
    parser.add_argument(
        "--paper", type=str, default=None,
        help="Path to paper target image A* (grayscale, any size).",
    )
    parser.add_argument(
        "--mirror", type=str, default=None,
        help="Path to mirror target image B* (grayscale, any size).",
    )
    parser.add_argument(
        "--compatible", action="store_true", default=False,
        help="Generate inherently compatible B* from A* via forward mapping.",
    )
    parser.add_argument(
        "--iterations", type=int, default=200,
        help="Number of optimization iterations (default: 200).",
    )
    parser.add_argument(
        "--lambda-paper", type=float, default=1.0, dest="lambda_paper",
        help="Weight for paper fidelity loss in D1 (default: 1.0).",
    )
    parser.add_argument(
        "--lambda-mirror", type=float, default=1.0, dest="lambda_mirror",
        help="Weight for mirror fidelity loss (default: 1.0).",
    )
    parser.add_argument(
        "--lambda-reg", type=float, default=0.05, dest="lambda_reg",
        help="Weight for TV regularization (default: 0.05).",
    )
    parser.add_argument(
        "--tau-sense", type=float, default=0.65, dest="tau_sense",
        help="Semantic recognizability threshold (default: 0.65).",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Output subdirectory tag (auto-generated if omitted). "
             "Use to prevent overwriting previous runs.",
    )

    args = parser.parse_args()

    # Override config defaults from CLI
    # (We create a custom config by modifying the dataclass post-init)
    cfg_override = Q3Config(
        iterations=args.iterations,
        lambda_paper=args.lambda_paper,
        lambda_mirror=args.lambda_mirror,
        lambda_reg=args.lambda_reg,
        tau_sense=args.tau_sense,
    )

    # Run with overridden config
    main(
        paper_target_path=args.paper,
        mirror_target_path=args.mirror,
        compatible_mode=args.compatible,
        cfg=cfg_override,
        tag=args.tag,
    )

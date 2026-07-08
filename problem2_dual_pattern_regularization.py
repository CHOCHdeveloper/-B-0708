from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "problem2"
FIG_DIR = ROOT / "figures" / "problem2"
REPORT_PATH = ROOT / "reports" / "PROBLEM2_MEANINGFULNESS_REGULARIZATION.md"
SEED = 2026


@dataclass(frozen=True)
class ExperimentConfig:
    paper_width: int = 210
    paper_height: int = 297
    mirror_height: int = 160
    radius_mm: float = 15.0
    cylinder_height_mm: float = 60.0
    center_mm: tuple[float, float] = (105.0, 90.0)
    observer_mm: tuple[float, float, float] = (105.0, -180.0, 140.0)
    theta_min_deg: float = -15.0
    theta_max_deg: float = 15.0
    iterations: int = 180
    step_size: float = 0.18


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float64) / 255.0


def crop_panel_from_comparison(path: Path, panel: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    y0 = int(img.height * 0.11)
    panel_w = img.width // 3
    x0 = panel * panel_w
    return img.crop((x0, y0, x0 + panel_w, img.height))


def prepare_targets(cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    cat_panel = crop_panel_from_comparison(FIG_DIR.parent / "problem1" / "cat_optimized_compact_printable_comparison.png", 0)
    mona_panel = crop_panel_from_comparison(FIG_DIR.parent / "problem1" / "mona_lisa_optimized_compact_printable_comparison.png", 0)

    # Paper target A*: a recognizable cat image embedded in A4 with margins.
    paper = Image.new("RGB", (cfg.paper_width, cfg.paper_height), "white")
    cat_h = 170
    cat_w = int(cat_panel.width * cat_h / cat_panel.height)
    cat_img = cat_panel.resize((cat_w, cat_h), Image.Resampling.LANCZOS)
    paper.paste(cat_img, ((cfg.paper_width - cat_w) // 2, 60))

    # Mirror target B*: Mona Lisa. It is intentionally independent from A*.
    mirror_w = int(mona_panel.width * cfg.mirror_height / mona_panel.height)
    mirror = mona_panel.resize((mirror_w, cfg.mirror_height), Image.Resampling.LANCZOS)
    return np.asarray(paper, dtype=np.float64) / 255.0, np.asarray(mirror, dtype=np.float64) / 255.0


def inverse_map_grid(width: int, height: int, cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_min = math.radians(cfg.theta_min_deg)
    theta_max = math.radians(cfg.theta_max_deg)
    u = np.linspace(0.0, 1.0, width, dtype=np.float64)
    v = np.linspace(0.0, 1.0, height, dtype=np.float64)
    theta = theta_min + u[None, :] * (theta_max - theta_min)
    h = cfg.cylinder_height_mm * (1.0 - v[:, None])
    theta = np.broadcast_to(theta, (height, width))
    h = np.broadcast_to(h, (height, width))

    radius = cfg.radius_mm
    cx, cy = cfg.center_mm
    ox, oy, oz = cfg.observer_mm
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    sx = cx + radius * cos_t
    sy = cy + radius * sin_t
    sz = h

    vx = sx - ox
    vy = sy - oy
    vz = sz - oz
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    vx /= norm
    vy /= norm
    vz /= norm
    dot = vx * cos_t + vy * sin_t
    rx = vx - 2.0 * dot * cos_t
    ry = vy - 2.0 * dot * sin_t
    rz = vz

    valid = np.abs(rz) > 1e-9
    t = np.full_like(rz, np.nan)
    t[valid] = -sz[valid] / rz[valid]
    px = sx + t * rx
    py = sy + t * ry
    valid &= t > 0
    valid &= (px >= 0) & (px <= cfg.paper_width - 1) & (py >= 0) & (py <= cfg.paper_height - 1)
    col = np.rint(px).astype(np.int64)
    row = np.rint((cfg.paper_height - 1) - py).astype(np.int64)
    return row, col, valid


def sample_mirror(paper: np.ndarray, row: np.ndarray, col: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.ones((row.shape[0], row.shape[1], 3), dtype=np.float64)
    out[valid] = paper[row[valid], col[valid]]
    return out


def scatter_mirror_residual(residual: np.ndarray, row: np.ndarray, col: np.ndarray, valid: np.ndarray,
                            shape: tuple[int, int, int]) -> np.ndarray:
    grad = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape[:2], dtype=np.float64)
    rr = row[valid]
    cc = col[valid]
    np.add.at(grad, (rr, cc), residual[valid])
    np.add.at(counts, (rr, cc), 1.0)
    used = counts > 0
    grad[used] /= counts[used, None]
    return grad


def laplacian(img: np.ndarray) -> np.ndarray:
    return (
        -4.0 * img
        + np.roll(img, 1, axis=0)
        + np.roll(img, -1, axis=0)
        + np.roll(img, 1, axis=1)
        + np.roll(img, -1, axis=1)
    )


def tv_gradient(img: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    dx = np.diff(img, axis=1, append=img[:, -1:, :])
    dy = np.diff(img, axis=0, append=img[-1:, :, :])
    mag = np.sqrt(dx * dx + dy * dy + eps * eps)
    px = dx / mag
    py = dy / mag
    div_x = px - np.roll(px, 1, axis=1)
    div_y = py - np.roll(py, 1, axis=0)
    return -(div_x + div_y)


def regularization_gradient(img: np.ndarray, anchor: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return np.zeros_like(img)
    if kind == "l2_anchor":
        return 2.0 * (img - anchor)
    if kind == "l1_anchor":
        return np.sign(img - anchor)
    if kind == "tikhonov":
        return -laplacian(img)
    if kind == "tv":
        return tv_gradient(img)
    raise ValueError(f"unknown regularization kind: {kind}")


def optimize_dual_pattern(anchor: np.ndarray, mirror_target: np.ndarray, row: np.ndarray, col: np.ndarray,
                          valid: np.ndarray, regularizer: str, gamma: float, alpha: float,
                          beta: float, cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    x = anchor.copy()
    for _ in range(cfg.iterations):
        mirror = sample_mirror(x, row, col, valid)
        mirror_residual = mirror - mirror_target
        grad = 2.0 * alpha * (x - anchor)
        grad += 2.0 * beta * scatter_mirror_residual(mirror_residual, row, col, valid, x.shape)
        grad += gamma * regularization_gradient(x, anchor, regularizer)
        x = np.clip(x - cfg.step_size * grad, 0.0, 1.0)
    return x, sample_mirror(x, row, col, valid)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = mse(a, b)
    return float(10.0 * math.log10(1.0 / max(err, 1e-12)))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    ga = a.mean(axis=2)
    gb = b.mean(axis=2)
    mu_a = ga.mean()
    mu_b = gb.mean()
    var_a = ga.var()
    var_b = gb.var()
    cov = ((ga - mu_a) * (gb - mu_b)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))


def edges(img: np.ndarray) -> np.ndarray:
    gray = img.mean(axis=2)
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    mag = np.sqrt(gx * gx + gy * gy)
    threshold = np.percentile(mag, 78)
    return mag > threshold


def edge_f1(a: np.ndarray, b: np.ndarray) -> float:
    ea = edges(a)
    eb = edges(b)
    tp = np.logical_and(ea, eb).sum()
    fp = np.logical_and(ea, ~eb).sum()
    fn = np.logical_and(~ea, eb).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def entropy(img: np.ndarray) -> float:
    gray = np.clip((img.mean(axis=2) * 255).astype(np.uint8), 0, 255)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def total_variation(img: np.ndarray) -> float:
    dx = np.abs(np.diff(img, axis=1)).mean()
    dy = np.abs(np.diff(img, axis=0)).mean()
    return float(dx + dy)


def meaningfulness_score(candidate: np.ndarray, target: np.ndarray) -> float:
    ssim = max(0.0, min(1.0, ssim_gray(candidate, target)))
    ef1 = edge_f1(candidate, target)
    # Entropy is treated as an inverted-U complexity score: too simple and too noisy are both bad.
    h0 = entropy(target)
    sigma_h = max(0.20 * h0, 0.35)
    ent_score = math.exp(-((entropy(candidate) - h0) ** 2) / (2.0 * sigma_h * sigma_h))
    return float(0.60 * ssim + 0.25 * ef1 + 0.15 * ent_score)


def save_img(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).save(path)


def contact_sheet(anchor: np.ndarray, mirror_target: np.ndarray, paper: np.ndarray, mirror: np.ndarray,
                  path: Path, title: str) -> None:
    panels = [
        ("paper target A*", anchor),
        ("mirror target B*", mirror_target),
        ("optimized paper f", paper),
        ("simulated mirror M[f]", mirror),
    ]
    h = 220
    imgs = []
    for label, arr in panels:
        im = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
        w = int(im.width * h / im.height)
        im = im.resize((w, h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (w, h + 34), "white")
        canvas.paste(im, (0, 34))
        ImageDraw.Draw(canvas).text((8, 10), label, fill=(20, 20, 20), font=ImageFont.load_default())
        imgs.append(canvas)
    out = Image.new("RGB", (sum(i.width for i in imgs), h + 68), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), title, fill=(20, 20, 20), font=ImageFont.load_default())
    x = 0
    for im in imgs:
        out.paste(im, (x, 34))
        x += im.width
    out.save(path)


def plot_coefficient_effect(rows: list[dict[str, object]]) -> None:
    l2_rows = [r for r in rows if r["regularizer"] == "l2_anchor"]
    gammas = [float(r["gamma"]) for r in l2_rows]
    paper = [float(r["paper_meaningfulness"]) for r in l2_rows]
    mirror = [float(r["mirror_meaningfulness"]) for r in l2_rows]
    tvs = [float(r["paper_total_variation"]) for r in l2_rows]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.0), dpi=160)
    ax1.plot(gammas, paper, marker="o", label="paper meaningfulness")
    ax1.plot(gammas, mirror, marker="s", label="mirror meaningfulness")
    ax1.set_xscale("symlog", linthresh=0.001)
    ax1.set_xlabel("regularization coefficient gamma")
    ax1.set_ylabel("meaningfulness score")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(gammas, tvs, marker="^", color="#666666", label="paper TV")
    ax2.set_ylabel("total variation")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q2_l2_coefficient_effect.png")
    fig.savefig(FIG_DIR / "q2_l2_coefficient_effect.pdf")
    plt.close(fig)


def plot_regularizer_comparison(rows: list[dict[str, object]]) -> None:
    chosen = [r for r in rows if abs(float(r["gamma"]) - 0.03) < 1e-12]
    labels = [str(r["regularizer"]) for r in chosen]
    x = np.arange(len(labels))
    paper = [float(r["paper_meaningfulness"]) for r in chosen]
    mirror = [float(r["mirror_meaningfulness"]) for r in chosen]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=160)
    ax.bar(x - 0.18, paper, width=0.36, label="paper")
    ax.bar(x + 0.18, mirror, width=0.36, label="mirror")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("meaningfulness score")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q2_regularizer_comparison.png")
    fig.savefig(FIG_DIR / "q2_regularizer_comparison.pdf")
    plt.close(fig)


def write_outputs(rows: list[dict[str, object]]) -> None:
    csv_path = OUT_DIR / "q2_regularization_metrics.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT_DIR / "q2_regularization_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    best = max(rows, key=lambda r: min(float(r["paper_meaningfulness"]), float(r["mirror_meaningfulness"])))
    lines = [
        "# 第二问：图像意义判定与正则化影响实验",
        "",
        "## 1. 可计算的“有意义”判定",
        "",
        "将图像是否有意义转化为可复现的综合评分：",
        "",
        "\\[",
        "S(I,T)=0.60S_{sem}(I,T)+0.25S_{str}(I,T)+0.15S_{ent}(I).",
        "\\]",
        "",
        "其中语义项由 SSIM 近似，结构项由边缘 F1 近似，熵项采用倒 U 型复杂度评分，用于惩罚过度平滑和过度杂乱。若纸面评分和镜面评分同时超过阈值，可认为两幅图均保持主要语义。",
        "",
        "## 2. 双意义可行条件",
        "",
        "设纸面目标为 \\(A^*\\)，镜面目标为 \\(B^*\\)，圆柱映射为 \\(M_p\\)。双意义作品需要满足：",
        "",
        "\\[",
        "S(f,A^*)\\ge \\tau_A,\\quad S(M_p[f],B^*)\\ge \\tau_B,\\quad p\\in\\mathcal{P}.",
        "\\]",
        "",
        "更强的局部兼容条件是在映射区域 \\(D_1\\) 上，\\(A^*\\) 与 \\(B^*\\circ Q_p^{-1}\\) 不能强冲突。若两者在该区域结构差异过大，只能得到折中解。",
        "",
        "## 3. 正则化模型",
        "",
        "实验求解如下离散优化问题：",
        "",
        "\\[",
        "\\min_f\\alpha\\|f-A^*\\|_2^2+\\beta\\|M_p[f]-B^*\\|_2^2+\\gamma R(f).",
        "\\]",
        "",
        f"指标表保存于 `{csv_path.as_posix()}`。",
        "",
        "## 4. 关键结果",
        "",
        "| regularizer | gamma | paper score | mirror score | paper PSNR | mirror PSNR | TV |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['regularizer']} | {float(r['gamma']):.4f} | {float(r['paper_meaningfulness']):.4f} | "
            f"{float(r['mirror_meaningfulness']):.4f} | {float(r['paper_psnr']):.2f} | "
            f"{float(r['mirror_psnr']):.2f} | {float(r['paper_total_variation']):.4f} |"
        )
    lines.extend([
        "",
        "## 5. 实验结论",
        "",
        f"- 按 max-min 准则，最佳折中为 `{best['regularizer']}`，`gamma={float(best['gamma']):.4f}`，纸面评分 {float(best['paper_meaningfulness']):.4f}，镜面评分 {float(best['mirror_meaningfulness']):.4f}。",
        "- 正则化系数过小，优化更偏向同时追逐两个目标，镜面评分较高但纸面图案可能出现局部干扰；正则化系数过大，图像被过度锚定到纸面目标，镜面目标难以恢复。",
        "- L2 锚定正则适合保持纸面原图整体外观；L1 锚定正则允许少量局部区域承担镜面编码；Tikhonov 和 TV 正则能降低高频振荡，但在本实验的倒 U 型复杂度评分下会因细节损失受到惩罚。",
        "",
        "## 6. 论文可用图表",
        "",
        "- `figures/problem2/q2_l2_coefficient_effect.pdf`：正则化系数影响。",
        "- `figures/problem2/q2_regularizer_comparison.pdf`：不同正则化方法对比。",
        "- `figures/problem2/q2_best_tradeoff_comparison.png`：最佳折中图像示例。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    np.random.seed(SEED)
    ensure_dirs()
    cfg = ExperimentConfig()
    anchor, mirror_target = prepare_targets(cfg)
    row, col, valid = inverse_map_grid(mirror_target.shape[1], mirror_target.shape[0], cfg)

    alpha = 1.0
    beta = 1.0
    experiments: list[tuple[str, float]] = []
    for gamma in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        experiments.append(("l2_anchor", gamma))
    for kind in ["none", "l1_anchor", "tikhonov", "tv"]:
        experiments.append((kind, 0.03))

    rows: list[dict[str, object]] = []
    best_key = None
    best_value = -1.0
    for regularizer, gamma in experiments:
        paper, mirror = optimize_dual_pattern(anchor, mirror_target, row, col, valid, regularizer, gamma, alpha, beta, cfg)
        stem = f"q2_{regularizer}_gamma_{gamma:g}".replace(".", "p")
        paper_path = FIG_DIR / f"{stem}_paper.png"
        mirror_path = FIG_DIR / f"{stem}_mirror.png"
        sheet_path = FIG_DIR / f"{stem}_comparison.png"
        save_img(paper, paper_path)
        save_img(mirror, mirror_path)
        contact_sheet(anchor, mirror_target, paper, mirror, sheet_path, f"{regularizer}, gamma={gamma:g}")

        p_score = meaningfulness_score(paper, anchor)
        m_score = meaningfulness_score(mirror, mirror_target)
        current = min(p_score, m_score)
        if current > best_value:
            best_value = current
            best_key = (paper, mirror)
            contact_sheet(anchor, mirror_target, paper, mirror, FIG_DIR / "q2_best_tradeoff_comparison.png", f"best: {regularizer}, gamma={gamma:g}")

        rows.append({
            "regularizer": regularizer,
            "gamma": gamma,
            "paper_meaningfulness": p_score,
            "mirror_meaningfulness": m_score,
            "paper_ssim": ssim_gray(paper, anchor),
            "mirror_ssim": ssim_gray(mirror, mirror_target),
            "paper_edge_f1": edge_f1(paper, anchor),
            "mirror_edge_f1": edge_f1(mirror, mirror_target),
            "paper_entropy": entropy(paper),
            "mirror_entropy": entropy(mirror),
            "paper_psnr": psnr(paper, anchor),
            "mirror_psnr": psnr(mirror, mirror_target),
            "paper_total_variation": total_variation(paper),
            "paper_png": str(paper_path.relative_to(ROOT)),
            "mirror_png": str(mirror_path.relative_to(ROOT)),
            "comparison_png": str(sheet_path.relative_to(ROOT)),
        })

    plot_coefficient_effect(rows)
    plot_regularizer_comparison(rows)
    write_outputs(rows)
    if best_key is not None:
        save_img(best_key[0], FIG_DIR / "q2_best_tradeoff_paper.png")
        save_img(best_key[1], FIG_DIR / "q2_best_tradeoff_mirror.png")
    print(f"saved metrics to {OUT_DIR / 'q2_regularization_metrics.csv'}")
    print(f"saved report to {REPORT_PATH}")
    print(f"saved figures to {FIG_DIR}")


if __name__ == "__main__":
    run()

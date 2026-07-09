"""
Q1: Cylindrical Mirror Anamorphosis — Complete Mirror Image Design
====================================================================
Physics:  eye -> mirror_cylinder -> reflect -> paper (z=0 plane)
Method:   Forward bilinear splat — mirror pixels deposit onto paper via rays.
Key:      Target designed only on VISIBLE theta range (~175 deg).
          R=18mm, H=50mm — 97%+ of visible mirror maps to A4.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata, RectBivariateSpline
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, time, warnings; warnings.filterwarnings('ignore')

A4_W, A4_L = 210.0, 297.0  # mm

# ═══════════════════════════════════════════════════════════════
# 1. PHYSICS — one function, used everywhere consistently
# ═══════════════════════════════════════════════════════════════
def ray_to_paper(theta, z, R, E, C):
    """Eye -> mirror -> paper(z=0). Returns (px,py) in mm. nan = invalid."""
    theta, z = np.float64(theta), np.float64(z)
    Mx = C[0] + R*np.cos(theta); My = C[1] + R*np.sin(theta); Mz = z
    Nx, Ny = np.cos(theta), np.sin(theta)
    # eye->mirror direction (= toward surface)
    dx, dy, dz = Mx-E[0], My-E[1], Mz-E[2]
    nrm = np.sqrt(dx*dx+dy*dy+dz*dz) + 1e-15
    ix, iy, iz = dx/nrm, dy/nrm, dz/nrm
    dot = ix*Nx + iy*Ny           # < 0 => front-face (visible)
    rx, ry, rz = ix-2*dot*Nx, iy-2*dot*Ny, iz
    t = np.where(np.abs(rz) > 1e-12, -Mz/rz, np.nan)
    px, py = Mx+t*rx, My+t*ry
    bad = (dot>=0) | (rz>=-1e-12) | (t<=0) | ~np.isfinite(t)
    return np.where(bad, np.nan, px), np.where(bad, np.nan, py)

# ═══════════════════════════════════════════════════════════════
# 2. VISIBLE THETA RANGE
# ═══════════════════════════════════════════════════════════════
def visible_range(R, E, C):
    """Find the contiguous visible theta span (~175 deg)."""
    th = np.linspace(0, 2*np.pi, 2000)
    dot = (C[0]-E[0])*np.cos(th) + (C[1]-E[1])*np.sin(th) + R
    front = dot < 0
    # find largest contiguous block in [0, 2pi]
    diff = np.diff(front.astype(int))
    starts = np.where(diff == 1)[0]+1; ends = np.where(diff == -1)[0]
    if len(starts)==0 or len(ends)==0:
        return 0.0, 2*np.pi
    lens = [(e-s if e>s else 2000-s+e) for s,e in zip(starts,ends)]
    bi = np.argmax(lens)
    s, e = starts[bi], ends[bi]
    t0 = float(th[s]); t1 = float(th[e])
    if t1 <= t0: t1 += 2*np.pi
    print(f"  Visible: {np.degrees(t0):.0f} to {np.degrees(t1):.0f} deg ({np.degrees(t1-t0):.0f} deg span)")
    return t0, t1-t0  # start, span (rad)

# ═══════════════════════════════════════════════════════════════
# 3. DESIGN — target pattern -> paper (splat)
# ═══════════════════════════════════════════════════════════════
def design(target_img, th0, dth, R, H, E, C, paper_dpi=6.0):
    """
    Splat target onto paper. Target defined on [th0, th0+dth] x [0, H].
    Returns paper (ny,nx) array and per-pixel fill count.
    """
    nzt, nth = target_img.shape
    nx = int(A4_W * paper_dpi); ny = int(A4_L * paper_dpi)

    # Ray grid: supersample target by 6x for density
    ss = 6
    nth_r = nth * ss; nzt_r = nzt * ss
    th_r = np.linspace(th0, th0+dth, nth_r); zv_r = np.linspace(0, H, nzt_r)
    TV, ZV = np.meshgrid(th_r, zv_r)

    px, py = ray_to_paper(TV.ravel(), ZV.ravel(), R, E, C)
    ok = np.isfinite(px) & (px>=0)&(px<=A4_W) & (py>=0)&(py<=A4_L)
    print(f"    Design splat: {ok.sum():,}/{ok.size:,} rays -> A4 ({ok.sum()/ok.size*100:.1f}%)")

    # Sample target at ray positions
    import scipy.ndimage as ndi
    ti = (TV.ravel()[ok] - th0) / dth * (nth-1)
    zi = ZV.ravel()[ok] / H * (nzt-1)
    vals = ndi.map_coordinates(target_img.astype(np.float64),
          np.stack([np.clip(zi, 0, nzt-1.001), np.clip(ti, 0, nth-1.001)]),
          order=1, mode='constant', cval=1.0)

    # Bilinear subpixel accumulation
    ix_f = px[ok]*paper_dpi; iy_f = py[ok]*paper_dpi
    ix0 = np.clip(np.floor(ix_f).astype(int), 0, nx-1)
    iy0 = np.clip(np.floor(iy_f).astype(int), 0, ny-1)
    ix1 = np.clip(ix0+1, 0, nx-1); iy1 = np.clip(iy0+1, 0, ny-1)
    fx, fy = ix_f-ix0, iy_f-iy0
    w00, w10, w01, w11 = (1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy

    acc = np.zeros((ny, nx), np.float64); cnt = np.zeros((ny, nx), np.float64)
    np.add.at(acc, (iy0,ix0), vals*w00); np.add.at(cnt, (iy0,ix0), w00)
    np.add.at(acc, (iy0,ix1), vals*w10); np.add.at(cnt, (iy0,ix1), w10)
    np.add.at(acc, (iy1,ix0), vals*w01); np.add.at(cnt, (iy1,ix0), w01)
    np.add.at(acc, (iy1,ix1), vals*w11); np.add.at(cnt, (iy1,ix1), w11)

    filled = cnt > 0; paper = np.full((ny, nx), 1.0, np.float64)
    paper[filled] = acc[filled]/cnt[filled]

    # Fill small holes
    hole = ~filled
    if hole.sum()>0 and hole.sum()<hole.size*0.3:
        dist = distance_transform_edt(hole)
        near = (dist>0)&(dist<=3.0)
        if near.sum()>0:
            gy,gx = np.where(filled); fy,fx = np.where(near)
            paper[fy,fx] = griddata(np.column_stack([gx,gy]), paper[gy,gx],
                                     np.column_stack([fx,fy]), method='nearest')
    return paper

# ═══════════════════════════════════════════════════════════════
# 4. VALIDATE — render mirror from paper
# ═══════════════════════════════════════════════════════════════
def render(paper, th0, dth, R, H, E, C, n_th=600, n_z=400):
    """Forward render: paper -> mirror using PHYSICS."""
    ny,nx = paper.shape; th=np.linspace(th0, th0+dth, n_th); zv=np.linspace(0,H,n_z)
    TV,ZV = np.meshgrid(th, zv)
    px,py = ray_to_paper(TV.ravel(), ZV.ravel(), R, E, C)
    ok = np.isfinite(px)&(px>=0)&(px<=A4_W)&(py>=0)&(py<=A4_L)
    cov = ok.sum()/len(ok)*100
    p = RectBivariateSpline(np.linspace(0,A4_L,ny), np.linspace(0,A4_W,nx), paper)
    rendered = np.ones((n_z, n_th), np.float64)
    if ok.sum()>0:
        rendered.ravel()[np.where(ok.ravel())[0]] = p.ev(
            np.clip(py[ok], 0, A4_L-0.01), np.clip(px[ok], 0, A4_W-0.01))
    return rendered, ok.reshape(n_z, n_th), cov, th, zv

# ═══════════════════════════════════════════════════════════════
# 5. PLOT
# ═══════════════════════════════════════════════════════════════
def plot_all(target, paper, rendered, ok_mask, cov, R, H, C, E,
             th0, dth, name, out):
    # error only in valid region
    nz_r, nt_r = rendered.shape
    tgt_rs = np.array(Image.fromarray((target*255).astype(np.uint8))
                      .resize((nt_r, nz_r), Image.LANCZOS))/255.0
    err = rendered - tgt_rs; vp = ok_mask
    mae = float(abs(err[vp]).mean()) if vp.sum()>0 else 0
    rmse = float(np.sqrt((err[vp]**2).mean())) if vp.sum()>0 else 0

    fig, ax = plt.subplots(2, 3, figsize=(24, 13))
    t0d, t1d = np.degrees(th0), np.degrees(th0+dth)

    # A: Target (cropped to visible range)
    ax[0,0].imshow(target, cmap='gray', aspect='auto', extent=[t0d, t1d, H, 0])
    ax[0,0].set_title('A. Target Mirror Pattern', fontsize=14, fontweight='bold')
    ax[0,0].set_xlabel('theta (deg)')

    # B: Paper
    ax[0,1].imshow(paper, cmap='gray', origin='lower', extent=[0,210,0,297])
    ax[0,1].set_title('B. Paper Pattern (A4, print 100%)', fontsize=14, fontweight='bold')
    ax[0,1].set_xlabel('x (mm)'); ax[0,1].set_ylabel('y (mm)')
    ax[0,1].add_patch(plt.Circle((C[0],C[1]), R+5, fc='none', ec='red', lw=2, ls='--'))
    ax[0,1].text(C[0], C[1]-R-16, 'MIRROR', ha='center', color='red', fontweight='bold', fontsize=10)

    # C: Paper zoom
    ax[0,2].imshow(paper, cmap='gray', origin='lower', extent=[0,210,0,297])
    m=90; ax[0,2].set_xlim(max(0,C[0]-m), min(210,C[0]+m))
    ax[0,2].set_ylim(max(0,C[1]-m), min(297,C[1]+m))
    ax[0,2].set_title('C. Paper Zoom (near cylinder)', fontsize=14)
    ax[0,2].add_patch(plt.Circle((C[0],C[1]), R, fc='none', ec='red', lw=2))

    # D: Mirror reflection
    ax[1,0].imshow(rendered, cmap='gray', aspect='auto', extent=[t0d, t1d, H, 0])
    ax[1,0].set_title(f'D. Mirror Reflection ({cov:.0f}% visible->A4)', fontsize=14, fontweight='bold')

    # E: Error
    vm = max(abs(err[vp].min()) if vp.sum()>0 else 0.01,
             abs(err[vp].max()) if vp.sum()>0 else 0.01, 0.01)
    ax[1,1].imshow(np.where(vp, err, 0.0), cmap='RdBu_r', aspect='auto',
                    vmin=-vm, vmax=vm, extent=[t0d, t1d, H, 0])
    ax[1,1].set_title(f'E. Error (MAE={mae:.5f}, RMSE={rmse:.5f})', fontsize=13)
    plt.colorbar(ax[1,1].images[0], ax=ax[1,1], shrink=0.85)

    # F: Info
    ax[1,2].axis('off')
    ax[1,2].text(0.05, 0.95, (
        f"DESIGN SPECIFICATIONS\n{'='*28}\n\n"
        f"  Mirror R = {R:.0f} mm, H = {H:.0f} mm\n"
        f"  Center  C = ({C[0]:.0f}, {C[1]:.0f}) mm\n"
        f"  Eye     E = ({E[0]:.0f}, {E[1]:.0f}, {E[2]:.0f}) mm\n"
        f"  Paper   A4: 210 x 297 mm\n\n"
        f"QUALITY\n{'='*28}\n\n"
        f"  Visible->A4:  {cov:.0f}%\n"
        f"  MAE:          {mae:.5f}\n"
        f"  RMSE:         {rmse:.5f}\n"
        f"  Mirror img:   {nz_r}x{nt_r} px\n"
        f"  Paper img:    {paper.shape[1]}x{paper.shape[0]} px\n\n"
        f"HOW TO USE\n{'='*28}\n\n"
        f"  1. Print paper_pattern at 100%\n"
        f"     scale on A4 paper.\n"
        f"  2. Place cylindrical mirror (R={R:.0f}\n"
        f"     mm) at red circle.\n"
        f"  3. View from ({E[0]:.0f},{E[1]:.0f},{E[2]:.0f}) mm.\n"
    ), transform=ax[1,2].transAxes, fontsize=11, fontfamily='monospace', va='top')

    plt.suptitle(f'Q1: {name}', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(); p = os.path.join(out, f'q1_{name}.png')
    fig.savefig(p, dpi=160, bbox_inches='tight'); plt.close(fig)
    Image.fromarray((np.clip(paper,0,1)*255).astype(np.uint8)).save(
        os.path.join(out, f'paper_{name}.png'))
    return mae, rmse

# ═══════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    out_dir = "results/Q1/experiments/round1"
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)

    # OPTIMIZED parameters
    E = np.array([0.0, -300.0, 250.0])
    R = 18.0; H = 50.0; C = np.array([105.0, 120.0])

    print("="*60)
    print(f"Q1: Cylindrical Mirror Anamorphosis")
    print(f"  R={R}mm H={H}mm C=({C[0]},{C[1]}) E=({E[0]},{E[1]},{E[2]})")
    print("="*60)

    # Visible range
    print("\nVisible theta range:")
    th0, dth = visible_range(R, E, C)

    # Load targets — target image IS the full visible mirror surface
    print("\nLoading target images...")
    targets = {}

    def load_target(path, label):
        if not os.path.exists(path): return
        im = Image.open(path).convert('L')
        im = im.resize((400, 300), Image.LANCZOS)  # target mirror resolution
        targets[label] = np.array(im, dtype=np.float64)/255.0
        print(f"  {label}: {path} -> {im.size}")

    load_target("d:/cc/hzbB/page3_img1.jpeg", "Figure_3")
    load_target("d:/cc/hzbB/page3_img2.jpeg", "Figure_4")

    # Text pattern
    txt = Image.new('L', (400, 300), 255); draw = ImageDraw.Draw(txt)
    for fn in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
        try: draw.text((50,100),"MCM 2026",fill=0,font=ImageFont.truetype(fn,72)); break
        except: pass
    targets["MCM"] = 1.0-np.array(txt, dtype=np.float64)/255.0

    # Ring
    yy,xx=np.ogrid[:300,:400]
    ring=np.abs((xx-200)**2/95**2+(yy-150)**2/65**2-1.0)
    targets["Ring"]=np.clip(ring*2.5, 0.15, 1.0)

    # Design + validate each
    print("\nDesigning paper patterns...")
    all_m = {}
    for nm, tgt in targets.items():
        print(f"\n  [{nm}]")
        paper = design(tgt, th0, dth, R, H, E, C, paper_dpi=6.0)
        rendered, ok, cov, _, _ = render(paper, th0, dth, R, H, E, C)
        mae, rmse = plot_all(tgt, paper, rendered, ok, cov, R, H, C, E, th0, dth, nm, fig_dir)
        all_m[nm] = {"MAE": mae, "RMSE": rmse, "Coverage%": cov}
        print(f"    MAE={mae:.5f}  RMSE={rmse:.5f}  Cov={cov:.0f}%")

    # Save
    json.dump({"Q":"Q1","method":"M1 Splat","params":{"R_mm":R,"H_mm":H,
        "Xc_mm":float(C[0]),"Yc_mm":float(C[1]),"E_mm":[0.0,-300.0,250.0]},
        "visible_range_deg":[np.degrees(th0),np.degrees(th0+dth)],
        "patterns":all_m},
        open(os.path.join(out_dir,"run_summary.json"),"w"), indent=2)

    print(f"\n{'='*60}\nQ1 DONE!")
    for nm,m in all_m.items():
        print(f"  {nm:20s} MAE={m['MAE']:.5f}  RMSE={m['RMSE']:.5f}  Cov={m['Coverage%']:.0f}%")
    print("="*60)

if __name__=="__main__":
    t0=time.time(); main()
    print(f"Runtime: {time.time()-t0:.1f}s")

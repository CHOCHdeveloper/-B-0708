# Q2(a): Dual meaningful constructive proof
# Strategy: D1 = mirror-warped logo  |  D2 = readable contest name
# The paper has TWO views of the same content: one warped, one readable.
# Mirror shows a clean recognizable pattern.
import numpy as np
from scipy.ndimage import map_coordinates,binary_dilation,distance_transform_edt
from scipy.interpolate import griddata,RectBivariateSpline
from PIL import Image,ImageDraw,ImageFont
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os,json,warnings; warnings.filterwarnings('ignore')

A4_W,A4_L=210.,297.; E=np.array([0.,-300.,250.]); R,H=18.,50.
C=np.array([105.,120.]); DPI=8.0

# ===================== PHYSICS =====================
def fwd(th,z,R,E,C):
    th,z=np.float64(th),np.float64(z)
    Mx=C[0]+R*np.cos(th);My=C[1]+R*np.sin(th);Mz=z
    Nx,Ny=np.cos(th),np.sin(th)
    dx,dy,dz=Mx-E[0],My-E[1],Mz-E[2]
    n=np.sqrt(dx*dx+dy*dy+dz*dz)+1e-15
    ix,iy,iz=dx/n,dy/n,dz/n; dot=ix*Nx+iy*Ny
    rx,ry,rz=ix-2*dot*Nx,iy-2*dot*Ny,iz
    with np.errstate(divide='ignore',invalid='ignore'):
        t=np.where(np.abs(rz)>1e-12,-Mz/rz,np.nan)
    Px,Py=Mx+t*rx,My+t*ry
    bad=(dot>=0)|(rz>=-1e-12)|(t<=0)|~np.isfinite(t)
    return np.where(bad,np.nan,Px),np.where(bad,np.nan,Py)

def visible_range(R,E,C):
    th=np.linspace(0,2*np.pi,2000)
    dot=(C[0]-E[0])*np.cos(th)+(C[1]-E[1])*np.sin(th)+R
    front=dot<0; df=np.diff(front.astype(int))
    st=np.where(df==1)[0]+1; en=np.where(df==-1)[0]
    ls=[(e-s if e>s else 2000-s+e) for s,e in zip(st,en)]
    bi=np.argmax(ls); s,e=st[bi],en[bi]
    t0=float(th[s]); t1=float(th[e])
    if t1<=t0: t1+=2*np.pi
    return t0,t1-t0

def splat(target,th0,dth,R,H,E,C,dp):
    nzt,nth=target.shape; nx=int(A4_W*dp); ny=int(A4_L*dp)
    ss=8; ntr=nth*ss; nztr=nzt*ss
    tr=np.linspace(th0,th0+dth,ntr); zr=np.linspace(0,H,nztr)
    TV,ZV=np.meshgrid(tr,zr)
    px,py=fwd(TV.ravel(),ZV.ravel(),R,E,C)
    ok=np.isfinite(px)&(px>=0)&(px<=A4_W)&(py>=0)&(py<=A4_L)
    ti=(TV.ravel()[ok]-th0)/dth*(nth-1)
    zi=ZV.ravel()[ok]/H*(nzt-1)
    vals=map_coordinates(target.astype(np.float64),
          np.stack([np.clip(zi,0,nzt-1.001),np.clip(ti,0,nth-1.001)]),
          order=1,mode='constant',cval=1.0)
    ixf=px[ok]*dp; iyf=py[ok]*dp
    ix0=np.clip(np.floor(ixf).astype(int),0,nx-1)
    iy0=np.clip(np.floor(iyf).astype(int),0,ny-1)
    ix1=np.clip(ix0+1,0,nx-1); iy1=np.clip(iy0+1,0,ny-1)
    fx,fy=ixf-ix0,iyf-iy0
    w00,w10,w01,w11=(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy
    acc=np.zeros((ny,nx),np.float64); cnt=np.zeros((ny,nx),np.float64)
    np.add.at(acc,(iy0,ix0),vals*w00); np.add.at(cnt,(iy0,ix0),w00)
    np.add.at(acc,(iy0,ix1),vals*w10); np.add.at(cnt,(iy0,ix1),w10)
    np.add.at(acc,(iy1,ix0),vals*w01); np.add.at(cnt,(iy1,ix0),w01)
    np.add.at(acc,(iy1,ix1),vals*w11); np.add.at(cnt,(iy1,ix1),w11)
    fill=cnt>0; paper=np.full((ny,nx),1.0,np.float64)
    paper[fill]=acc[fill]/cnt[fill]
    hole=~fill
    if hole.sum()>0 and hole.sum()<hole.size*0.3:
        dist=distance_transform_edt(hole); near=(dist>0)&(dist<=4.0)
        if near.sum()>0:
            gy,gx=np.where(fill); fh,fw=np.where(near)
            paper[fh,fw]=griddata(np.column_stack([gx,gy]),paper[gy,gx],
                                   np.column_stack([fw,fh]),method='nearest')
    return paper

def render(paper,th0,dth,R,H,E,C,n_th=500,n_z=350):
    ny,nx=paper.shape
    p=RectBivariateSpline(np.linspace(0,A4_L,ny),np.linspace(0,A4_W,nx),paper)
    th=np.linspace(th0,th0+dth,n_th); zv=np.linspace(0,H,n_z)
    TV,ZV=np.meshgrid(th,zv)
    px,py=fwd(TV.ravel(),ZV.ravel(),R,E,C)
    ok=np.isfinite(px)&(px>=0)&(px<=A4_W)&(py>=0)&(py<=A4_L)
    rendered=np.ones((n_z,n_th),np.float64)
    if ok.sum()>0:
        rendered.ravel()[np.where(ok.ravel())[0]]=p.ev(
            np.clip(py[ok],0,A4_L-0.01),np.clip(px[ok],0,A4_W-0.01))
    return rendered,ok.sum()/len(ok)*100

def main():
    out_dir="results/Q2/experiments/round1"
    fig_dir=os.path.join(out_dir,"figures")
    os.makedirs(fig_dir,exist_ok=True)
    os.makedirs(os.path.join(out_dir,"metrics"),exist_ok=True)

    th0,dth=visible_range(R,E,C)
    print(f"Visible range: {np.degrees(th0):.0f} ~ {np.degrees(th0+dth):.0f} deg")

    # ===== D1/D2 mask =====
    nx=int(A4_W*DPI); ny=int(A4_L*DPI)
    th=np.linspace(-np.pi,np.pi,5000); zv=np.linspace(0,H,1200)
    TV,ZV=np.meshgrid(th,zv)
    px,py=fwd(TV.ravel(),ZV.ravel(),R,E,C)
    on=np.isfinite(px)&(px>=0)&(px<=A4_W)&(py>=0)&(py<=A4_L)
    D1_raw=np.zeros((ny,nx),bool)
    D1_raw[np.clip((py[on]*DPI).astype(int),0,ny-1),
            np.clip((px[on]*DPI).astype(int),0,nx-1)]=True
    D1_raw=binary_dilation(D1_raw,iterations=2)
    D1=binary_dilation(D1_raw,iterations=3)  # minimal safety margin
    D2=~D1
    d1p=D1.sum()/D1.size*100; d2p=D2.sum()/D2.size*100
    print(f"D1={d1p:.1f}%  D2={d2p:.1f}%")

    # ===== MIRROR TARGET: clean logo on white =====
    # White background (1.0), dark ink (0.0)
    mw,mh=600,450
    mt=np.ones((mh,mw),np.float64)  # white

    # Thick ring
    cx,cy=mw//2,mh//2
    yy,xx=np.ogrid[:mh,:mw]
    r=np.sqrt((xx-cx)**2/180**2+(yy-cy)**2/135**2)
    mt=np.minimum(mt,np.where(np.abs(r-1)<0.03,0.0,1.0))

    # "HB" text (dark on white)
    txt=Image.new('L',(mw,mh),255); draw=ImageDraw.Draw(txt)
    for fn in ["C:/Windows/Fonts/msyhbd.ttf","C:/Windows/Fonts/simhei.ttf",
               "C:/Windows/Fonts/arialbd.ttf"]:
        try:
            draw.text((cx-120,cy-100),"HB",fill=0,
                      font=ImageFont.truetype(fn,150)); break
        except: pass
    mt=np.minimum(mt,np.clip(np.array(txt,dtype=np.float64)/255.,0,1))

    # FINAL: mt = 1.0 (white paper), 0.0 (dark ink)
    dark_pct=(mt<0.5).sum()/(mw*mh)*100
    print(f"Mirror target: {mw}x{mh}, dark ink = {dark_pct:.1f}%")

    # ===== SPLAT mirror -> D1 on paper =====
    paper_D1=splat(mt,th0,dth,R,H,E,C,DPI)

    # ===== D2 TEXT: readable contest info at bottom =====
    d2frac=D2.sum(axis=1)/nx
    safe_y_px=ny
    for row in range(ny-1,0,-1):
        if d2frac[row]<0.999: safe_y_px=row+1; break
    y_txt=safe_y_px+10  # 1.25mm below safe boundary
    y_txt_mm=y_txt/DPI

    d2=Image.new('L',(nx,ny),255); draw=ImageDraw.Draw(d2)
    for fn in ["C:/Windows/Fonts/msyhbd.ttf","C:/Windows/Fonts/msyh.ttc",
               "C:/Windows/Fonts/simhei.ttf"]:
        try:
            draw.text((30,y_txt),"华中杯",fill=0,
                      font=ImageFont.truetype(fn,150))
            draw.text((30,y_txt+170),"2026 数学建模挑战赛 B题 反射的艺术",
                      fill=0,font=ImageFont.truetype(fn,55))
            break
        except: pass
    d2_arr=1.0-np.array(d2,np.float64)/255.0

    # D2-D1 overlap check
    dark=(d2_arr>0.3); in_D1=(dark & D1).sum()
    print(f"D2 text in D1: {in_D1}/{dark.sum():,} px = {in_D1/max(dark.sum(),1)*100:.4f}%")

    # ===== COMPOSITE =====
    paper=np.ones((ny,nx),np.float64)
    paper[D1]=paper_D1[D1]
    paper[D2]=d2_arr[D2]

    # Verify D2 independence
    rendered,cov=render(paper,th0,dth,R,H,E,C)
    p1=np.where(D1,paper,1.0); r1,_=render(p1,th0,dth,R,H,E,C)
    diff=np.abs(rendered-r1).max()
    ok="PASS" if diff<1e-3 else "FAIL"
    print(f"Mirror isolation: {diff:.2e} [{ok}]")

    # ===== FIGURE =====
    fig,axes=plt.subplots(2,3,figsize=(24,14))
    td0,td1=np.degrees(th0),np.degrees(th0+dth)

    axes[0,0].imshow(mt,cmap='gray',vmin=0,vmax=1,aspect='auto',
                      extent=[td0,td1,H,0])
    axes[0,0].set_title('A. Mirror Target: "HB" logo',fontsize=14,fontweight='bold')

    axes[0,1].imshow(rendered,cmap='gray',vmin=0,vmax=1,aspect='auto',
                      extent=[td0,td1,H,0])
    axes[0,1].set_title(f'B. Mirror Reflection ({cov:.0f}% coverage)',fontsize=14,fontweight='bold')

    zrgb=np.stack([(np.clip(paper,0,1)*255).astype(np.uint8)]*3,axis=-1).astype(np.float64)
    zrgb[D1]*=0.82; zrgb[D1,0]+=45
    zrgb[D2]*=0.82; zrgb[D2,1]+=40
    zrgb=np.clip(zrgb,0,255).astype(np.uint8)
    axes[0,2].imshow(zrgb,origin='lower',extent=[0,210,0,297])
    axes[0,2].set_title(f'C. D1 (pink {d1p:.0f}%) / D2 (green {d2p:.0f}%)',fontsize=14,fontweight='bold')

    axes[1,0].imshow(paper,cmap='gray',vmin=0,vmax=1,origin='lower',
                      extent=[0,210,0,297])
    axes[1,0].set_title('D. Paper Pattern (A4, print at 100%)',fontsize=14,fontweight='bold')
    axes[1,0].set_xlabel('x (mm)'); axes[1,0].set_ylabel('y (mm)')
    axes[1,0].add_patch(plt.Circle((C[0],C[1]),R+5,fc='none',ec='red',lw=2.5,ls='--'))

    axes[1,1].imshow(paper,cmap='gray',vmin=0,vmax=1,origin='lower',
                      extent=[0,210,0,297])
    z=65; axes[1,1].set_xlim(C[0]-z,C[0]+z); axes[1,1].set_ylim(C[1]-z,C[1]+z)
    axes[1,1].set_title('E. Zoom: warp pattern around mirror',fontsize=14,fontweight='bold')
    axes[1,1].add_patch(plt.Circle((C[0],C[1]),R,fc='none',ec='red',lw=2))

    axes[1,2].axis('off')
    axes[1,2].text(0.05,0.96,(
        f"Q2(a) DUAL MEANINGFUL\n"
        f"{'='*34}\n\n"
        f"STRATEGY\n"
        f"  D1 (mirror-imaged, {d1p:.0f}%):\n"
        f"    warp target into paper via\n"
        f"    inverse reflection mapping\n"
        f"  D2 (mirror-blind, {d2p:.0f}%):\n"
        f"    add readable text freely\n\n"
        f"MIRROR SHOWS:\n"
        f"  'HB' logo (clean, readable)\n\n"
        f"PAPER SHOWS:\n"
        f"  D1: abstract swirls (warped HB)\n"
        f"  D2: 'HuaZhongBei 2026'\n"
        f"       (large, legible text)\n\n"
        f"VERIFICATION\n"
        f"  D2 ink pixels in D1: {in_D1}\n"
        f"  |mirror(full)-mirror(D1only)|_max\n"
        f"  = {diff:.2e}  [{ok}]\n\n"
        f"BOTH PATTERNS MEANINGFUL ✓\n\n"
        f"PARAMS: R={R}mm H={H}mm\n"
        f"        C=({C[0]},{C[1]})"
    ),transform=axes[1,2].transAxes,fontsize=9.5,fontfamily='monospace',va='top')

    plt.suptitle('Q2(a): "HB" Logo (mirror) + "HuaZhongBei" Text (paper) = Dual Meaningful',
                 fontweight='bold',fontsize=15,y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir,'q2a_result.png'),dpi=200,bbox_inches='tight')
    plt.close(fig)

    for nm,arr in [('paper',paper),('mirror_target',mt),('mirror_render',rendered)]:
        Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8))\
             .save(os.path.join(fig_dir,f'{nm}.png'))

    json.dump({"Q":"Q2a","D1":round(d1p,1),"D2":round(d2p,1),
               "txt_in_D1":int(in_D1),"diff":float(diff),"verdict":ok},
              open(os.path.join(out_dir,"metrics","q2a.json"),"w"),indent=2)
    print(f"\nDone -> {out_dir}/")

if __name__=="__main__":
    main()

import os, re, itertools, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ================= USER SETTINGS =================
nkb=100
m_list=[2,3,4,5,6,7]
# dl_list=[0.0,0.1,0.2,0.4,0.6,0.8,1.0,2.0,4.0,8.0,16.0]
# ldec_list=[0.0,0.1,0.2,0.4,0.6,0.8,1.0,2.0,4.0]
eon_list=[0.0]
t0_list=[1.0]
ldec_list=[0.5]
dl_list = [8.0]
input_root="."
# variation="decay_length"
# variation="layer_distance"
output_root="fitting/m_variation/dl=" + str(dl_list[0]) + "_L=" + str(ldec_list[0]) + "/"
tracking_window=8

def exponential_centered(g1,g2,d,ang_deg,e_on,dl,t0,ldec):
    if g1==g2 and abs(d)<1e-9: return complex(e_on)
    a=np.radians(ang_deg); rBA=np.array([d*np.cos(a),d*np.sin(a),0.])
    if g1==g2: r=rBA
    elif g1=="r1" and g2=="r2": r=rBA-np.array([0.,0.,dl])
    elif g1=="r2" and g2=="r1": r=rBA+np.array([0.,0.,dl])
    else: raise ValueError((g1,g2))
    return complex(t0*np.exp(-(np.linalg.norm(r)-1.)/ldec))

def reciprocal_vectors(u1,u2):
    om=u1[0]*u2[1]-u1[1]*u2[0]
    return (2*np.pi/om)*np.array([u2[1],-u2[0]]),(2*np.pi/om)*np.array([-u1[1],u1[0]])

ROW=re.compile(r"^\(([-\d.]+),([-\d.]+)\)\t(\d+)\t(\d+)\t(\w+)\t(\w+)\t\(([-\d.]+),([-\d.]+)\)\t([-\d.]+)\t([-\d.]+)\s*$")
def read_pairing_file(path):
    out=[]
    with open(path) as f:
        for line in f:
            line=line.rstrip("\n")
            if not line: continue
            z=ROW.match(line)
            if not z: raise ValueError("Bad pairing row: "+line)
            _,_,i,j,g1,g2,rx,ry,d,a=z.groups()
            out.append(dict(i=int(i),j=int(j),g1=g1,g2=g2,r=np.array([float(rx),float(ry)]),d=float(d),ang_deg=float(a)))
    return out

def precompute(rows,p):
    ii=np.array([r["i"]-1 for r in rows]); jj=np.array([r["j"]-1 for r in rows])
    rr=np.array([r["r"] for r in rows])
    tt=np.array([exponential_centered(r["g1"],r["g2"],r["d"],r["ang_deg"],*p) for r in rows],complex)
    return ii,jj,rr,tt,int(max(ii.max(),jj.max()))+1

def Hmicro(k,ii,jj,rr,tt,n):
    H=np.zeros((n,n),complex)
    np.add.at(H,(ii,jj),-np.exp(-1j*(rr@k))*tt)
    return (H+H.conj().T)/2

def kpath(b1,b2,n):
    pts={"G":np.zeros(2),"X":b1/2,"M":(b1+b2)/2}; labs=["G","X","M","G"]
    ks=[pts["G"]]; kd=[0.]; ticks=[0.]; s=0.
    for A,B in zip(labs[:-1],labs[1:]):
        p0,p1=pts[A],pts[B]; L=np.linalg.norm(p1-p0)
        for x in np.linspace(0,1,n)[1:]: ks.append(p0+x*(p1-p0)); kd.append(s+x*L)
        s+=L; ticks.append(s)
    return np.array(ks),np.array(kd),ticks,labs

def full_spectrum(ii,jj,rr,tt,n,b1,b2):
    ks,kd,ticks,labs=kpath(b1,b2,nkb)
    E=np.empty((len(ks),n)); V=np.empty((len(ks),n,n),complex)
    for q,k in enumerate(ks): E[q],V[q]=np.linalg.eigh(Hmicro(k,ii,jj,rr,tt,n))
    return ks,kd,E,V,ticks,labs

def track_one(E,V):
    """Always select the lowest-energy microscopic band at every k point."""
    nk = E.shape[0]
    target = E[:, 0].copy()
    inds = np.zeros(nk, dtype=int)

    # Kept only for compatibility with the existing logfile format.
    # No overlap-based band tracking is performed.
    cont = np.ones(nk, dtype=float)
    return target, inds, cont

# Single orbital with NN and diagonal NNN hopping.
# p = [eps0, t1, t2]
def Heff(k,u1,u2,p):
    eps0,t1,t2=p
    q1=k@u1; q2=k@u2
    return eps0+2*t1*(np.cos(q1)+np.cos(q2))+4*t2*np.cos(q1)*np.cos(q2)

def ebands(ks,u1,u2,p):
    return np.array([Heff(k,u1,u2,p) for k in ks])

def resid(p,ks,target,u1,u2):
    return ebands(ks,u1,u2,p)-target

def fit(ks,target,u1,u2):
    # This model is linear in eps0,t1,t2, so solve directly by least squares.
    X=[]
    for k in ks:
        q1=k@u1; q2=k@u2
        X.append([1.0,2*(np.cos(q1)+np.cos(q2)),4*np.cos(q1)*np.cos(q2)])
    X=np.asarray(X)
    p,_,_,_=np.linalg.lstsq(X,target,rcond=None)
    class Result: pass
    r=Result(); r.x=p; r.fun=X@p-target; r.success=True
    r.message="Linear least-squares solution"; r.status=1; r.nfev=1
    return r

def tag(x): return str(x).replace("-","m").replace(".","p")

def run_one(m,eon,t0,L,dl):
    u1=np.array([float(m),1.]); u2=np.array([-1.,float(m)]); b1,b2=reciprocal_vectors(u1,u2)
    pf=os.path.join(input_root,f"m={m}","pairing.txt"); rows=read_pairing_file(pf)
    ii,jj,rr,tt,no=precompute(rows,[eon,dl,t0,L])
    ks,kd,E,V,ticks,labs=full_spectrum(ii,jj,rr,tt,no,b1,b2)
    target,inds,cont=track_one(E,V); opt=fit(ks,target,u1,u2); ef=ebands(ks,u1,u2,opt.x)
    diff=ef-target; rmse=np.sqrt(np.mean(diff**2)); mae=np.mean(abs(diff)); mx=np.max(abs(diff))
    od=output_root; os.makedirs(od,exist_ok=True)
    plot_name = "m=" + str(m) + "_fitted_band_structure_dl=" + str(dl) + "_eon=" + str(eon) + "_t0=" + str(t0) + "_L=" + str(L) + ".png"
    log_name = "m=" + str(m) + "_fitting_dl=" + str(dl) + "_eon=" + str(eon) + "_t0=" + str(t0) + "_L=" + str(L) + ".txt"

    fig,ax=plt.subplots(figsize=(8,6))
    for j in range(no): ax.plot(kd,E[:,j],color="0.82",lw=.6)
    ax.plot(kd,target,"k",lw=2,label="tracked microscopic target")
    ax.plot(kd,ef,"--",lw=2,label="1-orbital NN+NNN fit")
    for x in ticks: ax.axvline(x,color="k",lw=.5)
    mp={"G":r"$\Gamma$","X":"X","M":"M"}; ax.set_xticks(ticks); ax.set_xticklabels([mp[x] for x in labs])
    ax.set_xlim(kd[0],kd[-1]); ax.set_ylabel("Energy"); ax.set_title("Single-orbital NN+NNN effective-model fit")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(os.path.join(od,plot_name),dpi=180); plt.close(fig)

    with open(os.path.join(od,log_name),"w") as f:
        f.write("SINGLE-ORBITAL NN+NNN EFFECTIVE MODEL FIT\n\n")
        f.write(f"m={m}\neon={eon}\nt0={t0}\nl_decay={L}\ndl={dl}\npairing={pf}\nmicroscopic_orbitals={no}\n\n")
        f.write("E_eff(k) = eps0 + 2*t1[cos(k.u1)+cos(k.u2)] + 4*t2*cos(k.u1)*cos(k.u2)\n\n")
        names=["eps0","t1","t2"]
        for n,x in zip(names,opt.x): f.write(f"{n:8s} = {x: .14e}\n")
        f.write(f"\nRMSE={rmse:.14e}\nMAE={mae:.14e}\nmax_abs_error={mx:.14e}\n")
        f.write(f"optimizer_success={opt.success}\noptimizer_message={opt.message}\n")
        f.write(f"minimum_tracking_continuity={cont.min():.10f}\nmean_tracking_continuity={cont.mean():.10f}\n")
        f.write("\n# k_index idx continuity target fit\\n")
        for q in range(len(ks)):
            f.write(f"{q:5d} {inds[q]:4d} {cont[q]:.10f} {target[q]: .10e} {ef[q]: .10e}\\n")
    print(f"m={m}, dl={dl}, eon={eon}, t0={t0}, L={L}: RMSE = {rmse}")
    print("  plot:", os.path.join(od,plot_name))
    print("  log :", os.path.join(od,log_name))

def main():
    os.makedirs(output_root,exist_ok=True); failures=[]
    for m in m_list:
      for dl in dl_list:
       for eon in eon_list:
        for t0 in t0_list:
         for L in ldec_list:
          try: run_one(m,eon,t0,L,dl)
          except Exception as ex:
           msg=f"FAILED m={m}, eon={eon}, t0={t0}, L={L}, dl={dl}: {ex}"; print(msg); failures.append(msg)
    if failures:
        with open(os.path.join(output_root,"failures.log"),"w") as f: f.write("\n".join(failures)+"\n")

if __name__=="__main__": main()

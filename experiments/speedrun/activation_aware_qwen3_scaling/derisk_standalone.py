import jax, jax.numpy as jnp, numpy as np
from functools import partial
from levanter.optim.util import zeropower_via_newtonschulz5
np.random.seed(0)

# --- synthetic anisotropic regression: y = W3 relu(W2 relu(W1 x)), x ~ N(0,C) anisotropic ---
din, h, dout, N = 64, 128, 16, 4096
# anisotropic input covariance (eigenvalues spread 1e-2 .. 1)
Q,_ = np.linalg.qr(np.random.randn(din,din)); evs = np.logspace(-2,0,din)
C = (Q*evs)@Q.T
L = np.linalg.cholesky(C+1e-9*np.eye(din))
def gen(n, key):
    x = (np.random.randn(n,din)@L.T).astype(np.float32)
    return jnp.asarray(x)
key=jax.random.PRNGKey(0)
Wt = [np.random.randn(h,din)/np.sqrt(din), np.random.randn(h,h)/np.sqrt(h), np.random.randn(dout,h)/np.sqrt(h)]
Wt=[jnp.asarray(w.astype(np.float32)) for w in Wt]
def teacher(x):
    a=jnp.maximum(x@Wt[0].T,0); a=jnp.maximum(a@Wt[1].T,0); return a@Wt[2].T
Xtr=gen(N,key); Ytr=teacher(Xtr)

def init():
    k=jax.random.PRNGKey(1); ks=jax.random.split(k,3)
    return [jax.random.normal(ks[0],(h,din))*0.1, jax.random.normal(ks[1],(h,h))*0.1, jax.random.normal(ks[2],(dout,h))*0.1]

def fwd(W,x):  # returns output + list of per-linear INPUTS (activations)
    ins=[]
    a=x; ins.append(a); a=jnp.maximum(a@W[0].T,0)
    ins.append(a); a=jnp.maximum(a@W[1].T,0)
    ins.append(a); out=a@W[2].T
    return out, ins
def loss_and_acts(W,x,y):
    out,ins=fwd(W,x); return jnp.mean((out-y)**2), ins

def sig_inv_half(AAt, damping):
    w,U=jnp.linalg.eigh(AAt); w=jnp.maximum(w,0.0)
    inv=1.0/jnp.sqrt(w+damping*jnp.mean(w)+1e-30)
    return (U*inv[None,:])@U.T
def actaware_D(M, AAt, damping):
    S=sig_inv_half(AAt,damping)
    o=zeropower_via_newtonschulz5((M@S).astype(jnp.float32),steps=5)
    D=o@S
    # normalize to Frobenius sqrt(min) like Muon for LR comparability
    k=min(M.shape); return D*(jnp.sqrt(float(k))/(jnp.linalg.norm(D)+1e-8))
def muon_D(M):
    o=zeropower_via_newtonschulz5(M.astype(jnp.float32),steps=5)
    return o*(jnp.sqrt(float(min(M.shape)))/(jnp.linalg.norm(o)+1e-8))

def run(method, lr, steps=300, beta=0.95, damping=1e-3):
    W=init(); buf=[jnp.zeros_like(w) for w in W]; Sig=[jnp.eye(w.shape[1]) for w in W]
    gloss=jax.jit(jax.value_and_grad(lambda W: loss_and_acts(W,Xtr,Ytr)[0]))
    accts=jax.jit(lambda W: loss_and_acts(W,Xtr,Ytr)[1])
    losses=[]
    for t in range(steps):
        l,g=gloss(W); losses.append(float(l))
        if not np.isfinite(float(l)): losses.append(float('nan')); break
        ins=accts(W)
        buf=[beta*b+(1-beta)*gi for b,gi in zip(buf,g)]
        if method=="act":
            Sig=[beta*S+(1-beta)*(a.T@a/a.shape[0]) for S,a in zip(Sig,ins)]
            D=[actaware_D(b,S,damping) for b,S in zip(buf,Sig)]
        else:
            D=[muon_D(b) for b in buf]
        W=[w-lr*d for w,d in zip(W,D)]
    return losses
print("method  lr      final_loss  min_loss  diverged")
for method in ["muon","act"]:
    for lr in [0.003,0.01,0.03,0.1]:
        ls=run(method,lr)
        fin=ls[-1]; mn=min([x for x in ls if np.isfinite(x)], default=float('nan'))
        div = (not np.isfinite(fin)) or fin>ls[0]*2
        print(f"{method:5s}  {lr:<6g}  {fin:.4f}     {mn:.4f}    {div}")

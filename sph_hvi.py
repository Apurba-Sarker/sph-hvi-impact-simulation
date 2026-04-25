"""
2D SPH Solver — High-Velocity Impact (HVI) of Aluminum Sphere on Metal Plates
AOE 5404 Course Project — Mashaekh Tausif Ehsan & Apurba Sarker

Physics:
  Wendland C2 kernel (C = 7/4πh², support 2h)   [Dehnen & Aly 2012]
  Mie-Gruneisen EOS                               [Kim et al. 2025 Eq.27]
  Johnson-Cook plasticity + Jaumann update        [Kim et al. 2025 Eq.13,20]
  Monaghan (1992) artificial viscosity            [Hiermaier 1997 Eq.12]
  Libersky-symmetric continuity                   [Hiermaier 1997 Eq.8]
  Monaghan-symmetric momentum & energy            [Hiermaier 1997 Eq.9-10]
  Leapfrog time integration
  Variable h: h0=1.2dx, dh/dt=(h/2)div_v        [Kim et al. 2025 Sec.3.1]

Benchmarks:
  Al-Al: 1cm Al sphere @ 6.18 km/s on 4mm Al plate  -> crater 3.1cm, l/w 1.39
  Al-Cu: 1cm Al sphere @ 5.75 km/s on 1.5mm Cu plate -> crater 2.12cm, l/w 1.39
"""

import numpy as np
import time, os, pickle

# ─── Material database  (Kim et al. 2025, Table 2) ────────────────────────
MATERIALS = {
    "Al": dict(
        rho0=2710., c0=5300., G=27.6e9,
        S=1.5,      Gamma=1.7,
        A=175e6,    B=380e6,  C=0.0015, n=0.34, m=1.0,
        T_room=273., T_melt=775., Cv=900.),
    "Cu": dict(
        rho0=8900., c0=3940., G=44.7e9,
        S=1.489,    Gamma=2.02,
        A=90e6,     B=292e6,  C=0.025,  n=0.31, m=1.09,
        T_room=293., T_melt=1356., Cv=385.),
}

KAPPA = 2.0   # kernel support = kappa * h

# ─── Wendland C2 kernel ───────────────────────────────────────────────────
# Normalization verified analytically: C * 2pi * int_0^{kh} W(r) r dr = 1
# => C = 7 / (4 pi h^2)  with support kh = 2h

def W(r, h):
    q = r / (KAPPA * h)
    C = 7.0 / (4.0 * np.pi * h**2)
    w = C * (1.0 - q)**4 * (1.0 + 4.0 * q)
    w[q >= 1.0] = 0.0
    return w

def dWdr(r, h):
    q  = r / (KAPPA * h)
    C  = 7.0 / (4.0 * np.pi * h**2)
    dw = -20.0 * C * q * (1.0 - q)**3 / (KAPPA * h)
    dw[q >= 1.0] = 0.0
    return dw

# ─── Mie-Gruneisen EOS ───────────────────────────────────────────────────
def eos_P(rho, e, mat):
    rho0, c0, S, Gam = mat["rho0"], mat["c0"], mat["S"], mat["Gamma"]
    mu    = rho / rho0 - 1.0
    denom = np.maximum((1.0 - S * mu)**2, 1e-30)
    ph    = rho0 * c0**2 * mu * (1.0 + (1.0 - Gam/2.0)*mu) / denom
    P     = ph + Gam * rho * e
    # No tension beyond 10% expansion
    # Tension cutoff: spall strength ~1 GPa for metals
    # Prevents tensile instability in thin plates (Swegle et al. 1994)
    P     = np.maximum(P, -1.0e9)
    return P

def eos_c(rho, mat):
    rho0, c0, S, Gam = mat["rho0"], mat["c0"], mat["S"], mat["Gamma"]
    mu  = rho / rho0 - 1.0
    num = c0**2 * (1.0 + (2.0*S - Gam)*mu)
    den = np.maximum((1.0 - S*mu)**2, 1e-30)
    return np.sqrt(np.maximum(num/den, c0**2*0.01))

# ─── Johnson-Cook yield ───────────────────────────────────────────────────
def jc_yield(ep, edot, T, mat):
    A,B,C   = mat["A"],  mat["B"],  mat["C"]
    n,m     = mat["n"],  mat["m"]
    Tr, Tm  = mat["T_room"], mat["T_melt"]
    fh = A + B * np.power(np.maximum(ep, 0.0), n)
    fr = 1.0 + C * np.log(np.maximum(edot, 1.0))
    Th = np.clip((T - Tr)/(Tm - Tr), 0.0, 0.9999)
    ft = 1.0 - Th**m
    return fh * fr * ft

# ─── Neighbor search (vectorized O(N²)) ──────────────────────────────────
def find_neighbors(x, h):
    """All pairs (i<j) within kappa*h_mean. Returns i,j,r,dxij."""
    diff   = x[:, None, :] - x[None, :, :]           # (N,N,2)
    dist2  = (diff**2).sum(axis=2)                    # (N,N)
    h_sym  = 0.5*(h[:,None] + h[None,:])
    mask   = np.triu((dist2 > 0) & (dist2 < (KAPPA*h_sym)**2), k=1)
    ii, jj = np.where(mask)
    r      = np.sqrt(dist2[ii, jj])
    dxij   = diff[ii, jj]          # xi - xj
    return ii.astype(np.int32), jj.astype(np.int32), r, dxij


# ─── Kernel-summation density ─────────────────────────────────────────────────
def summation_density(x, h, m, mid):
    """rho_i = sum_j m_j W(r_ij, h_i) -- stable, no drift."""
    N = len(x)
    rho = m * W(np.zeros(N), h)          # self-contribution
    ii, jj, r_arr, _ = find_neighbors(x, h)
    np.add.at(rho, ii, m[jj] * W(r_arr, h[ii]))
    np.add.at(rho, jj, m[ii] * W(r_arr, h[jj]))
    return rho

# ─── Per-particle EOS fields ──────────────────────────────────────────────
def compute_eos(rho, e, mid, mats):
    N  = len(rho)
    P  = np.zeros(N); cs = np.zeros(N); G = np.zeros(N)
    for k in np.unique(mid):
        idx = mid == k
        mat = mats[k]
        P[idx]  = eos_P(rho[idx], e[idx], mat)
        cs[idx] = eos_c(rho[idx], mat)
        G[idx]  = mat["G"]
    return P, cs, G

# ─── SPH derivatives (all physics in one pass) ───────────────────────────
def derivatives(x, v, rho, e, h, m, Sxx, Syy, Sxy, ep, T, mid, mats):
    N  = len(x)
    P, cs, G = compute_eos(rho, e, mid, mats)

    ii, jj, r_arr, dxij = find_neighbors(x, h)

    drho = np.zeros(N)
    dv   = np.zeros((N,2))
    de   = np.zeros(N)
    Lxx  = np.zeros(N); Lxy = np.zeros(N)
    Lyx  = np.zeros(N); Lyy = np.zeros(N)

    for k in range(len(ii)):
        i = ii[k];  j = jj[k]
        r  = r_arr[k];  ex = dxij[k]   # xi - xj
        if r < 1e-15: continue

        hm    = 0.5*(h[i] + h[j])
        dWr   = dWdr(np.array([r]), np.array([hm]))[0]
        gradW = dWr * ex / r            # grad_i W_ij  (points i->i direction, i.e. away from j)

        vij   = v[i] - v[j]
        vdotx = vij[0]*ex[0] + vij[1]*ex[1]

        # ── Artificial viscosity (Monaghan 1992) ──────────────────────
        Pi = 0.0
        if vdotx < 0.0:
            alpha = 1.5;  beta = 1.5
            cm    = 0.5*(cs[i]+cs[j])
            rhom  = 0.5*(rho[i]+rho[j])
            eps   = 0.01*hm
            mu    = hm * vdotx / (r**2 + eps**2)
            Pi    = (-alpha*cm*mu + beta*mu**2) / rhom

        # ── Continuity (Libersky, Hiermaier Eq.8) ────────────────────
        # drho_i = rho_i * sum_j (m_j/rho_j)(vi-vj).∇_i W_ij
        # By symmetry: ∇_j W_ji = -∇_i W_ij, (vj-vi) = -(vi-vj)
        # => drho_j += rho_j*(m_i/rho_i)*(vi-vj).∇_i W_ij  (same sign)
        cont = vij[0]*gradW[0] + vij[1]*gradW[1]
        drho[i] += rho[i] * (m[j]/rho[j]) * cont
        drho[j] += rho[j] * (m[i]/rho[i]) * cont

        # ── Momentum (Monaghan symmetric, Hiermaier Eq.9) ────────────
        # dv_i = -sum_j m_j (sig_i/rho_i^2 + sig_j/rho_j^2 + Pi_ij) ∇W
        fi = 1.0/rho[i]**2;  fj = 1.0/rho[j]**2
        # total stress sigma = S - P*I
        sxx_i = Sxx[i] - P[i];  syy_i = Syy[i] - P[i];  sxy_i = Sxy[i]
        sxx_j = Sxx[j] - P[j];  syy_j = Syy[j] - P[j];  sxy_j = Sxy[j]

        Axx = sxx_i*fi + sxx_j*fj - Pi   # Pi repulsive: adds to +P, subtracts from sigma
        Ayy = syy_i*fi + syy_j*fj - Pi
        Axy = sxy_i*fi + sxy_j*fj

        fx = Axx*gradW[0] + Axy*gradW[1]
        fy = Axy*gradW[0] + Ayy*gradW[1]

        dv[i,0] += m[j]*fx;  dv[i,1] += m[j]*fy
        dv[j,0] -= m[i]*fx;  dv[j,1] -= m[i]*fy   # Newton III

        # ── Energy (Hiermaier Eq.10, asymmetric per particle) ────────
        # de_i/dt = (sigma_i_ab/rho_i^2) * sum_j m_j * vij_a * gradW_b  + Pi term
        # de_j/dt = (sigma_j_ab/rho_j^2) * sum_i m_i * vij_a * gradW_b  + Pi term
        # Accumulating separately ensures sum_k m_k*de_k = -dKE/dt (energy conservation).
        # Using sigma_i only for de[i], sigma_j only for de[j], Pi split equally.
        Bxx_i = sxx_i*fi + 0.5*Pi;  Byy_i = syy_i*fi + 0.5*Pi;  Bxy_i = sxy_i*fi
        Bxx_j = sxx_j*fj + 0.5*Pi;  Byy_j = syy_j*fj + 0.5*Pi;  Bxy_j = sxy_j*fj
        dei = (Bxx_i*vij[0]*gradW[0] + Bxy_i*vij[0]*gradW[1] +
               Bxy_i*vij[1]*gradW[0] + Byy_i*vij[1]*gradW[1])
        dej = (Bxx_j*vij[0]*gradW[0] + Bxy_j*vij[0]*gradW[1] +
               Bxy_j*vij[1]*gradW[0] + Byy_j*vij[1]*gradW[1])
        de[i] += m[j]*dei
        de[j] += m[i]*dej

        # ── Velocity gradient (for Jaumann) ──────────────────────────
        # (dv_a/dx_b)_i = sum_j (m_j/rho_j)(v_i-v_j)_a (gradW)_b
        fij = m[j]/rho[j];  fji = m[i]/rho[i]
        Lxx[i] += fij*vij[0]*gradW[0];  Lxy[i] += fij*vij[0]*gradW[1]
        Lyx[i] += fij*vij[1]*gradW[0];  Lyy[i] += fij*vij[1]*gradW[1]
        # for j: (vj-vi) = -vij, gradW_ji = -gradW_ij → product is same
        Lxx[j] += fji*vij[0]*gradW[0];  Lxy[j] += fji*vij[0]*gradW[1]
        Lyx[j] += fji*vij[1]*gradW[0];  Lyy[j] += fji*vij[1]*gradW[1]

    # ── Jaumann deviatoric stress rate ────────────────────────────────────
    exx = Lxx;  eyy = Lyy
    exy = 0.5*(Lxy + Lyx)
    Omg = 0.5*(Lxy - Lyx)    # rotation rate Omega_12

    etr  = exx + eyy
    dSxx = 2.0*G*(exx - etr/3.0) + 2.0*Sxy*Omg
    dSyy = 2.0*G*(eyy - etr/3.0) - 2.0*Sxy*Omg
    dSxy = 2.0*G*exy + (Sxx - Syy)*Omg

    # ── Plasticity ────────────────────────────────────────────────────────
    edot_eff = np.sqrt(2.0/3.0*(exx**2 + eyy**2 + 2.0*exy**2) + 1e-30)
    svm      = np.sqrt(np.maximum(Sxx**2 - Sxx*Syy + Syy**2 + 3.0*Sxy**2, 0.0)) + 1e-30
    dep      = np.zeros(N);  dT_pl = np.zeros(N)
    for k in np.unique(mid):
        idx = mid == k
        mat = mats[k]
        sy      = jc_yield(ep[idx], edot_eff[idx], T[idx], mat)
        excess  = np.maximum(svm[idx] - sy, 0.0)
        dep[idx]   = excess / (3.0*mat["G"] + 1e-30)
        dT_pl[idx] = svm[idx]*dep[idx] / (rho[idx]*mat["Cv"] + 1e-30)

    return drho, dv, de, dSxx, dSyy, dSxy, dep, dT_pl, P, cs

# ─── Radial return ────────────────────────────────────────────────────────
def radial_return(Sxx, Syy, Sxy, sy):
    svm   = np.sqrt(np.maximum(Sxx**2 - Sxx*Syy + Syy**2 + 3.0*Sxy**2, 0.0)) + 1e-30
    scale = np.where(svm > sy, sy/svm, 1.0)
    return Sxx*scale, Syy*scale, Sxy*scale

# ─── CFL timestep ────────────────────────────────────────────────────────
def cfl_dt(h, v, cs, C_cfl=0.3):
    vsig = cs + np.sqrt((v**2).sum(axis=1))
    dt   = C_cfl * np.min(h / (vsig + 1e-10))
    return float(np.clip(dt, 1e-12, 1e-8))   # max 10 ns (Kim et al.)

# ─── Leapfrog step ───────────────────────────────────────────────────────
def step(x,v,rho,e,h,m,Sxx,Syy,Sxy,ep,T,mid,mats,dt):
    drho,dv,de,dSxx,dSyy,dSxy,dep,dT,P,cs = derivatives(
        x,v,rho,e,h,m,Sxx,Syy,Sxy,ep,T,mid,mats)

    # --- density (continuity ODE with robust floor) ─────────────────────
    rho = rho + drho*dt
    for k in np.unique(mid):
        idx = mid==k
        rho[idx] = np.maximum(rho[idx], mats[k]["rho0"]*0.05)  # 5% floor

    # --- velocity ---
    v = v + dv*dt

    # --- internal energy ---
    e = np.maximum(e + de*dt, 0.0)

    # --- deviatoric stress (Jaumann) ---
    Sxx = Sxx + dSxx*dt
    Syy = Syy + dSyy*dt
    Sxy = Sxy + dSxy*dt

    # radial return
    for k in np.unique(mid):
        idx = mid==k; mat = mats[k]
        sy            = jc_yield(ep[idx], dep[idx], T[idx], mat)
        Sxx[idx], Syy[idx], Sxy[idx] = radial_return(Sxx[idx],Syy[idx],Sxy[idx],sy)

    ep = ep + dep*dt
    T  = T  + dT*dt

    # --- position ---
    x  = x + v*dt

    # --- smoothing length ---
    div_v = -drho / np.maximum(rho, 1.0)
    h_new = h + (h/4.0)*div_v*dt   # gentler factor (0.25 vs 0.5) reduces tensile drift
    h_new = np.clip(h_new, h*0.8, h*3.0)    # tight: prevent free-surface h blowup
    h_new = np.maximum(h_new, 1e-5)

    KE = 0.5*np.sum(m*(v**2).sum(axis=1))
    IE = np.sum(m*e)
    return x,v,rho,e,h_new,Sxx,Syy,Sxy,ep,T, KE,IE,P,cs

# ─── Initial conditions ──────────────────────────────────────────────────
def init(dx, R, t_plate, w_plate, v_imp, sph_mat, plt_mat):
    mats = [MATERIALS[sph_mat], MATERIALS[plt_mat]]
    h0   = 1.2*dx

    xs, vs, rhos, hs, ms, mids = [], [], [], [], [], []

    # Sphere centre at (0, t_plate + R + 0.5*dx).
    # The 0.5*dx gap prevents coincident sphere-bottom / plate-top particles.
    rho_s = mats[0]["rho0"]
    cy    = t_plate + R + 0.5*dx
    Nr    = int(np.ceil(R/dx)) + 2
    for ix in range(-Nr, Nr+1):
        for iy in range(-Nr, Nr+1):
            px = ix*dx;  py = cy + iy*dx
            if px**2 + (py-cy)**2 <= R**2:
                xs.append([px, py]); vs.append([0., -v_imp])
                rhos.append(rho_s);  hs.append(h0)
                ms.append(rho_s*dx**2); mids.append(0)

    # Plate: rectangle x in [-w/2,w/2], y in [0,t_plate]
    rho_p = mats[1]["rho0"]
    nxp = int(np.round(w_plate/dx))+1
    nyp = int(np.round(t_plate/dx))+1
    for ix in range(nxp):
        for iy in range(nyp):
            xs.append([-w_plate/2 + ix*dx, iy*dx])
            vs.append([0., 0.])
            rhos.append(rho_p); hs.append(h0)
            ms.append(rho_p*dx**2); mids.append(1)

    N   = len(xs)
    mid = np.array(mids, dtype=np.int32)
    T0  = np.array([mats[m]["T_room"] for m in mids])
    return (np.array(xs), np.array(vs), np.array(rhos), np.zeros(N),
            np.array(hs), np.array(ms), np.zeros(N), np.zeros(N), np.zeros(N),
            np.zeros(N), T0, mid, mats, N)

# ─── Output metrics ──────────────────────────────────────────────────────
def measure(x, rho, mid, mats, t_plate_ref=None):
    """
    Crater diameter: x-span of LOW-DENSITY plate particles still near the plate
    (|y| < 1.5*t_plate) to avoid counting free-flying debris as crater width.
    Falls back to density threshold if nothing has exited yet.

    Debris cloud aspect ratio: percentile-trimmed bounding box (5th–95th percentile
    in both x and y) of particles below the plate, to reject outlier particles
    that fly far and inflate the bounding box.
    """
    plt_mask = mid == 1
    rho0p    = mats[1]["rho0"]
    crater   = 0.0

    # Plate thickness estimate from initial config (fallback 4mm)
    t_plate = 4e-3 if t_plate_ref is None else t_plate_ref

    # Primary: plate particles below the plate bottom (y < 0) but still
    # within 1.5x plate thickness — these are the rim/ejecta near the crater,
    # not far-flying debris. Use density < 80% rho0 as damage indicator.
    near_plate = plt_mask & (x[:,1] < 0) & (x[:,1] > -1.5*t_plate)
    damaged    = near_plate & (rho < 0.80*rho0p)
    if damaged.any():
        crater = x[damaged,0].max() - x[damaged,0].min()
    else:
        # Fallback: density-depleted particles near the axis (pre-perforation)
        near_axis = plt_mask & (np.abs(x[:,0]) < 3e-3)
        dam_near  = near_axis & (rho < 0.60*rho0p)
        if dam_near.any():
            crater = 2.0 * np.abs(x[dam_near,0]).max()

    # Debris cloud: particles below y = -0.5mm
    # Use 5th–95th percentile bounding box to reject lone outliers
    below = x[:,1] < -5e-4
    if below.any():
        xb = x[below]
        if len(xb) >= 10:
            x_lo, x_hi = np.percentile(xb[:,0], [5, 95])
            y_lo, y_hi = np.percentile(xb[:,1], [5, 95])
        else:
            x_lo, x_hi = xb[:,0].min(), xb[:,0].max()
            y_lo, y_hi = xb[:,1].min(), xb[:,1].max()
        length = max(y_hi - y_lo, 1e-9)
        width  = max(x_hi - x_lo, 1e-9)
        aspect = length / width
    else:
        aspect = 0.0

    return crater, aspect

# ─── Case config ─────────────────────────────────────────────────────────
CASES = {
    "Al-Al": dict(R=5e-3, t_plate=4e-3, w_plate=0.10, v_imp=6180.,
                  sph_mat="Al", plt_mat="Al",
                  exp_crater=0.031, exp_aspect=1.39),
    "Al-Cu": dict(R=5e-3, t_plate=1.5e-3, w_plate=0.08, v_imp=5750.,
                  sph_mat="Al", plt_mat="Cu",
                  exp_crater=0.0212, exp_aspect=1.39),
}

# ─── Main runner ─────────────────────────────────────────────────────────
def run(case="Al-Al", dx_mm=1.0, t_end_us=20.0, snap_us=2.0,
        outdir="/home/apurba/sph-hvi-impact-simulation/sph_output", verbose=True):

    os.makedirs(outdir, exist_ok=True)
    cfg = CASES[case];  dx = dx_mm*1e-3

    (x,v,rho,e,h,m,Sxx,Syy,Sxy,ep,T,mid,mats,N) = init(
        dx, cfg["R"], cfg["t_plate"], cfg["w_plate"],
        cfg["v_imp"], cfg["sph_mat"], cfg["plt_mat"])

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  {case}  |  dx={dx_mm}mm  |  N={N}  "
              f"(sphere={np.sum(mid==0)}, plate={np.sum(mid==1)})")
        print(f"  Impact {cfg['v_imp']/1e3:.2f}km/s  "
              f"R={cfg['R']*100:.1f}cm  t={cfg['t_plate']*1e3:.1f}mm")
        print(f"{'═'*60}")

    t_end  = t_end_us*1e-6;  t_snap = snap_us*1e-6
    t = 0.0;  step_n = 0;  next_snap = 0.0
    t0 = time.perf_counter()

    hist = dict(t=[], crater=[], aspect=[], KE=[], IE=[], TE=[], snaps=[])

    # initial dt
    _,cs0,_ = compute_eos(rho,e,mid,mats)
    dt = cfl_dt(h,v,cs0)

    while t < t_end:
        if t >= next_snap - 1e-14:
            _,cs_s,_ = compute_eos(rho,e,mid,mats)
            _P       = np.zeros(N)
            for k in np.unique(mid):
                idx=mid==k; _P[idx]=eos_P(rho[idx],e[idx],mats[k])
            cd, ar = measure(x,rho,mid,mats, t_plate_ref=cfg["t_plate"])
            hist["t"].append(t*1e6)
            hist["crater"].append(cd*100); hist["aspect"].append(ar)
            hist["snaps"].append(dict(t=t*1e6, x=x.copy(), v=v.copy(),
                                      rho=rho.copy(), mid=mid.copy(), P=_P.copy()))
            if verbose:
                print(f"  t={t*1e6:6.2f}µs  dt={dt*1e9:.3f}ns  "
                      f"crater={cd*100:.3f}cm  l/w={ar:.3f}  "
                      f"Pmax={_P.max()/1e9:.2f}GPa")
            next_snap += t_snap

        (x,v,rho,e,h,Sxx,Syy,Sxy,ep,T, KE,IE,P,cs) = step(
            x,v,rho,e,h,m,Sxx,Syy,Sxy,ep,T,mid,mats,dt)

        hist["KE"].append(KE); hist["IE"].append(IE); hist["TE"].append(KE+IE)
        dt = cfl_dt(h,v,cs)
        dt = min(dt, max(next_snap-t, 1e-12))
        t += dt;  step_n += 1

    wall = time.perf_counter()-t0
    cd, ar = measure(x,rho,mid,mats, t_plate_ref=cfg["t_plate"])
    err_c = abs(cd-cfg["exp_crater"])/cfg["exp_crater"]*100
    err_a = abs(ar-cfg["exp_aspect"])/cfg["exp_aspect"]*100 if ar>0 else 999.
    te0   = hist["TE"][0] if hist["TE"] and hist["TE"][0]!=0 else 1e-30
    edrift= abs(hist["TE"][-1]-hist["TE"][0])/abs(te0)*100 if hist["TE"] else 0.

    res = dict(case=case, dx_mm=dx_mm, N=N, n_steps=step_n, wall_s=wall,
               crater_cm=cd*100, aspect=ar,
               exp_crater_cm=cfg["exp_crater"]*100, exp_aspect=cfg["exp_aspect"],
               err_crater_pct=err_c, err_aspect_pct=err_a, e_drift_pct=edrift,
               history=hist, cfg=cfg,
               # final state (for post-processing)
               x_final=x, v_final=v, rho_final=rho, mid_final=mid)

    if verbose:
        print(f"\n  ── Results ──────────────────────────────────")
        print(f"  Wall     : {wall:.1f}s ({wall/60:.2f}min)")
        print(f"  Steps    : {step_n}")
        print(f"  Crater   : {cd*100:.3f}cm  (exp {cfg['exp_crater']*100:.2f}cm,  err {err_c:.1f}%)")
        print(f"  Debris   : {ar:.3f}    (exp {cfg['exp_aspect']:.2f},         err {err_a:.1f}%)")
        print(f"  E drift  : {edrift:.2f}%")
        print(f"  ────────────────────────────────────────────")

    fname = os.path.join(outdir, f"result_{case.replace('-','_')}_dx{dx_mm}mm.pkl")
    with open(fname,"wb") as f: pickle.dump(res, f)
    if verbose: print(f"  Saved → {fname}")
    return res

# ─── Plotting ────────────────────────────────────────────────────────────
def plot(res, outdir="/home/apurba/sph-hvi-impact-simulation/sph_output"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case=res["case"]; dx_mm=res["dx_mm"]; hist=res["history"]; cfg=res["cfg"]
    snaps = hist["snaps"]
    n_s   = len(snaps)
    idx_s = np.linspace(0, n_s-1, min(5, n_s), dtype=int)

    fig = plt.figure(figsize=(20,10))
    fig.suptitle(f"SPH HVI — {case}  (dx={dx_mm}mm, N={res['N']})",
                 fontsize=14, fontweight="bold")

    # Row 1: particle snapshots
    for col, si in enumerate(idx_s):
        ax = fig.add_subplot(2, 5, col+1)
        sn = snaps[si]
        c  = np.where(sn["mid"]==0, 0.9, 0.15)
        ax.scatter(sn["x"][:,0]*100, sn["x"][:,1]*100,
                   c=c, cmap="plasma", s=1.5, rasterized=True, vmin=0, vmax=1)
        ax.set_title(f"t={sn['t']:.1f}µs", fontsize=9)
        ax.set_xlabel("x (cm)", fontsize=7); ax.set_ylabel("y (cm)", fontsize=7)
        ax.tick_params(labelsize=6); ax.set_aspect("equal")

    # Crater vs time
    ax = fig.add_subplot(2,5,6)
    ax.plot(hist["t"], hist["crater"], "b-o", ms=3, lw=1.5, label="SPH")
    ax.axhline(cfg["exp_crater"]*100, color="r", ls="--", lw=2,
               label=f"Exp {cfg['exp_crater']*100:.2f}cm")
    ax.set_xlabel("t (µs)"); ax.set_ylabel("Crater (cm)")
    ax.set_title("Crater diameter"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Aspect ratio vs time
    ax = fig.add_subplot(2,5,7)
    ax.plot(hist["t"], hist["aspect"], "g-o", ms=3, lw=1.5, label="SPH")
    ax.axhline(cfg["exp_aspect"], color="r", ls="--", lw=2,
               label=f"Exp {cfg['exp_aspect']:.2f}")
    ax.set_xlabel("t (µs)"); ax.set_ylabel("l/w")
    ax.set_title("Debris aspect ratio"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Energy conservation
    ax = fig.add_subplot(2,5,8)
    if hist["TE"]:
        te=np.array(hist["TE"]); ke=np.array(hist["KE"]); ie=np.array(hist["IE"])
        te0=te[0] if te[0]!=0 else 1.0; st=np.arange(len(te))
        ax.plot(st, te/te0, "k-", lw=1.5, label="Total")
        ax.plot(st, ke/te0, "b--", lw=1, label="KE")
        ax.plot(st, ie/te0, "r--", lw=1, label="IE")
        ax.set_xlabel("Step"); ax.set_ylabel("E/E₀")
        ax.set_title("Energy conservation"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Speed field
    ax = fig.add_subplot(2,5,9)
    sn = snaps[-1]
    spd = np.sqrt((sn["v"]**2).sum(axis=1))
    sc  = ax.scatter(sn["x"][:,0]*100, sn["x"][:,1]*100,
                     c=spd/1e3, s=1.5, cmap="hot", rasterized=True)
    plt.colorbar(sc, ax=ax, label="km/s")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title(f"Speed at t={sn['t']:.0f}µs"); ax.set_aspect("equal")

    # Summary table
    ax = fig.add_subplot(2,5,10); ax.axis("off")
    rows = [["Metric","SPH","Exp","Err"],
            ["Crater(cm)",f"{res['crater_cm']:.3f}",
             f"{res['exp_crater_cm']:.2f}",f"{res['err_crater_pct']:.1f}%"],
            ["Debris l/w",f"{res['aspect']:.3f}",
             f"{res['exp_aspect']:.2f}",f"{res['err_aspect_pct']:.1f}%"],
            ["E drift",f"{res['e_drift_pct']:.2f}%","<5%",""],
            ["N parts",f"{res['N']}","",""],
            ["Time",f"{res['wall_s']:.0f}s","",""]]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r,c),cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor("#1a4e8c"); cell.set_text_props(color="w",fontweight="bold")
    ax.set_title(f"Results — {case}", fontsize=9)

    plt.tight_layout()
    fname = os.path.join(outdir, f"sph_{case.replace('-','_')}_dx{dx_mm}mm.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot → {fname}")
    return fname

# ─── Convergence study ───────────────────────────────────────────────────
def convergence_study(case="Al-Al", outdir="sph_output"):
    """
    O4: Particle-spacing convergence study at dx = 1.0, 0.75, 0.5 mm.

    Metrics are evaluated at a FIXED early snapshot time (eval_us) — the
    physically meaningful window before free debris scatters and inflates
    the bounding-box measurements.  For Al-Al: t=4µs; for Al-Cu: t=2µs.
    These match the evaluation times used in make_plots.py.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_us = {"Al-Al": 6.0, "Al-Cu": 2.0}[case]
    spacings = [1.0, 0.75, 0.5]
    craters  = []   # crater diameter (cm) at eval_us
    aspects  = []   # debris l/w at eval_us
    Ns       = []   # particle counts
    walls    = []   # wall-clock times

    cfg = CASES[case]

    for dx in spacings:
        print(f"\n>>> Convergence {case}  dx={dx}mm")
        # Run only to eval_us + a little margin (saves time vs full 20µs)
        t_end = eval_us + 2.0
        res = run(case=case, dx_mm=dx, t_end_us=t_end, snap_us=0.5,
                  outdir=outdir, verbose=True)
        Ns.append(res["N"])
        walls.append(res["wall_s"])

        # Find snapshot closest to eval_us
        hist   = res["history"]
        t_arr  = np.array(hist["t"])
        idx    = int(np.argmin(np.abs(t_arr - eval_us)))
        craters.append(hist["crater"][idx])
        aspects.append(hist["aspect"][idx])
        print(f"  @ t={t_arr[idx]:.1f}µs  crater={hist['crater'][idx]:.3f}cm"
              f"  l/w={hist['aspect'][idx]:.3f}")

    dx_arr = np.array(spacings)

    # ── convergence plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"O4 Convergence Study — {case}  (metrics at t={eval_us:.0f}µs)",
                 fontsize=13, fontweight="bold")

    # Crater diameter
    ax = axes[0]
    ax.plot(dx_arr, craters, "bo-", ms=8, lw=2, label="SPH")
    ax.axhline(cfg["exp_crater"]*100, color="r", ls="--", lw=2,
               label=f"Exp {cfg['exp_crater']*100:.2f} cm")
    # empirical order on log-log
    if all(c > 0 for c in craters):
        p = np.polyfit(np.log(dx_arr), np.log(craters), 1)
        order_str = f"order ≈ {p[0]:.2f}"
        # overlay fit line
        dx_fit = np.linspace(dx_arr.min()*0.9, dx_arr.max()*1.1, 50)
        ax.plot(dx_fit, np.exp(np.polyval(p, np.log(dx_fit))),
                "b--", lw=1, alpha=0.5, label=order_str)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Δx (mm)"); ax.set_ylabel("Crater diameter (cm)")
    ax.set_title(f"Crater diameter\n({order_str if all(c>0 for c in craters) else 'N/A'})")
    ax.legend(fontsize=8); ax.grid(which="both", alpha=0.3)

    # Debris aspect ratio
    ax = axes[1]
    valid_a = [a for a in aspects if a > 0.01]
    valid_dx = [dx_arr[i] for i,a in enumerate(aspects) if a > 0.01]
    if valid_a:
        ax.plot(valid_dx, valid_a, "gs-", ms=8, lw=2, label="SPH")
        ax.axhline(cfg["exp_aspect"], color="r", ls="--", lw=2,
                   label=f"Exp {cfg['exp_aspect']:.2f}")
        if len(valid_a) >= 2:
            p2 = np.polyfit(np.log(valid_dx), np.log(valid_a), 1)
            dx_fit = np.linspace(min(valid_dx)*0.9, max(valid_dx)*1.1, 50)
            ax.plot(dx_fit, np.exp(np.polyval(p2, np.log(dx_fit))),
                    "g--", lw=1, alpha=0.5, label=f"order ≈ {p2[0]:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Δx (mm)"); ax.set_ylabel("Debris l/w")
    ax.set_title("Debris aspect ratio"); ax.legend(fontsize=8); ax.grid(which="both", alpha=0.3)

    # Wall time vs N
    ax = axes[2]
    ax.plot(Ns, walls, "rs-", ms=8, lw=2)
    for n, w, dx in zip(Ns, walls, spacings):
        ax.annotate(f"dx={dx}mm\n{w:.0f}s", xy=(n, w),
                    xytext=(n*1.03, w*1.05), fontsize=8)
    if len(Ns) >= 2:
        p3 = np.polyfit(np.log(Ns), np.log(walls), 1)
        ax.set_title(f"Wall time vs N  (slope ≈ {p3[0]:.2f})")
    else:
        ax.set_title("Wall time vs N")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N particles"); ax.set_ylabel("Wall time (s)")
    ax.grid(which="both", alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(outdir, f"convergence_{case.replace('-','_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  Convergence plot → {fname}")

    # summary table to stdout
    print(f"\n  {'dx':>6}  {'N':>6}  {'crater(cm)':>12}  {'l/w':>8}  {'wall(s)':>8}")
    for dx, n, c, a, w in zip(spacings, Ns, craters, aspects, walls):
        print(f"  {dx:>6.2f}  {n:>6}  {c:>12.3f}  {a:>8.3f}  {w:>8.1f}")

    return dict(spacings=spacings, Ns=Ns, craters=craters,
                aspects=aspects, walls=walls, eval_us=eval_us)

# ─── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case",   default="Al-Al",
                   choices=["Al-Al","Al-Cu","both","convergence-AlAl","convergence-AlCu","convergence-both"])
    p.add_argument("--dx",     default=1.0,   type=float)
    p.add_argument("--t_end",  default=20.0,  type=float)
    p.add_argument("--snap",   default=2.0,   type=float)
    p.add_argument("--outdir", default="sph_output")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.case == "convergence-AlAl":
        convergence_study("Al-Al", args.outdir)
    elif args.case == "convergence-AlCu":
        convergence_study("Al-Cu", args.outdir)
    elif args.case == "convergence-both":
        convergence_study("Al-Al", args.outdir)
        convergence_study("Al-Cu", args.outdir)
    elif args.case == "both":
        for c in ["Al-Al","Al-Cu"]:
            res = run(c, args.dx, args.t_end, args.snap, args.outdir)
            plot(res, args.outdir)
    else:
        res = run(args.case, args.dx, args.t_end, args.snap, args.outdir)
        plot(res, args.outdir)
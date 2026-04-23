"""
wind_profiles.py  (v4 — proper CLE15)

Six parametric tropical cyclone wind profile models.

Changes from v3:
  * `chavas2015()` now literally implements Chavas, Lin & Emanuel (2015):
      - Inner region: ER11 Eq. (36) = CLE15 Eq. (6) — closed form
      - Outer region: E04 Eq. (2)   = CLE15 Eq. (2) — numerical Riccati ODE
      - Merge:        shooting on r0 for tangency (CLE15 §2b)
      - Cd(V):        CLE15 Eq. (11), piecewise Donelan 2004
      - Default Wcool: 2 mm/s (CLE15 Fig. 5 climatology)
      - Default Ck/Cd: CLE15 Fig. 7 quadratic fit in Vmax
  * REMOVED: `_chavas_outer_wind` and `_estimate_r_out_from_r34`
    (ad-hoc closed form + fixed-point r0 solver from v3 — neither
    corresponded to any published equation).
  * `chavas2015()` no longer accepts `r_out` or `r34_mean` — r0 is
    SOLVED from (Vm, rm) as prescribed in CLE15 §2b.
  * Callers should REMOVE the `Chavas_clim` configuration from
    compare_profiles.py — with r0 now internally solved,
    the obs vs. clim distinction no longer exists.

All functions return V(r) in knots given:
  r     : radius array (nm)
  vmax  : maximum sustained wind (kt)
  rmax  : radius of maximum wind (nm)
  pc    : central pressure (hPa) — needed for Holland models
  penv  : environmental pressure (hPa) — default 1013
  lat   : latitude (degrees N) — for Coriolis

References:
  [1] Rankine vortex (modified) — e.g. Holland 1980 §1
  [2] Holland 1980,            MWR 108, 1212–1218
  [3] Holland et al. 2010,     MWR 138, 4393–4401
  [4] Willoughby et al. 2006,  MWR 134, 1102–1120
  [5] Emanuel 2004,            in Atmos. Turbulence & Mesoscale Meteorology
  [6] Emanuel & Rotunno 2011,  JAS 68, 2236–2249  (ER11, inner region)
  [7] Chavas, Lin & Emanuel 2015, JAS 72, 3647–3662  (CLE15, merged model)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ── Constants ──
RHO      = 1.15          # air density (kg/m^3)
E_EULER  = np.e
NM_TO_M  = 1852.0        # 1 nautical mile in meters
KT_TO_MS = 0.5144        # 1 knot in m/s
MS_TO_KT = 1.0 / KT_TO_MS
OMEGA    = 7.2921e-5     # Earth rotation rate (rad/s)


def coriolis(lat):
    """Coriolis parameter f (s^-1) at given latitude."""
    return 2.0 * OMEGA * np.sin(np.radians(np.abs(lat)))


# ===============================================================
# 1. Modified Rankine Vortex
# ===============================================================
def rankine(r, vmax, rmax, alpha=0.5, **kwargs):
    """
    Modified Rankine vortex.
    V(r) = Vmax * (r/Rmax)         for r <= Rmax
    V(r) = Vmax * (Rmax/r)^alpha   for r >  Rmax
    """
    r = np.asarray(r, dtype=float)
    v = np.where(
        r <= rmax,
        vmax * (r / rmax),
        vmax * (rmax / r) ** alpha,
    )
    v[r == 0] = 0.0
    return v


# ===============================================================
# 2. Holland 1980
# ===============================================================
def holland1980(r, vmax, rmax, pc, lat, penv=1013.0, **kwargs):
    """
    Holland (1980) gradient wind profile.

    B is diagnosed from Vmax and pressure deficit:
      B = (Vmax_ms)^2 * rho * e / (penv - pc)

    V(r) = sqrt( B/rho * (Rmax/r)^B * dp * exp(-(Rmax/r)^B)
                 + (r_m*f/2)^2 ) - r_m*f/2
    """
    r = np.asarray(r, dtype=float)
    f = coriolis(lat)
    dp = (penv - pc) * 100.0  # hPa -> Pa

    if dp <= 0:
        return np.zeros_like(r)

    vmax_ms = vmax * KT_TO_MS
    B = (vmax_ms ** 2 * RHO * E_EULER) / dp
    B = np.clip(B, 1.0, 2.5)

    rmax_m = rmax * NM_TO_M
    r_m = np.maximum(r * NM_TO_M, 1.0)
    rr = (rmax_m / r_m) ** B

    v_ms = np.sqrt(
        (B / RHO) * rr * dp * np.exp(-rr) + (r_m * f / 2) ** 2
    ) - r_m * f / 2

    v_ms = np.maximum(v_ms, 0.0)
    return v_ms * MS_TO_KT


# ===============================================================
# 3. Holland et al. 2010 (simplified: fixed x=0.5, no vt/dPdt)
# ===============================================================
def holland2010(r, vmax, rmax, pc, lat, penv=1013.0, **kwargs):
    """
    Holland et al. (2010) revised wind profile.

    Simplified here (translation speed and pressure-tendency terms
    omitted; outer exponent x held at 0.5). This is the H1980 radial
    form with an H2010-style intensity/latitude-dependent peakedness
    bs, NOT the full H2010 profile.

      bs = -4.4e-5*dp^2 + 0.01*dp - 0.014*|lat| + 1.0
      V(r) = Vmax * [ (Rmax/r)^bs * exp(1 - (Rmax/r)^bs) ]^0.5
    """
    r = np.asarray(r, dtype=float)
    dp = penv - pc
    if not np.isfinite(dp) or dp <= 0:
        return np.full_like(r, np.nan)

    bs = -4.4e-5 * dp**2 + 0.01 * dp - 0.014 * abs(lat) + 1.0
    bs = np.clip(bs, 0.5, 2.5)

    rr = rmax / np.maximum(r, 0.01)
    v_frac = (rr ** bs * np.exp(1.0 - rr ** bs)) ** 0.5
    v = vmax * v_frac
    v[r == 0] = 0.0
    return v


# ===============================================================
# 4. Willoughby et al. 2006
# ===============================================================
def _bellramp(xi):
    """Degree-9 polynomial ramp from W06 Eq. (A2d)."""
    xi = np.clip(xi, 0.0, 1.0)
    return 126*xi**5 - 420*xi**6 + 540*xi**7 - 315*xi**8 + 70*xi**9


def _find_R1(rmax_km, X_eff, n, transition_width_km=25.0):
    """Bisection for R1 so dV/dr=0 at Rmax. W06 Eq. (3)."""
    w_target = n * X_eff / (n * X_eff + rmax_km)
    xi_lo, xi_hi = 0.0, 1.0
    for _ in range(60):
        xi_mid = 0.5 * (xi_lo + xi_hi)
        if _bellramp(xi_mid) < w_target:
            xi_lo = xi_mid
        else:
            xi_hi = xi_mid
    xi_sol = 0.5 * (xi_lo + xi_hi)
    return max(rmax_km - xi_sol * transition_width_km, 0.0)


def willoughby2006(r, vmax, rmax, lat, **kwargs):
    """
    Willoughby, Darling & Rahn (2006) piecewise-continuous profile.
    Three regions joined by a degree-9 bellramp.

    W06 Eqs. (10a)-(10c) — Vmax in m/s (knots converted internally),
    lat in degrees:
      n  = 0.4067 + 0.0144*Vmax_ms - 0.0038*|lat|
      X1 = 317.1  - 2.026*Vmax_ms  + 1.915*|lat|   (km)
      A  = 0.0696 + 0.0049*Vmax_ms - 0.0064*|lat|
      X2 = 25 km (fixed)
    """
    r = np.asarray(r, dtype=float)
    vmax_ms = vmax * KT_TO_MS
    lat_abs = abs(lat)

    n = 0.4067 + 0.0144 * vmax_ms - 0.0038 * lat_abs
    n = np.clip(n, 0.2, 2.4)
    X1_km = max(317.1 - 2.026 * vmax_ms + 1.915 * lat_abs, 50.0)
    X2_km = 25.0
    A = np.clip(0.0696 + 0.0049 * vmax_ms - 0.0064 * lat_abs, 0.0, 1.0)

    rmax_km = rmax * NM_TO_M / 1000.0
    r_km    = r    * NM_TO_M / 1000.0

    tw_km  = 25.0
    X_eff  = (1.0 - A) * X1_km + A * X2_km
    R1_km  = _find_R1(rmax_km, X_eff, n, tw_km)
    R2_km  = R1_km + tw_km

    r_safe = np.maximum(r_km, 1e-6)
    v_inner = vmax * (r_safe / rmax_km) ** n

    dr = r_km - rmax_km
    v_outer = vmax * (
        (1.0 - A) * np.exp(-dr / X1_km) +
        A         * np.exp(-dr / X2_km)
    )

    xi = np.where(tw_km > 0, (r_km - R1_km) / tw_km, 0.0)
    xi = np.clip(xi, 0.0, 1.0)
    w  = _bellramp(xi)
    v_trans = v_inner * (1.0 - w) + v_outer * w

    v = np.where(r_km <= R1_km, v_inner,
        np.where(r_km <= R2_km, v_trans, v_outer))
    v[r == 0] = 0.0
    return np.maximum(v, 0.0)


# ===============================================================
# 5. Emanuel 2004 (simplified hyperbolic form)
# ===============================================================
def emanuel2004(r, vmax, rmax, lat, **kwargs):
    """
    Simplified hyperbolic outer wind (not the full E04 radiative-
    subsidence model, which requires thermodynamic inputs absent
    from best-track data):
        V(r) = Vmax * sqrt( 2*Rmax*r / (Rmax^2 + r^2) ),   r > Rmax
    Solid-body rotation inside Rmax. Asymptotic decay at large r
    is r^(-1/2) — same rate as Rankine(alpha=0.5), but with a
    sqrt(2) larger prefactor.
    """
    r = np.asarray(r, dtype=float)
    v_inner = vmax * (r / rmax)
    v_outer = vmax * np.sqrt(2.0 * rmax * r / (rmax**2 + r**2))
    v = np.where(r <= rmax, v_inner, v_outer)
    v[r == 0] = 0.0
    return v


# ===============================================================
# 6. Chavas, Lin & Emanuel (2015) — proper implementation
# ===============================================================
#
#   Inner region (r <= ra) : ER11 Eq. (36) / CLE15 Eq. (6).
#   Outer region (ra < r < r0) : E04 Eq. (2) / CLE15 Eq. (2),
#       integrated numerically (Riccati ODE, no closed form).
#   Merge (ra, Va) found by shooting on r0: enforce continuity
#       of M AND dM/dr at the crossing of the two curves.
#   Drag Cd(V) : CLE15 Eq. (11), piecewise Donelan 2004 fit.
#   Default Wcool = 2 mm/s (CLE15 Sec. 5b, Fig. 5 climatology).
#   Default Ck/Cd = CLE15 Fig. 7 quadratic fit in Vmax (m/s).
# ===============================================================

def _cd_donelan_scalar(V_ms):
    """CLE15 Eq. (11). Scalar form for use inside ODE rhs."""
    if V_ms <= 6.0:
        return 6.16e-4
    if V_ms >= 35.4:
        return 2.4e-3
    return 5.91e-5 * V_ms + 2.614e-4


def _ck_cd_fit(vmax_ms):
    """CLE15 Fig. 7 quadratic fit (Vmax in m/s)."""
    return 0.00055 * vmax_ms**2 - 0.0259 * vmax_ms + 0.763


def _er11_M(r_m, rm_m, vmax_ms, f, Ck_Cd):
    """
    ER11 Eq. (36) = CLE15 Eq. (6).

        (M/Mm)^(2-alpha) = 2*(r/rm)^2 / [(2-alpha) + alpha*(r/rm)^2]

    alpha = Ck/Cd; Mm = rm*Vm + 0.5*f*rm^2.
    """
    Mm      = rm_m * vmax_ms + 0.5 * f * rm_m**2
    rhat_sq = (r_m / rm_m)**2
    alpha   = Ck_Cd
    base    = 2.0 * rhat_sq / ((2.0 - alpha) + alpha * rhat_sq)
    return Mm * np.power(np.maximum(base, 0.0), 1.0 / (2.0 - alpha))


def _er11_dMdr(r_m, rm_m, vmax_ms, f, Ck_Cd):
    """
    Analytical derivative of ER11 Eq. (36):

        dM/dr = 4*Mm*r / (rm^2 * D^2) * base^((1-alpha)/(2-alpha))

    where D = (2-alpha) + alpha*(r/rm)^2 and
          base = 2*(r/rm)^2 / D.
    """
    Mm      = rm_m * vmax_ms + 0.5 * f * rm_m**2
    rhat_sq = (r_m / rm_m)**2
    alpha   = Ck_Cd
    D       = (2.0 - alpha) + alpha * rhat_sq
    base    = 2.0 * rhat_sq / D
    return (4.0 * Mm * r_m / rm_m**2) / D**2 * \
           np.power(np.maximum(base, 1e-30),
                    (1.0 - alpha) / (2.0 - alpha))


def _e04_rhs(r, M, r0_m, f, Wcool, chi_const):
    """
    E04 Eq. (2) = CLE15 Eq. (2).

        dM/dr = chi * (r*V)^2 / (r0^2 - r^2)
        V     = M/r - f*r/2
        chi   = 2*Cd/Wcool  (V-dependent Cd)  OR  a constant
    """
    M_scalar = float(M[0])
    V = M_scalar / r - 0.5 * f * r
    if V < 0.0:
        V = 0.0
    denom = r0_m**2 - r**2
    if denom <= 0.0:
        return [0.0]
    if chi_const is not None:
        chi = chi_const
    else:
        chi = 2.0 * _cd_donelan_scalar(V) / Wcool
    return [chi * (r * V)**2 / denom]


def _integrate_e04(r0_m, f, Wcool, r_min_m, chi_const=None):
    """Integrate E04 ODE from r0 (V=0) inward to r_min."""
    r_start = r0_m * (1.0 - 1e-6)
    M_start = 0.5 * f * r_start**2
    sol = solve_ivp(
        lambda r, M: _e04_rhs(r, M, r0_m, f, Wcool, chi_const),
        t_span=(r_start, r_min_m),
        y0=[M_start],
        method='RK45',
        rtol=1e-6,
        atol=10.0,
        dense_output=True,
        max_step=(r_start - r_min_m) / 50.0,
    )
    return sol


def _find_merge_at_r0(r0_m, rm_m, vmax_ms, f, Ck_Cd, Wcool, chi_const,
                      sol=None):
    """
    For given r0, locate the radius ra that minimises M_outer - M_inner.
    At the correct (tangent) r0 this minimum is exactly zero.
    Returns (ra, Ma, sol). (None, None, sol) if r_grid empty.
    """
    r_min = rm_m * 1.001
    if sol is None:
        sol = _integrate_e04(r0_m, f, Wcool, r_min, chi_const)
    # Dense log-spaced grid across the integration range
    r_grid = np.geomspace(r_min, r0_m * (1.0 - 1e-6), 1000)
    M_out  = sol.sol(r_grid)[0]
    M_in   = _er11_M(r_grid, rm_m, vmax_ms, f, Ck_Cd)
    delta  = M_out - M_in
    # Interior minimum (exclude endpoints where curves may coincide
    # trivially or be poorly defined)
    interior = slice(5, -5)
    i_loc = int(np.argmin(delta[interior])) + 5
    ra = float(r_grid[i_loc])
    Ma = float(sol.sol(ra)[0])
    return ra, Ma, sol


def _min_delta(r0_m, rm_m, vmax_ms, f, Ck_Cd, Wcool, chi_const):
    """
    Tangency residual: min_r (M_outer - M_inner).
    Sign convention:
      < 0  =>  outer curve dips below inner somewhere  =>  r0 too small
      = 0  =>  curves tangent                          =>  correct r0
      > 0  =>  outer curve always above inner          =>  r0 too large
    """
    r_min = rm_m * 1.001
    sol = _integrate_e04(r0_m, f, Wcool, r_min, chi_const)
    r_grid = np.geomspace(r_min, r0_m * (1.0 - 1e-6), 1000)
    M_out  = sol.sol(r_grid)[0]
    M_in   = _er11_M(r_grid, rm_m, vmax_ms, f, Ck_Cd)
    delta  = M_out - M_in
    interior = slice(5, -5)
    return float(np.min(delta[interior]))


def _solve_r0(rm_m, vmax_ms, f, Ck_Cd, Wcool, chi_const=None):
    """
    Shoot on r0 via Brent's method to achieve ER11/E04 tangency,
    i.e. min_r (M_outer - M_inner) = 0.
    """
    def g(r0):
        return _min_delta(r0, rm_m, vmax_ms, f,
                          Ck_Cd, Wcool, chi_const)

    lo, hi = rm_m * 3.0, rm_m * 50.0
    try:
        g_lo = g(lo)
        g_hi = g(hi)
        # Expand upper bound if bracket not yet found
        tries = 0
        while g_lo * g_hi > 0 and tries < 6:
            hi *= 2.0
            g_hi = g(hi)
            tries += 1
        if g_lo * g_hi > 0:
            return None
        return brentq(g, lo, hi, rtol=1e-4, maxiter=60)
    except (ValueError, RuntimeError):
        return None


def chavas2015(r, vmax, rmax, lat, Ck_Cd=None, Wcool=2e-3,
               chi_const=None, **kwargs):
    """
    Chavas, Lin & Emanuel (2015) merged profile, implemented per
    CLE15 Sec. 2.

    Inputs:
      r         : radial grid (nm)
      vmax      : max sustained wind (kt)
      rmax      : radius of max wind (nm)
      lat       : latitude (deg)
      Ck_Cd     : ratio of exchange coefficients; if None, uses CLE15
                  Fig. 7 quadratic fit in Vmax (m/s), clipped to [0.1, 2.0]
      Wcool     : radiative-subsidence rate (m/s); default 2e-3 (2 mm/s)
      chi_const : if given, overrides chi = 2*Cd/Wcool with this constant;
                  used for reproducing CLE15 Fig. 2 (chi_const=1)

    Algorithm:
      1. Compute ER11 Eq. (36) inner region from (Vm, rm, f, Ck/Cd).
      2. Shoot on r0: integrate E04 Eq. (2) inward until the two
         curves are tangent at some ra (CLE15 Sec. 2b).
      3. Merge: ER11 for r <= ra, E04 for ra < r < r0, zero for r >= r0.

    Returns:
      V(r) in knots. NaN everywhere on convergence failure.
    """
    r = np.asarray(r, dtype=float)
    f = coriolis(lat)
    if f < 1e-7:
        f = 1e-7  # near-equator safety

    vmax_ms = vmax * KT_TO_MS
    rmax_m  = rmax * NM_TO_M
    r_m     = r    * NM_TO_M

    if Ck_Cd is None:
        Ck_Cd = float(np.clip(_ck_cd_fit(vmax_ms), 0.1, 2.0))

    # 1. Solve for r0
    r0_m = _solve_r0(rmax_m, vmax_ms, f, Ck_Cd, Wcool, chi_const)
    if r0_m is None:
        return np.full_like(r, np.nan)

    # 2. Get merge point and outer-region solution
    ra_m, Ma, sol_outer = _find_merge_at_r0(
        r0_m, rmax_m, vmax_ms, f, Ck_Cd, Wcool, chi_const
    )
    if ra_m is None:
        return np.full_like(r, np.nan)

    # 3. Assemble V(r)
    v_ms = np.zeros_like(r_m)

    # Inner: ER11 (0 < r <= ra)
    m_inner = (r_m > 0.0) & (r_m <= ra_m)
    if m_inner.any():
        M_in = _er11_M(r_m[m_inner], rmax_m, vmax_ms, f, Ck_Cd)
        V_in = M_in / r_m[m_inner] - 0.5 * f * r_m[m_inner]
        v_ms[m_inner] = np.maximum(V_in, 0.0)

    # Outer: E04 (ra < r < r0)
    m_outer = (r_m > ra_m) & (r_m < r0_m)
    if m_outer.any():
        M_out = sol_outer.sol(r_m[m_outer])[0]
        V_out = M_out / r_m[m_outer] - 0.5 * f * r_m[m_outer]
        v_ms[m_outer] = np.maximum(V_out, 0.0)

    # r >= r0: zero (already initialised)
    return v_ms * MS_TO_KT


# ===============================================================
# Registry
# ===============================================================
PROFILES = {
    'Rankine':        rankine,
    'Holland1980':    holland1980,
    'Holland2010':    holland2010,
    'Willoughby2006': willoughby2006,
    'Emanuel2004':    emanuel2004,
    'Chavas2015':     chavas2015,
}

REQUIRED = {
    'Rankine':        ['vmax', 'rmax'],
    'Holland1980':    ['vmax', 'rmax', 'pc', 'lat'],
    'Holland2010':    ['vmax', 'rmax', 'pc', 'lat'],
    'Willoughby2006': ['vmax', 'rmax', 'lat'],
    'Emanuel2004':    ['vmax', 'rmax'],
    'Chavas2015':     ['vmax', 'rmax', 'lat'],   # r0 is SOLVED, not input
}


# ===============================================================
# Self-test: reproduce CLE15 Fig. 2
# ===============================================================
if __name__ == '__main__':
    # CLE15 Fig. 2: (rm, Vm) = (30 km, 50 m/s), Ck/Cd = 1, chi = 1.
    # Expected: r0 ~ 847 km, merge at (ra, Va) ~ (79.1 km, 31.7 m/s).
    rm_km = 30.0
    Vm_ms = 50.0
    lat   = 20.0
    f_val = coriolis(lat)

    r0_m = _solve_r0(rm_km * 1000.0, Vm_ms, f_val,
                     Ck_Cd=1.0, Wcool=2e-3, chi_const=1.0)
    ra_m, Ma, _ = _find_merge_at_r0(
        r0_m, rm_km * 1000.0, Vm_ms, f_val,
        Ck_Cd=1.0, Wcool=2e-3, chi_const=1.0,
    )
    Va_ms = Ma / ra_m - 0.5 * f_val * ra_m

    print("CLE15 Fig. 2 self-test")
    print(f"  r0 = {r0_m/1000:7.1f} km   (expected ~847)")
    print(f"  ra = {ra_m/1000:7.1f} km   (expected ~79.1)")
    print(f"  Va = {Va_ms:7.2f} m/s  (expected ~31.7)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(0)

    r_nm = np.linspace(0.0, 300.0, 601)
    params = dict(vmax=100.0, rmax=20.0, pc=940.0, lat=18.0, penv=1013.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, func in PROFILES.items():
        if name == 'Chavas2015':
            v = func(r_nm, vmax=params['vmax'], rmax=params['rmax'],
                     lat=params['lat'])
        else:
            v = func(r_nm, **params)
        ax.plot(r_nm, v, label=name, linewidth=1.5)

    ax.axhline(34, color='gray', ls='--', lw=0.8, label='34 kt (TS)')
    ax.axhline(64, color='gray', ls=':',  lw=0.8, label='64 kt (TY)')
    ax.axvline(params['rmax'], color='black', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Radius (nm)')
    ax.set_ylabel('Wind speed (kt)')
    ax.set_title(f"Parametric profiles: Vmax={params['vmax']} kt, "
                 f"Rmax={params['rmax']} nm, Pc={params['pc']} hPa, "
                 f"Lat={params['lat']} deg N")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('profile_sanity_check.png', dpi=150)
    print("Saved profile_sanity_check.png")

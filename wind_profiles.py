"""
wind_profiles.py
Six parametric tropical cyclone wind profile models.

All functions return V(r) in knots given:
  r     : radius array (nm)
  vmax  : maximum sustained wind (kt)
  rmax  : radius of maximum wind (nm)
  pc    : central pressure (hPa) — needed for Holland models
  penv  : environmental pressure (hPa) — default 1013
  lat   : latitude (degrees N) — for Coriolis
  
References:
  [1] Rankine vortex (modified) — e.g. Holland 1980 §1
  [2] Holland 1980, MWR 108, 1212–1218
  [3] Holland et al. 2010, MWR 138, 4393–4401
  [4] Willoughby et al. 2006, MWR 134, 1102–1120
  [5] Emanuel 2004, J. Atmos. Sci. 61, 1849–1858
  [6] Chavas et al. 2015, J. Atmos. Sci. 72, 3012–3028
"""

import numpy as np

# ── Constants ──
RHO = 1.15          # air density (kg/m³)
E_EULER = np.e
NM_TO_M = 1852.0    # 1 nautical mile in meters
KT_TO_MS = 0.5144   # 1 knot in m/s
MS_TO_KT = 1.0 / KT_TO_MS
OMEGA = 7.2921e-5   # Earth rotation rate (rad/s)


def coriolis(lat):
    """Coriolis parameter f (s⁻¹) at given latitude."""
    return 2 * OMEGA * np.sin(np.radians(np.abs(lat)))


# ═══════════════════════════════════════════════════════════════════
# 1. Modified Rankine Vortex
# ═══════════════════════════════════════════════════════════════════
def rankine(r, vmax, rmax, alpha=0.5, **kwargs):
    """
    Modified Rankine vortex.
    V(r) = Vmax * (r/Rmax)         for r <= Rmax
    V(r) = Vmax * (Rmax/r)^alpha   for r >  Rmax
    
    alpha = 0.5 is typical; 0.4-0.6 range.
    """
    r = np.asarray(r, dtype=float)
    v = np.where(
        r <= rmax,
        vmax * (r / rmax),
        vmax * (rmax / r) ** alpha
    )
    v[r == 0] = 0.0
    return v


# ═══════════════════════════════════════════════════════════════════
# 2. Holland 1980
# ═══════════════════════════════════════════════════════════════════
def holland1980(r, vmax, rmax, pc, lat, penv=1013.0, **kwargs):
    """
    Holland (1980) gradient wind profile.
    
    B is diagnosed from Vmax and pressure deficit:
      B = (Vmax_ms)^2 * rho * e / (penv - pc) / 100
    
    Gradient wind (then reduced to surface via 0.8 factor already
    implicit if Vmax is surface wind — here we assume inputs are
    surface-level, so no reduction applied).
    
    V(r) = sqrt( B/rho * (Rmax/r)^B * dp*100 * exp(-(Rmax/r)^B)
                 + (r_m*f/2)^2 ) - r_m*f/2
    
    where dp = penv - pc (hPa), and distances in meters.
    """
    r = np.asarray(r, dtype=float)
    f = coriolis(lat)
    dp = (penv - pc) * 100.0  # Pa
    
    if dp <= 0:
        return np.zeros_like(r)
    
    vmax_ms = vmax * KT_TO_MS
    
    # Diagnose B from Vmax (Holland 1980 eq. 8 rearranged)
    B = (vmax_ms ** 2 * RHO * E_EULER) / dp
    B = np.clip(B, 1.0, 2.5)  # physical bounds
    
    rmax_m = rmax * NM_TO_M
    r_m = r * NM_TO_M
    r_m = np.maximum(r_m, 1.0)  # avoid division by zero
    
    rr = (rmax_m / r_m) ** B
    
    v_ms = np.sqrt(
        (B / RHO) * rr * dp * np.exp(-rr) + (r_m * f / 2) ** 2
    ) - r_m * f / 2
    
    v_ms = np.maximum(v_ms, 0.0)
    return v_ms * MS_TO_KT


# ═══════════════════════════════════════════════════════════════════
# 3. Holland et al. 2010
# ═══════════════════════════════════════════════════════════════════
def holland2010(r, vmax, rmax, pc, lat, penv=1013.0, **kwargs):
    """
    Holland et al. (2010) revised wind profile.

    Peakedness parameter bs from H10 Eq. 8 (Δp in hPa):
      bs = -4.4e-5 * Δp^2 + 0.01*Δp + 0.03*dP/dt
           - 0.014*|lat| + 0.15*vt^x + 1.0

    Simplified here (translation speed and pressure tendency omitted,
    as they are unavailable across all snapshots):
      bs = -4.4e-5 * Δp^2 + 0.01*Δp - 0.014*|lat| + 1.0

    Wind profile (H10 Eq. 6):
      Vs(r) = Vmax * [ (Rmax/r)^bs * exp(1 - (Rmax/r)^bs) ]^0.5

    NOTE: this revision now requires pc. NaN pc => NaN profile.
    """
    r = np.asarray(r, dtype=float)

    # Pressure deficit (hPa)
    dp = penv - pc
    if not np.isfinite(dp) or dp <= 0:
        return np.full_like(r, np.nan)

    # Peakedness parameter (H10 Eq. 8, Δp-based)
    bs = -4.4e-5 * dp**2 + 0.01 * dp - 0.014 * abs(lat) + 1.0
    bs = np.clip(bs, 0.5, 2.5)

    rr = rmax / np.maximum(r, 0.01)  # Rmax/r

    v_frac = (rr ** bs * np.exp(1.0 - rr ** bs)) ** 0.5
    v = vmax * v_frac
    v[r == 0] = 0.0
    return v


# ═══════════════════════════════════════════════════════════════════
# 4. Willoughby et al. 2006
# ═══════════════════════════════════════════════════════════════════
def _bellramp(xi):
    """Fourth-order ramp w(xi), W06 Eq. (A2d). Degree-9 polynomial."""
    xi = np.clip(xi, 0.0, 1.0)
    return 126*xi**5 - 420*xi**6 + 540*xi**7 - 315*xi**8 + 70*xi**9


def _find_R1(rmax_km, X_eff, n, transition_width_km=25.0):
    """
    Locate R1 (inner edge of transition zone) so that dV/dr = 0 at Rmax.
    From W06 Eq. (3):  w((Rmax-R1)/(R2-R1)) = n*X_eff / (n*X_eff + Rmax)
    where X_eff = (1-A)*X1 + A*X2 for dual-exponential.
    Solved by bisection on the bellramp inverse.
    """
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

    Three regions joined by a degree-9 bellramp:
      r <= R1:        V = Vmax * (r/Rmax)^n          (inner power law)
      R1 < r <= R2:   smooth polynomial transition   (bellramp blend)
      r > R2:         dual-exponential outer decay    (W06 Eq. 4)

    Empirical relations from W06 Eqs. (10a)-(10c):
      n  = 0.4067 + 0.0144*Vmax_ms - 0.0038*|lat|
      X1 = 317.1  - 2.026*Vmax_ms  + 1.915*|lat|   (km, slow decay)
      A  = 0.0696 + 0.0049*Vmax_ms  - 0.0064*|lat|  (A >= 0)
      X2 = 25 km                                      (fast decay, fixed)

    Outer profile (W06 Eq. 4):
      Vo = Vmax * [(1-A)*exp(-(r-Rmax)/X1) + A*exp(-(r-Rmax)/X2)]
    """
    r = np.asarray(r, dtype=float)
    vmax_ms = vmax * KT_TO_MS
    lat_abs = abs(lat)

    # ── Empirical parameters — W06 Eqs. (10a)-(10c) ──
    n = 0.4067 + 0.0144 * vmax_ms - 0.0038 * lat_abs
    n = np.clip(n, 0.2, 2.4)

    X1_km = 317.1 - 2.026 * vmax_ms + 1.915 * lat_abs
    X1_km = max(X1_km, 50.0)

    X2_km = 25.0          # fixed fast decay length (W06 §4)

    A = 0.0696 + 0.0049 * vmax_ms - 0.0064 * lat_abs
    A = np.clip(A, 0.0, 1.0)

    # ── Convert to km ──
    rmax_km = rmax * NM_TO_M / 1000.0
    r_km    = r    * NM_TO_M / 1000.0

    # ── Transition zone (25 km width for dual-exponential, W06 §4) ──
    tw_km  = 25.0
    X_eff  = (1.0 - A) * X1_km + A * X2_km
    R1_km  = _find_R1(rmax_km, X_eff, n, tw_km)
    R2_km  = R1_km + tw_km

    # ── Inner profile: power law ──
    r_safe = np.maximum(r_km, 1e-6)
    v_inner = vmax * (r_safe / rmax_km) ** n

    # ── Outer profile: dual-exponential (W06 Eq. 4) ──
    dr = r_km - rmax_km
    v_outer = vmax * (
        (1.0 - A) * np.exp(-dr / X1_km) +
        A          * np.exp(-dr / X2_km)
    )

    # ── Bellramp blend in transition zone ──
    xi = np.where(tw_km > 0, (r_km - R1_km) / tw_km, 0.0)
    xi = np.clip(xi, 0.0, 1.0)
    w  = _bellramp(xi)
    v_trans = v_inner * (1.0 - w) + v_outer * w

    # ── Assemble ──
    v = np.where(r_km <= R1_km, v_inner,
        np.where(r_km <= R2_km, v_trans, v_outer))

    v[r == 0] = 0.0
    return np.maximum(v, 0.0)


# ═══════════════════════════════════════════════════════════════════
# 5. Emanuel 2004
# ═══════════════════════════════════════════════════════════════════
def emanuel2004(r, vmax, rmax, lat, r_out=300.0, **kwargs):
    """
    Emanuel (2004) angular-momentum-based profile.
    
    Inside Rmax: solid body rotation 
      V(r) = Vmax * (r / Rmax)
    
    Outside Rmax: conservation of angular momentum modified by
    surface drag, following E04 eq. (5):
      V(r) = Vmax * sqrt( 2*Rmax*r / (Rmax^2 + r^2) )
    
    This gives a physically-motivated outer wind decay that is
    slower than Rankine but faster than Holland at large radii.
    
    Note: This is the simplified "hyperbolic" profile from E04
    that does not require potential intensity or thermodynamic input.
    """
    r = np.asarray(r, dtype=float)
    
    # Inner: solid body
    v_inner = vmax * (r / rmax)
    
    # Outer: E04 hyperbolic profile
    v_outer = vmax * np.sqrt(2.0 * rmax * r / (rmax**2 + r**2))
    
    v = np.where(r <= rmax, v_inner, v_outer)
    v[r == 0] = 0.0
    return v


# ═══════════════════════════════════════════════════════════════════
# 6. Chavas et al. 2015
# ═══════════════════════════════════════════════════════════════════
def _chavas_outer_wind(r_m, r_out_m, f, Ck_Cd=1.0):
    """
    Chavas & Lin (2013) outer wind model (m/s).
    V_outer(r) = f*r_out/2 * sqrt( (r_out/r)^(2*Ck_Cd/(1+Ck_Cd)) - 1 )
    
    Following CLE15 eq. (4): the exponent on (r0/r) is 2*Ck/Cd / (1 + Ck/Cd).
    For Ck/Cd = 1 this gives exponent = 1.
    """
    exponent = 2.0 * Ck_Cd / (1.0 + Ck_Cd)
    ratio = np.maximum(r_out_m / r_m, 1.0) ** exponent
    return (f * r_out_m / 2.0) * np.sqrt(np.maximum(ratio - 1.0, 0.0))


def _estimate_r_out_from_r34(r34_nm, vmax, rmax, lat, Ck_Cd=1.0):
    """
    Invert the Chavas-Lin outer wind equation to find r_out such that
    V_outer(R34) = 34 kt.  Solved analytically:
    
    34 kt -> v34_ms
    v34_ms = f*r0/2 * sqrt( (r0/r34)^exp - 1 )
    => (r0/r34)^exp = 1 + (2*v34/f/r0)^2
    
    Since r0 appears on both sides, solve iteratively.
    Falls back to 10*Rmax if no convergence or bad inputs.
    """
    f = coriolis(lat)
    if f < 1e-7:
        f = 1e-7  # near-equator safety
    
    v34_ms = 34.0 * KT_TO_MS
    r34_m = r34_nm * NM_TO_M
    exponent = 2.0 * Ck_Cd / (1.0 + Ck_Cd)
    
    # Initial guess: r_out ~ 2.5 * R34
    r0 = 2.5 * r34_m
    
    for _ in range(50):
        lhs = 1.0 + (2.0 * v34_ms / (f * r0)) ** 2
        r0_new = r34_m * lhs ** (1.0 / exponent)
        if abs(r0_new - r0) < 100.0:  # converge within 100 m
            return r0_new / NM_TO_M
        r0 = 0.5 * (r0 + r0_new)  # damped update
    
    # Fallback
    return 10.0 * rmax


def chavas2015(r, vmax, rmax, lat, r_out=None, r34_mean=None, **kwargs):
    """
    Chavas, Lin & Emanuel (2015) merged profile.
    
    Inner region (r <= Rmax): E04 profile (solid body core + hyperbolic)
    Outer region (r > Rmax): Chavas & Lin (2013) boundary-layer model
    Merge at Rmax with continuity enforced by scaling the outer branch.
    
    r_out is estimated from observed mean R34 by inverting the CL13
    outer wind equation. Falls back to 10*Rmax if R34 not available.
    
    Ck/Cd = 1.0 (ratio of exchange coefficients).
    """
    r = np.asarray(r, dtype=float)
    f = coriolis(lat)
    if f < 1e-7:
        f = 1e-7
    
    Ck_Cd = 1.0
    
    # Estimate r_out from R34 if available
    if r_out is None:
        if r34_mean is not None and r34_mean > 0:
            r_out = _estimate_r_out_from_r34(r34_mean, vmax, rmax, lat, Ck_Cd)
        else:
            r_out = 10.0 * rmax
    
    r_out_m = r_out * NM_TO_M
    rmax_m = rmax * NM_TO_M
    r_m = np.maximum(r * NM_TO_M, 1.0)
    
    # Inner: solid body for r <= Rmax
    v_inner = vmax * (r / rmax)
    
    # Outer: Chavas-Lin model
    v_outer_ms = _chavas_outer_wind(r_m, r_out_m, f, Ck_Cd)
    v_outer = v_outer_ms * MS_TO_KT
    
    # Scale outer to match Vmax at Rmax (continuity)
    v_outer_at_rmax_ms = _chavas_outer_wind(
        np.array([rmax_m]), r_out_m, f, Ck_Cd
    )[0]
    v_outer_at_rmax = v_outer_at_rmax_ms * MS_TO_KT
    
    if v_outer_at_rmax > 0:
        scale = vmax / v_outer_at_rmax
    else:
        scale = 1.0
    v_outer_scaled = v_outer * scale
    
    # Assemble
    v = np.where(r <= rmax, v_inner, v_outer_scaled)
    v[r == 0] = 0.0
    v = np.maximum(v, 0.0)
    return v


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════
PROFILES = {
    'Rankine':      rankine,
    'Holland1980':  holland1980,
    'Holland2010':  holland2010,
    'Willoughby2006': willoughby2006,
    'Emanuel2004':  emanuel2004,
    'Chavas2015':   chavas2015,
}

# Required input fields for each model
REQUIRED = {
    'Rankine':        ['vmax', 'rmax'],
    'Holland1980':    ['vmax', 'rmax', 'pc', 'lat'],
    'Holland2010':    ['vmax', 'rmax', 'pc', 'lat'],
    'Willoughby2006': ['vmax', 'rmax', 'lat'],
    'Emanuel2004':    ['vmax', 'rmax', 'lat'],
    'Chavas2015':     ['vmax', 'rmax', 'lat', 'r34_mean'],
}


# ═══════════════════════════════════════════════════════════════════
# Quick sanity plot
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Typical WNP typhoon
    r = np.linspace(0, 300, 601)  # 0-300 nm
    params = dict(vmax=100, rmax=20, pc=940, lat=18.0, penv=1013.0, r34_mean=120)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, func in PROFILES.items():
        v = func(r, **params)
        ax.plot(r, v, label=name, linewidth=1.5)
    
    ax.axhline(34, color='gray', ls='--', lw=0.8, label='34 kt (TS)')
    ax.axhline(64, color='gray', ls=':', lw=0.8, label='64 kt (TY)')
    ax.axvline(params['rmax'], color='black', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Radius (nm)')
    ax.set_ylabel('Wind speed (kt)')
    ax.set_title(f"Parametric profiles: Vmax={params['vmax']} kt, Rmax={params['rmax']} nm, "
                 f"Pc={params['pc']} hPa, Lat={params['lat']}°N")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('profile_sanity_check.png', dpi=150)
    print("Saved profile_sanity_check.png")

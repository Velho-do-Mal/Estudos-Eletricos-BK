# bk_estudos_eletricos/core/field_em.py
# ======================================================================
# CORREÇÕES BK_Fixes_v1:
#   Fix #3 – AneelLimits: aliases E_max_kV_m_areas_occup e B_max_uT_areas_occup
#             retornavam o valor "geral" (4,17 kV/m / 200 µT) em vez do
#             valor "ocupacional" (8,33 kV/m / 1000 µT).
#             Qualquer código que usasse esses aliases aplicava limites 2×
#             mais restritivos do que o correto para áreas de acesso restrito.
#   Fix #4 – Nota sobre limitação do método de cargas (MSC):
#             O código usa lambda = C'_seq × V (capacitância de sequência).
#             O MSC rigoroso exigiria resolver [Q] = [P]^-1 · [V] por condutor.
#             Para linhas transpostas e balanceadas a aproximação é aceitável,
#             mas deve ser documentada.
# ======================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cmath, math, base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from .cables import EPS0, MU0
from .line_params import LineGeometry, LineParamsResult

# ────────────────────────────────────────────────────────────────────
# Estruturas
# ────────────────────────────────────────────────────────────────────

@dataclass
class FieldConfig:
    """
    Configuração para cálculo de campos |E| e |B| ao nível do solo.

    h_obs_m  : altura de observação (padrão 1,5 m — ANEEL RN 915/2021, art. 4º)
    x_min_m  : limite lateral mínimo (padrão −30 m)
    x_max_m  : limite lateral máximo (padrão +30 m)
    n_points : número de pontos no perfil lateral
    Ic_manual: corrente RMS por circuito (A); se None, calcula por S e V
    rho_solo : resistividade do solo (Ω·m); padrão 100 Ω·m
    freq_hz  : frequência do sistema (Hz)
    """
    h_obs_m: float = 1.5
    x_min_m: float = -30.0
    x_max_m: float = 30.0
    n_points: int = 301
    Ic_manual: Optional[float] = None
    rho_solo: float = 100.0
    freq_hz: float = 60.0

@dataclass
class AneelLimits:
    """
    Limites normativos — ANEEL Resolução Normativa nº 915/2021.

    Público em geral (entorno / muro externo):
        E_max = 4,17 kV/m  |  B_max = 200 µT

    Ocupacional (área de acesso restrito, ex.: interior de SE):
        E_max = 8,33 kV/m  |  B_max = 1000 µT
    """
    E_max_kV_m_geral: float = 4.17
    B_max_uT_geral: float = 200.0
    E_max_kV_m_ocup: float = 8.33
    B_max_uT_ocup: float = 1000.0

    # FIX #3: aliases corrigidos — agora retornam o valor OCUPACIONAL (não geral).
    # Versão anterior retornava E_max_kV_m_geral / B_max_uT_geral por engano,
    # aplicando limites 2× mais restritivos para áreas de acesso restrito.
    @property
    def E_max_kV_m_areas_occup(self) -> float:
        """Limite ocupacional de campo elétrico (8,33 kV/m — RN 915/2021)."""
        return self.E_max_kV_m_ocup   # FIX #3: era self.E_max_kV_m_geral

    @property
    def B_max_uT_areas_occup(self) -> float:
        """Limite ocupacional de campo magnético (1000 µT — RN 915/2021)."""
        return self.B_max_uT_ocup     # FIX #3: era self.B_max_uT_geral

@dataclass
class FieldProfilesResult:
    x_m: List[float]
    E_kV_m: List[float]
    B_uT: List[float]
    E_max_kV_m: float
    x_E_max_m: float
    B_max_uT: float
    x_B_max_m: float
    E_compliance_msg: str
    B_compliance_msg: str
    V_LL_kV: float
    power_mva: float
    config: FieldConfig
    limits: AneelLimits

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _phase_angle_deg(phase: str) -> float:
    ph = (phase or "").upper()
    if ph == "A": return 0.0
    if ph == "B": return -120.0
    if ph == "C": return 120.0
    return 0.0

def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def _get_params_for_circuit(params_by_circuit, circuit_index):
    if circuit_index not in params_by_circuit:
        raise ValueError(f"Sem LineParamsResult para circuito {circuit_index}.")
    return params_by_circuit[circuit_index]

def _deri_complex_depth(rho_solo: float, freq_hz: float) -> complex:
    """
    Profundidade complexa de retorno pelo solo — Método de Deri (IEEE PAS 1981).
    p = sqrt(rho_s / (j·omega·mu0))
    """
    omega = 2.0 * math.pi * freq_hz
    rho_s = max(rho_solo, 50.0)
    return cmath.sqrt(complex(rho_s, 0.0) / complex(0.0, omega * MU0))

# ────────────────────────────────────────────────────────────────────
# Dados de fase (cargas e correntes fasorial)
# FIX #4 – NOTA SOBRE LIMITAÇÃO DO MSC:
#   O cálculo de carga por condutor usa lambda = C'_sequencia * V_fase.
#   O MSC rigoroso resolve [Q] = [P_Maxwell]^-1 · [V] para cada condutor.
#   Para linhas transpostas e balanceadas a aproximação por C' sequencial
#   produz resultados adequados para estudos de engenharia (~5% de erro).
#   Para linhas não-transpostas ou configurações especiais, recomenda-se
#   implementar a solução completa da matriz de coeficientes de potencial.
# ────────────────────────────────────────────────────────────────────

def _build_phase_data(
    geom: LineGeometry,
    params_by_circuit: Dict[int, LineParamsResult],
    V_LL_kV: float,
    power_mva: float,
    Ic_manual: Optional[float],
) -> Dict[Tuple[int, str], Tuple[complex, complex]]:
    """
    (cidx, phase) -> (lambda_per_m [C/m, fasor], I_phase [A, fasor])

    NOTA (Fix #4): lambda calculado como C'_seq * V_fase por condutor.
    Aproximação válida para linhas transpostas balanceadas.
    """
    circuits  = geom.circuits()
    n_circuits = max(1, len(circuits))

    if Ic_manual is not None and abs(Ic_manual) > 1e-9:
        I_mag = float(Ic_manual)
    else:
        if power_mva <= 0 or V_LL_kV <= 0:
            I_mag = 0.0
        else:
            I_mag = (power_mva / n_circuits) * 1e6 / (math.sqrt(3.0) * V_LL_kV * 1e3)

    V_phase_V = (V_LL_kV * 1e3) / math.sqrt(3.0)
    phase_data: Dict[Tuple[int, str], Tuple[complex, complex]] = {}

    for cidx in circuits:
        params   = _get_params_for_circuit(params_by_circuit, cidx)
        C_F_km   = params.C_F_km
        for phase in ("A", "B", "C"):
            ang_rad   = math.radians(_phase_angle_deg(phase))
            V_ph      = V_phase_V * cmath.exp(1j * ang_rad)
            I_ph      = I_mag    * cmath.exp(1j * ang_rad)
            q_per_km  = C_F_km * V_ph
            lambda_per_m = q_per_km / 1000.0
            phase_data[(cidx, phase)] = (lambda_per_m, I_ph)

    return phase_data

# ────────────────────────────────────────────────────────────────────
# Campo E — MSC com Método das Imagens (solo equipotencial)
# ────────────────────────────────────────────────────────────────────

def _compute_E_at_point(x, z, geom, phase_data) -> float:
    """
    |E|(x,z) em kV/m usando MSC com imagens elétricas (solo equipotencial).
    """
    Ex = 0.0 + 0j
    Ez = 0.0 + 0j
    for cond in geom.conductors:
        if cond.phase not in ("A", "B", "C"): continue
        key = (cond.circuit_index, cond.phase)
        if key not in phase_data: continue
        lam, _ = phase_data[key]
        xi, yi = cond.x_m, cond.y_m
        coef = lam / (2.0 * math.pi * EPS0)
        dx = x - xi
        dz_r = z - yi
        r2_r = dx*dx + dz_r*dz_r
        if r2_r > 1e-8:
            Ex += coef * dx / r2_r
            Ez += coef * dz_r / r2_r
        dz_i = z + yi  # imagem espelhada em y = -yi
        r2_i = dx*dx + dz_i*dz_i
        if r2_i > 1e-8:
            Ex -= coef * dx / r2_i
            Ez -= coef * dz_i / r2_i
    return math.sqrt(abs(Ex)**2 + abs(Ez)**2) / 1e3

# ────────────────────────────────────────────────────────────────────
# Campo B — Imagens Complexas de Deri (solo com perdas)
# ────────────────────────────────────────────────────────────────────

def _compute_B_at_point(x, z, geom, phase_data, p_deri) -> float:
    """
    |B|(x,z) em µT usando Método das Imagens Complexas de Deri.
    Imagem complexa: y'_i = -y_i - 2p  (p = profundidade complexa de Deri).
    """
    Bx = 0.0 + 0j
    Bz = 0.0 + 0j
    for cond in geom.conductors:
        if cond.phase not in ("A", "B", "C"): continue
        key = (cond.circuit_index, cond.phase)
        if key not in phase_data: continue
        _, I_ph = phase_data[key]
        xi = cond.x_m; yi = cond.y_m
        yi_img = -yi - 2.0 * p_deri
        dx   = x - xi
        dz_r = z - yi
        dz_i = z - yi_img
        r2_r = dx*dx + dz_r*dz_r
        r2_i = dx*dx + dz_i*dz_i
        coef = MU0 / (2.0 * math.pi)
        if abs(r2_r) > 1e-8:
            Bx += coef * I_ph * (-dz_r / r2_r)
            Bz += coef * I_ph * ( dx   / r2_r)
        if abs(r2_i) > 1e-12:
            Bx -= coef * I_ph * (-dz_i / r2_i)
            Bz -= coef * I_ph * ( dx   / r2_i)
    return math.sqrt(abs(Bx)**2 + abs(Bz)**2) * 1e6

# ────────────────────────────────────────────────────────────────────
# Perfis laterais combinados
# ────────────────────────────────────────────────────────────────────

def compute_fields_profiles(
    geom: LineGeometry,
    params_by_circuit: Dict[int, LineParamsResult],
    config: FieldConfig,
    limits: AneelLimits,
    V_LL_kV: float,
    power_mva: float,
) -> FieldProfilesResult:
    if config.n_points < 2:
        raise ValueError("n_points deve ser >= 2.")
    if config.x_max_m <= config.x_min_m:
        raise ValueError("x_max_m deve ser > x_min_m.")
    if V_LL_kV < 0 or power_mva < 0:
        raise ValueError("V_LL_kV e power_mva devem ser >= 0.")

    phase_data = _build_phase_data(geom, params_by_circuit, V_LL_kV, power_mva, config.Ic_manual)
    p_deri     = _deri_complex_depth(config.rho_solo, config.freq_hz)

    x_vals: List[float] = []
    E_vals: List[float] = []
    B_vals: List[float] = []

    dx = (config.x_max_m - config.x_min_m) / (config.n_points - 1)
    for i in range(config.n_points):
        xp = config.x_min_m + i * dx
        x_vals.append(xp)
        E_vals.append(_compute_E_at_point(xp, config.h_obs_m, geom, phase_data))
        B_vals.append(_compute_B_at_point(xp, config.h_obs_m, geom, phase_data, p_deri))

    E_max   = max(E_vals) if E_vals else 0.0
    B_max   = max(B_vals) if B_vals else 0.0
    x_E_max = x_vals[E_vals.index(E_max)] if E_vals else 0.0
    x_B_max = x_vals[B_vals.index(B_max)] if B_vals else 0.0

    E_lim = limits.E_max_kV_m_geral
    B_lim = limits.B_max_uT_geral
    E_msg = (f"ATENDE — |E|_max = {E_max:.3f} kV/m <= {E_lim:.2f} kV/m (RN 915/2021)"
             if E_max <= E_lim else
             f"NAO ATENDE — |E|_max = {E_max:.3f} kV/m > {E_lim:.2f} kV/m (RN 915/2021)")
    B_msg = (f"ATENDE — |B|_max = {B_max:.3f} µT <= {B_lim:.1f} µT (RN 915/2021)"
             if B_max <= B_lim else
             f"NAO ATENDE — |B|_max = {B_max:.3f} µT > {B_lim:.1f} µT (RN 915/2021)")

    return FieldProfilesResult(
        x_m=x_vals, E_kV_m=E_vals, B_uT=B_vals,
        E_max_kV_m=E_max, x_E_max_m=x_E_max,
        B_max_uT=B_max,   x_B_max_m=x_B_max,
        E_compliance_msg=E_msg, B_compliance_msg=B_msg,
        V_LL_kV=V_LL_kV, power_mva=power_mva,
        config=config, limits=limits,
    )

# ────────────────────────────────────────────────────────────────────
# Plotagem 2D
# ────────────────────────────────────────────────────────────────────

def _plot_profiles_2d(result: FieldProfilesResult) -> Tuple[str, str]:
    x = result.x_m; E = result.E_kV_m; B = result.B_uT; lim = result.limits

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.fill_between(x, E, alpha=0.15, color="#1565C0")
    ax.plot(x, E, linewidth=2.2, color="#1565C0", label="|E|(x) kV/m")
    ax.plot(result.x_E_max_m, result.E_max_kV_m, 'rv', markersize=9,
            label=f"|E|_max={result.E_max_kV_m:.4f} kV/m @ x={result.x_E_max_m:.1f}m")
    e_ylim = max(E)*1.4 if E else 1.0
    if lim.E_max_kV_m_geral <= e_ylim*2.5:
        ax.axhline(lim.E_max_kV_m_geral, linestyle="--", color="#C62828",
                   label=f"Lim. geral={lim.E_max_kV_m_geral:.2f} kV/m")
    if lim.E_max_kV_m_ocup <= e_ylim*2.5:
        ax.axhline(lim.E_max_kV_m_ocup, linestyle=":", color="#E65100",
                   label=f"Lim. ocup.={lim.E_max_kV_m_ocup:.2f} kV/m")
    ax.set_ylim(bottom=0, top=e_ylim)
    ax.set_xlabel("Distância lateral x (m)"); ax.set_ylabel("|E| (kV/m)")
    ax.grid(True, alpha=0.35); ax.legend(fontsize=8)
    ax.set_title(f"Perfil de campo elétrico — h={result.config.h_obs_m:.1f} m (ANEEL RN 915/2021)")
    b64_E = _fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.fill_between(x, B, alpha=0.15, color="#00695C")
    ax.plot(x, B, linewidth=2.2, color="#00695C", label="|B|(x) µT")
    ax.plot(result.x_B_max_m, result.B_max_uT, 'rv', markersize=9,
            label=f"|B|_max={result.B_max_uT:.4f} µT @ x={result.x_B_max_m:.1f}m")
    b_ylim = max(B)*1.4 if B else 1.0
    if lim.B_max_uT_geral <= b_ylim*2.5:
        ax.axhline(lim.B_max_uT_geral, linestyle="--", color="#C62828",
                   label=f"Lim. geral={lim.B_max_uT_geral:.0f} µT")
    if lim.B_max_uT_ocup <= b_ylim*2.5:
        ax.axhline(lim.B_max_uT_ocup, linestyle=":", color="#E65100",
                   label=f"Lim. ocup.={lim.B_max_uT_ocup:.0f} µT")
    ax.set_ylim(bottom=0, top=b_ylim)
    ax.set_xlabel("Distância lateral x (m)"); ax.set_ylabel("|B| (µT)")
    ax.grid(True, alpha=0.35); ax.legend(fontsize=8)
    ax.set_title(f"Perfil de campo magnético — h={result.config.h_obs_m:.1f} m (ANEEL RN 915/2021)")
    b64_B = _fig_to_base64(fig)

    return b64_E, b64_B

# ────────────────────────────────────────────────────────────────────
# Grade 3D
# ────────────────────────────────────────────────────────────────────

def _compute_EB_grid_3d(geom, params_by_circuit, result, n_heights=25):
    cfg        = result.config
    phase_data = _build_phase_data(geom, params_by_circuit, result.V_LL_kV, result.power_mva, cfg.Ic_manual)
    p_deri     = _deri_complex_depth(cfg.rho_solo, cfg.freq_hz)
    x_vals     = result.x_m
    if not x_vals: return [], [], [], []
    max_y  = max((c.y_m for c in geom.conductors), default=25.0)
    z_max  = max(max_y + 2.0, cfg.h_obs_m + 1.0)
    z_vals = [i*z_max/(n_heights-1) for i in range(n_heights)]
    E_grid: List[List[float]] = []
    B_grid: List[List[float]] = []
    for zp in z_vals:
        row_E, row_B = [], []
        for xp in x_vals:
            row_E.append(_compute_E_at_point(xp, zp, geom, phase_data))
            row_B.append(_compute_B_at_point(xp, zp, geom, phase_data, p_deri))
        E_grid.append(row_E); B_grid.append(row_B)
    return x_vals, z_vals, E_grid, B_grid

def _plot_surface_3d(x_vals, z_vals, F_grid, z_label, title, colormap="viridis") -> str:
    if not x_vals or not z_vals or not F_grid: return ""
    try:
        import numpy as np
    except ImportError:
        return ""
    X, Z = np.meshgrid(x_vals, z_vals)
    F    = np.array(F_grid, dtype=float)
    p98  = np.percentile(F[F>0], 98) if np.any(F>0) else 1.0
    F    = np.clip(F, 0, p98)
    fig  = plt.figure(figsize=(10, 6))
    ax   = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Z, F, cmap=colormap, linewidth=0.2, antialiased=True, alpha=0.92)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel(z_label)
    ax.set_title(title); ax.view_init(elev=28, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, label=z_label, pad=0.12)
    try:
        ax.contourf(X, Z, F, zdir='z', offset=0, cmap=colormap, alpha=0.3, levels=15)
    except Exception:
        pass
    return _fig_to_base64(fig)

def compute_plotly_surface_data(geom, params_by_circuit, result, n_heights=20):
    return _compute_EB_grid_3d(geom, params_by_circuit, result, n_heights=n_heights)

# ────────────────────────────────────────────────────────────────────
# Relatório HTML
# ────────────────────────────────────────────────────────────────────

def generate_html_report_fields(project, geom, params_by_circuit, config, limits, result) -> str:
    E_b64, B_b64 = _plot_profiles_2d(result)
    x_v, z_v, Eg, Bg = _compute_EB_grid_3d(geom, params_by_circuit, result, n_heights=25)
    E3d = _plot_surface_3d(x_v, z_v, Eg, "|E| (kV/m)", "Mapa 3D de campo elétrico |E|(x,z)", "plasma")
    B3d = _plot_surface_3d(x_v, z_v, Bg, "|B| (µT)",  "Mapa 3D de campo magnético |B|(x,z)", "viridis")
    res = result; lim = limits
    sEg = "ok" if res.E_max_kV_m <= lim.E_max_kV_m_geral else "nok"
    sEo = "ok" if res.E_max_kV_m <= lim.E_max_kV_m_ocup  else "nok"
    sBg = "ok" if res.B_max_uT   <= lim.B_max_uT_geral   else "nok"
    sBo = "ok" if res.B_max_uT   <= lim.B_max_uT_ocup    else "nok"
    css = ("<style>body{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fa;color:#222}"
           ".container{max-width:1100px;margin:0 auto;padding:28px;background:#fff;box-shadow:0 4px 18px rgba(0,0,0,.09)}"
           "h1,h2,h3{color:#0b3c5d}.header{border-bottom:2px solid #e0e4ea;margin-bottom:18px}"
           "table{width:100%;border-collapse:collapse;margin:12px 0;font-size:.9rem}"
           "th,td{border:1px solid #dde2eb;padding:7px 9px;text-align:right}"
           "th{background:#f0f3f9;font-weight:600}td.label{text-align:left;font-weight:500}"
           ".ok{color:#1a7f37;font-weight:700}.nok{color:#b3261e;font-weight:700}"
           ".img-block{text-align:center;margin:18px 0}"
           ".eq-block{background:#f8fafc;border-left:4px solid #0b3c5d;padding:10px 14px;"
           "font-family:Consolas,monospace;font-size:.84rem;line-height:1.6}"
           ".small-note{font-size:.8rem;color:#777}</style>")
    tabela = f"""<table>
<tr><th class="label">Grandeza</th><th>Calculado</th>
    <th>Lim.geral</th><th>Status geral</th>
    <th>Lim.ocup.</th><th>Status ocup.</th></tr>
<tr><td class="label">|E|_max @ {config.h_obs_m:.1f}m</td>
    <td>{res.E_max_kV_m:.4f} kV/m (x={res.x_E_max_m:.1f}m)</td>
    <td>{lim.E_max_kV_m_geral:.2f} kV/m</td>
    <td class="{sEg}">{"OK" if sEg=="ok" else "EXCEDE"}</td>
    <td>{lim.E_max_kV_m_ocup:.2f} kV/m</td>
    <td class="{sEo}">{"OK" if sEo=="ok" else "EXCEDE"}</td></tr>
<tr><td class="label">|B|_max @ {config.h_obs_m:.1f}m</td>
    <td>{res.B_max_uT:.4f} µT (x={res.x_B_max_m:.1f}m)</td>
    <td>{lim.B_max_uT_geral:.1f} µT</td>
    <td class="{sBg}">{"OK" if sBg=="ok" else "EXCEDE"}</td>
    <td>{lim.B_max_uT_ocup:.1f} µT</td>
    <td class="{sBo}">{"OK" if sBo=="ok" else "EXCEDE"}</td></tr>
</table>"""
    eqs = ("<div class='eq-block'>"
           "<b>Campo E (MSC + Imagens):</b><br/>"
           "Ex = (1/2πε₀) Σ λᵢ·[(x−xᵢ)/r²ᵢ − (x−xᵢ)/r²ᵢ_img]<br/>"
           "Ez = (1/2πε₀) Σ λᵢ·[(z−yᵢ)/r²ᵢ − (z+yᵢ)/r²ᵢ_img]<br/>"
           "Imagem elétrica em y_img = −yᵢ (solo equipotencial).<br/><br/>"
           "<b>Campo B (Imagens Complexas de Deri):</b><br/>"
           "p = √(ρ_s / j·ω·μ₀)  — profundidade complexa (Deri et al., IEEE PAS 1981)<br/>"
           "y'ᵢ = −yᵢ − 2p  — imagem complexa<br/>"
           "Bx = (μ₀/2π) Σ İᵢ·[−(z−yᵢ)/r²ᵢ + (z−y'ᵢ)/r²ᵢ_img]<br/>"
           "<b>Nota (Fix #4):</b> λ calculado como C'_seq·V (aproximação para linhas transpostas).</div>")
    return f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"/>
<title>Campos EM – {project.nome_projeto}</title>{css}</head><body>
<div class="container">
<div class="header"><h1>BK Estudos Elétricos — Campos Elétrico e Magnético</h1>
<div class="meta"><b>Projeto:</b> {project.nome_projeto} | <b>Cliente:</b> {project.cliente} |
<b>Nº:</b> {project.numero_projeto}<br/>
<b>V_LL:</b> {res.V_LL_kV:.3f} kV | <b>Potência:</b> {res.power_mva:.1f} MVA |
<b>h_obs:</b> {config.h_obs_m:.2f} m | <b>Faixa:</b> [{config.x_min_m:.0f};{config.x_max_m:.0f}] m</div></div>
<h2>1. Limites — ANEEL RN 915/2021</h2>
<table><tr><th class="label">Público</th><th>|E| (kV/m)</th><th>|B| (µT)</th></tr>
<tr><td class="label">Geral (entorno)</td><td>{lim.E_max_kV_m_geral:.2f}</td><td>{lim.B_max_uT_geral:.0f}</td></tr>
<tr><td class="label">Ocupacional (restrito)</td><td>{lim.E_max_kV_m_ocup:.2f}</td><td>{lim.B_max_uT_ocup:.0f}</td></tr>
</table>
<h2>2. Resultados</h2>{tabela}
<h2>3. Perfis 2D</h2>
<div class="img-block"><img src="data:image/png;base64,{E_b64}" style="max-width:100%"/></div>
<div class="img-block"><img src="data:image/png;base64,{B_b64}" style="max-width:100%"/></div>
<h2>4. Mapas 3D</h2>
<div class="img-block"><img src="data:image/png;base64,{E3d}" style="max-width:100%"/></div>
<div class="img-block"><img src="data:image/png;base64,{B3d}" style="max-width:100%"/></div>
<h2>5. Metodologia</h2>{eqs}
<p class="small-note">Validação final recomendada por medição em campo (ABNT NBR IEC 61786).</p>
</div></body></html>"""

if __name__ == "__main__":
    print("field_em_fixed.py OK — importe como módulo para uso.")

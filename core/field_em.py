# bk_estudos_eletricos/core/field_em.py
# ======================================================================
# Cálculo de campos elétricos e magnéticos em linhas aéreas
# Norma: ANEEL Resolução Normativa nº 915/2021
# Método: Simulação de Cargas (MSC) com Método das Imagens
#         Campo B: Método das Imagens Complexas de Deri (solo com perdas)
#
# Referências:
#   [1] ANEEL RN 915/2021 — Limites de exposição a campos EM
#   [2] ICNIRP 2010 — Guidelines 1 Hz–100 kHz
#   [3] PERRO B.D.S. — Campos em LTs à freq. industrial, RJ, 2007
#   [4] PINHO A.C. — Campo elétrico 2D em LTs, UFSC, 1994
#   [5] VELAME M.R. — MSC vs. Imagens, UFRB, 2019
#   [6] EPRI — AC Transmission Reference Book 200 kV and above, 3rd ed., 2005
#   [7] DERI A. et al. — Complex Ground Return Plane, IEEE PAS, 1981
#   [8] VIEIRA H.R. — Acoplamento Magnético em LTs, UFSJ, 2013
#
# Observações:
#   - Campo E: MSC com imagens elétricas (solo equipotencial)
#   - Campo B: superposição fasorial + Método das Imagens Complexas de Deri
#   - Altura de avaliação: 1,5 m (conforme ANEEL RN 915/2021, art. 4º)
#   - Distância mínima de análise lateral: ±30 m a partir do eixo (padrão)
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cmath
import math
import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .cables import EPS0, MU0
from .line_params import (
    LineGeometry,
    LineParamsResult,
)

# ────────────────────────────────────────────────────────────────────
# Estruturas
# ────────────────────────────────────────────────────────────────────

@dataclass
class FieldConfig:
    """
    Configuração para cálculo de campos |E| e |B| ao nível do solo.

    h_obs_m    : altura de observação (padrão: 1,5 m — ANEEL RN 915/2021)
    x_min_m    : limite lateral mínimo (padrão: -30 m)
    x_max_m    : limite lateral máximo (padrão: +30 m)
    n_points   : número de pontos no perfil lateral
    Ic_manual  : corrente RMS por circuito (A); se None, calcula por S e V
    rho_solo   : resistividade do solo (Ω·m); padrão 100 Ω·m (Vieira, ref.8)
    freq_hz    : frequência do sistema (Hz)
    """
    h_obs_m:   float = 1.5
    x_min_m:   float = -30.0
    x_max_m:   float =  30.0
    n_points:  int   = 301
    Ic_manual: Optional[float] = None
    rho_solo:  float = 100.0   # Ω·m
    freq_hz:   float = 60.0


@dataclass
class AneelLimits:
    """
    Limites normativos — ANEEL Resolução Normativa nº 915/2021.

    Público em geral (entorno / muro externo):
      E_max = 4,17 kV/m   |   B_max = 200 µT

    População ocupacional (área de acesso restrito, ex.: interior de SE):
      E_max = 8,33 kV/m   |   B_max = 1000 µT

    Valores em kV/m (campo elétrico) e µT (campo magnético).
    """
    # Público em geral
    E_max_kV_m_geral:  float = 4.17
    B_max_uT_geral:    float = 200.0
    # Ocupacional (acesso restrito)
    E_max_kV_m_ocup:   float = 8.33
    B_max_uT_ocup:     float = 1000.0

    # Aliases de compatibilidade (anteriores ao RN 915)
    @property
    def E_max_kV_m_areas_occup(self): return self.E_max_kV_m_geral
    @property
    def B_max_uT_areas_occup(self): return self.B_max_uT_geral


@dataclass
class FieldProfilesResult:
    """Resultado numérico dos perfis de campo elétrico e magnético."""
    x_m:      List[float]
    E_kV_m:   List[float]
    B_uT:     List[float]

    E_max_kV_m:  float
    x_E_max_m:   float
    B_max_uT:    float
    x_B_max_m:   float

    E_compliance_msg: str
    B_compliance_msg: str

    V_LL_kV:   float
    power_mva: float
    config:    FieldConfig
    limits:    AneelLimits


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


def _get_params_for_circuit(
    params_by_circuit: Dict[int, LineParamsResult],
    circuit_index: int,
) -> LineParamsResult:
    if circuit_index not in params_by_circuit:
        raise ValueError(f"Sem LineParamsResult para circuito {circuit_index}.")
    return params_by_circuit[circuit_index]


def _deri_complex_depth(rho_solo: float, freq_hz: float) -> complex:
    """
    Profundidade complexa de retorno pelo solo — Método de Deri (ref.[7]).
      p = sqrt(ρ_s / (j·ω·μ₀))
    Com ρ_s ≥ 50 Ω·m os resultados são equivalentes ao solo perfeito (ref.[8]).
    """
    omega = 2.0 * math.pi * freq_hz
    rho_s = max(rho_solo, 50.0)   # Vieira: ≥ 50 Ω·m → resultado idêntico
    return cmath.sqrt(complex(rho_s, 0.0) / complex(0.0, omega * MU0))


# ────────────────────────────────────────────────────────────────────
# Dados de fase (Q e I fasorial)
# ────────────────────────────────────────────────────────────────────

def _build_phase_data(
    geom: LineGeometry,
    params_by_circuit: Dict[int, LineParamsResult],
    V_LL_kV: float,
    power_mva: float,
    Ic_manual: Optional[float],
) -> Dict[Tuple[int, str], Tuple[complex, complex]]:
    """
    (cidx, phase) → (lambda_per_m [C/m, fasor], I_phase [A, fasor])
    """
    circuits   = geom.circuits()
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
            ang_rad = math.radians(_phase_angle_deg(phase))
            V_ph    = V_phase_V * cmath.exp(1j * ang_rad)
            I_ph    = I_mag     * cmath.exp(1j * ang_rad)
            q_per_km    = C_F_km * V_ph
            lambda_per_m = q_per_km / 1000.0
            phase_data[(cidx, phase)] = (lambda_per_m, I_ph)

    return phase_data


# ────────────────────────────────────────────────────────────────────
# Campo E — MSC com Método das Imagens (solo equipotencial)
# Referências: PINHO (ref.[4]), VELAME (ref.[5]), EPRI (ref.[6])
# ────────────────────────────────────────────────────────────────────

def _compute_E_at_point(
    x: float,
    z: float,
    geom: LineGeometry,
    phase_data: Dict[Tuple[int, str], Tuple[complex, complex]],
) -> float:
    """
    |E|(x,z) em kV/m usando o Método da Simulação de Cargas (MSC) com imagens.

    Componentes fasorial horizontal (Ex) e vertical (Ez):
      Ėxt = (1/2πε₀) Σ q̇ᵢ { (x−xᵢ)/[(x−xᵢ)²+(z−yᵢ)²]
                             − (x−xᵢ)/[(x−xᵢ)²+(z+yᵢ)²] }
      Ėyt = (1/2πε₀) Σ q̇ᵢ { (z−yᵢ)/[(x−xᵢ)²+(z−yᵢ)²]
                             − (z+yᵢ)/[(x−xᵢ)²+(z+yᵢ)²] }
    |E| = √(|Ėxt|² + |Ėyt|²)   [V/m]
    """
    Ex = 0.0 + 0j
    Ez = 0.0 + 0j

    for cond in geom.conductors:
        if cond.phase not in ("A", "B", "C"):
            continue
        key = (cond.circuit_index, cond.phase)
        if key not in phase_data:
            continue
        lam, _ = phase_data[key]

        xi, yi = cond.x_m, cond.y_m
        coef = lam / (2.0 * math.pi * EPS0)

        dx = x - xi
        # Contribuição real
        dz_r = z - yi
        r2_r = dx*dx + dz_r*dz_r
        if r2_r > 1e-8:
            Ex += coef * dx    / r2_r
            Ez += coef * dz_r  / r2_r
        # Imagem elétrica (condutor espelhado em y = −yᵢ)
        dz_i = z + yi
        r2_i = dx*dx + dz_i*dz_i
        if r2_i > 1e-8:
            Ex -= coef * dx    / r2_i
            Ez -= coef * dz_i  / r2_i

    E_V_m = math.sqrt(abs(Ex)**2 + abs(Ez)**2)
    return E_V_m / 1e3   # kV/m


# ────────────────────────────────────────────────────────────────────
# Campo B — Imagens Complexas de Deri (solo com perdas)
# Referência: DERI et al. IEEE PAS 1981 (ref.[7]), VIEIRA (ref.[8])
# ────────────────────────────────────────────────────────────────────

def _compute_B_at_point(
    x: float,
    z: float,
    geom: LineGeometry,
    phase_data: Dict[Tuple[int, str], Tuple[complex, complex]],
    p_deri: complex,
) -> float:
    """
    |B|(x,z) em µT usando Método das Imagens Complexas de Deri.

    y'ᵢ = −yᵢ − 2·p   (coordenada da imagem complexa)

    Ḃxt = (µ₀/2π) Σ İᵢ { −(z−yᵢ)/[(x−xᵢ)²+(z−yᵢ)²]
                          − (−(z−y'ᵢ))/[(x−xᵢ)²+(z−y'ᵢ)²] }
    Ḃyt = (µ₀/2π) Σ İᵢ { (x−xᵢ)/[(x−xᵢ)²+(z−yᵢ)²]
                          − (x−xᵢ)/[(x−xᵢ)²+(z−y'ᵢ)²] }
    |B| = √(|Ḃxt|² + |Ḃyt|²)   [T]
    """
    Bx = 0.0 + 0j
    Bz = 0.0 + 0j

    for cond in geom.conductors:
        if cond.phase not in ("A", "B", "C"):
            continue
        key = (cond.circuit_index, cond.phase)
        if key not in phase_data:
            continue
        _, I_ph = phase_data[key]

        xi  = cond.x_m
        yi  = cond.y_m
        yi_img = -yi - 2.0 * p_deri   # imagem complexa de Deri

        dx    = x - xi
        dz_r  = z - yi
        dz_i  = z - yi_img            # complexo

        r2_r  = dx*dx + dz_r*dz_r
        r2_i  = dx*dx + dz_i*dz_i     # complexo

        coef  = MU0 / (2.0 * math.pi)

        if abs(r2_r) > 1e-8:
            Bx += coef * I_ph * (-dz_r / r2_r)
            Bz += coef * I_ph * (  dx   / r2_r)

        if abs(r2_i) > 1e-12:
            # sinal negativo p/ a imagem (corrente oposta)
            Bx -= coef * I_ph * (-dz_i / r2_i)
            Bz -= coef * I_ph * (  dx   / r2_i)

    B_T = math.sqrt(abs(Bx)**2 + abs(Bz)**2)
    return B_T * 1e6   # µT


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
    """
    Calcula perfis |E|(x) e |B|(x) à altura h_obs_m.
    Usa MSC + Imagens para E e Imagens Complexas de Deri para B.
    Distância padrão de avaliação: ±30 m (pode ser ajustada pelo usuário).
    """
    if config.n_points < 2:
        raise ValueError("n_points deve ser ≥ 2.")
    if config.x_max_m <= config.x_min_m:
        raise ValueError("x_max_m deve ser > x_min_m.")
    if V_LL_kV < 0 or power_mva < 0:
        raise ValueError("V_LL_kV e power_mva devem ser ≥ 0.")

    phase_data = _build_phase_data(geom, params_by_circuit, V_LL_kV, power_mva, config.Ic_manual)
    p_deri     = _deri_complex_depth(config.rho_solo, config.freq_hz)

    x_vals: List[float] = []
    E_vals: List[float] = []
    B_vals: List[float] = []

    dx = (config.x_max_m - config.x_min_m) / (config.n_points - 1)
    for i in range(config.n_points):
        xp = config.x_min_m + i * dx
        E  = _compute_E_at_point(xp, config.h_obs_m, geom, phase_data)
        B  = _compute_B_at_point(xp, config.h_obs_m, geom, phase_data, p_deri)
        x_vals.append(xp)
        E_vals.append(E)
        B_vals.append(B)

    E_max = max(E_vals) if E_vals else 0.0
    B_max = max(B_vals) if B_vals else 0.0
    x_E_max = x_vals[E_vals.index(E_max)] if E_vals else 0.0
    x_B_max = x_vals[B_vals.index(B_max)] if B_vals else 0.0

    # Conformidade — usa limite "público em geral" (RN 915/2021)
    E_lim = limits.E_max_kV_m_geral
    B_lim = limits.B_max_uT_geral
    E_msg = (f"✅ ATENDE — |E|_máx = {E_max:.3f} kV/m ≤ {E_lim:.2f} kV/m (RN 915/2021)"
             if E_max <= E_lim else
             f"❌ NÃO ATENDE — |E|_máx = {E_max:.3f} kV/m > {E_lim:.2f} kV/m (RN 915/2021)")
    B_msg = (f"✅ ATENDE — |B|_máx = {B_max:.3f} µT ≤ {B_lim:.1f} µT (RN 915/2021)"
             if B_max <= B_lim else
             f"❌ NÃO ATENDE — |B|_máx = {B_max:.3f} µT > {B_lim:.1f} µT (RN 915/2021)")

    return FieldProfilesResult(
        x_m=x_vals, E_kV_m=E_vals, B_uT=B_vals,
        E_max_kV_m=E_max, x_E_max_m=x_E_max,
        B_max_uT=B_max,   x_B_max_m=x_B_max,
        E_compliance_msg=E_msg, B_compliance_msg=B_msg,
        V_LL_kV=V_LL_kV, power_mva=power_mva,
        config=config, limits=limits,
    )


# ────────────────────────────────────────────────────────────────────
# Plotagem 2D matplotlib (para relatório Word/HTML)
# ────────────────────────────────────────────────────────────────────

def _plot_profiles_2d(result: FieldProfilesResult) -> Tuple[str, str]:
    x   = result.x_m
    E   = result.E_kV_m
    B   = result.B_uT
    lim = result.limits

    # Campo elétrico — com autoscale e preenchimento
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.fill_between(x, E, alpha=0.15, color="#1565C0")
    ax.plot(x, E, linewidth=2.2, color="#1565C0", label="|E|(x) – kV/m")
    # Marcador de máximo
    ax.plot(result.x_E_max_m, result.E_max_kV_m, 'rv', markersize=9,
            label=f"|E|_máx = {result.E_max_kV_m:.4f} kV/m @ x={result.x_E_max_m:.1f}m")
    # Limites — só desenha se escala é compatível
    e_max_val = max(E) if E else 1.0
    e_ylim = e_max_val * 1.4
    if lim.E_max_kV_m_geral <= e_ylim * 2.5:
        ax.axhline(lim.E_max_kV_m_geral, linestyle="--", linewidth=1.5, color="#C62828",
                   label=f"Limite geral = {lim.E_max_kV_m_geral:.2f} kV/m")
    if lim.E_max_kV_m_ocup <= e_ylim * 2.5:
        ax.axhline(lim.E_max_kV_m_ocup, linestyle=":", linewidth=1.5, color="#E65100",
                   label=f"Limite ocupacional = {lim.E_max_kV_m_ocup:.2f} kV/m")
    ax.set_ylim(bottom=0, top=e_ylim)
    ax.set_xlabel("Distância lateral x (m)", fontsize=10)
    ax.set_ylabel("|E| (kV/m)", fontsize=10)
    ax.grid(True, alpha=0.35, linestyle='-')
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Perfil lateral de campo elétrico — h_obs = {result.config.h_obs_m:.1f} m (ANEEL RN 915/2021)", fontsize=10)
    b64_E = _fig_to_base64(fig)

    # Campo magnético — com autoscale e preenchimento
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.fill_between(x, B, alpha=0.15, color="#00695C")
    ax.plot(x, B, linewidth=2.2, color="#00695C", label="|B|(x) – µT")
    ax.plot(result.x_B_max_m, result.B_max_uT, 'rv', markersize=9,
            label=f"|B|_máx = {result.B_max_uT:.4f} µT @ x={result.x_B_max_m:.1f}m")
    b_max_val = max(B) if B else 1.0
    b_ylim = b_max_val * 1.4
    if lim.B_max_uT_geral <= b_ylim * 2.5:
        ax.axhline(lim.B_max_uT_geral, linestyle="--", linewidth=1.5, color="#C62828",
                   label=f"Limite geral = {lim.B_max_uT_geral:.1f} µT")
    if lim.B_max_uT_ocup <= b_ylim * 2.5:
        ax.axhline(lim.B_max_uT_ocup, linestyle=":", linewidth=1.5, color="#E65100",
                   label=f"Limite ocupacional = {lim.B_max_uT_ocup:.1f} µT")
    ax.set_ylim(bottom=0, top=b_ylim)
    ax.set_xlabel("Distância lateral x (m)", fontsize=10)
    ax.set_ylabel("|B| (µT)", fontsize=10)
    ax.grid(True, alpha=0.35, linestyle='-')
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Perfil lateral de campo magnético — h_obs = {result.config.h_obs_m:.1f} m (ANEEL RN 915/2021)", fontsize=10)
    b64_B = _fig_to_base64(fig)

    return b64_E, b64_B


# ────────────────────────────────────────────────────────────────────
# Grade 3D para superfícies
# ────────────────────────────────────────────────────────────────────

def _compute_EB_grid_3d(
    geom: LineGeometry,
    params_by_circuit: Dict[int, LineParamsResult],
    result: FieldProfilesResult,
    n_heights: int = 25,
) -> Tuple[List[float], List[float], List[List[float]], List[List[float]]]:
    cfg        = result.config
    phase_data = _build_phase_data(geom, params_by_circuit, result.V_LL_kV, result.power_mva, cfg.Ic_manual)
    p_deri     = _deri_complex_depth(cfg.rho_solo, cfg.freq_hz)

    x_vals = result.x_m
    if not x_vals:
        return [], [], [], []

    max_y = max((c.y_m for c in geom.conductors), default=25.0)
    z_max = max(max_y + 2.0, cfg.h_obs_m + 1.0)
    z_vals = [i * z_max / (n_heights - 1) for i in range(n_heights)]

    E_grid: List[List[float]] = []
    B_grid: List[List[float]] = []

    for zp in z_vals:
        row_E, row_B = [], []
        for xp in x_vals:
            row_E.append(_compute_E_at_point(xp, zp, geom, phase_data))
            row_B.append(_compute_B_at_point(xp, zp, geom, phase_data, p_deri))
        E_grid.append(row_E)
        B_grid.append(row_B)

    return x_vals, z_vals, E_grid, B_grid


def _plot_surface_3d(
    x_vals: List[float],
    z_vals: List[float],
    F_grid: List[List[float]],
    z_label: str,
    title: str,
    colormap: str = "viridis",
) -> str:
    if not x_vals or not z_vals or not F_grid:
        return ""
    try:
        import numpy as np
    except ImportError:
        return ""

    X, Z = np.meshgrid(x_vals, z_vals)
    F    = np.array(F_grid, dtype=float)
    # Clip outliers para melhor visualização (percentil 98)
    p98 = np.percentile(F[F > 0], 98) if np.any(F > 0) else 1.0
    F    = np.clip(F, 0, p98)

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Z, F, cmap=colormap, linewidth=0.2, antialiased=True,
                           alpha=0.92, edgecolor='gray', rstride=1, cstride=2)
    ax.set_xlabel("Distância lateral x (m)", fontsize=9, labelpad=10)
    ax.set_ylabel("Altura z (m)", fontsize=9, labelpad=10)
    ax.set_zlabel(z_label, fontsize=9, labelpad=10)
    ax.set_title(title, fontsize=10, pad=15)
    ax.view_init(elev=28, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, label=z_label, pad=0.12)

    # Adiciona contorno projetado no plano z=0
    try:
        ax.contourf(X, Z, F, zdir='z', offset=0, cmap=colormap, alpha=0.3, levels=15)
    except Exception:
        pass

    return _fig_to_base64(fig)


# ────────────────────────────────────────────────────────────────────
# Dados para gráfico 3D interativo Plotly (retorna dicts JSON-safe)
# ────────────────────────────────────────────────────────────────────

def compute_plotly_surface_data(
    geom: LineGeometry,
    params_by_circuit: Dict[int, LineParamsResult],
    result: FieldProfilesResult,
    n_heights: int = 20,
) -> Tuple[List[float], List[float], List[List[float]], List[List[float]]]:
    """
    Retorna (x_vals, z_vals, E_grid, B_grid) prontos para go.Surface() do Plotly.
    Usa grade reduzida (n_heights=20) para performance no browser.
    """
    return _compute_EB_grid_3d(geom, params_by_circuit, result, n_heights=n_heights)


# ────────────────────────────────────────────────────────────────────
# Relatório HTML (para download / arquivo)
# ────────────────────────────────────────────────────────────────────

def generate_html_report_fields(
    project,
    geom: LineGeometry,
    params_by_circuit: Dict[int, LineParamsResult],
    config: FieldConfig,
    limits: AneelLimits,
    result: FieldProfilesResult,
) -> str:
    E_b64, B_b64 = _plot_profiles_2d(result)

    x_vals, z_vals, E_grid, B_grid = _compute_EB_grid_3d(geom, params_by_circuit, result, n_heights=25)
    E3d_b64 = _plot_surface_3d(x_vals, z_vals, E_grid, "|E| (kV/m)", "Mapa 3D de campo elétrico |E|(x,z)", "plasma")
    B3d_b64 = _plot_surface_3d(x_vals, z_vals, B_grid, "|B| (µT)",   "Mapa 3D de campo magnético |B|(x,z)", "viridis")

    res = result
    lim = result.limits

    status_E_g = "ok"   if res.E_max_kV_m <= lim.E_max_kV_m_geral else "nok"
    status_E_o = "ok"   if res.E_max_kV_m <= lim.E_max_kV_m_ocup  else "nok"
    status_B_g = "ok"   if res.B_max_uT   <= lim.B_max_uT_geral   else "nok"
    status_B_o = "ok"   if res.B_max_uT   <= lim.B_max_uT_ocup    else "nok"

    n_gw = len(geom.shields())
    gw_info = " | ".join([f"{c.name}: ({c.x_m:.1f}, {c.y_m:.1f}) m" for c in geom.shields()])

    css = """
    <style>
      body{font-family:"Segoe UI",Arial,sans-serif;background:#f5f7fa;color:#222;margin:0;padding:0}
      .container{max-width:1100px;margin:0 auto;padding:28px;background:#fff;box-shadow:0 4px 18px rgba(0,0,0,.09)}
      h1,h2,h3{color:#0b3c5d}
      .header{border-bottom:2px solid #e0e4ea;margin-bottom:18px;padding-bottom:10px}
      .meta{font-size:.93rem;color:#444;line-height:1.7}
      .section{margin:22px 0}
      .img-block{text-align:center;margin:18px 0}
      .eq-block{background:#f8fafc;border-left:4px solid #0b3c5d;padding:10px 14px;font-family:Consolas,monospace;font-size:.84rem;line-height:1.6}
      table{width:100%;border-collapse:collapse;margin:12px 0;font-size:.9rem}
      th,td{border:1px solid #dde2eb;padding:7px 9px;text-align:right}
      th{background:#f0f3f9;font-weight:600}
      td.label{text-align:left;font-weight:500}
      .small-note{font-size:.8rem;color:#777}
      .ok{color:#1a7f37;font-weight:700}
      .nok{color:#b3261e;font-weight:700}
      .ref-list{font-size:.82rem;color:#444;line-height:1.8;padding-left:18px}
    </style>"""

    table_rows = f"""
    <table>
      <tr><th class="label">Grandeza</th><th>Calculado</th>
          <th>Lim. geral (kV/m ou µT)</th><th>Status geral</th>
          <th>Lim. ocup. (kV/m ou µT)</th><th>Status ocup.</th></tr>
      <tr>
        <td class="label">|E|<sub>máx</sub> @ {config.h_obs_m:.1f} m</td>
        <td>{res.E_max_kV_m:.4f} kV/m (x={res.x_E_max_m:.1f} m)</td>
        <td>{lim.E_max_kV_m_geral:.2f}</td>
        <td class="{status_E_g}">{"✅ OK" if status_E_g=="ok" else "❌ EXCEDE"}</td>
        <td>{lim.E_max_kV_m_ocup:.2f}</td>
        <td class="{status_E_o}">{"✅ OK" if status_E_o=="ok" else "❌ EXCEDE"}</td>
      </tr>
      <tr>
        <td class="label">|B|<sub>máx</sub> @ {config.h_obs_m:.1f} m</td>
        <td>{res.B_max_uT:.4f} µT (x={res.x_B_max_m:.1f} m)</td>
        <td>{lim.B_max_uT_geral:.1f}</td>
        <td class="{status_B_g}">{"✅ OK" if status_B_g=="ok" else "❌ EXCEDE"}</td>
        <td>{lim.B_max_uT_ocup:.1f}</td>
        <td class="{status_B_o}">{"✅ OK" if status_B_o=="ok" else "❌ EXCEDE"}</td>
      </tr>
    </table>"""

    eqs = """
    <div class="eq-block">
      <strong>3.1 Campo Elétrico — MSC com Método das Imagens</strong><br/>
      [Q̇] = [C] · [V̇]  → λᵢ = Qᵢ / 1000  (C/m, por condutor)<br/>
      Ėxt = (1/2πε₀) Σ q̇ᵢ { (x−xᵢ)/r²ᵢ − (x−xᵢ)/r²ᵢ_img }<br/>
      Ėyt = (1/2πε₀) Σ q̇ᵢ { (y−yᵢ)/r²ᵢ − (y+yᵢ)/r²ᵢ_img }<br/>
      |E| = √(|Ėxt|² + |Ėyt|²)  [V/m]  → relatório em kV/m<br/>
      <em>Solo equipotencial; imagem elétrica em y_img = −yᵢ (potencial nulo no solo)</em><br/><br/>
      <strong>3.2 Campo Magnético — Imagens Complexas de Deri</strong><br/>
      Ḃ = μ₀ · Ḧ  com μ = μ₀ (ar/solo não-magnético)<br/>
      Profundidade complexa: p = √(ρs / j·ω·μ₀)  (Deri et al., IEEE PAS 1981)<br/>
      Imagem complexa: y'ᵢ = −yᵢ − 2p<br/>
      Ḃxt = (μ₀/2π) Σ İᵢ { (−(y−yᵢ))/r²ᵢ − (+( y−y'ᵢ))/r²ᵢ_img }<br/>
      Ḃyt = (μ₀/2π) Σ İᵢ { (x−xᵢ)/r²ᵢ  − (x−xᵢ)/r²ᵢ_img }<br/>
      |B| = √(|Ḃxt|² + |Ḃyt|²)  [T]  → relatório em µT<br/>
      <em>Para ρs ≥ 50 Ω·m o resultado é equivalente ao solo perfeito (Vieira, UFSJ 2013)</em>
    </div>"""

    refs = """
    <ol class="ref-list">
      <li>ANEEL – Resolução Normativa nº 915, de 23/02/2021 (revoga nº 616/2014)</li>
      <li>ICNIRP – Guidelines for Limiting Exposure to Electric and Magnetic Fields (1 Hz–100 kHz), Dec. 2010</li>
      <li>PERRO B.D.S. – Estudo dos Campos Eletromagnéticos em LTs à Freq. Industrial. RJ, 2007</li>
      <li>PINHO A.C. – Cálculo do Campo Elétrico 2D em LTs. UFSC, 1994</li>
      <li>VELAME M.R. – Cálculo dos Campos EM: MSC vs. Imagens. UFRB, 2019</li>
      <li>EPRI – AC Transmission Line Reference Book 200 kV and above, 3rd ed., Palo Alto, 2005</li>
      <li>DERI A. et al. – The Complex Ground Return Plane. IEEE Trans. PAS, ago. 1981, pp. 3686–3694</li>
      <li>VIEIRA H.R. – Acoplamento Magnético em LTs e Dutos Metálicos. UFSJ, 2013</li>
    </ol>"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="utf-8"/>
<title>Estudo de Campos EM – {project.nome_projeto}</title>
{css}
</head>
<body><div class="container">

  <div class="header">
    <h1>BK Estudos Elétricos — Campos Elétrico e Magnético</h1>
    <div class="meta">
      <strong>Projeto:</strong> {project.nome_projeto} &nbsp;|&nbsp;
      <strong>Cliente:</strong> {project.cliente} &nbsp;|&nbsp;
      <strong>Nº:</strong> {project.numero_projeto}<br/>
      <strong>Tensão nominal:</strong> {res.V_LL_kV:.3f} kV (L-L) &nbsp;|&nbsp;
      <strong>Potência:</strong> {res.power_mva:.1f} MVA<br/>
      <strong>Altura de avaliação:</strong> {config.h_obs_m:.2f} m (ANEEL RN 915/2021, art. 4º)<br/>
      <strong>Faixa lateral avaliada:</strong> [{config.x_min_m:.1f} m ; {config.x_max_m:.1f} m]<br/>
      <strong>Resistividade do solo:</strong> {config.rho_solo:.0f} Ω·m &nbsp;|&nbsp;
      <strong>Frequência:</strong> {config.freq_hz:.0f} Hz<br/>
      <strong>Cabos-guarda:</strong> {n_gw} ({gw_info if gw_info else "—"})
    </div>
  </div>

  <div class="section">
    <h2>1. Limites Normativos — ANEEL RN 915/2021</h2>
    <table>
      <tr><th class="label">Tipo de público</th><th>|E| máx (kV/m)</th><th>|B| máx (µT)</th></tr>
      <tr><td class="label">Público em geral (entorno / muro externo)</td>
          <td>{lim.E_max_kV_m_geral:.2f}</td><td>{lim.B_max_uT_geral:.1f}</td></tr>
      <tr><td class="label">Ocupacional (acesso restrito a funcionários)</td>
          <td>{lim.E_max_kV_m_ocup:.2f}</td><td>{lim.B_max_uT_ocup:.1f}</td></tr>
    </table>
    <p class="small-note">Avaliação a 1,5 m de altura acima do nível do solo.</p>
  </div>

  <div class="section">
    <h2>2. Resultados e Verificação</h2>
    {table_rows}
  </div>

  <div class="section">
    <h2>3. Perfis Laterais (2D)</h2>
    <div class="img-block">
      <h3>Perfil de campo elétrico |E|(x)</h3>
      <img src="data:image/png;base64,{E_b64}" style="max-width:100%"/>
    </div>
    <div class="img-block">
      <h3>Perfil de campo magnético |B|(x)</h3>
      <img src="data:image/png;base64,{B_b64}" style="max-width:100%"/>
    </div>
  </div>

  <div class="section">
    <h2>4. Mapas 3D — |E| e |B| em função de (x, z)</h2>
    <div class="img-block">
      <h3>Mapa 3D de campo elétrico |E|(x,z)</h3>
      <img src="data:image/png;base64,{E3d_b64}" style="max-width:100%"/>
    </div>
    <div class="img-block">
      <h3>Mapa 3D de campo magnético |B|(x,z)</h3>
      <img src="data:image/png;base64,{B3d_b64}" style="max-width:100%"/>
    </div>
  </div>

  <div class="section">
    <h2>5. Metodologia de Cálculo</h2>
    {eqs}
  </div>

  <div class="section">
    <h2>6. Referências</h2>
    {refs}
  </div>

  <p class="small-note">
    Nota: modelo 2D de condutores infinitos com superposição fasorial.
    Para validação final recomenda-se medição em campo conforme ABNT NBR IEC 61786.
  </p>

</div></body></html>"""

    return html


if __name__ == "__main__":
    print("Módulo field_em carregado.")

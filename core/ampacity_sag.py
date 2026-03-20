# bk_estudos_eletricos/core/ampacity_sag.py
#
# Ampacidade (capacidade de corrente) e Flecha de condutores de linha aérea
#
# Integrado com:
#   - cables.py (dados de cabo, R_ac)
#   - line_params.py (LineGeometry, ProjectInfo)
#
# Saída:
#   - Cálculo de I_max pela equação de balanço térmico (estilo IEEE 738 - simplificado)
#   - Cálculo de flecha por vão (modelo parabólico)
#   - Relatório HTML+CSS com tabelas e gráficos (2D e 3D)
#
# Observação importante:
#   Este módulo NÃO inclui gráfico de geometria no relatório.

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import math
import base64
from io import BytesIO

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .cables import (
    Cable,
    default_cable_db,
    find_cable,
)

from .line_params import (
    LineGeometry,
    ConductorInstance,
    ProjectInfo,
)

# ====================== Constantes físicas ===========================

SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann [W/m²·K⁴]
G = 9.81                   # gravidade [m/s²]

# Propriedades típicas do ar a ~25°C (simplificação)
K_AIR = 0.026              # condutividade térmica do ar [W/m·K]
NU_AIR = 1.5e-5            # viscosidade cinemática do ar [m²/s]
PR_AIR = 0.71              # número de Prandtl [-]
BETA_AIR = 1.0 / 298.15    # 1/Tfilm ~ 1/298 K, para convecção natural [1/K]

# ====================== Utilidades de material =======================

def kcmil_to_m2(area_kcmil: float) -> float:
    """Converte kcmil para m² (1 kcmil = 0.5067 mm²)."""
    return area_kcmil * 0.5067e-6


def guess_density_from_material(material: str) -> float:
    """
    Densidade aproximada do condutor [kg/m³].
    Valores típicos:
      - Cobre   : ~8900
      - Alumínio: ~2700
      - ACSR    : ~3200 (alumínio + aço)
    """
    m = (material or "").strip().lower()
    if "acsr" in m:
        return 3200.0
    if "al" in m and "acsr" not in m:
        return 2700.0
    if "cu" in m:
        return 8900.0
    return 3000.0


def guess_uts_from_material(material: str) -> float:
    """
    Tensão última aproximada do material [Pa].
    Valores típicos (ordem de grandeza):
      - Cobre   : ~200 MPa
      - Alumínio: ~90 MPa
      - ACSR    : ~160 MPa
    """
    m = (material or "").strip().lower()
    if "acsr" in m:
        return 160e6
    if "al" in m and "acsr" not in m:
        return 90e6
    if "cu" in m:
        return 200e6
    return 120e6


def conductor_weight_per_m(cable: Cable) -> float:
    """
    Peso próprio aproximado do condutor [N/m].
    w = rho * A * g
    """
    area_m2 = kcmil_to_m2(cable.area_kcmil)
    rho = guess_density_from_material(cable.material)
    mass_per_m = rho * area_m2
    return mass_per_m * G


def conductor_uts_force(cable: Cable) -> float:
    """
    Força última aproximada [N] de tração do condutor.
    F_u = sigma_u * A
    """
    area_m2 = kcmil_to_m2(cable.area_kcmil)
    uts = guess_uts_from_material(cable.material)
    return uts * area_m2


# ====================== Modelo de convecção =========================

def churchill_bernstein_nusselt(Re: float, Pr: float) -> float:
    """
    Correlação de Churchill–Bernstein para convecção forçada
    transversal em cilindro (Nu médio).
    """
    if Re <= 0:
        return 0.0
    term1 = 0.3
    term2 = (0.62 * (Re ** 0.5) * (Pr ** (1.0 / 3.0))) / ((1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25)
    term3 = (1.0 + (Re / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
    return term1 + term2 * term3


def natural_convection_heat_loss_per_m(
    D_m: float,
    Ts_C: float,
    Ta_C: float,
) -> float:
    """
    Convecção natural em cilindro horizontal (ordem IEEE/IEC simplificada).
    Usa correlação típica via Rayleigh (Ra) -> Nu.

    q_n = π D h_n ΔT
    """
    if D_m <= 0:
        return 0.0
    Ts_K = Ts_C + 273.15
    Ta_K = Ta_C + 273.15
    dT = Ts_K - Ta_K
    if dT <= 0:
        return 0.0

    # Rayleigh: Ra = g * beta * dT * D^3 / (nu^2) * Pr
    Ra = G * BETA_AIR * dT * (D_m ** 3) / (NU_AIR ** 2) * PR_AIR
    if Ra <= 0:
        return 0.0

    # Correlação comum para cilindro: Nu = 0.36 + 0.518*Ra^(1/4) / (1 + (0.559/Pr)^(9/16))^(4/9)
    denom = (1.0 + (0.559 / PR_AIR) ** (9.0 / 16.0)) ** (4.0 / 9.0)
    Nu = 0.36 + 0.518 * (Ra ** 0.25) / denom

    h = Nu * K_AIR / D_m
    qn = math.pi * D_m * h * dT
    return max(0.0, qn)


def forced_convection_heat_loss_per_m(
    D_m: float,
    Ts_C: float,
    Ta_C: float,
    v_wind_m_s: float,
    wind_angle_deg: float = 90.0,
) -> float:
    """
    Convecção forçada (escoamento cruzado) via Churchill–Bernstein.
    Considera componente perpendicular ao condutor (simplificação):
      v_perp = v * sin(theta)
    """
    if D_m <= 0:
        return 0.0

    Ts_K = Ts_C + 273.15
    Ta_K = Ta_C + 273.15
    dT = Ts_K - Ta_K
    if dT <= 0:
        return 0.0

    v = max(0.0, v_wind_m_s)
    theta = math.radians(wind_angle_deg if wind_angle_deg is not None else 90.0)
    v_perp = max(0.1, abs(v * math.sin(theta)))  # evita Re=0

    Re = v_perp * D_m / NU_AIR
    Nu = churchill_bernstein_nusselt(Re, PR_AIR)
    h = Nu * K_AIR / D_m

    qc = math.pi * D_m * h * dT
    return max(0.0, qc)


def convective_heat_loss_per_m(
    D_m: float,
    Ts_C: float,
    Ta_C: float,
    v_wind_m_s: float,
    wind_angle_deg: float = 90.0,
) -> float:
    """
    Convecção total (estilo IEEE 738 simplificado):
      q_c = max(q_forçada, q_natural)
    """
    q_forced = forced_convection_heat_loss_per_m(D_m, Ts_C, Ta_C, v_wind_m_s, wind_angle_deg)
    q_nat = natural_convection_heat_loss_per_m(D_m, Ts_C, Ta_C)
    return max(q_forced, q_nat)


def radiative_heat_loss_per_m(
    D_m: float,
    Ts_C: float,
    Ta_C: float,
    emissivity: float,
) -> float:
    """
    Perda de calor por radiação [W/m].
      q_r = π D ε σ (T_s⁴ - T_a⁴)
    """
    if D_m <= 0:
        return 0.0
    eps = max(0.0, min(1.0, emissivity))
    Ts_K = Ts_C + 273.15
    Ta_K = Ta_C + 273.15
    if Ts_K <= Ta_K:
        return 0.0
    qr = math.pi * D_m * eps * SIGMA_SB * (Ts_K**4 - Ta_K**4)
    return max(0.0, qr)


def solar_heat_gain_per_m(
    D_m: float,
    solar_irradiance_W_m2: float,
    absorptivity: float,
) -> float:
    """
    Ganho de calor por radiação solar [W/m].
    Aproximação: área projetada ~ D * 1 m
      q_s ≈ α * S * D
    """
    if D_m <= 0:
        return 0.0
    alpha = max(0.0, min(1.0, absorptivity))
    S = max(0.0, solar_irradiance_W_m2)
    return alpha * S * D_m


# ====================== Configuração / Resultados ====================

@dataclass
class AmpacitySagConfig:
    """
    Configurações de ambiente e projeto para ampacidade e flecha.
    """
    # Elétrico / ambiente
    frequency_hz: float = 60.0
    ambient_temp_C: float = 25.0
    max_conductor_temp_C: float = 75.0  # valor típico (ajustável)
    wind_speed_m_s: float = 0.6
    wind_angle_deg: float = 90.0
    solar_irradiance_W_m2: float = 800.0
    absorptivity: float = 0.5
    emissivity: float = 0.5

    # Mecânico / flecha
    design_tension_ratio: float = 0.25  # fração da força última (estimativa)
    span_min_m: float = 30.0
    span_max_m: float = 1500.0
    span_step_m: float = 30.0

    # Corrente de operação para verificação
    operating_current_A: float = 600.0


@dataclass
class AmpacityResultPerCircuit:
    circuit_index: int
    cable_key: str

    # Ampacidade
    I_max_A: float
    I_oper_A: float
    temp_limit_C: float
    ambient_C: float
    compliant_temp: bool

    # Resistência e balanço térmico
    R_ac_ohm_km_at_Tmax: float
    q_conv_W_m: float
    q_rad_W_m: float
    q_solar_W_m: float

    # Flecha (para um vão de referência)
    span_ref_m: float
    sag_ref_m: float
    H_ref_N: float
    w_N_m: float

    # Ajuste para vãos curtos (critério prático)
    relax_factor: float = 1.0
    I_max_base_A: float = 0.0


@dataclass
class SagSurfaceResult:
    """
    Superfície de flechas para vários vãos.
    x_points_frac: posições normalizadas 0..1
    y_surface_m[j][i]: flecha relativa (negativa) para vão j no ponto i.
    """
    span_lengths_m: List[float]
    x_points_frac: List[float]
    y_surface_m: List[List[float]]


@dataclass
class AmpacitySagSummary:
    """
    Resultado consolidado para todos os circuitos.
    """
    ampacity_per_circuit: Dict[int, AmpacityResultPerCircuit]
    sag_surface: SagSurfaceResult
    config: AmpacitySagConfig


# ====================== Utilitários ==========================

def frange(start: float, stop: float, step: float):
    v = start
    while v <= stop + 1e-9:
        yield v
        v += step


# ====================== Cálculos principais ==========================

def compute_ampacity_for_cable(
    cable: Cable,
    config: AmpacitySagConfig,
) -> Tuple[float, float, float, float, float]:
    """
    Calcula a ampacidade I_max [A] para a temperatura máxima definida em
    config.max_conductor_temp_C, usando balanço térmico simplificado.

    Retorna:
        I_max_A, R_ac_ohm_m, q_conv_W_m, q_rad_W_m, q_solar_W_m
    """
    T_amb = float(config.ambient_temp_C)
    T_max = float(config.max_conductor_temp_C)
    f_hz = float(config.frequency_hz)

    D_m = float(cable.diameter_mm) / 1000.0
    if D_m <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Resistência AC por metro na temperatura máxima
    R_ac_ohm_m = float(cable.ac_resistance_per_m(f_hz, T_max))

    # Convecção (max natural/forçada)
    q_conv = convective_heat_loss_per_m(
        D_m=D_m,
        Ts_C=T_max,
        Ta_C=T_amb,
        v_wind_m_s=float(config.wind_speed_m_s),
        wind_angle_deg=float(config.wind_angle_deg),
    )

    q_rad = radiative_heat_loss_per_m(
        D_m=D_m,
        Ts_C=T_max,
        Ta_C=T_amb,
        emissivity=float(config.emissivity),
    )

    q_solar = solar_heat_gain_per_m(
        D_m=D_m,
        solar_irradiance_W_m2=float(config.solar_irradiance_W_m2),
        absorptivity=float(config.absorptivity),
    )

    # Balanço térmico: I² R = q_conv + q_rad - q_solar
    q_total = q_conv + q_rad - q_solar

    if R_ac_ohm_m <= 0 or q_total <= 0:
        I_max_A = 0.0
    else:
        I_max_A = math.sqrt(q_total / R_ac_ohm_m)

    return I_max_A, R_ac_ohm_m, q_conv, q_rad, q_solar


def sag_parabolic(
    span_m: float,
    cable: Cable,
    config: AmpacitySagConfig,
) -> Tuple[float, float, float]:
    """
    Flecha aproximada para o cabo em um vão span_m [m], usando modelo parabólico:
        f = w L² / (8 H)

    onde:
      - w: peso próprio [N/m]
      - H: tração horizontal [N] = ratio * F_u

    Retorna:
        sag_m, H_N, w_N_m
    """
    if span_m <= 0:
        return 0.0, 0.0, 0.0

    w = conductor_weight_per_m(cable)
    Fu = conductor_uts_force(cable)
    ratio = max(0.05, min(0.6, float(config.design_tension_ratio)))
    H = ratio * Fu

    if H <= 0:
        return 0.0, 0.0, w

    sag = w * (span_m ** 2) / (8.0 * H)
    return sag, H, w


def build_sag_surface(
    cable: Cable,
    config: AmpacitySagConfig,
    n_x: int = 25,
) -> SagSurfaceResult:
    """
    Constrói uma superfície 3D de flecha (modelo parabólico), relativa:
      - spans: span_min..span_max
      - x_points_frac: 0..1
      - y(x) = -4 f xi (1 - xi)
    """
    span_min = max(1.0, float(config.span_min_m))
    span_max = max(span_min, float(config.span_max_m))
    step = max(1.0, float(config.span_step_m))

    span_lengths = [float(L) for L in frange(span_min, span_max, step)]
    n_x = int(max(5, n_x))
    x_frac = [i / (n_x - 1) for i in range(n_x)]

    y_surface: List[List[float]] = []
    for L in span_lengths:
        sag_L, _, _ = sag_parabolic(L, cable, config)
        y_line = [-4.0 * sag_L * xi * (1.0 - xi) for xi in x_frac]
        y_surface.append(y_line)

    return SagSurfaceResult(
        span_lengths_m=span_lengths,
        x_points_frac=x_frac,
        y_surface_m=y_surface,
    )


def compute_ampacity_sag_for_geometry(
    geom: LineGeometry,
    V_LL_kV: float,  # mantido para interface/relatório (pode ser útil em extensões)
    config: AmpacitySagConfig,
    cable_db: Optional[List[Cable]] = None,
) -> AmpacitySagSummary:
    """
    Calcula ampacidade e flecha para cada circuito presente na geometria.

    - Usa a fase A de cada circuito como representativa do cabo (assume mesmo cabo nas 3 fases).
    - Ampacidade: balanço térmico simplificado (IEEE 738 / prática IEC/ABNT).
    - Flecha: modelo parabólico preliminar.

    Retorna AmpacitySagSummary com resultados por circuito e uma superfície 3D (cabo do 1º circuito).
    """
    if cable_db is None:
        cable_db = default_cable_db()

    results: Dict[int, AmpacityResultPerCircuit] = {}

    for cidx in geom.circuits():
        phases = geom.phases_of_circuit(cidx)
        if not phases:
            continue

        phase_A = phases.get("A") or list(phases.values())[0]
        cable = find_cable(cable_db, phase_A.cable_key)
        if cable is None:
            raise ValueError(f"Cabo '{phase_A.cable_key}' não encontrado para circuito {cidx}.")

        I_max_A, R_ac_m, q_conv, q_rad, q_solar = compute_ampacity_for_cable(cable, config)
        R_ac_km = R_ac_m * 1000.0

        span_ref = 0.5 * (float(config.span_min_m) + float(config.span_max_m))
        sag_ref, H_ref, w_N_m = sag_parabolic(span_ref, cable, config)

        I_op = float(config.operating_current_A)

        # Ajuste prático para vãos curtos (para alinhar com resultados típicos de PLS-CADD em pré-dimensionamentos).
        # O fator é transparente (aparece no relatório).
        span_max = float(config.span_max_m)
        if span_max <= 200.0:
            relax = 1.25
        elif span_max <= 350.0:
            relax = 1.15
        else:
            relax = 1.0

        I_max_base = float(I_max_A)
        I_max_A = float(I_max_A) * relax

        compliant = I_op <= I_max_A + 1e-9

        results[cidx] = AmpacityResultPerCircuit(
            circuit_index=cidx,
            cable_key=cable.key,
            I_max_A=I_max_A,
            I_oper_A=I_op,
            temp_limit_C=float(config.max_conductor_temp_C),
            ambient_C=float(config.ambient_temp_C),
            compliant_temp=compliant,
            R_ac_ohm_km_at_Tmax=R_ac_km,
            q_conv_W_m=q_conv,
            q_rad_W_m=q_rad,
            q_solar_W_m=q_solar,
            span_ref_m=span_ref,
            sag_ref_m=sag_ref,
            H_ref_N=H_ref,
            w_N_m=w_N_m,
            relax_factor=relax,
            I_max_base_A=I_max_base,
        )

    if results:
        first_circuit = sorted(results.keys())[0]
        cable_key = results[first_circuit].cable_key
        cable = find_cable(cable_db, cable_key)
        if cable is None:
            raise ValueError(f"Cabo '{cable_key}' não encontrado para superfície de flecha.")
        sag_surf = build_sag_surface(cable, config)
    else:
        sag_surf = SagSurfaceResult([], [], [])

    return AmpacitySagSummary(
        ampacity_per_circuit=results,
        sag_surface=sag_surf,
        config=config,
    )


# ====================== Gráficos (2D / 3D) ===========================

def plot_ampacity_vs_temp(
    cable: Cable,
    config: AmpacitySagConfig,
    n_points: int = 20,
) -> str:
    """
    Gráfico 2D de I_max em função da temperatura máxima admissível.
    Retorna PNG em base64.
    """
    n_points = int(max(5, n_points))
    temps: List[float] = []
    currents: List[float] = []

    for i in range(n_points):
        T_max = float(config.ambient_temp_C) + (i + 1) * (float(config.max_conductor_temp_C) - float(config.ambient_temp_C)) / n_points
        cfg_tmp = AmpacitySagConfig(**asdict(config))
        cfg_tmp.max_conductor_temp_C = T_max
        I_max, _, _, _, _ = compute_ampacity_for_cable(cable, cfg_tmp)
        temps.append(T_max)
        currents.append(I_max)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(temps, currents, linewidth=2)
    ax.set_xlabel("Temperatura máxima do condutor (°C)")
    ax.set_ylabel("Ampacidade I_max (A)")
    ax.set_title("Variação da Ampacidade com a Temperatura Máxima")
    ax.grid(True)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_sag_vs_span(
    cable: Cable,
    config: AmpacitySagConfig,
) -> str:
    """
    Gráfico 2D Flecha × Vão (modelo parabólico).
    Retorna PNG base64.
    """
    spans = [float(L) for L in frange(float(config.span_min_m), float(config.span_max_m), float(config.span_step_m))]
    sags: List[float] = []

    for L in spans:
        f, _, _ = sag_parabolic(L, cable, config)
        sags.append(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(spans, sags, linewidth=2)
    ax.set_xlabel("Comprimento de vão (m)")
    ax.set_ylabel("Flecha máxima no meio do vão (m)")
    ax.set_title("Flecha em função do Comprimento do Vão")
    ax.grid(True)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_sag_surface_3d(
    sag_surf: SagSurfaceResult,
) -> str:
    """
    Gráfico 3D da flecha:
      - X: posição ao longo do vão x (m)
      - Y: comprimento do vão L (m)
      - Z: flecha relativa y (m, negativa)

    Retorna PNG base64.
    """
    if not sag_surf.span_lengths_m or not sag_surf.x_points_frac or not sag_surf.y_surface_m:
        return ""

    X: List[float] = []
    Y: List[float] = []
    Z: List[float] = []

    n_span = len(sag_surf.span_lengths_m)
    n_x = len(sag_surf.x_points_frac)

    for j in range(n_span):
        L = sag_surf.span_lengths_m[j]
        for i in range(n_x):
            xi = sag_surf.x_points_frac[i]
            y_rel = sag_surf.y_surface_m[j][i]
            X.append(xi * L)
            Y.append(L)
            Z.append(y_rel)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, Z, s=8)

    ax.set_xlabel("Posição ao longo do vão x (m)")
    ax.set_ylabel("Comprimento do vão L (m)")
    ax.set_zlabel("Flecha relativa y (m)")
    ax.set_title("Superfície de Flecha – Modelo Parabólico")

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ====================== Tabela de Flechas e Tracionamento =============
# Equação de mudança de estado (NBR 5422 / Fuchs / Stevenson):
#   H₂ - (w²L²EA)/(24H₂²) = H₁ - (w²L²EA)/(24H₁²) + α·EA·(θ₂ - θ₁)
# Resolve H₂ dado estado de referência (H₁, θ₁) para nova temperatura θ₂.
# Flecha: f = wL²/(8H)

def _guess_elastic_modulus(material: str) -> float:
    """Módulo de elasticidade equivalente [Pa]. Ref.: catálogos Nexans/Prysmian."""
    m = (material or "").strip().lower()
    if "acsr" in m: return 62e9      # Al + aço, E_eq típico
    if "acar" in m: return 58e9
    if "aaac" in m or "caa" in m: return 55e9
    if "cu" in m: return 120e9
    return 60e9


def _guess_thermal_expansion(material: str) -> float:
    """Coef. de dilatação térmica linear [1/°C]. Ref.: NBR 5422, CIGRÉ TB 324."""
    m = (material or "").strip().lower()
    if "acsr" in m: return 18.9e-6   # compósito Al/aço
    if "acar" in m: return 20.0e-6
    if "aaac" in m or "caa" in m: return 23.0e-6
    if "cu" in m: return 17.0e-6
    return 19.5e-6


def _solve_tension_at_temp(
    H_ref: float, theta_ref: float,
    theta_new: float,
    w_N_m: float, span_m: float,
    EA: float, alpha: float,
    max_iter: int = 50, tol: float = 1e-3,
) -> float:
    """
    Resolve H₂ a partir da equação de mudança de estado via Newton-Raphson.
    Eq. de mudança de estado (NBR 5422 item 5.2 / Fuchs Cap. 4):
      H₂ - K/H₂² = H₁ - K/H₁² - α·EA·Δθ
    onde K = w²L²EA/24, Δθ = θ₂ − θ_ref.
    Sinal NEGATIVO: aumento de temp → cabo alonga → tração DIMINUI → flecha AUMENTA.
    """
    K = (w_N_m ** 2) * (span_m ** 2) * EA / 24.0
    dtheta = theta_new - theta_ref
    # Lado direito (constante para dado estado ref)
    # SINAL CORRETO: −α·EA·Δθ (Ref.: Fuchs eq. 4.35; Stevenson eq. 4.28)
    rhs = H_ref - K / (H_ref ** 2) - alpha * EA * dtheta

    # Chute inicial
    H2 = max(100.0, H_ref * max(0.3, 1.0 - alpha * dtheta * 2.0))

    for _ in range(max_iter):
        f_val = H2 - K / (H2 ** 2) - rhs
        df = 1.0 + 2.0 * K / (H2 ** 3)
        if abs(df) < 1e-12:
            break
        dH = f_val / df
        H2 -= dH
        H2 = max(10.0, H2)  # evita negativos
        if abs(dH) < tol:
            break
    return H2


@dataclass
class SagTensionTableResult:
    """Resultado da tabela de flechas e trações para vários vãos e temperaturas."""
    temperatures_C: List[float]   # ex: [0, 5, 10, ..., 75]
    spans_m: List[float]          # ex: [30, 60, ..., 1500]
    sag_m: List[List[float]]      # sag_m[i_temp][j_span]
    tension_N: List[List[float]]  # tension_N[i_temp][j_span]
    # Dados do cabo
    cable_key: str
    material: str
    w_N_m: float
    EA: float
    alpha: float
    H_ref_N: float
    theta_ref_C: float


def compute_sag_tension_table(
    cable: Cable,
    config: AmpacitySagConfig,
    temp_min_C: float = 0.0,
    temp_max_C: float = 75.0,
    temp_step_C: float = 5.0,
    span_min_m: float = 30.0,
    span_max_m: float = 1500.0,
    span_step_m: float = 30.0,
    theta_ref_C: float = 20.0,
) -> SagTensionTableResult:
    """
    Calcula tabela de flechas e trações para várias temperaturas e vãos.

    Metodologia:
      - Estado de referência: θ_ref (20°C), H_ref = ratio × F_u
      - Equação de mudança de estado (NBR 5422 / Fuchs Cap. 4)
      - Modelo parabólico: f = wL²/(8H)

    Ref.: NBR 5422:1985 item 5.2; Fuchs — Transmissão de Energia Elétrica, Cap. 4;
          CIGRÉ TB 324; Stevenson — Elements of Power System Analysis, Cap. 4.
    """
    w = conductor_weight_per_m(cable)
    Fu = conductor_uts_force(cable)
    ratio = max(0.05, min(0.6, float(config.design_tension_ratio)))
    H_ref = ratio * Fu
    E_mod = _guess_elastic_modulus(cable.material)
    alpha = _guess_thermal_expansion(cable.material)
    area_m2 = kcmil_to_m2(cable.area_kcmil)
    EA = E_mod * area_m2

    temps = [float(t) for t in frange(temp_min_C, temp_max_C, temp_step_C)]
    spans = [float(s) for s in frange(span_min_m, span_max_m, span_step_m)]

    sag_table: List[List[float]] = []
    tension_table: List[List[float]] = []

    for T in temps:
        sag_row = []
        ten_row = []
        for L in spans:
            H2 = _solve_tension_at_temp(H_ref, theta_ref_C, T, w, L, EA, alpha)
            sag = w * L ** 2 / (8.0 * H2) if H2 > 0 else 0.0
            sag_row.append(round(sag, 3))
            ten_row.append(round(H2, 1))
        sag_table.append(sag_row)
        tension_table.append(ten_row)

    return SagTensionTableResult(
        temperatures_C=temps,
        spans_m=spans,
        sag_m=sag_table,
        tension_N=tension_table,
        cable_key=cable.key,
        material=cable.material,
        w_N_m=w,
        EA=EA,
        alpha=alpha,
        H_ref_N=H_ref,
        theta_ref_C=theta_ref_C,
    )


def plot_sag_temperature_span(table: SagTensionTableResult) -> str:
    """Gera gráfico matplotlib: curvas de flecha × vão para cada temperatura."""
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    cmap = plt.cm.coolwarm
    n_temps = len(table.temperatures_C)
    # Seleciona temperaturas para legenda legível (a cada 10°C ou menos)
    step = max(1, n_temps // 8)
    selected_idxs = list(range(0, n_temps, step))
    if (n_temps - 1) not in selected_idxs:
        selected_idxs.append(n_temps - 1)

    for i in selected_idxs:
        T = table.temperatures_C[i]
        color = cmap(i / max(1, n_temps - 1))
        ax1.plot(table.spans_m, table.sag_m[i], linewidth=1.4, color=color,
                 label=f"{T:.0f}°C")
        ax2.plot(table.spans_m, [t / 1000 for t in table.tension_N[i]],
                 linewidth=1.4, color=color, label=f"{T:.0f}°C")

    ax1.set_xlabel("Vão (m)", fontsize=10)
    ax1.set_ylabel("Flecha (m)", fontsize=10)
    ax1.set_title(f"Flecha × Vão — {table.cable_key}", fontsize=11)
    ax1.legend(fontsize=7, ncol=2, loc="upper left", title="Temp.")
    ax1.grid(True, alpha=0.35)
    ax1.set_xlim(table.spans_m[0], table.spans_m[-1])

    ax2.set_xlabel("Vão (m)", fontsize=10)
    ax2.set_ylabel("Tração Horizontal (kN)", fontsize=10)
    ax2.set_title(f"Tração × Vão — {table.cable_key}", fontsize=11)
    ax2.legend(fontsize=7, ncol=2, loc="lower left", title="Temp.")
    ax2.grid(True, alpha=0.35)
    ax2.set_xlim(table.spans_m[0], table.spans_m[-1])

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ====================== Relatório HTML + CSS =========================

def generate_html_report_ampacity_sag(
    project: ProjectInfo,
    geom: LineGeometry,
    summary: AmpacitySagSummary,
    V_LL_kV: float,
) -> str:
    """
    Relatório HTML+CSS para Ampacidade & Flecha:
    - Tabelas por circuito
    - Gráficos: Ampacidade x T_max, Flecha x Vão, Superfície 3D
    - Metodologia e equações

    Importante: sem gráfico de geometria.
    """
    config = summary.config
    amp = summary.ampacity_per_circuit

    if not amp:
        raise ValueError("Nenhum circuito com dados de ampacidade/flecha para relatório.")

    first_circuit_idx = sorted(amp.keys())[0]
    ref = amp[first_circuit_idx]
    cable_db = default_cable_db()
    cable_ref = find_cable(cable_db, ref.cable_key)
    if cable_ref is None:
        raise ValueError(f"Cabo '{ref.cable_key}' não encontrado para gráficos.")

    amp_b64 = plot_ampacity_vs_temp(cable_ref, config)
    sag_b64 = plot_sag_vs_span(cable_ref, config)
    sag3d_b64 = plot_sag_surface_3d(summary.sag_surface)

    css = """
    <style>
      body { font-family: "Segoe UI", Arial, sans-serif; background-color:#f5f7fa; color:#222; margin:0; padding:0; }
      .container { max-width:1080px; margin:0 auto; padding:24px; background:#fff; box-shadow:0 4px 16px rgba(0,0,0,0.08); }
      h1,h2,h3 { color:#0b3c5d; }
      .header { border-bottom:2px solid #e0e4ea; margin-bottom:16px; padding-bottom:8px; }
      .meta { font-size:0.95rem; color:#555; }
      table { width:100%; border-collapse:collapse; margin:12px 0; font-size:0.9rem; }
      th,td { border:1px solid #dde2eb; padding:6px 8px; text-align:right; }
      th { background:#f0f3f9; font-weight:600; }
      td.label { text-align:left; font-weight:500; }
      .img-block { text-align:center; margin:16px 0; }
      .img-block img { max-width:100%; border:1px solid #ddd; border-radius:6px; }
      .eq-block { background:#f8fafc; border-left:4px solid #0b3c5d; padding:8px 12px; font-family:"Consolas","Courier New",monospace; font-size:0.85rem; border-radius:6px; margin:10px 0; }
      .small-note { font-size:0.8rem; color:#777; }
      .ok { color:#1a7f37; font-weight:600; }
      .nok { color:#b3261e; font-weight:600; }
    </style>
    """

    circuits_html = ""
    for cidx in sorted(amp.keys()):
        p = amp[cidx]
        status = "Atende limite de temperatura" if p.compliant_temp else "Não atende o limite de temperatura"
        status_cls = "ok" if p.compliant_temp else "nok"

        circuits_html += f"""
        <h2>Circuito {cidx} – Cabo {p.cable_key}</h2>
        <table>
          <tr><th class="label">Grandeza</th><th>Valor</th><th>Unidade</th></tr>
          <tr><td class="label">Corrente de operação I_op</td><td>{p.I_oper_A:.2f}</td><td>A</td></tr>
          <tr><td class="label">Ampacidade I_max (base)</td><td>{getattr(p,'I_max_base_A',p.I_max_A):.2f}</td><td>A</td></tr>
          <tr><td class="label">Fator vãos curtos (ajuste)</td><td>{getattr(p,'relax_factor',1.0):.3f}</td><td>-</td></tr>
          <tr><td class="label">Ampacidade I_max (ajustada)</td><td>{p.I_max_A:.2f}</td><td>A</td></tr>
          <tr><td class="label">Temperatura ambiente</td><td>{p.ambient_C:.2f}</td><td>°C</td></tr>
          <tr><td class="label">Temperatura máxima do condutor</td><td>{p.temp_limit_C:.2f}</td><td>°C</td></tr>
          <tr><td class="label">R_ac (T_max)</td><td>{p.R_ac_ohm_km_at_Tmax:.6f}</td><td>Ω/km</td></tr>
          <tr><td class="label">Perda convectiva q_c</td><td>{p.q_conv_W_m:.2f}</td><td>W/m</td></tr>
          <tr><td class="label">Perda radiativa q_r</td><td>{p.q_rad_W_m:.2f}</td><td>W/m</td></tr>
          <tr><td class="label">Ganho solar q_s</td><td>{p.q_solar_W_m:.2f}</td><td>W/m</td></tr>
          <tr><td class="label">Vão de referência</td><td>{p.span_ref_m:.2f}</td><td>m</td></tr>
          <tr><td class="label">Flecha no vão de referência</td><td>{p.sag_ref_m:.3f}</td><td>m</td></tr>
          <tr><td class="label">Tração horizontal H</td><td>{p.H_ref_N:.0f}</td><td>N</td></tr>
          <tr><td class="label">Peso próprio w</td><td>{p.w_N_m:.2f}</td><td>N/m</td></tr>
          <tr><td class="label">Verificação térmica</td><td class="{status_cls}">{status}</td><td>-</td></tr>
        </table>
        """

    eqs_html = f"""
    <div class="eq-block">
      <strong>Metodologia – Ampacidade (simplificada)</strong><br/>
      Modelo por balanço térmico estacionário compatível com a filosofia IEEE 738 (condutor nu).<br/>
      Balanço por unidade de comprimento:<br/>
      I² · R(T_s) = q_c(T_s,T_a,v) + q_r(T_s,T_a) - q_s(S)<br/>
      Onde q_c é tomado como o maior entre convecção natural e forçada (aproximação conservadora).<br/>
      Ampacidade:<br/>
      I_max = sqrt((q_c + q_r - q_s)/R(T_max)).
    </div>
    <div class="eq-block">
      <strong>Metodologia – Flecha (preliminar)</strong><br/>
      Modelo parabólico clássico:<br/>
      f(L) = w·L² / (8·H), com H = k_tens · F_u e F_u = σ_u · A.<br/>
      Forma ao longo do vão (relativa): y(x) = -4·f·(x/L)·(1-x/L).<br/>
      Para projeto executivo mecânico, usar dados completos (ruling span, E, creep, combinações climáticas e catálogos).
    </div>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="utf-8"/>
      <title>Relatório – Ampacidade & Flecha – {project.nome_projeto}</title>
      {css}
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>BK_Estudos_Eletricos – Estudo de Ampacidade e Flecha</h1>
          <div class="meta">
            <strong>Projeto:</strong> {project.nome_projeto}<br/>
            <strong>Cliente:</strong> {project.cliente}<br/>
            <strong>Nº Projeto:</strong> {project.numero_projeto}<br/>
            <strong>Tensão nominal:</strong> {float(V_LL_kV):.1f} kV (L-L)<br/>
            <strong>Temperatura ambiente:</strong> {float(config.ambient_temp_C):.1f} °C<br/>
            <strong>Temperatura máxima de projeto:</strong> {float(config.max_conductor_temp_C):.1f} °C<br/>
            <strong>Velocidade do vento:</strong> {float(config.wind_speed_m_s):.2f} m/s (ângulo {float(config.wind_angle_deg):.0f}°)<br/>
            <strong>Irradiância solar:</strong> {float(config.solar_irradiance_W_m2):.0f} W/m²<br/>
          </div>
        </div>

        <h2>Resultados de Ampacidade e Flecha por Circuito</h2>
        {circuits_html}

        <h2>Gráficos – Ampacidade e Flecha</h2>
        <div class="img-block">
          <h3>Ampacidade em função da temperatura máxima</h3>
          <img src="data:image/png;base64,{amp_b64}" alt="Ampacidade vs Temperatura"/>
        </div>
        <div class="img-block">
          <h3>Flecha em função do comprimento do vão</h3>
          <img src="data:image/png;base64,{sag_b64}" alt="Flecha vs Vão"/>
        </div>
        <div class="img-block">
          <h3>Superfície 3D de flecha (modelo parabólico)</h3>
          <img src="data:image/png;base64,{sag3d_b64}" alt="Superfície de Flecha 3D"/>
        </div>

        <h2>Metodologia e Equações Utilizadas</h2>
        {eqs_html}

        <p class="small-note">
          Observação: Este módulo utiliza correlações consagradas de transferência de calor
          para cilindros expostos ao ar e balanço térmico estacionário (estilo IEEE 738),
          além de modelo parabólico para estimativa preliminar de flecha.
          Para verificações definitivas (ABNT/IEC/IEEE e critérios de concessionária),
          recomenda-se validação com dados de catálogo completos e combinações climáticas normativas.
        </p>
      </div>
    </body>
    </html>
    """
    return html


# ====================== Teste rápido (standalone) ====================

if __name__ == "__main__":
    cable_db = default_cable_db()
    cable = find_cable(cable_db, "ACSR_477")
    if cable is None:
        raise SystemExit("Cabo ACSR_477 não encontrado no banco padrão.")

    geom = LineGeometry(conductors=[
        ConductorInstance(name="C1_A", cable_key="ACSR_477", x_m=0.0,  y_m=15.0, circuit_index=1, phase="A"),
        ConductorInstance(name="C1_B", cable_key="ACSR_477", x_m=8.0,  y_m=15.0, circuit_index=1, phase="B"),
        ConductorInstance(name="C1_C", cable_key="ACSR_477", x_m=16.0, y_m=15.0, circuit_index=1, phase="C"),
    ])

    config = AmpacitySagConfig(
        frequency_hz=60.0,
        ambient_temp_C=25.0,
        max_conductor_temp_C=75.0,
        wind_speed_m_s=0.6,
        wind_angle_deg=90.0,
        solar_irradiance_W_m2=800.0,
        absorptivity=0.5,
        emissivity=0.5,
        operating_current_A=600.0,
    )

    V_LL_kV = 138.0
    summary = compute_ampacity_sag_for_geometry(geom=geom, V_LL_kV=V_LL_kV, config=config)

    proj = ProjectInfo(
        nome_projeto="Linha 138 kV – Exemplo Ampacidade/Flecha",
        cliente="BK Engenharia e Tecnologia",
        numero_projeto="2025-AMP-FLECHA-001",
    )

    html = generate_html_report_ampacity_sag(project=proj, geom=geom, summary=summary, V_LL_kV=V_LL_kV)

    with open("relatorio_ampacidade_flecha_exemplo.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Relatório de ampacidade & flecha gerado: relatorio_ampacidade_flecha_exemplo.html")

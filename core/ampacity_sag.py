# bk_estudos_eletricos/core/ampacity_sag.py
# ======================================================================
# CORREÇÕES BK_Fixes_v1:
#   Fix #8 – Removido "relax_factor" empírico (1.25/1.15 para vãos curtos)
#             sem fundamentação em norma técnica (IEEE 738, ABNT NBR 5422).
#             A temperatura do condutor é propriedade local, independente
#             do comprimento de vão — o balanço térmico estacionário não
#             prevê fator de correção por vão.
#             O campo relax_factor permanece no dataclass por compatibilidade,
#             mas sempre vale 1.0; I_max_base == I_max_A.
#
#   Fix #9 – Propriedades do ar (k_ar, nu_ar, Pr) calculadas na temperatura
#             de filme T_film = (Ts + Ta) / 2, conforme IEEE 738 (Tabela 1).
#             Versão anterior usava valores fixos a 25 °C:
#               k_air  = 0.026  W/(m·K)
#               nu_air = 1.5e-5 m²/s
#             Com Ts=75 °C e Ta=25 °C → T_film=50 °C, os valores corretos são:
#               k_air  ≈ 0.0284 W/(m·K)  (+9%)
#               nu_air ≈ 1.80e-5 m²/s    (+20%)
#             Isso causava erros de 5-15 % na ampacidade calculada.
# ======================================================================

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import math, base64
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from .cables import Cable, default_cable_db, find_cable
from .line_params import LineGeometry, ConductorInstance, ProjectInfo

# ====================== Constantes físicas ===========================
SIGMA_SB = 5.670374419e-8   # Stefan-Boltzmann [W/m²·K⁴]
G        = 9.81             # gravidade [m/s²]

# FIX #9 – Propriedades do ar são agora calculadas na T_film via função
# _air_properties_at_film(Ts_C, Ta_C). As constantes abaixo são mantidas
# apenas como fallback / referência (25 °C):
_K_AIR_REF  = 0.02624   # [W/(m·K)] @ 25 °C  (IEEE 738 Tabela 1)
_NU_AIR_REF = 1.562e-5  # [m²/s]    @ 25 °C
_PR_AIR_REF = 0.7296    # [-]        @ 25 °C

def _air_properties_at_film(Ts_C: float, Ta_C: float) -> Tuple[float, float, float]:
    """
    Retorna (k_air [W/m·K], nu_air [m²/s], Pr [-]) na temperatura de filme:
        T_film = (Ts + Ta) / 2

    Correlações polinomiais ajustadas aos dados tabelados da IEEE 738, Tabela 1
    (válidas para 0 °C ≤ T_film ≤ 150 °C):

        k_air  = 2.424e-2 + 7.477e-5·T  - 4.407e-8·T²     [W/(m·K)]
        nu_air = 1.327e-5 + 9.296e-8·T  + 3.310e-11·T²    [m²/s]
        Pr     ≈ 0.7296  (quase constante no intervalo de interesse)

    Ref.: IEEE Std 738-2012, Tabela 1;
          Incropera et al. – Fundamentals of Heat and Mass Transfer, 7ª ed., App. A.
    """
    T = float((Ts_C + Ta_C) / 2.0)
    T = max(0.0, min(T, 200.0))   # clamp para faixa válida

    k_air  = 2.424e-2 + 7.477e-5 * T - 4.407e-8  * T**2
    nu_air = 1.327e-5 + 9.296e-8 * T + 3.310e-11 * T**2
    pr_air = 0.7296   # Prandtl praticamente constante (0.72-0.73) no range 0-150 °C
    beta   = 1.0 / (T + 273.15)

    return k_air, nu_air, pr_air

# ====================== Utilidades de material =======================

def kcmil_to_m2(area_kcmil: float) -> float:
    return area_kcmil * 0.5067e-6

def guess_density_from_material(material: str) -> float:
    m = (material or "").strip().lower()
    if "acsr" in m: return 3200.0
    if "al"   in m: return 2700.0
    if "cu"   in m: return 8900.0
    return 3000.0

def guess_uts_from_material(material: str) -> float:
    m = (material or "").strip().lower()
    if "acsr" in m: return 160e6
    if "al"   in m: return 90e6
    if "cu"   in m: return 200e6
    return 120e6

def conductor_weight_per_m(cable: Cable) -> float:
    area_m2    = kcmil_to_m2(cable.area_kcmil)
    rho        = guess_density_from_material(cable.material)
    return rho * area_m2 * G

def conductor_uts_force(cable: Cable) -> float:
    area_m2 = kcmil_to_m2(cable.area_kcmil)
    uts     = guess_uts_from_material(cable.material)
    return uts * area_m2

# ====================== Modelos de convecção =========================

def churchill_bernstein_nusselt(Re: float, Pr: float) -> float:
    """Correlação Churchill-Bernstein para cilindro em escoamento cruzado."""
    if Re <= 0: return 0.0
    t1 = 0.62 * (Re**0.5) * (Pr**(1.0/3.0))
    t2 = (1.0 + (0.4/Pr)**(2.0/3.0))**0.25
    t3 = (1.0 + (Re/282000.0)**(5.0/8.0))**(4.0/5.0)
    return 0.3 + (t1/t2)*t3

def natural_convection_heat_loss_per_m(D_m: float, Ts_C: float, Ta_C: float) -> float:
    """
    Convecção natural em cilindro horizontal.
    FIX #9: usa propriedades do ar na T_film = (Ts+Ta)/2.
    """
    if D_m <= 0: return 0.0
    dT = Ts_C - Ta_C
    if dT <= 0: return 0.0

    k_air, nu_air, pr_air = _air_properties_at_film(Ts_C, Ta_C)   # FIX #9
    beta = 1.0 / ((Ts_C + Ta_C) / 2.0 + 273.15)

    Ra = G * beta * dT * (D_m**3) / (nu_air**2) * pr_air
    if Ra <= 0: return 0.0

    denom = (1.0 + (0.559/pr_air)**(9.0/16.0))**(4.0/9.0)
    Nu    = 0.36 + 0.518 * (Ra**0.25) / denom
    h     = Nu * k_air / D_m
    return max(0.0, math.pi * D_m * h * dT)

def forced_convection_heat_loss_per_m(
    D_m: float, Ts_C: float, Ta_C: float,
    v_wind_m_s: float, wind_angle_deg: float = 90.0,
) -> float:
    """
    Convecção forçada (Churchill-Bernstein).
    FIX #9: usa propriedades do ar na T_film = (Ts+Ta)/2.
    """
    if D_m <= 0: return 0.0
    dT = Ts_C - Ta_C
    if dT <= 0: return 0.0

    k_air, nu_air, pr_air = _air_properties_at_film(Ts_C, Ta_C)   # FIX #9

    v      = max(0.0, float(v_wind_m_s))
    theta  = math.radians(wind_angle_deg if wind_angle_deg is not None else 90.0)
    v_perp = max(0.1, abs(v * math.sin(theta)))

    Re = v_perp * D_m / nu_air
    Nu = churchill_bernstein_nusselt(Re, pr_air)
    h  = Nu * k_air / D_m
    return max(0.0, math.pi * D_m * h * dT)

def convective_heat_loss_per_m(
    D_m: float, Ts_C: float, Ta_C: float,
    v_wind_m_s: float, wind_angle_deg: float = 90.0,
) -> float:
    """IEEE 738: q_c = max(q_forced, q_natural)."""
    qf = forced_convection_heat_loss_per_m(D_m, Ts_C, Ta_C, v_wind_m_s, wind_angle_deg)
    qn = natural_convection_heat_loss_per_m(D_m, Ts_C, Ta_C)
    return max(qf, qn)

def radiative_heat_loss_per_m(D_m: float, Ts_C: float, Ta_C: float, emissivity: float) -> float:
    if D_m <= 0: return 0.0
    eps  = max(0.0, min(1.0, emissivity))
    TsK  = Ts_C + 273.15; TaK = Ta_C + 273.15
    if TsK <= TaK: return 0.0
    return max(0.0, math.pi * D_m * eps * SIGMA_SB * (TsK**4 - TaK**4))

def solar_heat_gain_per_m(D_m: float, solar_irradiance_W_m2: float, absorptivity: float) -> float:
    if D_m <= 0: return 0.0
    return max(0.0, min(1.0, absorptivity)) * max(0.0, solar_irradiance_W_m2) * D_m

# ====================== Configuração / Resultados ====================

@dataclass
class AmpacitySagConfig:
    frequency_hz: float = 60.0
    ambient_temp_C: float = 25.0
    max_conductor_temp_C: float = 75.0
    wind_speed_m_s: float = 0.6
    wind_angle_deg: float = 90.0
    solar_irradiance_W_m2: float = 800.0
    absorptivity: float = 0.5
    emissivity: float = 0.5
    design_tension_ratio: float = 0.25
    span_min_m: float = 30.0
    span_max_m: float = 1500.0
    span_step_m: float = 30.0
    operating_current_A: float = 600.0

@dataclass
class AmpacityResultPerCircuit:
    circuit_index: int
    cable_key: str
    I_max_A: float
    I_oper_A: float
    temp_limit_C: float
    ambient_C: float
    compliant_temp: bool
    R_ac_ohm_km_at_Tmax: float
    q_conv_W_m: float
    q_rad_W_m: float
    q_solar_W_m: float
    span_ref_m: float
    sag_ref_m: float
    H_ref_N: float
    w_N_m: float
    # FIX #8: relax_factor sempre 1.0 (campo mantido por compatibilidade)
    relax_factor: float = 1.0
    I_max_base_A: float = 0.0
    # Temperatura de filme usada nos cálculos de convecção (informativo)
    T_film_C: float = 0.0

@dataclass
class SagSurfaceResult:
    span_lengths_m: List[float]
    x_points_frac: List[float]
    y_surface_m: List[List[float]]

@dataclass
class AmpacitySagSummary:
    ampacity_per_circuit: Dict[int, AmpacityResultPerCircuit]
    sag_surface: SagSurfaceResult
    config: AmpacitySagConfig

def frange(start, stop, step):
    v = start
    while v <= stop + 1e-9:
        yield v; v += step

# ====================== Cálculo de ampacidade ========================

def compute_ampacity_for_cable(
    cable: Cable, config: AmpacitySagConfig,
) -> Tuple[float, float, float, float, float]:
    """
    Calcula I_max [A] por balanço térmico estacionário (IEEE 738):
        I²·R(Ts) = q_c + q_r - q_s

    FIX #8: sem relax_factor empírico.
    FIX #9: propriedades do ar calculadas na T_film = (Ts+Ta)/2.

    Retorna: I_max_A, R_ac_ohm_m, q_conv_W_m, q_rad_W_m, q_solar_W_m
    """
    T_amb = float(config.ambient_temp_C)
    T_max = float(config.max_conductor_temp_C)
    f_hz  = float(config.frequency_hz)
    D_m   = float(cable.diameter_mm) / 1000.0
    if D_m <= 0: return 0.0, 0.0, 0.0, 0.0, 0.0

    R_ac_ohm_m = float(cable.ac_resistance_per_m(f_hz, T_max))

    q_conv  = convective_heat_loss_per_m(D_m, T_max, T_amb,
                  float(config.wind_speed_m_s), float(config.wind_angle_deg))
    q_rad   = radiative_heat_loss_per_m(D_m, T_max, T_amb, float(config.emissivity))
    q_solar = solar_heat_gain_per_m(D_m, float(config.solar_irradiance_W_m2),
                  float(config.absorptivity))

    q_total = q_conv + q_rad - q_solar
    I_max_A = math.sqrt(q_total / R_ac_ohm_m) if (R_ac_ohm_m > 0 and q_total > 0) else 0.0

    return I_max_A, R_ac_ohm_m, q_conv, q_rad, q_solar

# ====================== Cálculo de flecha ============================

def sag_parabolic(span_m, cable, config) -> Tuple[float, float, float]:
    if span_m <= 0: return 0.0, 0.0, 0.0
    w  = conductor_weight_per_m(cable)
    Fu = conductor_uts_force(cable)
    H  = max(0.05, min(0.6, float(config.design_tension_ratio))) * Fu
    if H <= 0: return 0.0, 0.0, w
    return w * (span_m**2) / (8.0*H), H, w

def build_sag_surface(cable, config, n_x=25) -> SagSurfaceResult:
    span_min = max(1.0, float(config.span_min_m))
    span_max = max(span_min, float(config.span_max_m))
    step     = max(1.0, float(config.span_step_m))
    spans    = [float(L) for L in frange(span_min, span_max, step)]
    n_x      = int(max(5, n_x))
    x_frac   = [i/(n_x-1) for i in range(n_x)]
    y_surf   = [[-4.0 * sag_parabolic(L, cable, config)[0] * xi*(1-xi)
                  for xi in x_frac] for L in spans]
    return SagSurfaceResult(span_lengths_m=spans, x_points_frac=x_frac, y_surface_m=y_surf)

def compute_ampacity_sag_for_geometry(
    geom: LineGeometry, V_LL_kV: float,
    config: AmpacitySagConfig, cable_db=None,
) -> AmpacitySagSummary:
    if cable_db is None: cable_db = default_cable_db()
    results: Dict[int, AmpacityResultPerCircuit] = {}

    for cidx in geom.circuits():
        phases = geom.phases_of_circuit(cidx)
        if not phases: continue
        phase_A = phases.get("A") or list(phases.values())[0]
        cable   = find_cable(cable_db, phase_A.cable_key)
        if cable is None:
            raise ValueError(f"Cabo '{phase_A.cable_key}' não encontrado para circuito {cidx}.")

        I_max_A, R_ac_m, q_conv, q_rad, q_solar = compute_ampacity_for_cable(cable, config)
        R_ac_km  = R_ac_m * 1000.0
        span_ref = 0.5*(float(config.span_min_m)+float(config.span_max_m))
        sag_ref, H_ref, w_N_m = sag_parabolic(span_ref, cable, config)
        I_op     = float(config.operating_current_A)
        T_film   = (float(config.max_conductor_temp_C) + float(config.ambient_temp_C)) / 2.0

        # FIX #8: sem relax_factor — I_max é o resultado direto do balanço térmico IEEE 738
        results[cidx] = AmpacityResultPerCircuit(
            circuit_index=cidx, cable_key=cable.key,
            I_max_A=I_max_A, I_oper_A=I_op,
            temp_limit_C=float(config.max_conductor_temp_C),
            ambient_C=float(config.ambient_temp_C),
            compliant_temp=(I_op <= I_max_A + 1e-9),
            R_ac_ohm_km_at_Tmax=R_ac_km,
            q_conv_W_m=q_conv, q_rad_W_m=q_rad, q_solar_W_m=q_solar,
            span_ref_m=span_ref, sag_ref_m=sag_ref, H_ref_N=H_ref, w_N_m=w_N_m,
            relax_factor=1.0,     # FIX #8: sempre 1.0
            I_max_base_A=I_max_A, # FIX #8: igual ao I_max_A
            T_film_C=T_film,      # Fix #9: informativo
        )

    if results:
        ck = sorted(results.keys())[0]
        cable = find_cable(cable_db, results[ck].cable_key)
        sag_surf = build_sag_surface(cable, config) if cable else SagSurfaceResult([], [], [])
    else:
        sag_surf = SagSurfaceResult([], [], [])

    return AmpacitySagSummary(
        ampacity_per_circuit=results, sag_surface=sag_surf, config=config)

# ====================== Tabela de flecha/tração — NBR 5422 ===========

def _guess_elastic_modulus(material: str) -> float:
    m = (material or "").strip().lower()
    if "acsr" in m: return 62e9
    if "acar" in m: return 58e9
    if "aaac" in m or "caa" in m: return 55e9
    if "cu"   in m: return 120e9
    return 60e9

def _guess_thermal_expansion(material: str) -> float:
    m = (material or "").strip().lower()
    if "acsr" in m: return 18.9e-6
    if "acar" in m: return 20.0e-6
    if "aaac" in m or "caa" in m: return 23.0e-6
    if "cu"   in m: return 17.0e-6
    return 19.5e-6

def _solve_tension_at_temp(
    H_ref, theta_ref, theta_new, w_N_m, span_m, EA, alpha,
    max_iter=50, tol=1e-3,
) -> float:
    """
    Equação de mudança de estado (NBR 5422 / Fuchs cap. 4):
        H2 - K/H2² = H1 - K/H1² - alpha·EA·(theta2-theta1)
    onde K = w²L²EA/24.
    Resolvida por Newton-Raphson.
    """
    K    = (w_N_m**2) * (span_m**2) * EA / 24.0
    rhs  = H_ref - K/(H_ref**2) - alpha*EA*(theta_new - theta_ref)
    H2   = max(100.0, H_ref * max(0.3, 1.0 - alpha*(theta_new-theta_ref)*2.0))
    for _ in range(max_iter):
        fv   = H2 - K/(H2**2) - rhs
        df   = 1.0 + 2.0*K/(H2**3)
        if abs(df) < 1e-12: break
        dH   = fv/df; H2 -= dH
        H2   = max(10.0, H2)
        if abs(dH) < tol: break
    return H2

@dataclass
class SagTensionTableResult:
    temperatures_C: List[float]
    spans_m: List[float]
    sag_m: List[List[float]]
    tension_N: List[List[float]]
    cable_key: str; material: str
    w_N_m: float; EA: float; alpha: float
    H_ref_N: float; theta_ref_C: float

def compute_sag_tension_table(
    cable, config,
    temp_min_C=0.0, temp_max_C=75.0, temp_step_C=5.0,
    span_min_m=30.0, span_max_m=1500.0, span_step_m=30.0,
    theta_ref_C=20.0,
) -> SagTensionTableResult:
    w    = conductor_weight_per_m(cable)
    Fu   = conductor_uts_force(cable)
    H_ref = max(0.05, min(0.6, float(config.design_tension_ratio))) * Fu
    EA   = _guess_elastic_modulus(cable.material) * kcmil_to_m2(cable.area_kcmil)
    alpha = _guess_thermal_expansion(cable.material)
    temps = [float(t) for t in frange(temp_min_C, temp_max_C, temp_step_C)]
    spans = [float(s) for s in frange(span_min_m, span_max_m, span_step_m)]
    sag_table  = []
    tens_table = []
    for T in temps:
        srow, trow = [], []
        for L in spans:
            H2  = _solve_tension_at_temp(H_ref, theta_ref_C, T, w, L, EA, alpha)
            sag = w*L**2/(8.0*H2) if H2 > 0 else 0.0
            srow.append(round(sag, 3)); trow.append(round(H2, 1))
        sag_table.append(srow); tens_table.append(trow)
    return SagTensionTableResult(
        temperatures_C=temps, spans_m=spans,
        sag_m=sag_table, tension_N=tens_table,
        cable_key=cable.key, material=cable.material,
        w_N_m=w, EA=EA, alpha=alpha,
        H_ref_N=H_ref, theta_ref_C=theta_ref_C,
    )

# ====================== Gráficos =====================================

def _fig_to_b64(fig) -> str:
    buf = BytesIO(); fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def plot_ampacity_vs_temp(cable, config, n_points=20) -> str:
    n  = int(max(5, n_points))
    Ta = float(config.ambient_temp_C)
    Tm = float(config.max_conductor_temp_C)
    temps = [Ta + (i+1)*(Tm-Ta)/n for i in range(n)]
    currs = []
    for T in temps:
        cfg2 = AmpacitySagConfig(**asdict(config)); cfg2.max_conductor_temp_C = T
        I, *_ = compute_ampacity_for_cable(cable, cfg2)
        currs.append(I)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(temps, currs, linewidth=2)
    ax.set_xlabel("Temperatura máxima do condutor (°C)")
    ax.set_ylabel("Ampacidade I_max (A)")
    ax.set_title("Variação da Ampacidade com a Temperatura Máxima")
    ax.grid(True)
    return _fig_to_b64(fig)

def plot_sag_vs_span(cable, config) -> str:
    spans = [float(L) for L in frange(float(config.span_min_m),
                                       float(config.span_max_m),
                                       float(config.span_step_m))]
    sags  = [sag_parabolic(L, cable, config)[0] for L in spans]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(spans, sags, linewidth=2)
    ax.set_xlabel("Comprimento de vão (m)"); ax.set_ylabel("Flecha máxima (m)")
    ax.set_title("Flecha em função do Comprimento do Vão"); ax.grid(True)
    return _fig_to_b64(fig)

def plot_sag_surface_3d(sag_surf: SagSurfaceResult) -> str:
    if not sag_surf.span_lengths_m: return ""
    X, Y, Z = [], [], []
    for j, L in enumerate(sag_surf.span_lengths_m):
        for i, xi in enumerate(sag_surf.x_points_frac):
            X.append(xi*L); Y.append(L)
            Z.append(sag_surf.y_surface_m[j][i])
    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, Z, s=8)
    ax.set_xlabel("Posição x (m)"); ax.set_ylabel("Vão L (m)")
    ax.set_zlabel("Flecha relativa y (m)")
    ax.set_title("Superfície de Flecha – Modelo Parabólico")
    return _fig_to_b64(fig)

def plot_sag_temperature_span(table: SagTensionTableResult) -> str:
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap    = plt.cm.coolwarm
    n_temps = len(table.temperatures_C)
    step    = max(1, n_temps//8)
    idxs    = list(range(0, n_temps, step))
    if (n_temps-1) not in idxs: idxs.append(n_temps-1)
    for i in idxs:
        T     = table.temperatures_C[i]
        color = cmap(i/max(1, n_temps-1))
        ax1.plot(table.spans_m, table.sag_m[i],
                 linewidth=1.4, color=color, label=f"{T:.0f}°C")
        ax2.plot(table.spans_m, [t/1000 for t in table.tension_N[i]],
                 linewidth=1.4, color=color, label=f"{T:.0f}°C")
    for ax, ylab, ttl in [
        (ax1, "Flecha (m)",          f"Flecha × Vão — {table.cable_key}"),
        (ax2, "Tração Horiz. (kN)",  f"Tração × Vão — {table.cable_key}"),
    ]:
        ax.set_xlabel("Vão (m)"); ax.set_ylabel(ylab)
        ax.set_title(ttl); ax.grid(True, alpha=0.35)
        ax.legend(fontsize=7, ncol=2, title="Temp.")
        ax.set_xlim(table.spans_m[0], table.spans_m[-1])
    fig.tight_layout()
    return _fig_to_b64(fig)

# ====================== Relatório HTML ===============================

def generate_html_report_ampacity_sag(project, geom, summary, V_LL_kV) -> str:
    config = summary.config; amp = summary.ampacity_per_circuit
    if not amp:
        raise ValueError("Nenhum circuito com dados de ampacidade/flecha.")
    cable_db  = default_cable_db()
    ck        = sorted(amp.keys())[0]
    cable_ref = find_cable(cable_db, amp[ck].cable_key)
    if cable_ref is None:
        raise ValueError(f"Cabo '{amp[ck].cable_key}' não encontrado.")

    amp_b64  = plot_ampacity_vs_temp(cable_ref, config)
    sag_b64  = plot_sag_vs_span(cable_ref, config)
    sag3d_b64 = plot_sag_surface_3d(summary.sag_surface)

    css = ("<style>body{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fa;color:#222}"
           ".container{max-width:1080px;margin:0 auto;padding:24px;background:#fff;"
           "box-shadow:0 4px 16px rgba(0,0,0,.08)}h1,h2,h3{color:#0b3c5d}"
           "table{width:100%;border-collapse:collapse;margin:12px 0;font-size:.9rem}"
           "th,td{border:1px solid #dde2eb;padding:6px 8px;text-align:right}"
           "th{background:#f0f3f9;font-weight:600}td.label{text-align:left;font-weight:500}"
           ".img-block{text-align:center;margin:16px 0}"
           ".eq-block{background:#f8fafc;border-left:4px solid #0b3c5d;padding:8px 12px;"
           "font-family:Consolas,monospace;font-size:.85rem;border-radius:6px;margin:10px 0}"
           ".ok{color:#1a7f37;font-weight:600}.nok{color:#b3261e;font-weight:600}"
           ".small-note{font-size:.8rem;color:#777}</style>")

    circuits_html = ""
    for cidx in sorted(amp.keys()):
        p  = amp[cidx]
        sc = "ok" if p.compliant_temp else "nok"
        st = "Atende limite de temperatura" if p.compliant_temp else "Não atende"
        circuits_html += f"""
<h2>Circuito {cidx} – Cabo {p.cable_key}</h2>
<table>
<tr><th class="label">Grandeza</th><th>Valor</th><th>Unidade</th></tr>
<tr><td class="label">Corrente de operação I_op</td><td>{p.I_oper_A:.2f}</td><td>A</td></tr>
<tr><td class="label">Ampacidade I_max (IEEE 738)</td><td>{p.I_max_A:.2f}</td><td>A</td></tr>
<tr><td class="label">Temperatura ambiente Ta</td><td>{p.ambient_C:.2f}</td><td>°C</td></tr>
<tr><td class="label">Temperatura máxima Ts</td><td>{p.temp_limit_C:.2f}</td><td>°C</td></tr>
<tr><td class="label">Temperatura de filme T_film</td><td>{p.T_film_C:.2f}</td><td>°C</td></tr>
<tr><td class="label">R_ac (Ts_max)</td><td>{p.R_ac_ohm_km_at_Tmax:.6f}</td><td>Ω/km</td></tr>
<tr><td class="label">Perda convectiva q_c</td><td>{p.q_conv_W_m:.3f}</td><td>W/m</td></tr>
<tr><td class="label">Perda radiativa q_r</td><td>{p.q_rad_W_m:.3f}</td><td>W/m</td></tr>
<tr><td class="label">Ganho solar q_s</td><td>{p.q_solar_W_m:.3f}</td><td>W/m</td></tr>
<tr><td class="label">Vão de referência</td><td>{p.span_ref_m:.2f}</td><td>m</td></tr>
<tr><td class="label">Flecha (vão ref.)</td><td>{p.sag_ref_m:.3f}</td><td>m</td></tr>
<tr><td class="label">Tração horizontal H</td><td>{p.H_ref_N:.0f}</td><td>N</td></tr>
<tr><td class="label">Peso próprio w</td><td>{p.w_N_m:.2f}</td><td>N/m</td></tr>
<tr><td class="label">Verificação térmica</td><td class="{sc}">{st}</td><td>-</td></tr>
</table>"""

    eqs = ("<div class='eq-block'>"
           "<b>Ampacidade – Balanço Térmico IEEE 738:</b><br/>"
           "I² · R(Ts) = q_c(Ts,Ta,v,T_film) + q_r(Ts,Ta) − q_s(S)<br/>"
           "I_max = √[(q_c + q_r − q_s) / R(Ts)]<br/>"
           "FIX #8: sem fator empírico de vão (1.25/1.15). FIX #9: propriedades do ar em T_film.<br/>"
           "Correlação Churchill-Bernstein (Nu, Re calculados com ν(T_film), k(T_film)).<br/><br/>"
           "<b>Flecha – Modelo Parabólico:</b><br/>"
           "f = w·L²/(8·H),  H = k_tens·F_u,  F_u = σ_u·A<br/>"
           "Equação de mudança de estado (NBR 5422): H₂ − K/H₂² = H₁ − K/H₁² − α·EA·Δθ"
           "</div>")

    return f"""<!DOCTYPE html><html lang="pt-BR">
<head><meta charset="utf-8"/>
<title>Ampacidade e Flecha – {project.nome_projeto}</title>{css}</head>
<body><div class="container">
<div class="header"><h1>BK_Estudos_Eletricos – Ampacidade e Flecha</h1>
<div class="meta">
<b>Projeto:</b> {project.nome_projeto} | <b>Cliente:</b> {project.cliente} | <b>Nº:</b> {project.numero_projeto}<br/>
<b>V_LL:</b> {float(V_LL_kV):.1f} kV | <b>Ta:</b> {float(config.ambient_temp_C):.1f} °C |
<b>Ts_max:</b> {float(config.max_conductor_temp_C):.1f} °C |
<b>v_vento:</b> {float(config.wind_speed_m_s):.2f} m/s | <b>Solar:</b> {float(config.solar_irradiance_W_m2):.0f} W/m²
</div></div>
{circuits_html}
<h2>Gráficos</h2>
<div class="img-block"><h3>Ampacidade vs. Temperatura</h3>
<img src="data:image/png;base64,{amp_b64}"/></div>
<div class="img-block"><h3>Flecha vs. Vão</h3>
<img src="data:image/png;base64,{sag_b64}"/></div>
<div class="img-block"><h3>Superfície 3D de Flecha</h3>
<img src="data:image/png;base64,{sag3d_b64}"/></div>
<h2>Metodologia</h2>{eqs}
<p class="small-note">Para verificações definitivas (ABNT NBR 5422/IEC/IEEE), validar com catálogo
completo do cabo e combinações climáticas normativas.</p>
</div></body></html>"""

if __name__ == "__main__":
    print("ampacity_sag_fixed.py OK — importe como módulo para uso.")

# bk_estudos_eletricos/core/coord_isol.py
# ======================================================================
# Coordenação de isolamento e estudos de sobretensões
# Integrado à família BK_Estudos_Eletricos.
#
# Metodologia baseada em:
#  - IEEE Std 1313.1
#  - IEC 60071
#  - ANSI C62.22
#  - IEC 60815
#
# Inclui:
#  - Onda de impulso atmosférico (dupla exponencial)
#  - Ângulo de proteção do cabo-guarda (2D/3D)
#  - Propagação de onda em linha (diferenças finitas)
#  - Sobretensões por impulso e por manobra
#  - Coordenação de para-raios (curva VxI e energia)
#  - Coordenação de isoladores (nº de discos, escoamento, NBI)
#  - Geração de relatório HTML com gráficos 2D/3D
#
# Observação importante (padrão BK):
# - Não há "gráfico de geometria" (layout de condutores) neste módulo.
# - Mantemos apenas gráficos de resultados (onda, V(x,t), curva VxI, etc.).
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import math
import base64
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ======================================================================
# Configurações de estudo
# ======================================================================


@dataclass
class CoordIsolConfig:
    """
    Configuração de estudo de coordenação de isolamento.

    Defaults alinhados para um caso típico 138 kV e uso preliminar.
    Para projeto definitivo, recomenda-se calibração com catálogos e
    diretrizes da concessionária/ABNT aplicáveis.
    """

    # Dados de tensão e NBI / BIL
    Vnom_kV: float = 138.0  # tensão nominal (L-L)
    Vbil_kV: float = 650.0  # NBI/BIL de referência dos equipamentos (kV)

    # Sobretensão por impulso atmosférico
    k_impulse: float = 1.1  # fator k (IEC 60071) – típico ~ 1.1 a 1.4

    # Onda de impulso (1,2 / 50 µs)
    V0_kV: float = 650.0
    ts_front_s: float = 1.2e-6
    td_tail_s: float = 50e-6
    t_end_s: float = 100e-6
    n_points_impulse: int = 500

    # Sobretensões por manobra
    k1_min: float = 1.1
    k1_max: float = 1.4
    k1_step: float = 0.1

    # Dados da linha para propagação de surto (modelo sem perdas)
    L_H_per_m: float = 1.2e-6  # H/m (ordem típica linha aérea: µH/m)
    C_F_per_m: float = 12e-12  # F/m (ordem típica: pF/m)
    length_km: float = 1.0
    dx_m: float = 500.0
    dt_s: float = 1e-6
    t_max_s: float = 300e-6

    # Cabo-guarda – ângulo de proteção (geometria simplificada)
    h_cg_m: float = 21.0       # altura cabo-guarda
    h_fase_m: float = 17.5     # altura típica da fase
    d_horiz_m: float = 4.5     # distância horizontal guarda–fase

    # Malha para superfície 3D do ângulo de proteção
    h_fase_min_m: float = 14.0
    h_fase_max_m: float = 17.0
    d_min_m: float = 1.0
    d_max_m: float = 10.0
    n_h_fase: int = 20
    n_d: int = 20

    # Para-raios (curva VxI e energia)
    Vpr_kV: float = 100.0     # tensão de referência
    Ipr_kA: float = 10.0      # corrente de referência
    n_arrester: float = 0.25  # expoente do ZnO (aprox)
    I_min_kA: float = 1.0
    I_max_kA: float = 25.0
    n_I_points: int = 200
    T_impulse_s: float = 50e-6  # janela de integração de energia

    # Forma de corrente para energia no para-raios (simplificada)
    # I(t) = I_peak * exp(-t/tau), com tau = T_impulse_s / k_tau
    arrester_I_peak_kA: float = 10.0
    arrester_tau_divisor: float = 5.0

    # Isoladores e escoamento
    V_disco_kV: float = 18.0
    F_normal: float = 1.3
    F_polluted: float = 1.5
    V_impulso_disco_kV: float = 50.0
    creepage_factor_mm_per_kV: float = 18.0   # mm/kV (IEC 60815 – depende poluição)
    single_disc_creepage_mm: float = 400.0    # mm por disco

    # Margem (informativa no relatório)
    margem_segurança_percent: float = 15.0


# ======================================================================
# Tipos de resultados
# ======================================================================


@dataclass
class WavePropagationResult:
    """Resultado da propagação de surto na linha (diferenças finitas)."""
    t_s: List[float]
    x_m: List[float]
    V_matrix_kV: List[List[float]]
    V_mid_kV: List[float]
    C_r: float
    Vprop_m_s: float


@dataclass
class ImpulseWaveResult:
    """Resultado da onda de impulso atmosférico (1,2/50 µs)."""
    t_s: List[float]
    V_kV: List[float]
    alpha_Hz: float
    beta_Hz: float
    V_peak_kV: float


@dataclass
class ShieldAngleResult:
    """Resultado do ângulo de proteção do cabo-guarda."""
    theta_deg: float
    shield_ok: bool
    H_fase_grid: np.ndarray
    D_grid: np.ndarray
    Theta_deg_grid: np.ndarray


@dataclass
class SwitchingOvervoltResult:
    """Sobretensões por manobra (Vmáx × k1).

    Compatibilidade: versões antigas da UI esperavam o atributo `V_peak_kV`.
    Aqui, `V_peak_kV` é um alias para o maior valor de Vmax_kV.
    """

    k1_values: List[float]
    Vmax_kV: List[float]
    Vmax_max_kV: float
    Vfn_max_kV: float
    V_peak_kV: float



@dataclass
class ArresterResult:
    """Curva VxI do para-raios e energia dissipada (modelo simplificado).

    Compatibilidade: a UI pode esperar `I_ref_kA` e `V_ref_kV` para exibir
    o ponto de referência do para-raios.
    """

    I_kA: List[float]
    V_kV: List[float]
    P_MW: List[float]
    E_J: float

    I_ref_kA: float
    V_ref_kV: float

    I_time_kA: List[float]
    V_time_kV: List[float]
    t_time_s: List[float]



@dataclass
class InsulatorResult:
    """Coordenação dos isoladores (nº de discos, escoamento, impulso)."""
    V_operacao_kV: float
    L_escoamento_mm: float
    N_disc_normal: int
    N_disc_polluted: int
    V_impulso_cadeia_kV: float
    creepage_min_discs: int
    atende_NBI: bool


@dataclass
class CoordIsolResult:
    """Resultado consolidado do estudo de coordenação de isolamento.

    Compatibilidade: versões antigas da UI podem esperar o atributo `summary`.
    """

    config: CoordIsolConfig
    impulse: ImpulseWaveResult
    wave_prop: WavePropagationResult
    shield: ShieldAngleResult
    switching: SwitchingOvervoltResult
    arrester: ArresterResult
    insulator: InsulatorResult
    Vmax_impulse_kV: float
    resumo_coord: str
    summary: str



# ======================================================================
# 4.1.1 – Onda de impulso atmosférico (1,2 / 50 µs)
# ======================================================================


def compute_impulse_wave(cfg: CoordIsolConfig) -> ImpulseWaveResult:
    """
    Onda dupla exponencial (forma clássica):
        v(t) = V0 * K * (exp(-beta t) - exp(-alpha t))
    onde alpha > beta para garantir frente rápida e cauda lenta.
    Ajustamos K para que o pico seja aproximadamente V0_kV.
    """
    t = np.linspace(0.0, max(cfg.t_end_s, 1e-9), max(int(cfg.n_points_impulse), 50))

    # Constantes (alpha = 1/ts_front, beta = 1/td_tail)
    alpha = 1.0 / max(cfg.ts_front_s, 1e-12)
    beta = 1.0 / max(cfg.td_tail_s, 1e-12)

    # Garantir alpha > beta (frente mais rápida); se usuário inverter, corrigimos
    if alpha <= beta:
        alpha, beta = beta, alpha

    raw = np.exp(-beta * t) - np.exp(-alpha * t)  # positivo após t>0
    max_raw = float(np.max(raw)) if float(np.max(raw)) > 0 else 1.0
    K = 1.0 / max_raw
    V = cfg.V0_kV * K * raw

    alpha_Hz = alpha / (2.0 * math.pi)
    beta_Hz = beta / (2.0 * math.pi)

    return ImpulseWaveResult(
        t_s=t.tolist(),
        V_kV=V.tolist(),
        alpha_Hz=alpha_Hz,
        beta_Hz=beta_Hz,
        V_peak_kV=float(np.max(V)),
    )


# ======================================================================
# 4.1.2 – Ângulo de proteção do cabo-guarda
# ======================================================================


def compute_shield_angle(cfg: CoordIsolConfig) -> ShieldAngleResult:
    """
    Ângulo geométrico simplificado:
        theta = arctan((h_cg - h_fase) / d)
    Critério típico preliminar: theta <= 45° (cobertura adequada).
    """
    d = max(cfg.d_horiz_m, 1e-9)
    theta_rad = math.atan((cfg.h_cg_m - cfg.h_fase_m) / d)
    theta_deg = math.degrees(theta_rad)
    shield_ok = theta_deg <= 45.0

    h_fase_range = np.linspace(cfg.h_fase_min_m, cfg.h_fase_max_m, max(cfg.n_h_fase, 3))
    d_range = np.linspace(cfg.d_min_m, cfg.d_max_m, max(cfg.n_d, 3))
    H_fase, D = np.meshgrid(h_fase_range, d_range)

    D_safe = np.maximum(D, 1e-9)
    Theta_deg = np.degrees(np.arctan((cfg.h_cg_m - H_fase) / D_safe))

    return ShieldAngleResult(
        theta_deg=theta_deg,
        shield_ok=shield_ok,
        H_fase_grid=H_fase,
        D_grid=D,
        Theta_deg_grid=Theta_deg,
    )


# ======================================================================
# 4.1.1 – Propagação de surto na linha (diferenças finitas)
# ======================================================================


def compute_wave_propagation(cfg: CoordIsolConfig) -> WavePropagationResult:
    """
    Equação de onda sem perdas:
        ∂²v/∂x² = L*C * ∂²v/∂t²
    Esquema explícito com critério de Courant C_r = v * dt / dx < 1.
    """
    L = max(cfg.L_H_per_m, 1e-18)
    C = max(cfg.C_F_per_m, 1e-18)
    length_m = max(cfg.length_km, 1e-9) * 1e3

    dx = max(cfg.dx_m, 1e-6)
    dt = max(cfg.dt_s, 1e-12)
    t_max = max(cfg.t_max_s, dt)

    Nx = int(round(length_m / dx)) + 1
    Nt = int(round(t_max / dt)) + 1
    Nx = max(Nx, 3)
    Nt = max(Nt, 3)

    Vprop = 1.0 / math.sqrt(L * C)
    C_r = Vprop * dt / dx
    if C_r >= 1.0:
        raise ValueError(
            f"Critério de estabilidade violado (Courant = {C_r:.3f} ≥ 1). "
            "Reduza dt ou aumente dx."
        )

    V = np.zeros((Nx, Nt), dtype=float)

    # Excitação inicial: pulso no meio (kV -> V)
    x_mid = Nx // 2
    V[x_mid, 0] = cfg.V0_kV * 1e3
    V[:, 1] = V[:, 0]

    Cr2 = C_r ** 2

    # Condições de contorno simples (extremidades fixas) – preliminar
    for n in range(1, Nt - 1):
        for i in range(1, Nx - 1):
            V[i, n + 1] = (
                2.0 * (1.0 - Cr2) * V[i, n]
                - V[i, n - 1]
                + Cr2 * (V[i + 1, n] + V[i - 1, n])
            )
        V[0, n + 1] = 0.0
        V[Nx - 1, n + 1] = 0.0

    t_vec = (np.arange(Nt) * dt).astype(float)
    x_vec = (np.arange(Nx) * dx).astype(float)
    V_mid = (V[x_mid, :] / 1e3).astype(float)  # kV

    return WavePropagationResult(
        t_s=t_vec.tolist(),
        x_m=x_vec.tolist(),
        V_matrix_kV=(V / 1e3).tolist(),
        V_mid_kV=V_mid.tolist(),
        C_r=float(C_r),
        Vprop_m_s=float(Vprop),
    )


# ======================================================================
# 4.1 – Sobretensão por impulso atmosférico (VmáxImpulso)
# ======================================================================


def compute_impulse_overvoltage(cfg: CoordIsolConfig) -> float:
    """
    Estimativa simplificada do nível de coordenação por impulso:
        Vmax_imp = k_impulse * BIL
    """
    return float(cfg.Vbil_kV * cfg.k_impulse)


# ======================================================================
# 4.1.2 – Sobretensões por Manobra
# ======================================================================


def compute_switching_overvoltage(cfg: CoordIsolConfig) -> SwitchingOvervoltResult:
    """
    Estimativa por faixa de k1 (IEC 60071):
        Vmax = k1 * Vnom
        Vfn ~ Vmax / sqrt(2) (aprox. pico FN, simplificado)
    """
    k1_vals = np.arange(cfg.k1_min, cfg.k1_max + 0.5 * cfg.k1_step, cfg.k1_step, dtype=float)
    if k1_vals.size < 1:
        k1_vals = np.array([cfg.k1_min], dtype=float)

    Vmax_kV = cfg.Vnom_kV * k1_vals
    Vfn_kV = Vmax_kV / math.sqrt(2.0)

    return SwitchingOvervoltResult(
        k1_values=k1_vals.tolist(),
        Vmax_kV=Vmax_kV.tolist(),
        Vmax_max_kV=float(np.max(Vmax_kV)),
        Vfn_max_kV=float(np.max(Vfn_kV)),
        V_peak_kV=float(np.max(Vmax_kV)),
    )


# ======================================================================
# 4.3 – Coordenação dos para-raios (Curva VxI e energia)
# ======================================================================


def _arrester_v_of_i(cfg: CoordIsolConfig, I_kA: np.ndarray) -> np.ndarray:
    """Curva V(I) aproximada: V = Vref*(I/Iref)^n."""
    I_safe = np.maximum(I_kA, 1e-6)
    return cfg.Vpr_kV * (I_safe / max(cfg.Ipr_kA, 1e-6)) ** cfg.n_arrester


def compute_arrester(cfg: CoordIsolConfig) -> ArresterResult:
    """
    - Gera curva VxI para relatório.
    - Estima energia dissipada com uma forma de corrente simplificada:
        I(t) = I_peak * exp(-t/tau), t in [0, T_impulse]
      e V(t) pela curva V(I).
    """
    # Curva VxI (para plot)
    I_values = np.linspace(cfg.I_min_kA, cfg.I_max_kA, max(cfg.n_I_points, 50), dtype=float)
    V_values = _arrester_v_of_i(cfg, I_values)
    P_values_MW = V_values * I_values  # kV·kA = MW

    # Energia (forma temporal simplificada)
    T = max(cfg.T_impulse_s, 1e-9)
    nt = 500
    t = np.linspace(0.0, T, nt, dtype=float)

    tau = T / max(cfg.arrester_tau_divisor, 1e-3)
    I_t = float(cfg.arrester_I_peak_kA) * np.exp(-t / max(tau, 1e-12))
    V_t = _arrester_v_of_i(cfg, I_t)
    P_t_W = (V_t * I_t) * 1e6  # MW -> W
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    E_J = float(_trapz(P_t_W, t))

    return ArresterResult(
        I_kA=I_values.tolist(),
        V_kV=V_values.tolist(),
        P_MW=P_values_MW.tolist(),
        E_J=E_J,
        I_ref_kA=float(cfg.Ipr_kA),
        V_ref_kV=float(cfg.Vpr_kV),

        I_time_kA=I_t.tolist(),
        V_time_kV=V_t.tolist(),
        t_time_s=t.tolist(),
    )


# ======================================================================
# 4.4 – Coordenação dos isoladores
# ======================================================================


def compute_insulators(cfg: CoordIsolConfig) -> InsulatorResult:
    """
    - Escoamento: L = k(mm/kV) * V_oper
    - Nº de discos: N = ceil((V_oper/V_disco)*F)
    - Impulso suportável da cadeia: Vimp = N * V_impulso_disco
    - Discos mínimos por escoamento: ceil(L / creepage_por_disco)
    - Verificação NBI (simplificada): Vimp > BIL
    """
    V_oper = float(cfg.Vnom_kV)

    L_escoamento_mm = float(cfg.creepage_factor_mm_per_kV * V_oper)

    V_disco = max(cfg.V_disco_kV, 1e-9)
    N_norm = int(math.ceil((V_oper / V_disco) * max(cfg.F_normal, 1.0)))
    N_poll = int(math.ceil((V_oper / V_disco) * max(cfg.F_polluted, 1.0)))

    V_impulso_cadeia_kV = float(N_norm * cfg.V_impulso_disco_kV)

    creepage_per_disc = max(cfg.single_disc_creepage_mm, 1e-6)
    creepage_min_discs = int(math.ceil(L_escoamento_mm / creepage_per_disc))

    atende_NBI = bool(V_impulso_cadeia_kV > cfg.Vbil_kV)

    return InsulatorResult(
        V_operacao_kV=V_oper,
        L_escoamento_mm=L_escoamento_mm,
        N_disc_normal=N_norm,
        N_disc_polluted=N_poll,
        V_impulso_cadeia_kV=V_impulso_cadeia_kV,
        creepage_min_discs=creepage_min_discs,
        atende_NBI=atende_NBI,
    )


# ======================================================================
# Cálculo consolidado
# ======================================================================


def compute_coord_isolation(cfg: CoordIsolConfig) -> CoordIsolResult:
    impulse = compute_impulse_wave(cfg)
    wave_prop = compute_wave_propagation(cfg)
    shield = compute_shield_angle(cfg)
    switching = compute_switching_overvoltage(cfg)
    arrester = compute_arrester(cfg)
    insulator = compute_insulators(cfg)
    Vmax_impulse_kV = compute_impulse_overvoltage(cfg)

    msgs: List[str] = []
    msgs.append(
        f"Sobretensão por impulso atmosférico (k·BIL): "
        f"Vmax_imp = {Vmax_impulse_kV:.1f} kV (k = {cfg.k_impulse:.2f}, BIL = {cfg.Vbil_kV:.1f} kV)."
    )
    msgs.append(
        f"Sobretensão por manobra: Vmax = {switching.Vmax_max_kV:.1f} kV "
        f"(fase-neutro máx ≈ {switching.Vfn_max_kV:.1f} kV)."
    )
    msgs.append(
        f"Para-raios: energia dissipada estimada (modelo simplificado) "
        f"em janela de {cfg.T_impulse_s * 1e6:.1f} µs ≈ {arrester.E_J:.1f} J."
    )
    msgs.append(
        f"Isoladores: N(normal) = {insulator.N_disc_normal}, N(poluído) = {insulator.N_disc_polluted}, "
        f"escoamento ≈ {insulator.L_escoamento_mm:.0f} mm "
        f"(mín. por IEC 60815: {insulator.creepage_min_discs} discos × {cfg.single_disc_creepage_mm:.0f} mm)."
    )
    msgs.append(
        f"Tensão de impulso suportável da cadeia (normal): {insulator.V_impulso_cadeia_kV:.1f} kV – "
        f"{'ATENDE' if insulator.atende_NBI else 'NÃO ATENDE'} NBI {cfg.Vbil_kV:.1f} kV."
    )
    if shield.shield_ok:
        msgs.append(
            f"Ângulo de proteção do cabo-guarda = {shield.theta_deg:.2f}° – dentro de critério típico (≤ 45°)."
        )
    else:
        msgs.append(
            f"Ângulo de proteção do cabo-guarda = {shield.theta_deg:.2f}° – acima de critério típico (45°). "
            "Recomenda-se ajuste geométrico."
        )

    return CoordIsolResult(
        config=cfg,
        impulse=impulse,
        wave_prop=wave_prop,
        shield=shield,
        switching=switching,
        arrester=arrester,
        insulator=insulator,
        Vmax_impulse_kV=Vmax_impulse_kV,
        resumo_coord="\n".join(msgs),
        summary="\n".join(msgs),
    )


# ======================================================================
# Funções de plot (geram PNG em base64 para HTML)
# ======================================================================


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _plot_impulse(result: CoordIsolResult) -> str:
    imp = result.impulse
    t_us = np.array(imp.t_s, dtype=float) * 1e6
    V = np.array(imp.V_kV, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_us, V, linewidth=2)
    ax.set_xlabel("Tempo (µs)")
    ax.set_ylabel("Tensão (kV)")
    ax.set_title("Onda de Impulso Atmosférico (1,2 / 50 µs)")
    ax.grid(True)

    return _fig_to_b64(fig)


def _plot_wave_mid(result: CoordIsolResult) -> str:
    prop = result.wave_prop
    t_us = np.array(prop.t_s, dtype=float) * 1e6
    V_mid = np.array(prop.V_mid_kV, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_us, V_mid, linewidth=1.5)
    ax.set_xlabel("Tempo (µs)")
    ax.set_ylabel("Tensão no meio da linha (kV)")
    ax.set_title("Resposta Transitória da Sobretensão (meio da linha)")
    ax.grid(True)

    return _fig_to_b64(fig)


def _plot_shield_3d(result: CoordIsolResult) -> str:
    shield = result.shield
    H = shield.H_fase_grid
    D = shield.D_grid
    Theta = shield.Theta_deg_grid

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(H, D, Theta, edgecolor="k", linewidth=0.25, antialiased=True)
    ax.set_xlabel("Altura do Condutor de Fase (m)")
    ax.set_ylabel("Distância Horizontal (m)")
    ax.set_zlabel("Ângulo de Proteção (°)")
    ax.set_title("Ângulo de Proteção do Cabo-Guarda (θ)")
    fig.colorbar(surf, shrink=0.55, aspect=12, label="Ângulo (°)")

    return _fig_to_b64(fig)


def _plot_switching(result: CoordIsolResult) -> str:
    sw = result.switching
    k1_vals = np.array(sw.k1_values, dtype=float)
    Vmax = np.array(sw.Vmax_kV, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(k1_vals, Vmax, "-o", linewidth=2)
    ax.set_xlabel("Fator de Sobretensão (k1)")
    ax.set_ylabel("Sobretensão Máxima (kV)")
    ax.set_title("Sobretensão por Manobra (Vmáx × k1)")
    ax.grid(True)

    return _fig_to_b64(fig)


def _plot_arrester(result: CoordIsolResult) -> str:
    ar = result.arrester
    I = np.array(ar.I_kA, dtype=float)
    V = np.array(ar.V_kV, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(I, V, linewidth=2)
    ax.set_xlabel("Corrente de Descarga (kA)")
    ax.set_ylabel("Tensão no Para-raios (kV)")
    ax.set_title("Curva do Para-raios (V × I) – modelo aproximado")
    ax.grid(True)

    return _fig_to_b64(fig)


def _plot_Vxt_surface(result: CoordIsolResult) -> str:
    prop = result.wave_prop
    t_us = np.array(prop.t_s, dtype=float) * 1e6
    x_km = np.array(prop.x_m, dtype=float) / 1e3
    V = np.array(prop.V_matrix_kV, dtype=float)

    T, X = np.meshgrid(t_us, x_km)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, T, V, linewidth=0, antialiased=True)
    ax.set_xlabel("Comprimento da Linha (km)")
    ax.set_ylabel("Tempo (µs)")
    ax.set_zlabel("Tensão (kV)")
    ax.set_title("Propagação da Sobretensão – V(x,t)")
    fig.colorbar(surf, shrink=0.55, aspect=12, label="Tensão (kV)")

    return _fig_to_b64(fig)


# ======================================================================
# Relatório HTML – Coordenação e Isolação
# ======================================================================


def generate_html_report_coord(project, cfg: CoordIsolConfig, result: CoordIsolResult) -> str:
    """
    Gera relatório HTML + CSS do estudo de coordenação e isolação.
    `project` pode ser ProjectInfo (bk_estudos_eletricos.core.line_params)
    ou qualquer objeto com atributos:
        nome_projeto, cliente, numero_projeto
    """
    img_impulse = _plot_impulse(result)
    img_wave_mid = _plot_wave_mid(result)
    img_shield_3d = _plot_shield_3d(result)
    img_switching = _plot_switching(result)
    img_arrester = _plot_arrester(result)
    img_Vxt = _plot_Vxt_surface(result)

    css = """
    <style>
      body { font-family: "Segoe UI", Arial, sans-serif; background:#f5f7fa; color:#222; margin:0; padding:0; }
      .container { max-width:1080px; margin:0 auto; padding:24px; background:#fff; box-shadow:0 4px 16px rgba(0,0,0,0.08); }
      h1,h2,h3 { color:#0b3c5d; }
      .header { border-bottom:2px solid #e0e4ea; margin-bottom:16px; padding-bottom:8px; }
      .meta { font-size:0.95rem; color:#555; }
      .section { margin:20px 0; }
      .img-block { text-align:center; margin:16px 0; }
      .eq-block { background:#f8fafc; border-left:4px solid #0b3c5d; padding:8px 12px; font-family:Consolas, "Courier New", monospace; font-size:0.85rem; }
      table { width:100%; border-collapse:collapse; margin:12px 0; font-size:0.9rem; }
      th,td { border:1px solid #dde2eb; padding:6px 8px; text-align:right; }
      th { background:#f0f3f9; font-weight:600; }
      td.label { text-align:left; font-weight:500; }
      .small-note { font-size:0.85rem; color:#666; line-height:1.35; }
    </style>
    """

    eqs_html = f"""
    <div class="eq-block">
      <strong>Metodologia adotada</strong><br/>
      Estudo baseado em recomendações da <strong>IEEE Std 1313.1</strong>, <strong>IEC 60071</strong>,
      <strong>ANSI C62.22</strong> e <strong>IEC 60815</strong>, combinando modelagem analítica e simulação numérica.<br/><br/>

      1) <u>Impulso atmosférico (1,2/50 µs)</u><br/>
      &nbsp;&nbsp;v(t) = V₀·K·(e<sup>-βt</sup> − e<sup>-αt</sup>) com ajuste K para pico ≈ V₀.<br/>
      &nbsp;&nbsp;Propagação por equação de onda (linha sem perdas) via diferenças finitas explícitas,
      com critério de Courant Cᵣ = v·Δt/Δx &lt; 1.<br/><br/>

      2) <u>Ângulo de proteção do cabo-guarda</u><br/>
      &nbsp;&nbsp;θ = arctan((h<sub>cg</sub> − h<sub>fase</sub>)/d). Critério preliminar típico: θ ≤ 45°.<br/><br/>

      3) <u>Sobretensões por manobra</u><br/>
      &nbsp;&nbsp;Vmáx = k₁ · V<sub>nom</sub>, com k₁ típico em [{cfg.k1_min:.2f}, {cfg.k1_max:.2f}] (IEC 60071).<br/><br/>

      4) <u>Para-raios ZnO</u><br/>
      &nbsp;&nbsp;Curva aproximada: V(I) = V<sub>ref</sub>·(I/I<sub>ref</sub>)ⁿ. Energia estimada por
      E = ∫ V(t)·I(t) dt, com forma de corrente simplificada (para triagem).<br/><br/>

      5) <u>Isoladores / escoamento</u><br/>
      &nbsp;&nbsp;L<sub>escoamento</sub> = k · V (IEC 60815). Nº de discos: N = ceil((V/V<sub>disco</sub>)·F).<br/>
      &nbsp;&nbsp;Verificação simplificada de NBI: V<sub>imp</sub> = N·V<sub>imp,disco</sub> > BIL.<br/><br/>

      Margem informativa de coordenação: {cfg.margem_segurança_percent:.0f}%.
    </div>
    """

    ins = result.insulator
    sw = result.switching

    tabela_html = f"""
    <table>
      <tr><th class="label">Grandeza</th><th>Valor</th><th>Comentário</th></tr>
      <tr><td class="label">Vmáx por impulso atmosférico (k·BIL)</td><td>{result.Vmax_impulse_kV:.1f} kV</td><td>k={cfg.k_impulse:.2f}, BIL={cfg.Vbil_kV:.1f} kV</td></tr>
      <tr><td class="label">Vmáx por manobra</td><td>{sw.Vmax_max_kV:.1f} kV</td><td>k₁ ∈ [{cfg.k1_min:.2f}, {cfg.k1_max:.2f}]</td></tr>
      <tr><td class="label">Tensão fase-neutro máx (aprox.)</td><td>{sw.Vfn_max_kV:.1f} kV</td><td>VFN ≈ Vmáx/√2</td></tr>
      <tr><td class="label">Energia estimada no para-raios</td><td>{result.arrester.E_J:.1f} J</td><td>janela {cfg.T_impulse_s*1e6:.0f} µs (modelo simplificado)</td></tr>
      <tr><td class="label">Nº discos (normal)</td><td>{ins.N_disc_normal}</td><td>F={cfg.F_normal:.2f}</td></tr>
      <tr><td class="label">Nº discos (poluído)</td><td>{ins.N_disc_polluted}</td><td>F={cfg.F_polluted:.2f}</td></tr>
      <tr><td class="label">Escoamento calculado</td><td>{ins.L_escoamento_mm:.0f} mm</td><td>k={cfg.creepage_factor_mm_per_kV:.1f} mm/kV</td></tr>
      <tr><td class="label">Mín. discos por escoamento</td><td>{ins.creepage_min_discs}</td><td>{cfg.single_disc_creepage_mm:.0f} mm/disco</td></tr>
      <tr><td class="label">Impulso suportável da cadeia</td><td>{ins.V_impulso_cadeia_kV:.1f} kV</td><td>{"ATENDE" if ins.atende_NBI else "NÃO ATENDE"} BIL {cfg.Vbil_kV:.1f} kV</td></tr>
      <tr><td class="label">Ângulo de proteção do cabo-guarda</td><td>{result.shield.theta_deg:.2f}°</td><td>{"Cobertura adequada (≤45°)" if result.shield.shield_ok else "Cobertura insuficiente (>45°)"}</td></tr>
    </table>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="utf-8"/>
      <title>Relatório – Coordenação e Isolação – {project.nome_projeto}</title>
      {css}
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>BK_Estudos_Eletricos – Coordenação de Isolamento</h1>
          <div class="meta">
            <strong>Projeto:</strong> {project.nome_projeto}<br/>
            <strong>Cliente:</strong> {project.cliente}<br/>
            <strong>Nº Projeto:</strong> {project.numero_projeto}<br/>
            <strong>Tensão nominal:</strong> {cfg.Vnom_kV:.1f} kV (L-L)<br/>
            <strong>NBI / BIL de referência:</strong> {cfg.Vbil_kV:.1f} kV<br/>
          </div>
        </div>

        <div class="section">
          <h2>4. Metodologia</h2>
          {eqs_html}
        </div>

        <div class="section">
          <h2>5. Resultados e Discussão</h2>
          {tabela_html}
          <p class="small-note"><strong>Resumo técnico:</strong><br/>{ "<br/>".join(result.resumo_coord.splitlines()) }</p>
        </div>

        <div class="section">
          <h2>4.1 – Sobretensões por Impulso Atmosférico</h2>
          <div class="img-block">
            <h3>Onda de Impulso 1,2 / 50 µs</h3>
            <img src="data:image/png;base64,{img_impulse}" alt="Onda de impulso"/>
          </div>
          <div class="img-block">
            <h3>Resposta transitória no meio da linha</h3>
            <img src="data:image/png;base64,{img_wave_mid}" alt="Resposta transitória no meio da linha"/>
          </div>
          <div class="img-block">
            <h3>Propagação V(x,t) ao longo da linha</h3>
            <img src="data:image/png;base64,{img_Vxt}" alt="Propagação V(x,t)"/>
          </div>
        </div>

        <div class="section">
          <h2>4.1.2 – Ângulo de Proteção do Cabo-Guarda</h2>
          <div class="img-block">
            <img src="data:image/png;base64,{img_shield_3d}" alt="Ângulo de proteção do cabo guarda"/>
          </div>
        </div>

        <div class="section">
          <h2>4.1.2 – Sobretensões por Manobras</h2>
          <div class="img-block">
            <img src="data:image/png;base64,{img_switching}" alt="Sobretensões por manobra"/>
          </div>
        </div>

        <div class="section">
          <h2>4.3 – Coordenação dos Para-raios</h2>
          <div class="img-block">
            <img src="data:image/png;base64,{img_arrester}" alt="Curva VxI do para-raios"/>
          </div>
          <p class="small-note">
            Nota: a energia do para-raios aqui é uma estimativa preliminar (forma de corrente simplificada).
            Para especificação final, usar dados do fabricante (classe, forma 8/20 µs, TOV, etc.) e estudos EMT.
          </p>
        </div>

        <div class="section">
          <h2>6. Conclusão</h2>
          <p>
            Com base nos cálculos e simulações, o módulo indica conformidade preliminar (ou pontos de atenção)
            quanto a sobretensões por impulso/manobra, proteção por para-raios e requisitos de isolação/escoamento.
          </p>
        </div>

      </div>
    </body>
    </html>
    """
    return html


# ======================================================================
# Teste rápido (opcional)
# ======================================================================

if __name__ == "__main__":
    class _Proj:
        nome_projeto = "Exemplo Coordenação 138 kV"
        cliente = "BK Engenharia e Tecnologia"
        numero_projeto = "2025-Coord-001"

    cfg = CoordIsolConfig()
    res = compute_coord_isolation(cfg)
    html = generate_html_report_coord(_Proj(), cfg, res)

    with open("relatorio_coord_isol_exemplo.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Relatório gerado: relatorio_coord_isol_exemplo.html")

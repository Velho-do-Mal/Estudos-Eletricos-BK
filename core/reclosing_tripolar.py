# bk_estudos_eletricos/core/reclosing_tripolar.py
# ======================================================================
# CORREÇÕES BK_Fixes_v1:
#   Fix #10 – Frequência natural do modelo LC concentrado vs. linha distribuída.
#              O modelo usa omega0 = 1/sqrt(L_total*C_total) [modelo lumped].
#              Para linha distribuída sem perdas, a frequência do modo fundamental
#              (quarto de onda, extremo aberto) é:
#                  f0_qw = c / (4 * comprimento_km)   c ≈ 2.998e5 km/s
#              O modelo concentrado superstima f0 por fator ~pi/2 ≈ 1.57×.
#              CORREÇÃO: adiciona f0_quarter_wave ao resultado para comparação
#              e emite aviso quando a divergência supera 20%.
#              Para estudos executivos recomenda-se modelo EMT distribuído.
#
#   Fix #11 – FO(t) = |ΔV(t)| / V_base_fase(RMS) representa a tensão de passo
#              nos contatos do disjuntor, NÃO a sobretensão real na linha
#              após o fechamento. A sobretensão real depende da relação entre
#              a impedância de surto da linha e a impedância da fonte — em
#              redes fortes (Zsource << Zc), pode ser até 2× menor que FO.
#              CORREÇÃO:
#                a) FO agora é normalizado por V_pico (amplitude), não V_RMS,
#                   alinhando com a prática IEC 60071 para sobretensões transitórias.
#                b) Adicionado fator de atenuação por fonte (ks = Zsource/(Zsource+Zc))
#                   configurável em ReclosingConfig.Zsource_ohm. Se None, usa
#                   modelo conservativo (ks=1, equivale ao comportamento anterior).
# ======================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math, base64, warnings
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from .line_params import ProjectInfo, LineGeometry, LineParamsResult

# Velocidade de propagação em linha aérea (aprox. 95-98% da velocidade da luz)
_WAVE_SPEED_KM_S = 2.998e5 * 0.97   # km/s  (97% da velocidade da luz no vácuo)

# ======================= Configuração ================================

@dataclass
class ReclosingConfig:
    """
    Configuração do estudo de religamento tripolar.

    Parâmetros adicionados (Fix #10 e Fix #11):
      Zsource_ohm : impedância da fonte [Ω] (para cálculo do fator de atenuação ks).
                    Se None, usa modelo conservativo (ks=1 = pior caso).
                    Típico: 5-20 Ω para barramentos de transmissão.
    """
    V_LL_kV: float
    f_hz: float
    length_km: float
    dead_time_s: float

    trapped_kpu: float = 1.0
    damping_alpha: float = 0.0
    overvoltage_limit_pu: float = 2.0

    t_sim_s: float = 0.3
    dt_s: float = 1e-4
    t_min_window_s: float = 0.02
    t_max_window_s: float = 0.5

    grid_n_dead: int = 40
    grid_n_trap: int = 30

    circuit_indices: Optional[List[int]] = None

    # Fix #11: impedância da fonte para cálculo do fator de atenuação ks
    # Se None → ks=1 (conservativo — pior caso)
    Zsource_ohm: Optional[float] = None

@dataclass
class AcceptableWindow:
    t_start_s: float
    t_end_s: float

@dataclass
class ReclosingCircuitResult:
    circuit_index: int

    L_total_H: float
    C_total_F: float
    omega_sys_rad_s: float

    # Fix #10: dois valores de frequência natural
    omega0_rad_s: float       # modelo LC concentrado (usado na simulação)
    f0_hz: float              # = omega0/(2pi) — modelo concentrado
    f0_quarter_wave_hz: float # frequência de quarto de onda da linha distribuída
    f0_warning: str           # aviso se divergência > 20%

    # Fix #11: fator de atenuação e Zc
    Zc_ohm: float             # impedância de surto Zc = sqrt(L/C) [Ω]
    ks_factor: float          # fator de atenuação = Zsource/(Zsource+Zc), 0<ks<=1
    FO_method: str            # descrição do método de normalização usado

    dead_time_s: float
    trapped_kpu: float
    overvoltage_limit_pu: float

    FO_dead_pu: float
    FO_max_pu: float
    is_dead_time_acceptable: bool

    acceptable_windows: List[AcceptableWindow]

    t_s: List[float]
    Vs_V: List[float]
    Vline_V: List[float]
    dV_V: List[float]
    FO_pu: List[float]

@dataclass
class ReclosingStudyResult:
    config: ReclosingConfig
    per_circuit: Dict[int, ReclosingCircuitResult]

# ======================= Núcleo de cálculo ===========================

def _simulate_reclosing_time_series(
    V_LL_kV, f_hz, L_total_H, C_total_F,
    trapped_kpu, damping_alpha, t_sim_s, dt_s,
    Zc_ohm: float = 0.0, Zsource_ohm: Optional[float] = None,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Simula Vs(t), Vline(t), ΔV(t) e FO(t) no período de disjuntor aberto.

    Modelo monofásico equivalente LC por fase:
        Vs(t)    = Vpico · sin(omega_s · t)
        Vline(t) = k_trap · Vpico · cos(omega0 · t) · exp(-alpha·t)
        ΔV(t)    = Vs(t) - Vline(t)

    Fix #11 – FO normalizado por V_pico (amplitude) conforme IEC 60071
    (prática para sobretensões transitórias), e atenuado pelo fator ks:
        FO(t) = ks · |ΔV(t)| / Vpico
    onde ks = Zsource/(Zsource + Zc).  Se Zsource=None → ks=1 (conservativo).
    """
    if V_LL_kV <= 0 or f_hz <= 0 or L_total_H <= 0 or C_total_F <= 0:
        raise ValueError("Parâmetros inválidos para simulação de religamento.")
    trapped_kpu = min(max(float(trapped_kpu), 0.0), 1.0)

    omega_sys   = 2.0 * math.pi * f_hz
    omega0      = 1.0 / math.sqrt(L_total_H * C_total_F)
    V_phase_rms = (V_LL_kV * 1e3) / math.sqrt(3.0)
    Vpico       = math.sqrt(2.0) * V_phase_rms
    V_line0     = trapped_kpu * Vpico

    # Fix #11: fator de atenuação ks
    if Zsource_ohm is not None and Zc_ohm > 0:
        ks = float(Zsource_ohm) / (float(Zsource_ohm) + Zc_ohm)
        ks = max(0.0, min(1.0, ks))
    else:
        ks = 1.0   # conservativo (pior caso)

    n_steps = int(round(t_sim_s / dt_s)) + 1
    t_s     = []; Vs_V = []; Vline_V = []; dV_V = []; FO_pu = []

    for i in range(n_steps):
        t  = i * dt_s
        Vs = Vpico * math.sin(omega_sys * t)
        if damping_alpha > 0.0:
            Vline = V_line0 * math.cos(omega0*t) * math.exp(-damping_alpha*t)
        else:
            Vline = V_line0 * math.cos(omega0*t)
        dV = Vs - Vline
        # Fix #11: normalizado por Vpico (amplitude), atenuado por ks
        FO = ks * abs(dV) / Vpico if Vpico > 0 else 0.0
        t_s.append(t); Vs_V.append(Vs); Vline_V.append(Vline)
        dV_V.append(dV); FO_pu.append(FO)

    return t_s, Vs_V, Vline_V, dV_V, FO_pu

def _extract_acceptable_windows(t_s, FO_pu, limit_pu, t_min, t_max) -> List[AcceptableWindow]:
    if not t_s or not FO_pu or len(t_s) != len(FO_pu): return []
    if t_max < t_min: t_min, t_max = t_max, t_min
    windows, in_window, t_start, t_prev = [], False, 0.0, t_s[0]
    for t, fo in zip(t_s, FO_pu):
        if t < t_min or t > t_max:
            if in_window:
                windows.append(AcceptableWindow(t_start, t_prev))
                in_window = False
        elif fo <= limit_pu:
            if not in_window: in_window = True; t_start = t
        else:
            if in_window:
                windows.append(AcceptableWindow(t_start, t_prev)); in_window = False
        t_prev = t
    if in_window: windows.append(AcceptableWindow(t_start, t_prev))
    return [w for w in windows if w.t_end_s >= w.t_start_s]

def compute_reclosing_for_circuit(
    params: LineParamsResult, config: ReclosingConfig,
) -> ReclosingCircuitResult:
    if config.length_km <= 0:   raise ValueError("length_km deve ser > 0.")
    if config.dead_time_s < 0:  raise ValueError("dead_time_s deve ser >= 0.")

    L_total_H = float(params.L_H_km) * float(config.length_km)
    C_total_F = float(params.C_F_km) * float(config.length_km)
    if L_total_H <= 0 or C_total_F <= 0:
        raise ValueError(f"Circuito {params.circuit_index}: L_total ou C_total inválidos.")

    omega_sys = 2.0 * math.pi * config.f_hz
    omega0    = 1.0 / math.sqrt(L_total_H * C_total_F)
    f0_lump   = omega0 / (2.0 * math.pi)

    # Fix #10: frequência de quarto de onda da linha distribuída real
    f0_qw = _WAVE_SPEED_KM_S / (4.0 * float(config.length_km))
    ratio  = f0_lump / f0_qw if f0_qw > 0 else 1.0
    if abs(ratio - 1.0) > 0.20:
        f0_warn = (
            f"ATENÇÃO (Fix #10): f0_lumped={f0_lump:.1f} Hz diverge {(ratio-1)*100:.0f}% "
            f"de f0_quarter_wave={f0_qw:.1f} Hz. "
            f"Para estudos executivos use modelo EMT com parâmetros distribuídos."
        )
        warnings.warn(f0_warn, UserWarning, stacklevel=2)
    else:
        f0_warn = f"f0_lumped={f0_lump:.1f} Hz ≈ f0_quarter_wave={f0_qw:.1f} Hz (divergência < 20%)."

    # Fix #11: impedância de surto Zc = sqrt(L'/C') (por unidade de comprimento cancel.)
    Zc_ohm = math.sqrt(float(params.L_H_km) / float(params.C_F_km)) if params.C_F_km > 0 else 0.0
    if config.Zsource_ohm is not None and Zc_ohm > 0:
        ks_factor = config.Zsource_ohm / (config.Zsource_ohm + Zc_ohm)
        ks_factor = max(0.0, min(1.0, ks_factor))
        FO_method = f"FO = ks·|ΔV|/Vpico, ks={ks_factor:.3f} (Zsource={config.Zsource_ohm:.1f}Ω, Zc={Zc_ohm:.1f}Ω)"
    else:
        ks_factor = 1.0
        FO_method = "FO = |ΔV|/Vpico (ks=1, conservativo — Zsource não informado)"

    t_s, Vs_V, Vline_V, dV_V, FO_pu = _simulate_reclosing_time_series(
        V_LL_kV=config.V_LL_kV, f_hz=config.f_hz,
        L_total_H=L_total_H, C_total_F=C_total_F,
        trapped_kpu=config.trapped_kpu, damping_alpha=config.damping_alpha,
        t_sim_s=config.t_sim_s, dt_s=config.dt_s,
        Zc_ohm=Zc_ohm, Zsource_ohm=config.Zsource_ohm,
    )

    dead_index   = min(range(len(t_s)), key=lambda i: abs(t_s[i] - config.dead_time_s))
    FO_dead      = FO_pu[dead_index]
    FO_max       = max(FO_pu) if FO_pu else 0.0
    is_ok        = FO_dead <= config.overvoltage_limit_pu
    windows      = _extract_acceptable_windows(
        t_s, FO_pu, config.overvoltage_limit_pu,
        config.t_min_window_s, config.t_max_window_s)

    return ReclosingCircuitResult(
        circuit_index=int(params.circuit_index),
        L_total_H=L_total_H, C_total_F=C_total_F,
        omega_sys_rad_s=omega_sys,
        omega0_rad_s=omega0, f0_hz=f0_lump,
        f0_quarter_wave_hz=f0_qw, f0_warning=f0_warn,
        Zc_ohm=Zc_ohm, ks_factor=ks_factor, FO_method=FO_method,
        dead_time_s=config.dead_time_s, trapped_kpu=config.trapped_kpu,
        overvoltage_limit_pu=config.overvoltage_limit_pu,
        FO_dead_pu=float(FO_dead), FO_max_pu=float(FO_max),
        is_dead_time_acceptable=bool(is_ok),
        acceptable_windows=windows,
        t_s=t_s, Vs_V=Vs_V, Vline_V=Vline_V, dV_V=dV_V, FO_pu=FO_pu,
    )

def compute_reclosing_study(
    params_by_circuit: Dict[int, LineParamsResult], config: ReclosingConfig,
) -> ReclosingStudyResult:
    if not params_by_circuit:
        return ReclosingStudyResult(config=config, per_circuit={})
    cidxs = sorted(params_by_circuit.keys()) if config.circuit_indices is None \
            else sorted([c for c in config.circuit_indices if c in params_by_circuit])
    return ReclosingStudyResult(
        config=config,
        per_circuit={cidx: compute_reclosing_for_circuit(params_by_circuit[cidx], config)
                     for cidx in cidxs},
    )

# ======================= Gráficos ====================================

def _fig_to_b64(fig) -> str:
    buf = BytesIO(); fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def plot_reclosing_time_series(circuit_result: ReclosingCircuitResult) -> Tuple[str, str]:
    t_s      = circuit_result.t_s
    Vs_V     = circuit_result.Vs_V
    Vline_V  = circuit_result.Vline_V
    FO_pu    = circuit_result.FO_pu
    dead_time = circuit_result.dead_time_s

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(t_s, [v/1e3 for v in Vs_V], label="Vs (kV)", linewidth=1.5)
    ax1.plot(t_s, [v/1e3 for v in Vline_V], label="Vlinha (kV)", linewidth=1.2, linestyle="--")
    ax1.axvline(dead_time, linestyle=":", linewidth=1.5, label="Religamento")
    ax1.set_xlabel("Tempo (s)"); ax1.set_ylabel("Tensão (kV)")
    ax1.set_title(f"Circuito {circuit_result.circuit_index} – Tensões antes do religamento")
    ax1.grid(True); ax1.legend()
    img1 = _fig_to_b64(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(t_s, FO_pu, linewidth=1.5, label="FO(t)")
    ax2.axhline(circuit_result.overvoltage_limit_pu, linestyle="--", linewidth=1.2,
                label=f"Limite = {circuit_result.overvoltage_limit_pu:.2f} pu")
    ax2.axvline(dead_time, linestyle=":", linewidth=1.5, label=f"T_dead={dead_time:.3f}s")
    ax2.set_xlabel("Tempo (s)"); ax2.set_ylabel("FO(t) (pu)")
    ax2.set_title(f"Circuito {circuit_result.circuit_index} – Fator de Sobretensão\n"
                  f"({circuit_result.FO_method})")
    ax2.grid(True); ax2.legend()
    img2 = _fig_to_b64(fig2)
    return img1, img2

def plot_overvoltage_surface(params, config, trapped_min=0.0, trapped_max=1.0) -> str:
    if config.grid_n_dead < 2 or config.grid_n_trap < 2:
        raise ValueError("grid_n_dead e grid_n_trap devem ser >= 2.")
    t_dead_vals   = np.linspace(config.t_min_window_s, config.t_max_window_s, config.grid_n_dead).tolist()
    trapped_vals  = np.linspace(float(trapped_min), float(trapped_max), config.grid_n_trap).tolist()
    L_total_H = float(params.L_H_km) * float(config.length_km)
    C_total_F = float(params.C_F_km) * float(config.length_km)
    if L_total_H <= 0 or C_total_F <= 0:
        raise ValueError("L_total ou C_total inválidos.")
    Zc_ohm = math.sqrt(float(params.L_H_km)/float(params.C_F_km)) if params.C_F_km > 0 else 0.0
    t_sim_loc = max(config.t_max_window_s, config.dt_s)
    Z = []
    for kt in trapped_vals:
        t_s, _, _, _, FO_pu = _simulate_reclosing_time_series(
            config.V_LL_kV, config.f_hz, L_total_H, C_total_F,
            kt, config.damping_alpha, t_sim_loc, config.dt_s,
            Zc_ohm=Zc_ohm, Zsource_ohm=config.Zsource_ohm,
        )
        row = [float(FO_pu[min(range(len(t_s)), key=lambda i: abs(t_s[i]-td))]) for td in t_dead_vals]
        Z.append(row)
    T_dead, K_trap = np.meshgrid(t_dead_vals, trapped_vals)
    Z_np = np.array(Z, dtype=float)
    fig  = plt.figure(figsize=(7, 5))
    ax   = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T_dead, K_trap, Z_np, edgecolor="none", alpha=0.9)
    ax.set_xlabel("Tempo de religamento (s)"); ax.set_ylabel("k_trap (pu)")
    ax.set_zlabel("FO(t_dead) (pu)")
    ax.set_title(f"Circuito {params.circuit_index} – FO vs. T_dead e k_trap")
    fig.colorbar(surf, shrink=0.6, aspect=10)
    return _fig_to_b64(fig)

# ======================= Relatório HTML ==============================

def generate_html_report_reclosing(project, geom, params_by_circuit, study) -> str:
    config = study.config
    css = ("<style>body{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fa;color:#222}"
           ".container{max-width:1080px;margin:0 auto;padding:24px;background:#fff;"
           "box-shadow:0 4px 16px rgba(0,0,0,.08)}h1,h2,h3{color:#0b3c5d}"
           "table{width:100%;border-collapse:collapse;margin:12px 0;font-size:.9rem}"
           "th,td{border:1px solid #dde2eb;padding:6px 8px;text-align:right}"
           "th{background:#f0f3f9;font-weight:600}td.label{text-align:left;font-weight:500}"
           ".ok{color:#0b8f55;font-weight:600}.nok{color:#c0392b;font-weight:600}"
           ".img-block{text-align:center;margin:16px 0}"
           ".eq-block{background:#f8fafc;border-left:4px solid #0b3c5d;padding:8px 12px;"
           "font-family:Consolas,monospace;font-size:.85rem}"
           ".warn{background:#fff8e1;border-left:4px solid #f9a825;padding:8px 12px;"
           "font-size:.85rem;margin:8px 0}"
           ".small-note{font-size:.8rem;color:#777}</style>")

    metodologia = ("<div class='eq-block'>"
        "<b>Modelo simplificado LC por fase:</b><br/>"
        "L_total = L'·ℓ [H],  C_total = C'·ℓ [F]<br/>"
        "ω₀ = 1/√(L·C) [rad/s],  f₀ = ω₀/(2π) [Hz] — modelo concentrado<br/>"
        "f₀_qw = c/(4·ℓ) [Hz] — linha distribuída, quarto de onda (Fix #10)<br/>"
        "V_s(t) = Vpico·sin(ω_s·t),  V_linha(t) = k_trap·Vpico·cos(ω₀·t)·e^(-α·t)<br/>"
        "ΔV(t) = V_s(t) − V_linha(t)<br/>"
        "FO(t) = ks·|ΔV(t)|/Vpico  (Fix #11: normalizado por Vpico; ks = fator de atenuação)<br/>"
        "ks = Zsource/(Zsource+Zc),  Zc = √(L'/C')  [Ω]<br/>"
        "Se Zsource=None → ks=1 (conservativo — pior caso)."
        "</div>")

    circuits_html = ""
    for cidx, cr in study.per_circuit.items():
        img1, img2 = plot_reclosing_time_series(cr)
        img3d      = plot_overvoltage_surface(params_by_circuit[cidx], config)
        sc   = "ok" if cr.is_dead_time_acceptable else "nok"
        st   = (f"Conforme (FO_dead={cr.FO_dead_pu:.2f} pu ≤ {cr.overvoltage_limit_pu:.2f} pu)"
                if cr.is_dead_time_acceptable else
                f"Não conforme (FO_dead={cr.FO_dead_pu:.2f} pu > {cr.overvoltage_limit_pu:.2f} pu)")
        warn_html = (f"<div class='warn'><b>Fix #10:</b> {cr.f0_warning}</div>"
                     if "ATENÇÃO" in cr.f0_warning else "")
        wins_html = ("".join(f"<tr><td>{w.t_start_s:.4f}</td><td>{w.t_end_s:.4f}</td></tr>"
                             for w in cr.acceptable_windows)
                     if cr.acceptable_windows else
                     "<tr><td colspan='2'>Nenhuma janela aceitável no intervalo analisado.</td></tr>")
        circuits_html += f"""
<h2>Circuito {cidx}</h2>
{warn_html}
<table>
<tr><th class="label">Parâmetro</th><th>Valor</th><th>Unidade</th></tr>
<tr><td class="label">L_total</td><td>{cr.L_total_H:.4e}</td><td>H</td></tr>
<tr><td class="label">C_total</td><td>{cr.C_total_F:.4e}</td><td>F</td></tr>
<tr><td class="label">f_sistema</td><td>{config.f_hz:.3f}</td><td>Hz</td></tr>
<tr><td class="label">f₀ (modelo concentrado)</td><td>{cr.f0_hz:.2f}</td><td>Hz</td></tr>
<tr><td class="label">f₀ (quarto de onda — Fix #10)</td><td>{cr.f0_quarter_wave_hz:.2f}</td><td>Hz</td></tr>
<tr><td class="label">Zc (impedância de surto)</td><td>{cr.Zc_ohm:.2f}</td><td>Ω</td></tr>
<tr><td class="label">ks (fator de atenuação — Fix #11)</td><td>{cr.ks_factor:.3f}</td><td>-</td></tr>
<tr><td class="label">T_dead</td><td>{cr.dead_time_s:.4f}</td><td>s</td></tr>
<tr><td class="label">k_trap</td><td>{cr.trapped_kpu:.3f}</td><td>pu</td></tr>
<tr><td class="label">FO_dead</td><td>{cr.FO_dead_pu:.3f}</td><td>pu</td></tr>
<tr><td class="label">FO_max</td><td>{cr.FO_max_pu:.3f}</td><td>pu</td></tr>
<tr><td class="label">Limite adotado</td><td>{cr.overvoltage_limit_pu:.3f}</td><td>pu</td></tr>
<tr><td class="label">Avaliação</td><td class="{sc}" colspan="2">{st}</td></tr>
</table>
<p><small>{cr.FO_method}</small></p>
<div class="img-block"><img src="data:image/png;base64,{img1}"/></div>
<div class="img-block"><img src="data:image/png;base64,{img2}"/></div>
<div class="img-block"><img src="data:image/png;base64,{img3d}"/></div>
<h3>Janelas aceitáveis (FO(t) ≤ limite)</h3>
<table><tr><th>Início (s)</th><th>Fim (s)</th></tr>{wins_html}</table>
"""

    return f"""<!DOCTYPE html><html lang="pt-BR">
<head><meta charset="utf-8"/>
<title>Religamento Tripolar – {project.nome_projeto}</title>{css}</head>
<body><div class="container">
<div class="header"><h1>BK_Estudos_Eletricos – Estudo de Religamento Tripolar</h1>
<div class="meta">
<b>Projeto:</b> {project.nome_projeto} | <b>Cliente:</b> {project.cliente} | <b>Nº:</b> {project.numero_projeto}<br/>
<b>V_LL:</b> {config.V_LL_kV:.1f} kV | <b>f:</b> {config.f_hz:.2f} Hz |
<b>Comprimento:</b> {config.length_km:.3f} km | <b>T_dead:</b> {config.dead_time_s:.4f} s |
<b>Limite:</b> {config.overvoltage_limit_pu:.2f} pu
</div></div>
{circuits_html}
<h2>Metodologia</h2>{metodologia}
<p class="small-note">Para estudos executivos (IEC 60071 / IEEE 1313.1) recomenda-se
simulação EMT com parâmetros distribuídos e modelo de fonte detalhado.</p>
</div></body></html>"""

if __name__ == "__main__":
    print("reclosing_tripolar_fixed.py OK — importe como módulo para uso.")

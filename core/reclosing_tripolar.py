# bk_estudos_eletricos/core/reclosing_tripolar.py

"""
Módulo de estudo de religamento tripolar de linhas de transmissão.

Integração:
- Usa ProjectInfo, LineGeometry e LineParamsResult do módulo line_params.py
- Pode ser chamado a partir da UI (aba "Religamento Tripolar") usando:
    - parâmetros elétricos já calculados (L', C', Zc, etc.)
    - dados do projeto (tensão, frequência, comprimento da linha)

Funcionalidades:
- Modelo simplificado de sobretensão de religamento tripolar com carga presa (trapped charge)
- Cálculo do fator de sobretensão em função do tempo de religamento
- Avaliação de conformidade com limite de sobretensão (pu) (IEC 60071 / IEEE 1313.1 – conceitual)
- Geração de gráficos:
    - 2D: Vs(t), Vlinha(t) e ΔV(t) com marcação do instante de religamento
    - 2D: fator de sobretensão FO(t) ao longo do tempo
    - 3D: FO(t_dead) vs (tempo de religamento, fator de carga presa)
- Geração de relatório HTML + CSS com tabelas e imagens embutidas (base64)

Observação:
- Este módulo é intencionalmente simplificado (equivalente LC por fase).
  Para estudos avançados, recomenda-se modelo distribuído com ondas viajantes, resistências, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import base64
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessário para projeção 3D)

from .line_params import (
    ProjectInfo,
    LineGeometry,
    LineParamsResult,
)

# ======================= Configuração e resultados ====================


@dataclass
class ReclosingConfig:
    """
    Configuração global do estudo de religamento tripolar.

    Todos os valores são por fase (fase-terra), mas a tensão é informada em kV L-L.
    O modelo é monofásico equivalente, representando o circuito tripolar em religamento
    simultâneo das três fases (religamento tripolar).
    """

    V_LL_kV: float                  # Tensão nominal do sistema (kV L-L)
    f_hz: float                     # Frequência do sistema (Hz)
    length_km: float                # Comprimento da linha (km)
    dead_time_s: float              # Tempo de religamento (dead time) em segundos

    trapped_kpu: float = 1.0        # Fator de carga presa (0..1), 1 = pior caso
    damping_alpha: float = 0.0      # Amortecimento exponencial (Np/s)

    overvoltage_limit_pu: float = 2.0  # Limite de sobretensão (pu) para avaliação

    t_sim_s: float = 0.3            # Janela de simulação (s)
    dt_s: float = 1e-4              # Passo de tempo (s)

    t_min_window_s: float = 0.02    # Tempo mínimo considerado para janelas (s)
    t_max_window_s: float = 0.5     # Tempo máximo considerado para janelas (s)

    grid_n_dead: int = 40           # Pontos no eixo tempo (superfície 3D)
    grid_n_trap: int = 30           # Pontos no eixo k_trap (superfície 3D)

    circuit_indices: Optional[List[int]] = None  # Se None, usa todos os circuitos em params_by_circuit


@dataclass
class AcceptableWindow:
    t_start_s: float
    t_end_s: float


@dataclass
class ReclosingCircuitResult:
    """
    Resultado detalhado do estudo de religamento para um circuito.
    """

    circuit_index: int

    # Parâmetros equivalentes
    L_total_H: float
    C_total_F: float
    omega_sys_rad_s: float
    omega0_rad_s: float
    f0_hz: float

    # Fatores de sobretensão
    dead_time_s: float
    trapped_kpu: float
    overvoltage_limit_pu: float
    FO_dead_pu: float
    FO_max_pu: float
    is_dead_time_acceptable: bool

    # Janelas de religamento aceitáveis (FO(t) <= limite)
    acceptable_windows: List[AcceptableWindow]

    # Sinais temporais para plot
    t_s: List[float]
    Vs_V: List[float]
    Vline_V: List[float]
    dV_V: List[float]
    FO_pu: List[float]


@dataclass
class ReclosingStudyResult:
    """Resultado agregado do estudo de religamento para todos os circuitos."""
    config: ReclosingConfig
    per_circuit: Dict[int, ReclosingCircuitResult]


# ======================= Núcleo de cálculo ===========================


def _simulate_reclosing_time_series(
    V_LL_kV: float,
    f_hz: float,
    L_total_H: float,
    C_total_F: float,
    trapped_kpu: float,
    damping_alpha: float,
    t_sim_s: float,
    dt_s: float,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Simula Vs(t), Vlinha(t), ΔV(t) e FO(t) antes do religamento (disjuntor aberto).

    Modelo:
      - Vs(t): senoide de frequência fundamental (fonte ideal, fase A)
      - Vlinha(t): oscilação LC com carga presa, com amortecimento exponencial opcional
      - ΔV(t) = Vs(t) - Vlinha(t)
      - FO(t) = |ΔV(t)| / V_base_fase(RMS)

    Base por fase (RMS): V_phase_base = V_LL / sqrt(3).
    """
    if V_LL_kV <= 0:
        raise ValueError("V_LL_kV deve ser > 0.")
    if f_hz <= 0:
        raise ValueError("f_hz deve ser > 0.")
    if L_total_H <= 0 or C_total_F <= 0:
        raise ValueError("L_total_H e C_total_F devem ser > 0.")
    if t_sim_s <= 0:
        raise ValueError("t_sim_s deve ser > 0.")
    if dt_s <= 0:
        raise ValueError("dt_s deve ser > 0.")

    trapped_kpu = min(max(trapped_kpu, 0.0), 1.0)

    omega_sys = 2.0 * math.pi * f_hz
    omega0 = 1.0 / math.sqrt(L_total_H * C_total_F)

    # Base por fase (RMS) e amplitude
    V_phase_rms = (V_LL_kV * 1e3) / math.sqrt(3.0)
    V_phase_amp = math.sqrt(2.0) * V_phase_rms

    # Carga presa inicial na linha (pior caso = 1 pu)
    V_line0 = trapped_kpu * V_phase_amp

    n_steps = int(round(t_sim_s / dt_s)) + 1

    t_s: List[float] = []
    Vs_V: List[float] = []
    Vline_V: List[float] = []
    dV_V: List[float] = []
    FO_pu: List[float] = []

    for i in range(n_steps):
        t = i * dt_s
        t_s.append(t)

        Vs = V_phase_amp * math.sin(omega_sys * t)

        if damping_alpha > 0.0:
            Vline = V_line0 * math.cos(omega0 * t) * math.exp(-damping_alpha * t)
        else:
            Vline = V_line0 * math.cos(omega0 * t)

        dV = Vs - Vline
        FO = abs(dV) / V_phase_rms if V_phase_rms > 0 else 0.0

        Vs_V.append(Vs)
        Vline_V.append(Vline)
        dV_V.append(dV)
        FO_pu.append(FO)

    return t_s, Vs_V, Vline_V, dV_V, FO_pu


def _extract_acceptable_windows(
    t_s: List[float],
    FO_pu: List[float],
    limit_pu: float,
    t_min: float,
    t_max: float,
) -> List[AcceptableWindow]:
    """
    Extrai intervalos [t_ini, t_fim] em que FO(t) <= limit_pu,
    considerando apenas t dentro de [t_min, t_max].
    """
    if not t_s or not FO_pu or len(t_s) != len(FO_pu):
        return []

    if t_max < t_min:
        t_min, t_max = t_max, t_min

    windows: List[AcceptableWindow] = []
    in_window = False
    t_start = 0.0
    t_prev = t_s[0]

    for t, fo in zip(t_s, FO_pu):
        # Fora da janela de interesse
        if t < t_min or t > t_max:
            if in_window:
                windows.append(AcceptableWindow(t_start_s=t_start, t_end_s=t_prev))
                in_window = False
            t_prev = t
            continue

        # Dentro da janela de interesse
        if fo <= limit_pu:
            if not in_window:
                in_window = True
                t_start = t
        else:
            if in_window:
                windows.append(AcceptableWindow(t_start_s=t_start, t_end_s=t_prev))
                in_window = False

        t_prev = t

    if in_window:
        windows.append(AcceptableWindow(t_start_s=t_start, t_end_s=t_prev))

    # Remove janelas degeneradas
    windows = [w for w in windows if w.t_end_s >= w.t_start_s]
    return windows


def compute_reclosing_for_circuit(
    params: LineParamsResult,
    config: ReclosingConfig,
) -> ReclosingCircuitResult:
    """
    Calcula o estudo de religamento tripolar para um circuito específico.

    Equivalentemente por fase:
      L_total = L_H_km * length_km
      C_total = C_F_km * length_km
    """
    if config.length_km <= 0:
        raise ValueError("length_km deve ser > 0.")
    if config.dead_time_s < 0:
        raise ValueError("dead_time_s deve ser >= 0.")
    if config.overvoltage_limit_pu <= 0:
        raise ValueError("overvoltage_limit_pu deve ser > 0.")

    L_total_H = float(params.L_H_km) * float(config.length_km)
    C_total_F = float(params.C_F_km) * float(config.length_km)

    if L_total_H <= 0 or C_total_F <= 0:
        raise ValueError(f"Circuito {params.circuit_index}: L_total_H ou C_total_F inválidos.")

    omega_sys = 2.0 * math.pi * config.f_hz
    omega0 = 1.0 / math.sqrt(L_total_H * C_total_F)
    f0 = omega0 / (2.0 * math.pi)

    t_s, Vs_V, Vline_V, dV_V, FO_pu = _simulate_reclosing_time_series(
        V_LL_kV=config.V_LL_kV,
        f_hz=config.f_hz,
        L_total_H=L_total_H,
        C_total_F=C_total_F,
        trapped_kpu=config.trapped_kpu,
        damping_alpha=config.damping_alpha,
        t_sim_s=config.t_sim_s,
        dt_s=config.dt_s,
    )

    # Índice mais próximo do dead time
    dead_index = min(range(len(t_s)), key=lambda i: abs(t_s[i] - config.dead_time_s))
    FO_dead = FO_pu[dead_index]
    FO_max = max(FO_pu) if FO_pu else 0.0

    is_acceptable = FO_dead <= config.overvoltage_limit_pu

    windows = _extract_acceptable_windows(
        t_s=t_s,
        FO_pu=FO_pu,
        limit_pu=config.overvoltage_limit_pu,
        t_min=config.t_min_window_s,
        t_max=config.t_max_window_s,
    )

    return ReclosingCircuitResult(
        circuit_index=int(params.circuit_index),
        L_total_H=L_total_H,
        C_total_F=C_total_F,
        omega_sys_rad_s=omega_sys,
        omega0_rad_s=omega0,
        f0_hz=f0,
        dead_time_s=config.dead_time_s,
        trapped_kpu=config.trapped_kpu,
        overvoltage_limit_pu=config.overvoltage_limit_pu,
        FO_dead_pu=float(FO_dead),
        FO_max_pu=float(FO_max),
        is_dead_time_acceptable=bool(is_acceptable),
        acceptable_windows=windows,
        t_s=t_s,
        Vs_V=Vs_V,
        Vline_V=Vline_V,
        dV_V=dV_V,
        FO_pu=FO_pu,
    )


def compute_reclosing_study(
    params_by_circuit: Dict[int, LineParamsResult],
    config: ReclosingConfig,
) -> ReclosingStudyResult:
    """
    Calcula o estudo de religamento para todos os circuitos em params_by_circuit,
    respeitando config.circuit_indices (se fornecido).
    """
    if not params_by_circuit:
        return ReclosingStudyResult(config=config, per_circuit={})

    if config.circuit_indices is None:
        circuit_indices = sorted(params_by_circuit.keys())
    else:
        circuit_indices = sorted([c for c in config.circuit_indices if c in params_by_circuit])

    per_circuit: Dict[int, ReclosingCircuitResult] = {}
    for cidx in circuit_indices:
        per_circuit[cidx] = compute_reclosing_for_circuit(
            params=params_by_circuit[cidx],
            config=config,
        )

    return ReclosingStudyResult(config=config, per_circuit=per_circuit)


# ======================= Gráficos (para relatório) ===================


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_reclosing_time_series(
    circuit_result: ReclosingCircuitResult,
) -> Tuple[str, str]:
    """
    Gera dois gráficos 2D em base64:
      1) Vs(t) e Vlinha(t) com marcação do instante de religamento
      2) FO(t) (pu) com limite e marcação do religamento
    Retorna (img1_b64, img2_b64).
    """
    t_s = circuit_result.t_s
    Vs_V = circuit_result.Vs_V
    Vline_V = circuit_result.Vline_V
    FO_pu = circuit_result.FO_pu
    dead_time = circuit_result.dead_time_s

    # Gráfico 1: tensões Vs e Vlinha
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(t_s, [v / 1e3 for v in Vs_V], label="Vs (kV)", linewidth=1.5)
    ax1.plot(t_s, [v / 1e3 for v in Vline_V], label="Vlinha (kV)", linewidth=1.2, linestyle="--")
    ax1.axvline(dead_time, linestyle=":", linewidth=1.5, label="Religamento")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Tensão (kV)")
    ax1.set_title(f"Circuito {circuit_result.circuit_index} – Tensões antes do religamento")
    ax1.grid(True)
    ax1.legend(loc="best")
    img1_b64 = _fig_to_base64(fig1)

    # Gráfico 2: fator de sobretensão FO(t)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(t_s, FO_pu, linewidth=1.5)
    ax2.axhline(
        circuit_result.overvoltage_limit_pu,
        linestyle="--",
        linewidth=1.2,
        label=f"Limite = {circuit_result.overvoltage_limit_pu:.2f} pu",
    )
    ax2.axvline(dead_time, linestyle=":", linewidth=1.5, label="Religamento")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("FO(t) (pu)")
    ax2.set_title(f"Circuito {circuit_result.circuit_index} – Fator de sobretensão")
    ax2.grid(True)
    ax2.legend(loc="best")
    img2_b64 = _fig_to_base64(fig2)

    return img1_b64, img2_b64


def plot_overvoltage_surface(
    params: LineParamsResult,
    config: ReclosingConfig,
    trapped_min: float = 0.0,
    trapped_max: float = 1.0,
) -> str:
    """
    Gera gráfico 3D de FO(t_dead) em função de:
      - tempo de religamento (t_dead)
      - fator de carga presa (k_trap)
    Retorna base64 PNG.
    """
    if config.grid_n_dead < 2:
        raise ValueError("grid_n_dead deve ser >= 2.")
    if config.grid_n_trap < 2:
        raise ValueError("grid_n_trap deve ser >= 2.")

    trapped_min = float(trapped_min)
    trapped_max = float(trapped_max)
    if trapped_max < trapped_min:
        trapped_min, trapped_max = trapped_max, trapped_min

    t_dead_values = np.linspace(config.t_min_window_s, config.t_max_window_s, config.grid_n_dead).tolist()
    trapped_values = np.linspace(trapped_min, trapped_max, config.grid_n_trap).tolist()

    L_total_H = float(params.L_H_km) * float(config.length_km)
    C_total_F = float(params.C_F_km) * float(config.length_km)
    if L_total_H <= 0 or C_total_F <= 0:
        raise ValueError(f"Circuito {params.circuit_index}: L_total_H ou C_total_F inválidos.")

    t_sim_local = max(config.t_max_window_s, config.dt_s)
    dt_local = config.dt_s

    Z: List[List[float]] = []
    for k_trap in trapped_values:
        t_s, _Vs_V, _Vline_V, _dV_V, FO_pu = _simulate_reclosing_time_series(
            V_LL_kV=config.V_LL_kV,
            f_hz=config.f_hz,
            L_total_H=L_total_H,
            C_total_F=C_total_F,
            trapped_kpu=k_trap,
            damping_alpha=config.damping_alpha,
            t_sim_s=t_sim_local,
            dt_s=dt_local,
        )

        row: List[float] = []
        for t_dead in t_dead_values:
            idx = min(range(len(t_s)), key=lambda i: abs(t_s[i] - t_dead))
            row.append(float(FO_pu[idx]))
        Z.append(row)

    T_dead, K_trap = np.meshgrid(t_dead_values, trapped_values)
    Z_np = np.array(Z, dtype=float)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        T_dead,
        K_trap,
        Z_np,
        edgecolor="none",
        alpha=0.9,
    )
    ax.set_xlabel("Tempo de religamento (s)")
    ax.set_ylabel("Fator de carga presa (pu)")
    ax.set_zlabel("FO(t_dead) (pu)")
    ax.set_title(f"Circuito {params.circuit_index} – FO(t_dead) vs tempo e carga presa")
    fig.colorbar(surf, shrink=0.6, aspect=10)

    return _fig_to_base64(fig)


# ======================= Relatório HTML + CSS ========================


def generate_html_report_reclosing(
    project: ProjectInfo,
    geom: Optional[LineGeometry],
    params_by_circuit: Dict[int, LineParamsResult],
    study: ReclosingStudyResult,
) -> str:
    """
    Gera relatório HTML completo do estudo de religamento tripolar.
    """
    config = study.config

    css = """
    <style>
      body { font-family: "Segoe UI", Arial, sans-serif; background-color:#f5f7fa; color:#222; margin:0; padding:0; }
      .container { max-width:1080px; margin:0 auto; padding:24px; background:#fff; box-shadow:0 4px 16px rgba(0,0,0,0.08); }
      h1,h2,h3 { color:#0b3c5d; }
      .header { border-bottom:2px solid #e0e4ea; margin-bottom:16px; padding-bottom:8px; }
      .meta { font-size:0.95rem; color:#555; }
      table { width:100%; border-collapse:collapse; margin:12px 0; font-size:0.9rem; }
      th,td { border:1px solid #dde2eb; padding:6px 8px; text-align:right; }
      th { background-color:#f0f3f9; font-weight:600; }
      td.label { text-align:left; font-weight:500; }
      .ok { color:#0b8f55; font-weight:600; }
      .nok { color:#c0392b; font-weight:600; }
      .img-block { text-align:center; margin:16px 0; }
      .eq-block { background-color:#f8fafc; border-left:4px solid #0b3c5d; padding:8px 12px; font-family:Consolas, "Courier New", monospace; font-size:0.85rem; }
      .small-note { font-size:0.8rem; color:#777; }
      .section { margin-top:18px; }
    </style>
    """

    metodologia_html = f"""
    <div class="section">
      <h2>Metodologia do Estudo de Religamento Tripolar</h2>
      <p>
        O estudo foi desenvolvido com base em conceitos de coordenação de isolamento (IEC 60071, IEEE Std 1313.1)
        e recomendações de religamento automático (ex.: IEEE C37.104). O objetivo é avaliar o fator de sobretensão
        no instante do fechamento do disjuntor após um "dead time", considerando a carga presa (trapped charge).
      </p>
      <p>
        Cada circuito é representado por um equivalente monofásico (por fase) com indutância total L<sub>total</sub>
        e capacitância total C<sub>total</sub>, calculadas a partir dos parâmetros por km (L' e C') do estudo de
        parâmetros elétricos:
      </p>
      <div class="eq-block">
        L_total = L' · ℓ  [H]<br/>
        C_total = C' · ℓ  [F]<br/>
        ω₀ = 1 / √(L_total · C_total)   [rad/s]<br/>
        f₀ = ω₀ / (2π)                  [Hz]<br/>
        V_fase,base = V_LL / √3         [V(RMS)]<br/>
        V_fase,amp = √2 · V_fase,base   [V(amp)]<br/>
        V_s(t) = V_fase,amp · sen(ω_s t),  ω_s = 2π f_s<br/>
        V_linha(t) ≈ k_trap · V_fase,amp · cos(ω₀ t) · e^(-α t)<br/>
        ΔV(t) = V_s(t) − V_linha(t)<br/>
        FO(t) = |ΔV(t)| / V_fase,base   [pu]<br/>
      </div>
      <p>
        O valor FO(T<sub>dead</sub>) é comparado ao limite de sobretensão adotado.
        Também são determinadas janelas de tempo em que FO(t) permanece abaixo do limite, e geradas superfícies 3D de FO
        em função de T<sub>dead</sub> e k<sub>trap</sub>.
      </p>
    </div>
    """

    geom_html = ""
    if geom is not None and getattr(geom, "conductors", None):
        rows = []
        for c in geom.conductors:
            is_shield = bool(getattr(c, "is_shield", False))
            phase = getattr(c, "phase", None)
            tipo = "Cabo-guarda" if is_shield else f"Fase {phase or '-'}"
            rows.append(
                "<tr>"
                f"<td class='label'>{getattr(c, 'name', '')}</td>"
                f"<td>{getattr(c, 'circuit_index', '')}</td>"
                f"<td>{tipo}</td>"
                f"<td>{float(getattr(c, 'x_m', 0.0)):.3f}</td>"
                f"<td>{float(getattr(c, 'y_m', 0.0)):.3f}</td>"
                "</tr>"
            )
        geom_html = (
            "<div class='section'>"
            "<h2>Resumo da Geometria dos Condutores</h2>"
            "<table>"
            "<tr>"
            "<th class='label'>Condutor</th><th>Circuito</th><th>Tipo</th><th>x (m)</th><th>y (m)</th>"
            "</tr>"
            + "\n".join(rows)
            + "</table>"
            "</div>"
        )

    circuits_html = ""
    for cidx, circ_res in study.per_circuit.items():
        img_ts_1, img_ts_2 = plot_reclosing_time_series(circ_res)
        img_3d = plot_overvoltage_surface(
            params=params_by_circuit[cidx],
            config=config,
            trapped_min=0.0,
            trapped_max=1.0,
        )

        if circ_res.is_dead_time_acceptable:
            status_text = (
                f"<span class='ok'>Conforme (FO_dead = {circ_res.FO_dead_pu:.2f} pu ≤ "
                f"{circ_res.overvoltage_limit_pu:.2f} pu)</span>"
            )
        else:
            status_text = (
                f"<span class='nok'>Não conforme (FO_dead = {circ_res.FO_dead_pu:.2f} pu &gt; "
                f"{circ_res.overvoltage_limit_pu:.2f} pu)</span>"
            )

        if circ_res.acceptable_windows:
            win_rows = [
                f"<tr><td>{w.t_start_s:.4f}</td><td>{w.t_end_s:.4f}</td></tr>"
                for w in circ_res.acceptable_windows
            ]
            windows_html = (
                "<table>"
                "<tr><th>Início (s)</th><th>Fim (s)</th></tr>"
                + "\n".join(win_rows)
                + "</table>"
            )
        else:
            windows_html = (
                "<p>Não foram identificadas janelas em que FO(t) permaneça abaixo do limite "
                "no intervalo analisado.</p>"
            )

        circuits_html += f"""
        <div class="section">
          <h2>Circuito {cidx}</h2>

          <table>
            <tr><th class="label">Parâmetro</th><th>Valor</th><th>Unidade</th></tr>
            <tr><td class="label">L_total</td><td>{circ_res.L_total_H:.6e}</td><td>H</td></tr>
            <tr><td class="label">C_total</td><td>{circ_res.C_total_F:.6e}</td><td>F</td></tr>
            <tr><td class="label">f_sistema</td><td>{config.f_hz:.3f}</td><td>Hz</td></tr>
            <tr><td class="label">f_natural (f₀)</td><td>{circ_res.f0_hz:.3f}</td><td>Hz</td></tr>
            <tr><td class="label">T_dead</td><td>{circ_res.dead_time_s:.4f}</td><td>s</td></tr>
            <tr><td class="label">k_trap</td><td>{circ_res.trapped_kpu:.3f}</td><td>pu</td></tr>
            <tr><td class="label">FO_dead</td><td>{circ_res.FO_dead_pu:.3f}</td><td>pu</td></tr>
            <tr><td class="label">FO_max (0–{config.t_sim_s:.3f}s)</td><td>{circ_res.FO_max_pu:.3f}</td><td>pu</td></tr>
            <tr><td class="label">Limite adotado</td><td>{circ_res.overvoltage_limit_pu:.3f}</td><td>pu</td></tr>
            <tr><td class="label">Avaliação em T_dead</td><td colspan="2">{status_text}</td></tr>
          </table>

          <h3>Tensões Vs(t) e Vlinha(t)</h3>
          <div class="img-block">
            <img src="data:image/png;base64,{img_ts_1}" alt="Vs e Vlinha – Circuito {cidx}"/>
          </div>

          <h3>Fator de Sobretensão FO(t)</h3>
          <div class="img-block">
            <img src="data:image/png;base64,{img_ts_2}" alt="FO(t) – Circuito {cidx}"/>
          </div>

          <h3>Superfície 3D – FO(t_dead) em função de T_dead e k_trap</h3>
          <div class="img-block">
            <img src="data:image/png;base64,{img_3d}" alt="FO 3D – Circuito {cidx}"/>
          </div>

          <h3>Janelas recomendadas (FO(t) ≤ limite)</h3>
          {windows_html}
        </div>
        """

    conclusao_html = f"""
    <div class="section">
      <h2>Discussão e Conclusões</h2>
      <p>
        Os resultados de FO_dead e FO_max por circuito permitem avaliar se o esquema de religamento tripolar proposto
        mantém as sobretensões dentro do limite adotado (≈ {config.overvoltage_limit_pu:.2f} pu), apoiando a coordenação
        de isolamento.
      </p>
      <p class="small-note">
        Nota: modelo simplificado (equivalente LC). Para validação final, recomenda-se estudo EMT detalhado quando aplicável.
      </p>
    </div>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="utf-8"/>
      <title>Relatório – Estudo de Religamento Tripolar – {project.nome_projeto}</title>
      {css}
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>BK_Estudos_Eletricos – Estudo de Religamento Tripolar</h1>
          <div class="meta">
            <strong>Projeto:</strong> {project.nome_projeto}<br/>
            <strong>Cliente:</strong> {project.cliente}<br/>
            <strong>Nº Projeto:</strong> {project.numero_projeto}<br/>
            <strong>Tensão nominal:</strong> {config.V_LL_kV:.1f} kV (L-L)<br/>
            <strong>Frequência:</strong> {config.f_hz:.2f} Hz<br/>
            <strong>Comprimento:</strong> {config.length_km:.3f} km<br/>
            <strong>T_dead:</strong> {config.dead_time_s:.4f} s<br/>
            <strong>Limite adotado:</strong> {config.overvoltage_limit_pu:.2f} pu<br/>
          </div>
        </div>

        {geom_html}
        {metodologia_html}
        {circuits_html}
        {conclusao_html}
      </div>
    </body>
    </html>
    """

    return html


# ======================= Teste rápido (opcional) =====================

if __name__ == "__main__":
    # Mantido propositalmente simples: o uso típico é via UI/main_app.
    print("Módulo reclosing_tripolar carregado. Use via UI ou rotinas de estudo.")

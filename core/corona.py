# bk_estudos_eletricos/core/corona.py
# Estudo de efeito corona em linhas aéreas (Peek) – BK_Estudos_Eletricos
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import math
import base64
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt

from .cables import (
    Cable,
    default_cable_db,
    find_cable,
)
from .line_params import (
    LineGeometry,
    compute_GMD_for_circuit,
    ProjectInfo,
)

# -------------------- Configuração do estudo -------------------------


@dataclass
class CoronaConfig:
    """
    Configuração do estudo de corona para um circuito específico.

    - circuit_index: índice do circuito (1..)
    - V_LL_kV: tensão linha-linha (kV)
    - f_hz: frequência (Hz)
    - temp_C: temperatura ambiente (°C)
    - pressure_kPa: pressão local (kPa) — se <=0, será estimada pela altitude
    - altitude_m: altitude aproximada (m) (usada se pressure_kPa <= 0)
    - weather: condição de superfície do condutor (string)
    - k_factor: fator de sobretensão (IEC) para verificação de coordenação
    """
    circuit_index: int
    V_LL_kV: float
    f_hz: float
    temp_C: float
    pressure_kPa: float
    altitude_m: float
    weather: str = "normal"
    k_factor: float = 1.1


# -------------------- Resultados por circuito ------------------------


@dataclass
class CoronaCircuitResult:
    circuit_index: int

    V_phase_kV: float
    Vd_phase_kV: float
    Vd_LL_kV: float
    Ec_crit_kV_cm: float
    Esurface_kV_cm: float

    corona_loss_kW_km_phase: float
    # Observação: "total" aqui é a potência total estimada ao longo do comprimento analisado (kW),
    # mantendo o nome para compatibilidade com módulos/relatórios existentes.
    corona_loss_kW_km_total: float

    delta_air: float
    m0: float

    GMD_m: float
    r_eq_m: float
    r_eq_cm: float

    Ve_surge_LL_kV: float
    margin_Vd_percent: float

    corona_ok: bool
    corona_message: str


# -------------------- Helpers ---------------------------------------


def weather_to_m0(weather: str) -> float:
    """
    Fator de irregularidade superficial m0 (Peek), aproximado por condição do condutor.
    """
    w = (weather or "").strip().lower()
    if "bril" in w:
        return 1.0
    if "limp" in w:
        return 0.98
    if "rug" in w or "velho" in w or "envelhec" in w:
        return 0.85
    return 0.96


def air_density_factor(temp_C: float, pressure_kPa: float, altitude_m: float) -> float:
    """
    Estima fator de densidade do ar δ.
    Se pressure_kPa <= 0, estima pressão pelo modelo exponencial com altitude.
    """
    if pressure_kPa is None or pressure_kPa <= 0:
        pressure_kPa = 101.3 * math.exp(-altitude_m / 8150.0)
    p_atm = pressure_kPa / 101.3
    delta = p_atm * 293.0 / (273.0 + temp_C)
    return max(0.3, min(delta, 1.3))


def peek_critical_voltage_phase_kV(
    r_eq_cm: float,
    GMD_cm: float,
    m0: float,
    delta: float,
) -> Tuple[float, float]:
    """
    Fórmula de Peek: retorna (Vd_phase_kV, Ec_crit_kV_cm).
    """
    if r_eq_cm <= 0 or GMD_cm <= r_eq_cm:
        return 0.0, 0.0

    base_ln = math.log(GMD_cm / r_eq_cm)
    if base_ln <= 0:
        return 0.0, 0.0

    aux = 1.0 + 0.301 / math.sqrt(max(1e-12, delta * r_eq_cm))
    Vd_phase_kV = 21.1 * m0 * delta * r_eq_cm * aux * base_ln

    Ec_kV_cm = Vd_phase_kV / (r_eq_cm * base_ln) if base_ln > 0 else 0.0
    return Vd_phase_kV, Ec_kV_cm


def peek_corona_loss_kW_km_phase(
    f_hz: float,
    V_phase_kV: float,
    Vd_phase_kV: float,
    r_eq_cm: float,
    GMD_cm: float,
    m0: float,
    delta: float,
) -> float:
    """
    Perda por corona por fase (kW/km) segundo aproximação de Peek.
    """
    if V_phase_kV <= Vd_phase_kV or r_eq_cm <= 0 or GMD_cm <= 0:
        return 0.0

    denom = m0 * delta * r_eq_cm
    if denom <= 0:
        return 0.0

    term = (V_phase_kV / denom) - (Vd_phase_kV / denom)
    Pc = 241e-5 * (f_hz + 25.0) * (term**2) * math.sqrt(r_eq_cm / GMD_cm)
    return max(0.0, Pc)


# -------------------- Seleção de cabo/fase ---------------------------


def _select_phase_A_cable(
    geom: LineGeometry,
    circuit_index: int,
    cable_db: List[Cable],
) -> Tuple[Cable, float, float]:
    """
    Retorna (cable, r_eq_m, GMD_m) para o circuito (usa fase A ou primeira fase disponível).
    """
    phases = geom.phases_of_circuit(circuit_index)
    if not phases:
        raise ValueError(f"Nenhum condutor de fase cadastrado para o circuito {circuit_index}.")

    phase_A = phases.get("A") or list(phases.values())[0]
    cable = find_cable(cable_db, phase_A.cable_key)
    if cable is None:
        raise ValueError(
            f"Cabo '{phase_A.cable_key}' não encontrado no banco para o circuito {circuit_index}."
        )

    GMD_m = compute_GMD_for_circuit(geom, circuit_index)

    # tenta obter r_eq do bundle; se falhar, utiliza raio físico
    r_eq_m: float
    try:
        _GMR_eq_m, r_eq_m = cable.bundle_equivalents(
            n_bundle=max(1, getattr(phase_A, "bundle_n", 1)),
            ds_m=getattr(phase_A, "ds_bundle_m", 0.0),
        )
    except Exception:
        r_eq_m = getattr(cable, "radius_m", 0.0)

    if not r_eq_m or r_eq_m <= 0:
        r_eq_m = getattr(cable, "radius_m", 0.0)

    if r_eq_m <= 0:
        raise ValueError("Raio equivalente do condutor inválido (r_eq_m <= 0).")

    return cable, r_eq_m, GMD_m


# -------------------- Cálculo por circuito ---------------------------


def compute_corona_for_circuit(
    geom: LineGeometry,
    circuit_index: int,
    config: CoronaConfig,
    cable_db: Optional[List[Cable]] = None,
    length_km: float = 1.0,
) -> CoronaCircuitResult:
    """
    Calcula corona para um circuito (Peek), devolve CoronaCircuitResult.
    """
    if cable_db is None:
        cable_db = default_cable_db()

    _cable, r_eq_m, GMD_m = _select_phase_A_cable(geom, circuit_index, cable_db)
    if r_eq_m <= 0 or GMD_m <= 0:
        raise ValueError("Dados geométricos inválidos para cálculo de corona (r_eq_m ou GMD_m <= 0).")

    r_eq_cm = r_eq_m * 100.0
    GMD_cm = GMD_m * 100.0

    delta = air_density_factor(config.temp_C, config.pressure_kPa, config.altitude_m)
    m0 = weather_to_m0(config.weather)

    V_phase_kV = config.V_LL_kV / math.sqrt(3.0)

    Vd_phase_kV, Ec_crit_kV_cm = peek_critical_voltage_phase_kV(
        r_eq_cm=r_eq_cm,
        GMD_cm=GMD_cm,
        m0=m0,
        delta=delta,
    )
    Vd_LL_kV = Vd_phase_kV * math.sqrt(3.0)

    base_ln = math.log(GMD_cm / r_eq_cm) if (GMD_cm > r_eq_cm and r_eq_cm > 0) else 0.0
    Esurface_kV_cm = V_phase_kV / (r_eq_cm * base_ln) if base_ln > 0 else 0.0

    corona_loss_kW_km_phase = peek_corona_loss_kW_km_phase(
        f_hz=config.f_hz,
        V_phase_kV=V_phase_kV,
        Vd_phase_kV=Vd_phase_kV,
        r_eq_cm=r_eq_cm,
        GMD_cm=GMD_cm,
        m0=m0,
        delta=delta,
    )

    corona_loss_total_kW = corona_loss_kW_km_phase * 3.0 * max(length_km, 0.0)

    Ve_surge_LL_kV = config.k_factor * config.V_LL_kV
    margin_Vd_percent = ((Vd_LL_kV - Ve_surge_LL_kV) / Vd_LL_kV * 100.0) if Vd_LL_kV > 0 else -999.0

    corona_ok = Vd_LL_kV >= Ve_surge_LL_kV
    if corona_ok:
        corona_message = (
            "A coordenação de corona atende ao critério: tensão crítica disruptiva ≥ tensão de sobretensão considerada."
        )
    else:
        corona_message = (
            "A coordenação de corona NÃO atende ao critério: tensão crítica disruptiva < tensão de sobretensão considerada."
        )

    return CoronaCircuitResult(
        circuit_index=circuit_index,
        V_phase_kV=V_phase_kV,
        Vd_phase_kV=Vd_phase_kV,
        Vd_LL_kV=Vd_LL_kV,
        Ec_crit_kV_cm=Ec_crit_kV_cm,
        Esurface_kV_cm=Esurface_kV_cm,
        corona_loss_kW_km_phase=corona_loss_kW_km_phase,
        corona_loss_kW_km_total=corona_loss_total_kW,
        delta_air=delta,
        m0=m0,
        GMD_m=GMD_m,
        r_eq_m=r_eq_m,
        r_eq_cm=r_eq_cm,
        Ve_surge_LL_kV=Ve_surge_LL_kV,
        margin_Vd_percent=margin_Vd_percent,
        corona_ok=corona_ok,
        corona_message=corona_message,
    )


def compute_corona_all_circuits(
    geom: LineGeometry,
    config_by_circuit: Dict[int, CoronaConfig],
    cable_db: Optional[List[Cable]] = None,
    length_km: float = 1.0,
) -> Dict[int, CoronaCircuitResult]:
    results: Dict[int, CoronaCircuitResult] = {}
    for cidx, cfg in config_by_circuit.items():
        results[cidx] = compute_corona_for_circuit(
            geom=geom,
            circuit_index=cidx,
            config=cfg,
            cable_db=cable_db,
            length_km=length_km,
        )
    return results


# -------------------- Plots / imagens -------------------------------


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_corona_voltage_loss_curve(
    result: CoronaCircuitResult,
    config: CoronaConfig,
    n_points: int = 40,
) -> str:
    Vnom_LL = config.V_LL_kV
    V_LL_vals = np.linspace(0.8 * Vnom_LL, 1.4 * Vnom_LL, n_points)
    V_phase_vals = V_LL_vals / math.sqrt(3.0)

    Pc_vals: List[float] = []
    GMD_cm = result.GMD_m * 100.0

    for Vph in V_phase_vals:
        Pc = peek_corona_loss_kW_km_phase(
            f_hz=config.f_hz,
            V_phase_kV=float(Vph),
            Vd_phase_kV=result.Vd_phase_kV,
            r_eq_cm=result.r_eq_cm,
            GMD_cm=GMD_cm,
            m0=result.m0,
            delta=result.delta_air,
        )
        Pc_vals.append(Pc)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(V_LL_vals, Pc_vals, linewidth=2)
    ax.set_xlabel("Tensão linha-linha V_LL (kV)")
    ax.set_ylabel("Perda por corona por fase (kW/km)")
    ax.set_title(f"Circuito {result.circuit_index} – Perda por corona vs. V_LL")
    ax.grid(True)
    return _fig_to_base64(fig)


def plot_corona_Esurface_curve(
    result: CoronaCircuitResult,
    config: CoronaConfig,
    n_points: int = 40,
) -> str:
    Vnom_LL = config.V_LL_kV
    V_LL_vals = np.linspace(0.5 * Vnom_LL, 1.4 * Vnom_LL, n_points)
    V_phase_vals = V_LL_vals / math.sqrt(3.0)

    base_ln = math.log((result.GMD_m * 100.0) / result.r_eq_cm) if (result.GMD_m > 0 and result.r_eq_cm > 0) else 0.0
    Es_vals: List[float] = []
    for Vph in V_phase_vals:
        Es_vals.append(float(Vph) / (result.r_eq_cm * base_ln) if base_ln > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(V_LL_vals, Es_vals, linewidth=2, label="E_surface (kV/cm)")
    ax.axhline(result.Ec_crit_kV_cm, linestyle="--", label="E_c crítico (Peek)")
    ax.set_xlabel("Tensão linha-linha V_LL (kV)")
    ax.set_ylabel("Gradiente na superfície (kV/cm)")
    ax.set_title(f"Circuito {result.circuit_index} – Gradiente na superfície vs. V_LL")
    ax.grid(True)
    ax.legend()
    return _fig_to_base64(fig)


def plot_corona_loss_3d_delta_voltage(
    result: CoronaCircuitResult,
    config: CoronaConfig,
    n_delta: int = 20,
    n_V: int = 20,
) -> str:
    """
    Superfície 3D: perda por corona (kW/km/fase) em função de V_LL e δ.
    """
    Vnom_LL = config.V_LL_kV
    V_phase_vals = np.linspace(0.8 * Vnom_LL, 1.3 * Vnom_LL, n_V) / math.sqrt(3.0)
    delta_vals = np.linspace(0.8 * result.delta_air, 1.2 * result.delta_air, n_delta)

    V_phase_grid, delta_grid = np.meshgrid(V_phase_vals, delta_vals)  # (n_delta, n_V)
    Pc_grid = np.zeros_like(V_phase_grid, dtype=float)

    GMD_cm = result.GMD_m * 100.0

    for i in range(delta_grid.shape[0]):
        for j in range(delta_grid.shape[1]):
            Vd_ph, _Ec = peek_critical_voltage_phase_kV(
                r_eq_cm=result.r_eq_cm,
                GMD_cm=GMD_cm,
                m0=result.m0,
                delta=float(delta_grid[i, j]),
            )
            Pc_grid[i, j] = peek_corona_loss_kW_km_phase(
                f_hz=config.f_hz,
                V_phase_kV=float(V_phase_grid[i, j]),
                Vd_phase_kV=Vd_ph,
                r_eq_cm=result.r_eq_cm,
                GMD_cm=GMD_cm,
                m0=result.m0,
                delta=float(delta_grid[i, j]),
            )

    V_LL_grid = V_phase_grid * math.sqrt(3.0)

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(V_LL_grid, delta_grid, Pc_grid, edgecolor="k", linewidth=0.2)
    ax.set_xlabel("Tensão V_LL (kV)")
    ax.set_ylabel("Fator δ")
    ax.set_zlabel("Perda por corona (kW/km/fase)")
    ax.set_title(f"Circuito {result.circuit_index} – Perda por corona vs. V_LL e δ")
    fig.colorbar(surf, shrink=0.6, aspect=16)
    return _fig_to_base64(fig)


# -------------------- Relatório HTML -------------------------------


def generate_corona_html_report(
    project: ProjectInfo,
    geom: LineGeometry,
    corona_results: Dict[int, CoronaCircuitResult],
    config_by_circuit: Dict[int, CoronaConfig],
    length_km: float,
) -> str:
    """
    Relatório HTML (padrão BK) para Corona.

    Observação: conforme padrão solicitado, este relatório NÃO inclui gráfico de geometria.
    """
    corona_volt_loss_imgs: Dict[int, str] = {}
    corona_Esurface_imgs: Dict[int, str] = {}
    corona_loss_3d_imgs: Dict[int, str] = {}

    for cidx, res in corona_results.items():
        cfg = config_by_circuit[cidx]
        corona_volt_loss_imgs[cidx] = plot_corona_voltage_loss_curve(res, cfg)
        corona_Esurface_imgs[cidx] = plot_corona_Esurface_curve(res, cfg)
        corona_loss_3d_imgs[cidx] = plot_corona_loss_3d_delta_voltage(res, cfg)

    css = """
    <style>
      body { font-family:"Segoe UI", Arial, sans-serif; background:#f5f7fa; color:#222; margin:0; padding:0; }
      .container { max-width:1100px; margin:0 auto; padding:24px; background:#fff; box-shadow:0 4px 16px rgba(0,0,0,0.08); }
      h1,h2,h3 { color:#0b3c5d; }
      .header { border-bottom:2px solid #e0e4ea; margin-bottom:16px; padding-bottom:10px; }
      .meta { font-size:0.95rem; color:#555; }
      table { width:100%; border-collapse:collapse; margin:12px 0; font-size:0.9rem; }
      th,td { border:1px solid #dde2eb; padding:6px 8px; text-align:right; }
      th { background:#f0f3f9; font-weight:600; }
      td.label { text-align:left; font-weight:500; }
      .img-block { text-align:center; margin:16px 0; }
      .eq-block { background:#f8fafc; border-left:4px solid #0b3c5d; padding:8px 12px; font-family:Consolas,monospace; font-size:0.85rem; margin:10px 0; }
      .small-note { font-size:0.8rem; color:#777; }
      .ok { color:#0b7a34; font-weight:600; }
      .not-ok { color:#c0392b; font-weight:600; }
    </style>
    """

    metodologia_html = """
    <div class="eq-block">
      <strong>Metodologia do Estudo de Corona</strong><br/><br/>
      Modelagem baseada nas formulações de Peek para tensão crítica disruptiva e perdas por corona,
      com conversões de unidades (cm) e estimativa do fator δ (densidade do ar) para condições locais.
      A verificação de coordenação é realizada comparando-se Vd_LL com k·V_LL (sobretensão considerada).
    </div>
    """

    circuits_html = ""
    for cidx in sorted(corona_results.keys()):
        res = corona_results[cidx]
        cfg = config_by_circuit[cidx]
        status_class = "ok" if res.corona_ok else "not-ok"
        status_text = (
            "ATENDE ao critério de coordenação de isolamento por corona."
            if res.corona_ok
            else "NÃO atende ao critério de coordenação de isolamento por corona."
        )

        circuits_html += f"""
        <h2>Circuito {cidx}</h2>
        <table>
          <tr><th class="label">Grandeza</th><th>Valor</th><th>Unidade</th></tr>
          <tr><td class="label">Tensão nominal V_LL</td><td>{cfg.V_LL_kV:.3f}</td><td>kV</td></tr>
          <tr><td class="label">Tensão por fase V_fase</td><td>{res.V_phase_kV:.3f}</td><td>kV</td></tr>
          <tr><td class="label">Tensão crítica Vd_fase</td><td>{res.Vd_phase_kV:.3f}</td><td>kV</td></tr>
          <tr><td class="label">Tensão crítica Vd_LL</td><td>{res.Vd_LL_kV:.3f}</td><td>kV</td></tr>
          <tr><td class="label">Fator k de sobretensão</td><td>{cfg.k_factor:.3f}</td><td>-</td></tr>
          <tr><td class="label">V_e (k·V_LL)</td><td>{res.Ve_surge_LL_kV:.3f}</td><td>kV</td></tr>
          <tr><td class="label">Margem (Vd_LL − V_e)/Vd_LL</td><td>{res.margin_Vd_percent:.2f}</td><td>%</td></tr>
          <tr><td class="label">δ (fator de densidade do ar)</td><td>{res.delta_air:.3f}</td><td>-</td></tr>
          <tr><td class="label">m₀ (rugosidade)</td><td>{res.m0:.3f}</td><td>-</td></tr>
          <tr><td class="label">r_eq</td><td>{res.r_eq_m:.4f}</td><td>m</td></tr>
          <tr><td class="label">GMD</td><td>{res.GMD_m:.3f}</td><td>m</td></tr>
          <tr><td class="label">E<sub>c, crit</sub></td><td>{res.Ec_crit_kV_cm:.4f}</td><td>kV/cm</td></tr>
          <tr><td class="label">E<sub>surface</sub> (V_LL nominal)</td><td>{res.Esurface_kV_cm:.4f}</td><td>kV/cm</td></tr>
          <tr><td class="label">P<sub>c</sub> por fase</td><td>{res.corona_loss_kW_km_phase:.6f}</td><td>kW/km/fase</td></tr>
          <tr><td class="label">P<sub>c</sub> total (3φ, {length_km:.3f} km)</td><td>{res.corona_loss_kW_km_total:.6f}</td><td>kW</td></tr>
        </table>

        <p class="{status_class}">{status_text}</p>
        <p>{res.corona_message}</p>

        <h3>Gráfico 1 – Perda por corona vs. Tensão V_LL</h3>
        <div class="img-block"><img src="data:image/png;base64,{corona_volt_loss_imgs[cidx]}" alt="Perda por corona"/></div>

        <h3>Gráfico 2 – Gradiente na superfície vs. Tensão V_LL</h3>
        <div class="img-block"><img src="data:image/png;base64,{corona_Esurface_imgs[cidx]}" alt="Gradiente na superfície"/></div>

        <h3>Gráfico 3 – Perda por corona (3D) vs. V_LL e δ</h3>
        <div class="img-block"><img src="data:image/png;base64,{corona_loss_3d_imgs[cidx]}" alt="Perda 3D"/></div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="utf-8"/>
      <title>Estudo de Corona – {project.nome_projeto}</title>
      {css}
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>BK_Estudos_Eletricos – Estudo de Efeito Corona</h1>
          <div class="meta">
            <strong>Projeto:</strong> {project.nome_projeto}<br/>
            <strong>Cliente:</strong> {project.cliente}<br/>
            <strong>Nº Projeto:</strong> {project.numero_projeto}<br/>
            <strong>Comprimento da linha analisada:</strong> {length_km:.3f} km
          </div>
        </div>

        <h2>Resultados do Estudo de Corona</h2>
        {circuits_html}

        <h2>Metodologia</h2>
        {metodologia_html}

        <h2>Conclusões</h2>
        <p>Resumo das verificações de coordenação de isolamento com base em corona e recomendações quando necessário.</p>
        <p class="small-note">Se critérios não atendidos, reveja geometria, tipo/diâmetro do condutor, bundle ou coordenação (k) adotada.</p>
      </div>
    </body>
    </html>
    """
    return html


# -------------------- Teste rápido (opcional) ------------------------

if __name__ == "__main__":
    from .line_params import ConductorInstance

    db_cabos = default_cable_db()

    geom = LineGeometry(conductors=[
        ConductorInstance(name="C1_A", cable_key="ACSR_477", x_m=0.0,  y_m=15.0, circuit_index=1, phase="A", bundle_n=1),
        ConductorInstance(name="C1_B", cable_key="ACSR_477", x_m=8.0,  y_m=15.0, circuit_index=1, phase="B", bundle_n=1),
        ConductorInstance(name="C1_C", cable_key="ACSR_477", x_m=16.0, y_m=15.0, circuit_index=1, phase="C", bundle_n=1),
    ])

    cfg = CoronaConfig(
        circuit_index=1,
        V_LL_kV=138.0,
        f_hz=60.0,
        temp_C=25.0,
        pressure_kPa=101.3,
        altitude_m=800.0,
        weather="normal",
        k_factor=1.15,
    )

    res = compute_corona_for_circuit(
        geom=geom,
        circuit_index=1,
        config=cfg,
        cable_db=db_cabos,
        length_km=100.0,
    )

    proj = ProjectInfo(
        nome_projeto="Linha 138 kV – Corona Exemplo",
        cliente="BK Engenharia",
        numero_projeto="2025-001",
    )

    html = generate_corona_html_report(
        project=proj,
        geom=geom,
        corona_results={1: res},
        config_by_circuit={1: cfg},
        length_km=100.0,
    )

    with open("relatorio_corona_exemplo.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Relatório de corona gerado: relatorio_corona_exemplo.html")

# bk_estudos_eletricos/core/line_params.py
# =====================================================================
# Cálculo de parâmetros elétricos de linhas aéreas multi-circuito
# =====================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import base64
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .cables import (
    Cable,
    default_cable_db,
    find_cable,
    calc_line_params_from_cable,
    EPS0,
    MU0,
)

# =====================================================================
# Estruturas de dados
# =====================================================================

@dataclass
class ProjectInfo:
    nome_projeto: str
    cliente: str
    numero_projeto: str
    # novos campos (opcionais; usados para cálculo de queda de tensão e layout)
    power_mva: float = 0.0           # potência total do projeto (MVA)
    n_circuits: int = 1              # número de circuitos do projeto
    pf: float = 1.0                  # fator de potência (0..1)
    circuits_layout: str = "side"    # "side" (lado a lado) ou "stacked" (embaixo)

@dataclass
class VoltageSpec:
    magnitude_kV: float
    angle_deg: float

    @property
    def angle_rad(self) -> float:
        return math.radians(self.angle_deg)

@dataclass
class ConductorInstance:
    name: str
    cable_key: str
    x_m: float
    y_m: float
    circuit_index: int
    phase: Optional[str] = None
    bundle_n: int = 1
    ds_bundle_m: float = 0.4
    is_shield: bool = False

@dataclass
class LineGeometry:
    conductors: List[ConductorInstance]

    def circuits(self) -> List[int]:
        return sorted({c.circuit_index for c in self.conductors})

    def phases_of_circuit(self, circuit_index: int) -> Dict[str, ConductorInstance]:
        phases: Dict[str, ConductorInstance] = {}
        for c in self.conductors:
            if c.circuit_index == circuit_index and c.phase in ("A", "B", "C"):
                phases[c.phase] = c
        return phases

    def shields(self) -> List[ConductorInstance]:
        return [c for c in self.conductors if c.is_shield]

@dataclass
class LineParamsResult:
    circuit_index: int
    R_ohm_km: float
    X_ohm_km: float
    B_S_km: float
    L_H_km: float
    C_F_km: float
    L_mH_km: float
    C_nF_km: float
    Zc_ohm: float
    SIL_MW: float
    lambda_m: float
    Q_mC_km_por_fase: float
    Ec_kV_cm: float
    GMD_m: float
    GMR_eq_m: float
    r_eq_m: float
    Vs: Optional[VoltageSpec] = None
    Vr: Optional[VoltageSpec] = None

# =====================================================================
# Funções auxiliares
# =====================================================================

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

def compute_GMD_for_circuit(geom: LineGeometry, circuit_index: int) -> float:
    phases = geom.phases_of_circuit(circuit_index)
    if not all(p in phases for p in ("A", "B", "C")):
        raise ValueError(f"Circuito {circuit_index}: faltam fases A/B/C na geometria.")
    A = (phases["A"].x_m, phases["A"].y_m)
    B = (phases["B"].x_m, phases["B"].y_m)
    C = (phases["C"].x_m, phases["C"].y_m)
    D_ab = distance(A, B)
    D_bc = distance(B, C)
    D_ca = distance(C, A)
    if min(D_ab, D_bc, D_ca) <= 0:
        raise ValueError(f"Circuito {circuit_index}: distâncias entre fases inválidas.")
    GMD = (D_ab * D_bc * D_ca) ** (1.0 / 3.0)
    return GMD

def compute_surface_field_Ec(V_LL_kV: float, GMD_m: float, r_eq_m: float) -> float:
    if GMD_m <= r_eq_m or r_eq_m <= 0:
        return 0.0
    V_phase_V = (V_LL_kV * 1e3) / math.sqrt(3.0)
    denom = r_eq_m * math.log(GMD_m / r_eq_m)
    if denom <= 0:
        return 0.0
    E_surface_V_m = V_phase_V / denom
    Ec_kV_cm = E_surface_V_m / 1e5
    return Ec_kV_cm

def compute_line_charge_per_phase(C_F_km: float, V_LL_kV: float) -> float:
    C_per_phase_F_km = C_F_km
    V_phase_V = (V_LL_kV * 1e3) / math.sqrt(3.0)
    q_C_km = C_per_phase_F_km * V_phase_V
    q_mC_km = q_C_km * 1e3
    return q_mC_km

def approximate_lambda(L_H_km: float, C_F_km: float, f_hz: float) -> float:
    if L_H_km <= 0 or C_F_km <= 0 or f_hz <= 0:
        return 0.0
    L_p_m = L_H_km / 1000.0
    C_p_m = C_F_km / 1000.0
    w = 2 * math.pi * f_hz
    beta = w * math.sqrt(L_p_m * C_p_m)
    if beta <= 0:
        return 0.0
    lambda_m = 2 * math.pi / beta
    return lambda_m

# =====================================================================
# Cálculo por circuito
# =====================================================================

def compute_circuit_params(
    geom: LineGeometry,
    circuit_index: int,
    cable_db: List[Cable],
    V_LL_kV: float,
    f_hz: float,
    temp_C: float,
    default_bundle_n: int = 1,
    default_ds_bundle_m: float = 0.4,
    eps_r_ambiente: float = 1.0,
    Vs: Optional[VoltageSpec] = None,
    Vr: Optional[VoltageSpec] = None,
) -> LineParamsResult:
    phases = geom.phases_of_circuit(circuit_index)
    if not phases:
        raise ValueError(f"Nenhum condutor de fase cadastrado para o circuito {circuit_index}.")
    phase_A = phases.get("A") or list(phases.values())[0]
    cable = find_cable(cable_db, phase_A.cable_key)
    if cable is None:
        raise ValueError(f"Cabo '{phase_A.cable_key}' não encontrado no banco para circuito {circuit_index}.")
    GMD_m = compute_GMD_for_circuit(geom, circuit_index)
    n_bundle = max(1, phase_A.bundle_n or default_bundle_n)
    ds_bundle_m = phase_A.ds_bundle_m or default_ds_bundle_m
    raw = calc_line_params_from_cable(
        cable=cable,
        GMD_m=GMD_m,
        f_hz=f_hz,
        temp_C=temp_C,
        n_bundle=n_bundle,
        ds_bundle_m=ds_bundle_m,
        eps_r_ambiente=eps_r_ambiente,
    )
    R_ohm_km = raw["R_ohm_km"]
    X_ohm_km = raw["X_ohm_km"]
    B_S_km = raw["B_S_km"]
    L_H_km = raw["L_H_km"]
    C_F_km = raw["C_F_km"]
    GMR_eq_m = raw.get("GMR_eq_m", 0.0)
    r_eq_m = raw.get("r_eq_m", 0.0)
    L_mH_km = L_H_km * 1e3
    C_nF_km = C_F_km * 1e9
    L_p_m = L_H_km / 1000.0
    C_p_m = C_F_km / 1000.0
    if L_p_m > 0 and C_p_m > 0:
        Zc_ohm = math.sqrt(L_p_m / C_p_m)
    else:
        Zc_ohm = 0.0
    SIL_MW = 0.0
    if Zc_ohm > 0:
        SIL_MW = (V_LL_kV ** 2) / Zc_ohm
    lambda_m = approximate_lambda(L_H_km, C_F_km, f_hz)
    Q_mC_km = compute_line_charge_per_phase(C_F_km, V_LL_kV)
    Ec_kV_cm = compute_surface_field_Ec(V_LL_kV, GMD_m, r_eq_m)
    res = LineParamsResult(
        circuit_index=circuit_index,
        R_ohm_km=R_ohm_km,
        X_ohm_km=X_ohm_km,
        B_S_km=B_S_km,
        L_H_km=L_H_km,
        C_F_km=C_F_km,
        L_mH_km=L_mH_km,
        C_nF_km=C_nF_km,
        Zc_ohm=Zc_ohm,
        SIL_MW=SIL_MW,
        lambda_m=lambda_m,
        Q_mC_km_por_fase=Q_mC_km,
        Ec_kV_cm=Ec_kV_cm,
        GMD_m=GMD_m,
        GMR_eq_m=GMR_eq_m,
        r_eq_m=r_eq_m,
        Vs=Vs,
        Vr=Vr,
    )
    return res

def compute_all_circuits_params(
    geom: LineGeometry,
    cable_db: Optional[List[Cable]],
    V_LL_kV: float,
    f_hz: float,
    temp_C: float,
    default_bundle_n: int = 1,
    default_ds_bundle_m: float = 0.4,
    eps_r_ambiente: float = 1.0,
    Vs_by_circuit: Optional[Dict[int, VoltageSpec]] = None,
    Vr_by_circuit: Optional[Dict[int, VoltageSpec]] = None,
) -> Dict[int, LineParamsResult]:
    if cable_db is None:
        cable_db = default_cable_db()
    results: Dict[int, LineParamsResult] = {}
    for cidx in geom.circuits():
        Vs = Vs_by_circuit.get(cidx) if Vs_by_circuit else None
        Vr = Vr_by_circuit.get(cidx) if Vr_by_circuit else None
        results[cidx] = compute_circuit_params(
            geom=geom,
            circuit_index=cidx,
            cable_db=cable_db,
            V_LL_kV=V_LL_kV,
            f_hz=f_hz,
            temp_C=temp_C,
            default_bundle_n=default_bundle_n,
            default_ds_bundle_m=default_ds_bundle_m,
            eps_r_ambiente=eps_r_ambiente,
            Vs=Vs,
            Vr=Vr,
        )
    return results

# =====================================================================
# Plot helpers
# =====================================================================

def plot_geometry_2d(geom: LineGeometry, title: str = "Geometria da Linha (todos os circuitos)") -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_aspect('equal', adjustable='datalim')
    ax.axhline(0.0, color='lightgray', linewidth=0.8)
    xs = [c.x_m for c in geom.conductors]
    ys = [c.y_m for c in geom.conductors]
    if xs:
        xmin, xmax = min(xs), max(xs)
        xpad = max(1.0, (xmax - xmin) * 0.1)
    else:
        xmin, xmax, xpad = -10, 10, 1.0
    if ys:
        ymin, ymax = min(ys), max(ys)
        ypad = max(1.0, (ymax - ymin) * 0.2)
    else:
        ymin, ymax, ypad = -1, 30, 5.0
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(max(0.0, ymin - ypad), ymax + ypad)
    for c in geom.conductors:
        if c.is_shield:
            ax.plot(c.x_m, c.y_m, marker='s', markersize=6)
            ax.text(c.x_m, c.y_m + 0.3, c.name, fontsize=8, ha='center')
        else:
            marker = 'o' if c.phase in ('A','B','C') else 'x'
            ax.plot(c.x_m, c.y_m, marker=marker, markersize=6)
            label = f"{c.name} ({c.phase or '-'})"
            ax.text(c.x_m, c.y_m + 0.2, label, fontsize=8, ha='center')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle=':', linewidth=0.5)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def plot_pi_equivalent_2d(params: LineParamsResult, length_km: float, title: str = "Circuito π Equivalente (por fase)") -> str:
    R = params.R_ohm_km * length_km
    X = params.X_ohm_km * length_km
    B = params.B_S_km * length_km
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axis("off")
    x_left = 0.1
    x_series_left = 0.3
    x_series_right = 0.7
    x_right = 0.9
    y_line = 0.6
    y_ground = 0.15
    ax.plot([x_left, x_series_left], [y_line, y_line], linewidth=2)
    ax.plot([x_series_right, x_right], [y_line, y_line], linewidth=2)
    series_width = x_series_right - x_series_left
    series_height = 0.25
    series_rect = Rectangle((x_series_left, y_line - series_height / 2), series_width, series_height, fill=False, linewidth=2)
    ax.add_patch(series_rect)
    ax.text((x_series_left + x_series_right) / 2, y_line, f"Zs = {R:.3f} + j{X:.3f} Ω", ha="center", va="center", fontsize=9)
    ax.plot([x_left, x_left], [y_line, y_ground + 0.4], linewidth=2)
    shunt_left_rect = Rectangle((x_left - 0.14, y_ground + 0.4), 0.28, 0.2, fill=False, linewidth=2)
    ax.add_patch(shunt_left_rect)
    ax.text(x_left, y_ground + 0.75, f"Y·ℓ/2 = j{(B / 2):.3e} S", ha="center", va="bottom", fontsize=8)
    ax.plot([x_right, x_right], [y_line, y_ground + 0.4], linewidth=2)
    shunt_right_rect = Rectangle((x_right - 0.14, y_ground + 0.4), 0.28, 0.2, fill=False, linewidth=2)
    ax.add_patch(shunt_right_rect)
    ax.text(x_right, y_ground + 0.75, f"Y·ℓ/2 = j{(B / 2):.3e} S", ha="center", va="bottom", fontsize=8)
    ax.plot([x_left, x_right], [y_ground, y_ground], linewidth=1.5)
    for xg in (x_left, x_right):
        ax.plot([xg, xg], [y_ground, y_ground - 0.05], linewidth=1.5)
        ax.plot([xg - 0.03, xg + 0.03], [y_ground - 0.05, y_ground - 0.05], linewidth=1.5)
        ax.plot([xg - 0.02, xg + 0.02], [y_ground - 0.07, y_ground - 0.07], linewidth=1.5)
    ax.text(x_left - 0.02, y_line + 0.04, "Vs", ha="right", va="bottom", fontsize=9)
    ax.text(x_right + 0.02, y_line + 0.04, "Vr", ha="left", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

# =====================================================================
# Relatório HTML atualizado (premium + Vr por queda)
# =====================================================================

def generate_html_report(
    project: ProjectInfo,
    geom: LineGeometry,
    params_by_circuit: Dict[int, LineParamsResult],
    length_km: float,
    f_hz: float,
) -> str:
    """Gera HTML (fragmento) para o relatório de **Parâmetros Elétricos**.

    - Retorna um *fragmento* (sem <html>/<head>), pois a UI aplica o template BK.
    - Inclui as informações e gráficos (condutores, geometria e circuito π).
    """

    # Gráfico da geometria (todos os circuitos)
    geom_img_b64 = None
    try:
        geom_img_b64 = plot_geometry_2d(geom, "Geometria da Linha")
    except Exception:
        geom_img_b64 = None

    # Imagens do circuito π equivalente por circuito
    pi_imgs_b64: Dict[int, Optional[str]] = {}
    for cidx, p in params_by_circuit.items():
        try:
            pi_imgs_b64[cidx] = plot_pi_equivalent_2d(
                p,
                length_km,
                f"Circuito π Equivalente – Circuito {cidx}",
            )
        except Exception:
            pi_imgs_b64[cidx] = None

    power_mva = float(getattr(project, "power_mva", 0.0) or 0.0)
    n_circuits_proj = int(getattr(project, "n_circuits", 1) or 1)
    pf_proj = float(getattr(project, "pf", 1.0) or 1.0)
    S_per_circuit_MVA = power_mva / max(1, n_circuits_proj)

    html_parts: List[str] = []

    # ------------------------------------------------------------------
    # Dados gerais
    # ------------------------------------------------------------------
    html_parts.append("<h3>Dados gerais</h3>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Item</th><th>Valor</th></tr>")
    html_parts.append(f"<tr><td>Projeto</td><td>{project.nome_projeto}</td></tr>")
    html_parts.append(f"<tr><td>Cliente</td><td>{project.cliente}</td></tr>")
    html_parts.append(f"<tr><td>Nº Projeto</td><td>{project.numero_projeto}</td></tr>")
    html_parts.append(f"<tr><td>Comprimento</td><td>{float(length_km):.3f} km</td></tr>")
    html_parts.append(f"<tr><td>Frequência</td><td>{float(f_hz):.3f} Hz</td></tr>")
    html_parts.append(f"<tr><td>Potência total</td><td>{power_mva:.3f} MVA</td></tr>")
    html_parts.append(f"<tr><td>Nº de circuitos</td><td>{n_circuits_proj}</td></tr>")
    html_parts.append(f"<tr><td>FP (carga)</td><td>{pf_proj:.3f}</td></tr>")
    html_parts.append("</table>")

    # ------------------------------------------------------------------
    # Condutores / Geometria
    # ------------------------------------------------------------------
    html_parts.append("<h3>Condutores</h3>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Nome</th><th>Circuito</th><th>Fase</th><th>Cabo</th><th>x (m)</th><th>y (m)</th><th>Feixe</th><th>Tipo</th></tr>")
    for c in (geom.conductors or []):
        phase = c.phase if c.phase is not None else "GW"
        typ = "Cabo guarda" if getattr(c, "is_shield", False) else "Fase"
        html_parts.append(
            "<tr>"
            f"<td>{c.name}</td>"
            f"<td>{getattr(c, 'circuit_index', '')}</td>"
            f"<td>{phase}</td>"
            f"<td>{getattr(c, 'cable_key', '')}</td>"
            f"<td>{float(c.x_m):.3f}</td>"
            f"<td>{float(c.y_m):.3f}</td>"
            f"<td>{int(getattr(c, 'bundle_n', 1) or 1)}</td>"
            f"<td>{typ}</td>"
            "</tr>"
        )
    html_parts.append("</table>")

    if geom_img_b64:
        html_parts.append("<p><b>Geometria (vista 2D):</b></p>")
        html_parts.append(
            "<div style='text-align:center;margin:10px 0;'>"
            f"<img style='max-width:100%;border:1px solid #ddd;border-radius:6px' "
            f"src='data:image/png;base64,{geom_img_b64}' alt='Geometria da linha'>"
            "</div>"
        )

    # ------------------------------------------------------------------
    # Resultados por circuito
    # ------------------------------------------------------------------
    for cidx, p in params_by_circuit.items():
        html_parts.append(f"<h3>Resultados – Circuito {cidx}</h3>")

        # Vs (se não definido, tenta usar a tensão do projeto)
        default_vk = float(getattr(project, "voltage_kv", 0.0) or 0.0)
        if default_vk <= 0:
            default_vk = 138.0
        Vs_spec = p.Vs if getattr(p, "Vs", None) is not None else VoltageSpec(magnitude_kV=default_vk, angle_deg=0.0)

        # Vr calculada (modelo π + carga S no terminal receptor)
        Vr_LL_kV, Vr_ang_deg, I_series_A, P_loss_MW = solve_vr_pi(
            Vs_spec=Vs_spec,
            params=p,
            length_km_loc=length_km,
            S_total_MVA_per_circ=S_per_circuit_MVA,
            pf_local=pf_proj,
        )

        Vs_str = f"{Vs_spec.magnitude_kV:.3f} kV ∠ {Vs_spec.angle_deg:.2f}°"
        Vr_str = f"{Vr_LL_kV:.3f} kV ∠ {Vr_ang_deg:.2f}°" if Vr_LL_kV is not None else "-"

        theta_deg = math.degrees(math.atan2(float(p.X_ohm_km), max(1e-12, float(p.R_ohm_km))))
        R_total_ohm = float(p.R_ohm_km) * float(length_km)
        X_total_ohm = float(p.X_ohm_km) * float(length_km)

        html_parts.append("<table>")
        html_parts.append("<tr><th>Grandeza</th><th>Valor</th><th>Unidade</th></tr>")
        html_parts.append(f"<tr><td>R'</td><td>{float(p.R_ohm_km):.6f}</td><td>Ω/km</td></tr>")
        html_parts.append(f"<tr><td>X'</td><td>{float(p.X_ohm_km):.6f}</td><td>Ω/km</td></tr>")
        html_parts.append(f"<tr><td>|Z'|</td><td>{math.hypot(float(p.R_ohm_km), float(p.X_ohm_km)):.6f}</td><td>Ω/km</td></tr>")
        html_parts.append(f"<tr><td>∠Z'</td><td>{theta_deg:.3f}</td><td>°</td></tr>")
        html_parts.append(f"<tr><td>R_total</td><td>{R_total_ohm:.4f}</td><td>Ω</td></tr>")
        html_parts.append(f"<tr><td>X_total</td><td>{X_total_ohm:.4f}</td><td>Ω</td></tr>")
        html_parts.append(f"<tr><td>B'</td><td>{float(p.B_S_km):.6e}</td><td>S/km</td></tr>")
        html_parts.append(f"<tr><td>L'</td><td>{float(p.L_mH_km):.6f}</td><td>mH/km</td></tr>")
        html_parts.append(f"<tr><td>C'</td><td>{float(p.C_nF_km):.6f}</td><td>nF/km</td></tr>")
        html_parts.append(f"<tr><td>Zc</td><td>{float(p.Zc_ohm):.3f}</td><td>Ω</td></tr>")
        html_parts.append(f"<tr><td>SIL</td><td>{float(p.SIL_MW):.3f}</td><td>MW</td></tr>")
        html_parts.append(f"<tr><td>λ</td><td>{float(p.lambda_m):.3f}</td><td>m</td></tr>")
        html_parts.append(f"<tr><td>Q</td><td>{float(p.Q_mC_km_por_fase):.6f}</td><td>mC/km/fase</td></tr>")
        html_parts.append(f"<tr><td>Ec</td><td>{float(p.Ec_kV_cm):.6f}</td><td>kV/cm</td></tr>")
        html_parts.append(f"<tr><td>GMD</td><td>{float(p.GMD_m):.3f}</td><td>m</td></tr>")
        html_parts.append(f"<tr><td>GMR_eq</td><td>{float(p.GMR_eq_m):.4f}</td><td>m</td></tr>")
        html_parts.append(f"<tr><td>r_eq</td><td>{float(p.r_eq_m):.4f}</td><td>m</td></tr>")
        html_parts.append(f"<tr><td>Vs</td><td>{Vs_str}</td><td>kV</td></tr>")
        html_parts.append(f"<tr><td>Vr (calculada)</td><td>{Vr_str}</td><td>kV</td></tr>")
        html_parts.append(f"<tr><td>I (série)</td><td>{float(I_series_A or 0.0):.2f}</td><td>A</td></tr>")
        html_parts.append(f"<tr><td>Perdas Joule (3φ)</td><td>{float(P_loss_MW or 0.0):.4f}</td><td>MW</td></tr>")

        Vr_input = getattr(p, "Vr", None)
        if Vr_input is not None:
            try:
                Vr_in_str = f"{Vr_input.magnitude_kV:.3f} kV ∠ {Vr_input.angle_deg:.2f}°"
            except Exception:
                Vr_in_str = "-"
            html_parts.append(f"<tr><td>Vr (entrada)</td><td>{Vr_in_str}</td><td>kV</td></tr>")
        html_parts.append("</table>")

        if pi_imgs_b64.get(cidx):
            html_parts.append("<p><b>Circuito π equivalente:</b></p>")
            html_parts.append(
                "<div style='text-align:center;margin:10px 0;'>"
                f"<img style='max-width:100%;border:1px solid #ddd;border-radius:6px' "
                f"src='data:image/png;base64,{pi_imgs_b64[cidx]}' alt='Circuito π'>"
                "</div>"
            )

    # ------------------------------------------------------------------
    # Equações / Referências
    # ------------------------------------------------------------------
    html_parts.append("<h3>Equações fundamentais</h3>")
    html_parts.append(
        "<div style='background:#f4f7fb;border-left:4px solid #0d47a1;padding:10px 16px;"
        "font-family:Consolas,monospace;font-size:13px;border-radius:6px;'>"
        "L' = (μ₀ / 2π) · ln(GMD / GMR_eq)  [H/m]<br>"
        "C' = (2π·ε₀·εᵣ) / ln(GMD / r_eq)  [F/m]<br>"
        "X' = 2π·f·L'  [Ω/m]<br>"
        "B' = 2π·f·C'  [S/m]<br>"
        "Zc = √(L'/C') [Ω]<br>"
        "SIL = V_LL² / Zc [MW]<br>"
        "λ = 2π / (ω · √(L'·C')) [m]<br>"
        "E_c = V_fase / (r_eq · ln(GMD / r_eq)) [V/m]<br>"
        "q = C · V_fase [C/km]<br>"
        f"f = {float(f_hz):.3f} Hz"
        "</div>"
    )

    html_parts.append("<h3>Metodologia</h3>")
    html_parts.append(
        "<p>Os parâmetros elétricos foram obtidos a partir da modelagem geométrica dos condutores "
        "de fase e cabo-guarda, seguindo formulações clássicas de linhas aéreas (GMD/GMR, "
        "indutância, capacitância e modelo π).</p>"
    )

    return "".join(html_parts)


def solve_vr_pi(
    Vs_spec: VoltageSpec,
    params: LineParamsResult,
    length_km_loc: float,
    S_total_MVA_per_circ: float,
    pf_local: float,
    max_iter: int = 25,
):
    """Resolve Vr (por fase) a partir de Vs (Slack) e carga S no terminal receptor,
    usando modelo π (shunt B/2 em cada extremidade) e Newton numérico (2 variáveis).

    Retorna:
      Vr_LL_kV, Vr_ang_deg, I_series_A (módulo), P_loss_MW
    """
    # Série e shunt (por fase)
    Z = complex(float(params.R_ohm_km), float(params.X_ohm_km)) * float(length_km_loc)  # Ω
    B_total = float(params.B_S_km) * float(length_km_loc)  # S
    Ysh = 1j * B_total / 2.0

    Vs_LL_V = float(Vs_spec.magnitude_kV) * 1e3
    if Vs_LL_V <= 0:
        return None, None, 0.0, 0.0

    Vs_ang = Vs_spec.angle_rad
    Vs_phase_mag = Vs_LL_V / math.sqrt(3.0)
    Vs = complex(Vs_phase_mag * math.cos(Vs_ang), Vs_phase_mag * math.sin(Vs_ang))

    # Potência (por circuito) no terminal receptor
    S_va_total = max(0.0, float(S_total_MVA_per_circ)) * 1e6
    pf_local = float(pf_local) if pf_local is not None else 1.0
    pf_local = max(-1.0, min(1.0, pf_local))
    phi = math.acos(abs(pf_local))
    # Convenção: Q>0 indutivo (atraso). Se pf_local < 0, trata como adiantado (Q<0)
    Q_sign = 1.0 if pf_local >= 0 else -1.0
    P_total = S_va_total * abs(pf_local)
    Q_total = S_va_total * math.sin(phi) * Q_sign
    S_phase = complex(P_total / 3.0, Q_total / 3.0)

    # Sem carga → Vr = Vs
    if abs(S_phase) < 1e-9:
        Vr_phase = Vs
        I_series = 0j
    else:
        Vr_phase = Vs  # chute inicial

        def f(Vr: complex) -> complex:
            # I_load = conj(S/V)
            I_load = (S_phase.conjugate() / Vr.conjugate()) if abs(Vr) > 1e-9 else complex(1e9, 0)
            I_series_loc = I_load + Ysh * Vr
            return Vr + Z * I_series_loc - Vs  # deve ir a 0

        for _ in range(max_iter):
            F = f(Vr_phase)
            if abs(F) < 1e-6:
                break

            # Jacobiano numérico (2x2) em (Re, Im)
            x0, y0 = Vr_phase.real, Vr_phase.imag
            eps = max(1e-3, 1e-6 * abs(Vr_phase))
            # d/dx
            Fx = f(complex(x0 + eps, y0))
            # d/dy
            Fy = f(complex(x0, y0 + eps))

            dFdx = (Fx - F) / eps
            dFdy = (Fy - F) / eps

            # Sistema real 2x2:
            # [Re(dFdx) Re(dFdy)] [dx] = -Re(F)
            # [Im(dFdx) Im(dFdy)] [dy]   -Im(F)
            a = dFdx.real; b = dFdy.real
            c = dFdx.imag; d = dFdy.imag
            det = a * d - b * c
            if abs(det) < 1e-14:
                break

            rhs1 = -F.real
            rhs2 = -F.imag
            dx = (rhs1 * d - b * rhs2) / det
            dy = (a * rhs2 - rhs1 * c) / det

            Vr_phase = complex(x0 + dx, y0 + dy)

        I_load = (S_phase.conjugate() / Vr_phase.conjugate()) if abs(Vr_phase) > 1e-9 else 0j
        I_series = I_load + Ysh * Vr_phase

    Vr_LL_kV = abs(Vr_phase) * math.sqrt(3.0) / 1e3
    Vr_ang_deg = math.degrees(math.atan2(Vr_phase.imag, Vr_phase.real))

    R_total_ohm = float(params.R_ohm_km) * float(length_km_loc)
    P_loss_MW = 3.0 * (abs(I_series) ** 2) * R_total_ohm / 1e6 if R_total_ohm > 0 else 0.0

    return Vr_LL_kV, Vr_ang_deg, float(abs(I_series)), float(P_loss_MW)



# =====================================================================
# Teste rápido quando executado diretamente
# =====================================================================

if __name__ == "__main__":
    db_cabos = default_cable_db()
    geom = LineGeometry(
        conductors=[
            ConductorInstance(name="C1_A", cable_key="ACSR_477", x_m=0.0, y_m=15.0, circuit_index=1, phase="A", bundle_n=1),
            ConductorInstance(name="C1_B", cable_key="ACSR_477", x_m=8.0, y_m=15.0, circuit_index=1, phase="B", bundle_n=1),
            ConductorInstance(name="C1_C", cable_key="ACSR_477", x_m=16.0, y_m=15.0, circuit_index=1, phase="C", bundle_n=1),
            ConductorInstance(name="GW1",  cable_key="ACSR_266.8", x_m=8.0, y_m=20.0, circuit_index=1, phase=None, bundle_n=1, is_shield=True),
        ]
    )
    params_all = compute_all_circuits_params(geom, db_cabos, V_LL_kV=138.0, f_hz=60.0, temp_C=50.0)
    proj = ProjectInfo(nome_projeto="Teste", cliente="BK", numero_projeto="000", power_mva=100.0, n_circuits=1, pf=0.95, circuits_layout="side")
    html = generate_html_report(proj, geom, params_all, length_km=100.0, f_hz=60.0)
    with open("relatorio_exemplo.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    print("Teste gerado: relatorio_exemplo.html")

# bk_estudos_eletricos/core/shielding.py

# Estudo de blindagem contra descargas atmosféricas
# - Uso eficiente de cabos guarda e hastes para-raios
# - Geometria baseada em LineGeometry / ConductorInstance
# - Critério por ângulo de proteção + análise simples de aterramento
# - Relatório HTML com gráficos 2D/3D

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import math
from io import BytesIO
import base64

import numpy as np
import matplotlib.pyplot as plt

# import Axes3D apenas para registrar projeção 3D, evita lint warning
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Tentativa de importar tipos de geometria de forma robusta (geometry_model preferencial)
try:
    from .geometry_model import LineGeometry, ConductorInstance  # type: ignore
except Exception:
    from .line_params import LineGeometry, ConductorInstance  # type: ignore


# ========================= Dataclasses de entrada =========================


@dataclass
class ShieldingConfig:
    """
    Configuração do estudo de blindagem contra descargas atmosféricas.

    theta_max_deg:
        Ângulo máximo admissível de proteção entre cabo-guarda e fase.

    Notas:
      - Critério geométrico simplificado (ângulo de proteção).
      - Avaliação de aterramento: V_torre(kV) = I(kA) * R_terra(Ω), com comparação ao BIL.
    """

    V_LL_kV: float                   # Tensão nominal da linha
    theta_max_deg: float = 40.0      # Critério de proteção (padrão conservador)
    h_fase_min_m: float = 10.0       # Altura mínima típica de fase (para gráficos paramétricos)
    h_fase_max_m: float = 25.0
    d_min_m: float = 1.0             # Distância horizontal mínima para varredura (gráfico 3D)
    d_max_m: float = 20.0
    n_points_surface: int = 40       # Resolução da malha 3D (ângulo vs altura x distância)

    # Parâmetros para estudo de aterramento / backflash (modelo simplificado)
    tower_footing_R_ohm: float = 10.0    # Resistência de aterramento da estrutura (Ω)
    I_kA_min: float = 5.0                # Corrente de descarga mínima (kA)
    I_kA_max: float = 50.0               # Corrente de descarga máxima (kA)
    n_I_points: int = 80                 # Pontos para curva de V_torre(I)
    BIL_kV: float = 650.0                # NBI / BIL típico da isolação (kV)

    norma_ref: str = (
        "Baseado em recomendações IEC 60071, IEEE Std 998, "
        "ABNT NBR 5422 e ABNT NBR 5419 para blindagem de linhas aéreas."
    )


@dataclass
class ShieldingPhaseResult:
    """Resultado de blindagem para uma fase específica em um circuito."""
    circuit_index: int
    phase: str

    x_m: float
    y_m: float

    nearest_shield_name: Optional[str]
    nearest_shield_x_m: Optional[float]
    nearest_shield_y_m: Optional[float]

    horizontal_distance_m: float
    delta_h_m: float
    theta_deg: float
    is_protected: bool  # True se θ <= θ_max


@dataclass
class ShieldingGroundingResult:
    """
    Resultado simplificado de avaliação do aterramento / backflash
    considerando a tensão no topo da estrutura em função da corrente de descarga.
    """
    I_kA: np.ndarray
    V_tower_kV: np.ndarray
    exceeds_BIL: np.ndarray
    fraction_exceeds: float


@dataclass
class ShieldingResult:
    """Resultado consolidado do estudo de blindagem."""
    config: ShieldingConfig
    per_phase: List[ShieldingPhaseResult]
    grounding: ShieldingGroundingResult

    all_phases_protected: bool
    worst_theta_deg: float
    worst_phase: Optional[ShieldingPhaseResult]


# ========================= Funções utilitárias =========================


def _get_project_field(project: Any, *names: str, default: str = "") -> str:
    """
    Recupera de forma robusta um campo do objeto project.
    Aceita objetos com atributos ou dicionários com chaves equivalentes.
    """
    if project is None:
        return default

    # dict-like
    try:
        if isinstance(project, dict):
            for n in names:
                if n in project and project[n] is not None:
                    return str(project[n])
                nl = n.lower()
                if nl in project and project[nl] is not None:
                    return str(project[nl])
    except Exception:
        pass

    # objeto com atributos
    for n in names:
        try:
            if hasattr(project, n):
                val = getattr(project, n)
                if val is not None:
                    return str(val)
            nl = n.lower()
            if hasattr(project, nl):
                val = getattr(project, nl)
                if val is not None:
                    return str(val)
        except Exception:
            continue

    return default


def _horizontal_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return abs(p1[0] - p2[0])


def _angle_protection_deg(shield: Tuple[float, float], phase: Tuple[float, float]) -> float:
    """
    θ = arctan((h_cg - h_fase) / d), em graus.

    Se a fase estiver acima do cabo-guarda (h_fase >= h_cg) -> retorna 90° (sem proteção).
    Se a fase estiver exatamente abaixo (d = 0) -> retorna 0° (proteção máxima).
    """
    xg, yg = shield
    xf, yf = phase
    if yf >= yg:
        return 90.0

    d = _horizontal_distance((xg, yg), (xf, yf))
    if d <= 0.0:
        return 0.0

    theta_rad = math.atan((yg - yf) / max(d, 1e-9))
    return math.degrees(theta_rad)


def _safe_shields_list(geom: LineGeometry) -> List[ConductorInstance]:
    """Obtém lista de cabos-guarda de forma robusta."""
    try:
        shields = geom.shields()
        return list(shields) if shields is not None else []
    except Exception:
        return [c for c in getattr(geom, "conductors", []) if getattr(c, "is_shield", False)]


def _safe_circuits_list(geom: LineGeometry) -> List[int]:
    """Obtém lista de circuitos presentes na geometria de forma robusta."""
    try:
        circs = geom.circuits()
        return list(circs) if circs is not None else []
    except Exception:
        return sorted(set(getattr(c, "circuit_index", 1) for c in getattr(geom, "conductors", [])))


def _phases_of_circuit(geom: LineGeometry, cidx: int) -> Dict[str, ConductorInstance]:
    """Mapeia fases A/B/C do circuito cidx para seus condutores."""
    try:
        mapping = geom.phases_of_circuit(cidx)
        return dict(mapping) if mapping is not None else {}
    except Exception:
        mapping2: Dict[str, ConductorInstance] = {}
        for c in getattr(geom, "conductors", []):
            if getattr(c, "circuit_index", None) == cidx and getattr(c, "phase", None) in ("A", "B", "C"):
                mapping2[str(c.phase)] = c
        return mapping2


# ========================= Grounding / backflash ==============================


def _compute_grounding_response(config: ShieldingConfig) -> ShieldingGroundingResult:
    """
    Modelo simplificado:
      V_torre(kV) = I(kA) * R_terra(Ω)
    """
    I = np.linspace(config.I_kA_min, config.I_kA_max, max(int(config.n_I_points), 2))
    V_tower = I * config.tower_footing_R_ohm  # kA*Ω = kV
    exceeds = V_tower > config.BIL_kV
    fraction = float(np.sum(exceeds)) / float(len(exceeds)) if len(exceeds) > 0 else 0.0
    return ShieldingGroundingResult(I_kA=I, V_tower_kV=V_tower, exceeds_BIL=exceeds, fraction_exceeds=fraction)


# ========================= Cálculo principal de blindagem =======================


def compute_shielding(geom: LineGeometry, config: ShieldingConfig) -> ShieldingResult:
    """
    Computa a blindagem para cada fase (A,B,C) de cada circuito
    via ângulo de proteção em relação ao cabo-guarda mais favorável.

    Também avalia, de forma simplificada, a tensão no topo da estrutura em função
    da corrente de descarga (possível risco de backflashover).
    """
    shields_list = _safe_shields_list(geom)
    if not shields_list:
        raise ValueError(
            "Não há cabos guarda definidos na geometria (is_shield=True). "
            "Cadastre pelo menos um cabo guarda para usar o estudo de blindagem."
        )

    per_phase: List[ShieldingPhaseResult] = []
    theta_max = float(config.theta_max_deg)

    circuit_indices = _safe_circuits_list(geom)

    for cidx in circuit_indices:
        phases = _phases_of_circuit(geom, cidx)
        for phase_name in ("A", "B", "C"):
            if phase_name not in phases:
                continue

            ph = phases[phase_name]
            px, py = float(ph.x_m), float(ph.y_m)

            best_theta = 90.0
            best_shield_name: Optional[str] = None
            best_shield_x: Optional[float] = None
            best_shield_y: Optional[float] = None
            best_d = 0.0
            best_dh = 0.0

            for gw in shields_list:
                gx, gy = float(gw.x_m), float(gw.y_m)
                theta = _angle_protection_deg((gx, gy), (px, py))
                d = _horizontal_distance((gx, gy), (px, py))
                dh = gy - py

                if theta < best_theta:
                    best_theta = theta
                    best_shield_name = (
                        getattr(gw, "name", None)
                        or getattr(gw, "cable_key", None)
                        or "GW"
                    )
                    best_shield_x = gx
                    best_shield_y = gy
                    best_d = d
                    best_dh = dh

            is_prot = best_theta <= theta_max

            per_phase.append(
                ShieldingPhaseResult(
                    circuit_index=int(cidx),
                    phase=str(phase_name),
                    x_m=px,
                    y_m=py,
                    nearest_shield_name=best_shield_name,
                    nearest_shield_x_m=best_shield_x,
                    nearest_shield_y_m=best_shield_y,
                    horizontal_distance_m=float(best_d),
                    delta_h_m=float(best_dh),
                    theta_deg=float(best_theta),
                    is_protected=bool(is_prot),
                )
            )

    all_prot = all(p.is_protected for p in per_phase) if per_phase else False
    worst = max(per_phase, key=lambda p: p.theta_deg) if per_phase else None
    worst_theta = worst.theta_deg if worst else 0.0

    grounding = _compute_grounding_response(config)

    return ShieldingResult(
        config=config,
        per_phase=per_phase,
        grounding=grounding,
        all_phases_protected=all_prot,
        worst_theta_deg=worst_theta,
        worst_phase=worst,
    )


# ========================= Gráficos (base64 para HTML) =========================


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_shielding_angles_2d(result: ShieldingResult, title: str = "Ângulo de proteção por fase") -> str:
    labels: List[str] = []
    angles: List[float] = []
    colors: List[str] = []

    for p in result.per_phase:
        labels.append(f"C{p.circuit_index}-{p.phase}")
        angles.append(p.theta_deg)
        colors.append("#2ecc71" if p.is_protected else "#e74c3c")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(labels)) if labels else np.arange(1)
    ax.bar(x, angles if angles else [0.0], color=colors if colors else ["#95a5a6"])
    ax.axhline(
        result.config.theta_max_deg,
        linestyle="--",
        color="#34495e",
        linewidth=1.5,
        label=f"Limite θ_max = {result.config.theta_max_deg:.1f}°",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels else ["-"], rotation=45, ha="right")
    ax.set_ylabel("Ângulo θ (graus)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()

    return _fig_to_base64(fig)


def plot_protection_surface_3d(config: ShieldingConfig, h_cg_m: float) -> str:
    """
    Gráfico 3D do ângulo de proteção θ em função da altura da fase (H_fase) e distância (D)
    para um cabo guarda com altura h_cg_m (fixo).
    """
    n = max(int(config.n_points_surface), 10)
    h_fase_range = np.linspace(config.h_fase_min_m, config.h_fase_max_m, n)
    d_range = np.linspace(max(config.d_min_m, 1e-6), max(config.d_max_m, config.d_min_m + 1e-6), n)
    H_fase, D = np.meshgrid(h_fase_range, d_range)

    Theta_deg = np.where(
        H_fase >= h_cg_m,
        90.0,
        np.degrees(np.arctan((h_cg_m - H_fase) / np.maximum(D, 1e-9))),
    )

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(H_fase, D, Theta_deg, cmap="viridis", edgecolor="k", linewidth=0.3, alpha=0.9)
    ax.set_xlabel("Altura do condutor de fase (m)")
    ax.set_ylabel("Distância horizontal (m)")
    ax.set_zlabel("Ângulo de proteção θ (graus)")
    ax.set_title("Ângulo de proteção do cabo guarda em função da geometria")
    fig.colorbar(surf, shrink=0.7, aspect=14, label="θ (graus)")

    return _fig_to_base64(fig)


def plot_grounding_response(result: ShieldingResult) -> str:
    g = result.grounding
    config = result.config

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(g.I_kA, g.V_tower_kV, linewidth=2, label="V_torre (kV)")
    ax.axhline(config.BIL_kV, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"BIL = {config.BIL_kV:.1f} kV")
    ax.set_xlabel("Corrente de descarga (kA)")
    ax.set_ylabel("Tensão no topo da estrutura (kV)")
    ax.set_title("Resposta do aterramento da estrutura a descargas atmosféricas")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    return _fig_to_base64(fig)


# ========================= Relatório HTML + CSS ==============================


def generate_html_report(project: Any, geom: LineGeometry, result: ShieldingResult) -> str:
    """
    Gera relatório HTML + CSS para estudo de blindagem.
    Aceita 'project' como objeto com atributos (nome_projeto/cliente/numero_projeto) ou dict.
    """
    project_name = _get_project_field(project, "nome_projeto", "name", default="Projeto")
    project_client = _get_project_field(project, "cliente", "client", default="")
    project_number = _get_project_field(project, "numero_projeto", "project_number", default="")

    # altura média dos cabos guarda para a superfície 3D
    shields = _safe_shields_list(geom)
    if shields:
        try:
            h_cg_mean = float(np.mean([float(s.y_m) for s in shields]))
        except Exception:
            h_cg_mean = result.config.h_fase_max_m + 2.0
    else:
        h_cg_mean = result.config.h_fase_max_m + 2.0

    img_angles_b64 = plot_shielding_angles_2d(result)
    img_surface_b64 = plot_protection_surface_3d(result.config, h_cg_mean)
    img_ground_b64 = plot_grounding_response(result)

    css = """
    <style>
      body { font-family: "Segoe UI", Arial, sans-serif; background-color: #f5f7fa; color: #222; margin: 0; padding: 0; }
      .container { max-width: 1100px; margin: 0 auto; padding: 24px; background-color: #ffffff; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06); }
      h1,h2,h3 { color: #0b3c5d; }
      .header { border-bottom: 2px solid #e0e4ea; margin-bottom: 18px; padding-bottom: 6px; }
      .meta { font-size: 0.95rem; color: #555; }
      table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9rem; }
      th, td { border: 1px solid #dde2eb; padding: 6px 8px; text-align: right; }
      th { background-color: #f0f3f9; font-weight: 600; }
      td.label { text-align: left; font-weight: 500; }
      .ok { color: #27ae60; font-weight: 600; }
      .fail { color: #c0392b; font-weight: 600; }
      .img-block { text-align: center; margin: 16px 0; }
      .img-block img { max-width: 100%; border-radius: 6px; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.12); }
      .eq-block { background-color: #f8fafc; border-left: 4px solid #0b3c5d; padding: 8px 12px; font-family: "Consolas", "Courier New", monospace; font-size: 0.85rem; margin-top: 10px; }
      .small-note { font-size: 0.78rem; color: #777; }
    </style>
    """

    # tabela per-phase
    rows_html = ""
    for p in result.per_phase:
        cls = "ok" if p.is_protected else "fail"
        status = "Protegido" if p.is_protected else "Não protegido"
        rows_html += f"""
        <tr>
          <td class="label">Circuito {p.circuit_index} – Fase {p.phase}</td>
          <td>{p.x_m:.2f}</td>
          <td>{p.y_m:.2f}</td>
          <td>{p.nearest_shield_name or "-"}</td>
          <td>{(p.nearest_shield_x_m if p.nearest_shield_x_m is not None else 0.0):.2f}</td>
          <td>{(p.nearest_shield_y_m if p.nearest_shield_y_m is not None else 0.0):.2f}</td>
          <td>{p.horizontal_distance_m:.2f}</td>
          <td>{p.delta_h_m:.2f}</td>
          <td>{p.theta_deg:.2f}</td>
          <td class="{cls}">{status}</td>
        </tr>
        """

    per_phase_table = f"""
    <table>
      <tr>
        <th class="label">Ponto analisado</th>
        <th>x_fase (m)</th>
        <th>y_fase (m)</th>
        <th>Cabo guarda</th>
        <th>x_cg (m)</th>
        <th>y_cg (m)</th>
        <th>d horiz. (m)</th>
        <th>Δh (m)</th>
        <th>θ (graus)</th>
        <th>Situação</th>
      </tr>
      {rows_html}
    </table>
    """

    all_ok_text = (
        "<span class='ok'>Todas as fases atendem ao critério de blindagem "
        f"(θ ≤ {result.config.theta_max_deg:.1f}°).</span>"
        if result.all_phases_protected
        else "<span class='fail'>Existem fases com ângulo de proteção acima do limite "
             f"(θ_max = {result.config.theta_max_deg:.1f}°). Recomenda-se revisar a posição dos cabos guarda.</span>"
    )

    worst_phase_str = ""
    if result.worst_phase:
        wp = result.worst_phase
        worst_phase_str = (
            f"Caso mais crítico: circuito {wp.circuit_index}, fase {wp.phase}, "
            f"ângulo θ = {wp.theta_deg:.2f}° em relação ao cabo guarda {wp.nearest_shield_name or '-'}."
        )

    eq_block = f"""
    <div class="eq-block">
      <strong>Metodologia de blindagem contra descargas atmosféricas</strong><br/>
      • Para cada fase calcula-se θ = arctan((h_cg − h_fase) / d) [graus].<br/>
      • Critério: θ ≤ θ_max = {result.config.theta_max_deg:.1f}°. Referências: {result.config.norma_ref}<br/>
      • Tensão no topo (modelo simplificado): V_torre(kV) = I(kA) · R_terra(Ω). BIL = {result.config.BIL_kV:.1f} kV.
    </div>
    """

    g = result.grounding
    if g.fraction_exceeds == 0.0:
        grounding_text = (
            "<span class='ok'>Para a faixa de correntes analisada, a tensão no topo da estrutura "
            "permanece abaixo do BIL. O risco de backflashover é reduzido com a resistência de "
            f"aterramento de {result.config.tower_footing_R_ohm:.1f} Ω (modelo simplificado).</span>"
        )
    else:
        grounding_text = (
            "<span class='fail'>Para parte da faixa de correntes de descarga considerada, "
            "a tensão no topo da estrutura excede o BIL (modelo simplificado). Recomenda-se reduzir a resistência de "
            f"aterramento (atualmente {result.config.tower_footing_R_ohm:.1f} Ω) e reforçar o sistema.</span>"
        )

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="utf-8"/>
      <title>Relatório – Estudo de Blindagem – {project_name}</title>
      {css}
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Estudo de Blindagem contra Descargas Atmosféricas</h1>
          <div class="meta">
            <strong>Projeto:</strong> {project_name}<br/>
            <strong>Cliente:</strong> {project_client}<br/>
            <strong>Nº Projeto:</strong> {project_number}<br/>
            <strong>Tensão nominal:</strong> {result.config.V_LL_kV:.1f} kV<br/>
            <strong>Critério de ângulo máximo:</strong> θ_max = {result.config.theta_max_deg:.1f}°
          </div>
        </div>

        <h2>Resultados de blindagem geométrica (cabos guarda x fases)</h2>
        <p>{all_ok_text}<br/>{worst_phase_str}</p>

        {per_phase_table}

        <div class="img-block">
          <h3>Ângulo de proteção por fase / circuito</h3>
          <img src="data:image/png;base64,{img_angles_b64}" alt="Ângulo de proteção por fase"/>
        </div>

        <div class="img-block">
          <h3>Superfície 3D – Ângulo de proteção em função da altura da fase e distância horizontal</h3>
          <img src="data:image/png;base64,{img_surface_b64}" alt="Superfície do ângulo de proteção"/>
        </div>

        <h2>Avaliação do aterramento e risco de backflashover</h2>
        <p>{grounding_text}</p>
        <div class="img-block">
          <img src="data:image/png;base64,{img_ground_b64}" alt="Curva V_torre x I"/>
        </div>

        <h2>Metodologia e equações utilizadas</h2>
        {eq_block}

        <p class="small-note">
          Observação: o critério geométrico e o modelo de aterramento aqui apresentados são simplificados,
          adequados para estudos preliminares e comparativos. Para análises detalhadas recomenda-se uso de
          métodos eletrogeométricos (EGM), modelos de torre/aterramento dependentes de frequência e simulações EMT.
        </p>
      </div>
    </body>
    </html>
    """
    return html


# ========================= Teste rápido ======================================


if __name__ == "__main__":
    # teste em modo standalone (exemplo simples)
    geom = LineGeometry(
        conductors=[
            ConductorInstance(name="C1_A", cable_key="ACSR_477", x_m=0.0, y_m=17.5, circuit_index=1, phase="A"),
            ConductorInstance(name="C1_B", cable_key="ACSR_477", x_m=4.5, y_m=17.5, circuit_index=1, phase="B"),
            ConductorInstance(name="C1_C", cable_key="ACSR_477", x_m=9.0, y_m=17.5, circuit_index=1, phase="C"),
            ConductorInstance(name="GW1", cable_key="ACSR_266.8", x_m=4.5, y_m=21.0, circuit_index=1, phase=None, is_shield=True),
        ]
    )

    cfg = ShieldingConfig(V_LL_kV=138.0, theta_max_deg=45.0, tower_footing_R_ohm=10.0, BIL_kV=650.0)
    res = compute_shielding(geom, cfg)

    # project pode ser dict ou objeto com atributos
    proj = {"nome_projeto": "SE Exemplo – Linha 138 kV", "cliente": "BK Engenharia", "numero_projeto": "AGC-XXX-001"}
    html = generate_html_report(proj, geom, res)
    with open("relatorio_blindagem_exemplo.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Relatório de blindagem gerado: relatorio_blindagem_exemplo.html")

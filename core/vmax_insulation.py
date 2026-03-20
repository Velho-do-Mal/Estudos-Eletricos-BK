# bk_estudos_eletricos/core/vmax_insulation.py
#
# Estudo de Isolamento à Tensão Máxima Operativa (V_max / Um) + TOV
#
# - Verifica coordenação do isolamento em frequência industrial frente à maior tensão do sistema (Um)
#   e sobretensões temporárias (TOV) via fator k_TOV.
# - Avalia margens de segurança e comprimento de escoamento conforme níveis típicos de poluição (IEC 60815).
# - Gera relatório HTML + CSS com metodologia, equações, resultados e gráficos 2D/3D.
#
# Referências conceituais:
#   - IEC 60071-1 / IEC 60071-2 (Insulation Coordination)
#   - IEC 60815 (Selection and dimensioning of high-voltage insulators)
#   - IEC 62271 (HV switchgear – clearances and withstand voltages)
#
# Observação:
#   - Este módulo usa aproximações típicas de projeto (margens e mm/kV). Para aderência estrita,
#     ajuste os parâmetros conforme tabelas normativas / especificações do cliente.
#   - Por padrão, a tensão aplicada para TOV é calculada a partir de Um (maior tensão do sistema),
#     pois é o caso mais conservador para coordenação em frequência industrial.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any, Tuple
import math
import numpy as np
from io import BytesIO
import base64

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ======================= Estruturas de dados =========================


@dataclass
class VmaxProjectInfo:
    """Informações básicas do projeto para o relatório de isolamento."""
    nome_projeto: str
    cliente: str
    numero_projeto: str


@dataclass
class VmaxConfig:
    """
    Configuração global do estudo de isolamento à tensão máxima operativa.

    Convenções:
      - Vnom_kV: tensão nominal do sistema (kV L-L)
      - Um_kV: maior tensão do sistema conforme IEC 60071 (kV L-L)
      - k_TOV: fator multiplicativo sobre a tensão fase-terra baseada em Um (padrão conservador)
    """

    # Tensão nominal do sistema (kV L-L)
    Vnom_kV: float = 138.0

    # Maior tensão do sistema (Um), conforme IEC 60071 (kV L-L)
    Um_kV: float = 145.0

    # Fator de sobretensão temporária (TOV) – faixa para análises paramétricas
    k_TOV_min: float = 1.10
    k_TOV_max: float = 1.40

    # Fator "referência" para verificação principal
    k_TOV_ref: float = 1.20

    # Margem mínima recomendada entre tensão suportável e tensão aplicada (frequência industrial)
    min_safety_margin_percent: float = 15.0

    # Altitude do local (m) – para eventual correção da tensão suportável
    altitude_m: float = 0.0

    # Níveis de poluição típicos (IEC 60815) e mm/kV fase-terra (valores padrão típicos)
    creepage_mm_per_kV_level_II: float = 22.0
    creepage_mm_per_kV_level_III: float = 27.0
    creepage_mm_per_kV_level_IV: float = 31.0

    # Pontos para gráficos
    n_points_2d: int = 100
    n_points_3d_k: int = 40
    n_points_3d_creep: int = 40

    # Nota metodológica (texto curto para relatório)
    norma_ref: str = (
        "Referências conceituais: IEC 60071-1/2 (coordenação do isolamento), "
        "IEC 60815 (escoamento/poluição) e IEC 62271 (equipamentos AT)."
    )


@dataclass
class InsulationItem:
    """
    Elemento de isolamento a ser avaliado (barramento, isolador, bucha, cadeia, etc.).
    """

    name: str

    # Tensão suportável em frequência industrial (1 min) – fase-terra (kV rms)
    U_pf_withstand_kV: float

    # Tensão suportável ao impulso atmosférico (BIL/NBI) – fase-terra (kV crest)
    U_impulse_withstand_kV: float

    # Comprimento de escoamento total (mm)
    creepage_mm: float

    # Nível de poluição (1–4), conforme IEC 60815
    pollution_level: int = 2

    # Texto opcional
    description: str = ""

    # Altitude específica do equipamento (se None, usa cfg.altitude_m)
    altitude_m: Optional[float] = None


@dataclass
class InsulationItemResult:
    """Resultado calculado para um item de isolamento."""
    item: InsulationItem

    # Tensões de referência
    V_phase_nom_kV: float          # Vnom/√3
    V_phase_um_kV: float           # Um/√3

    # Tensão máxima temporária (TOV) aplicada (kV rms) baseada em Um
    k_TOV_used: float
    V_TOV_kV: float

    # Correção por altitude
    altitude_m_used: float
    Ka: float
    U_pf_corr_kV: float

    # Margem PF (%)
    margin_pf_percent: float
    meets_pf_margin: bool

    # Escoamento
    creepage_required_mm: float
    creepage_mm_per_kV: float
    creepage_required_mm_per_kV: float
    meets_creepage: bool


@dataclass
class VmaxStudyResult:
    """Resultado global do estudo de isolamento à tensão máxima operativa."""
    project: VmaxProjectInfo
    config: VmaxConfig
    item_results: List[InsulationItemResult]
    html_report: str


# ======================= Funções auxiliares =========================


def _get_project_field(project: Any, *names: str, default: str = "") -> str:
    """
    Recupera de forma robusta um campo do objeto project (atributo ou dict).
    """
    if project is None:
        return default

    # dict-like
    try:
        if isinstance(project, dict):
            for n in names:
                if n in project:
                    return str(project[n] or default)
                if n.lower() in project:
                    return str(project[n.lower()] or default)
    except Exception:
        pass

    # objeto com atributos
    for n in names:
        try:
            if hasattr(project, n):
                val = getattr(project, n)
                return str(val if val is not None else default)
            if hasattr(project, n.lower()):
                val = getattr(project, n.lower())
                return str(val if val is not None else default)
        except Exception:
            continue

    # fallback genérico
    for alt in ("nome_projeto", "name", "cliente", "client", "numero_projeto", "project_number"):
        try:
            if hasattr(project, alt):
                val = getattr(project, alt)
                return str(val if val is not None else default)
            if isinstance(project, dict) and alt in project:
                return str(project[alt] or default)
        except Exception:
            continue

    return default


def _clamp_pollution_level(level: int) -> int:
    try:
        lv = int(level)
    except Exception:
        lv = 2
    return max(1, min(4, lv))


def _altitude_correction_factor(altitude_m: float) -> float:
    """
    Fator de correção por altitude (Ka), aplicado como:
        U_pf_corr = U_pf_ref / Ka

    Aproximação conservadora simples:
        Ka = 1 + 0.1*(h/1000)

    (Para aplicação estrita, substituir por formulações/tabelas normativas adequadas.)
    """
    h = max(0.0, float(altitude_m or 0.0))
    return 1.0 + 0.1 * (h / 1000.0)


def _required_creepage_mm_per_kV(cfg: VmaxConfig, level: int) -> float:
    """
    Retorna o mm/kV fase-terra requerido conforme nível de poluição (valores típicos).
    """
    lv = _clamp_pollution_level(level)
    if lv <= 1:
        return 16.0
    if lv == 2:
        return float(cfg.creepage_mm_per_kV_level_II)
    if lv == 3:
        return float(cfg.creepage_mm_per_kV_level_III)
    return float(cfg.creepage_mm_per_kV_level_IV)


def _safe_positive(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if v > 0 else default


# ======================= Cálculo por item ============================


def compute_item_insulation(cfg: VmaxConfig, item: InsulationItem) -> InsulationItemResult:
    """
    Calcula as grandezas de isolamento para um item específico.

    Critério PF (frequência industrial):
      - tensão aplicada = V_TOV = k_TOV_ref * (Um/√3)
      - tensão suportável corrigida por altitude: U_pf_corr = U_pf / Ka
      - margem (%) = (U_pf_corr / V_TOV - 1)*100

    Critério de escoamento:
      - mm/kV disponível = creepage_mm / (Um/√3)   (base em fase-terra)
      - mm/kV requerido conforme nível de poluição (típico)
    """

    Vnom = _safe_positive(cfg.Vnom_kV, 0.0)
    Um = _safe_positive(cfg.Um_kV, Vnom if Vnom > 0 else 0.0)

    V_phase_nom_kV = Vnom / math.sqrt(3.0) if Vnom > 0 else 0.0
    V_phase_um_kV = Um / math.sqrt(3.0) if Um > 0 else 0.0

    k_TOV_used = float(cfg.k_TOV_ref) if (cfg.k_TOV_ref and cfg.k_TOV_ref > 0) else 1.0
    V_TOV_kV = k_TOV_used * V_phase_um_kV

    # Altitude
    alt_m = float(item.altitude_m) if item.altitude_m is not None else float(cfg.altitude_m or 0.0)
    Ka = _altitude_correction_factor(alt_m)

    U_pf = _safe_positive(item.U_pf_withstand_kV, 0.0)
    U_pf_corr_kV = U_pf / Ka if Ka > 0 else 0.0

    # Margem
    if V_TOV_kV > 1e-9:
        margin_pf = (U_pf_corr_kV / V_TOV_kV - 1.0) * 100.0
    else:
        margin_pf = 0.0

    meets_pf = margin_pf >= float(cfg.min_safety_margin_percent)

    # Escoamento (baseado em Um fase-terra; evita dividir por zero)
    denom = V_phase_um_kV if V_phase_um_kV > 1e-6 else 1e-6
    creepage_mm = max(0.0, float(item.creepage_mm or 0.0))
    creepage_mm_per_kV = creepage_mm / denom

    req_mm_per_kV = _required_creepage_mm_per_kV(cfg, item.pollution_level)
    creepage_required_mm = req_mm_per_kV * denom

    meets_creep = creepage_mm_per_kV >= req_mm_per_kV

    return InsulationItemResult(
        item=InsulationItem(
            name=item.name,
            U_pf_withstand_kV=item.U_pf_withstand_kV,
            U_impulse_withstand_kV=item.U_impulse_withstand_kV,
            creepage_mm=item.creepage_mm,
            pollution_level=_clamp_pollution_level(item.pollution_level),
            description=item.description,
            altitude_m=item.altitude_m,
        ),
        V_phase_nom_kV=V_phase_nom_kV,
        V_phase_um_kV=V_phase_um_kV,
        k_TOV_used=k_TOV_used,
        V_TOV_kV=V_TOV_kV,
        altitude_m_used=alt_m,
        Ka=Ka,
        U_pf_corr_kV=U_pf_corr_kV,
        margin_pf_percent=margin_pf,
        meets_pf_margin=meets_pf,
        creepage_required_mm=creepage_required_mm,
        creepage_mm_per_kV=creepage_mm_per_kV,
        creepage_required_mm_per_kV=req_mm_per_kV,
        meets_creepage=meets_creep,
    )


def compute_all_items_insulation(cfg: VmaxConfig, items: List[InsulationItem]) -> List[InsulationItemResult]:
    """Calcula os resultados de isolamento para todos os itens."""
    return [compute_item_insulation(cfg, it) for it in (items or [])]


# ======================= Gráficos 2D / 3D ===========================


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _plot_margin_bar_chart(results: List[InsulationItemResult], cfg: VmaxConfig) -> str:
    """Gráfico de barras: margem PF (%) por componente."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    if not results:
        ax.text(0.5, 0.5, "Sem dados para plotagem", ha="center", va="center")
        ax.axis("off")
        return _fig_to_base64(fig)

    names = [r.item.name for r in results]
    margins = [r.margin_pf_percent for r in results]

    bars = ax.bar(range(len(results)), margins)
    for i, r in enumerate(results):
        bars[i].set_color("tab:green" if r.meets_pf_margin else "tab:red")

    ax.axhline(cfg.min_safety_margin_percent, color="k", linestyle="--", linewidth=1.2,
               label=f"Margem mínima = {cfg.min_safety_margin_percent:.1f}%")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Margem PF (%)")
    ax.set_title("Margem de Isolamento em Frequência Industrial (base Um + TOV)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    return _fig_to_base64(fig)


def _plot_creepage_bar_chart(results: List[InsulationItemResult]) -> str:
    """Comparação: mm/kV disponível vs requerido por componente."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    if not results:
        ax.text(0.5, 0.5, "Sem dados para plotagem", ha="center", va="center")
        ax.axis("off")
        return _fig_to_base64(fig)

    names = [r.item.name for r in results]
    mmkv = [r.creepage_mm_per_kV for r in results]
    mmkv_req = [r.creepage_required_mm_per_kV for r in results]

    x = list(range(len(results)))
    width = 0.38

    ax.bar([xi - width / 2 for xi in x], mmkv, width, label="Disponível (mm/kV)")
    ax.bar([xi + width / 2 for xi in x], mmkv_req, width, label="Requerido (mm/kV)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Comprimento de escoamento específico (mm/kV)")
    ax.set_title("Comprimento de Escoamento – Disponível x Requerido (base Um fase-terra)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    return _fig_to_base64(fig)


def _plot_margin_surface_3d(cfg: VmaxConfig, item: InsulationItem) -> str:
    """
    Superfície 3D: margem PF (%) em função de:
      - k_TOV (cfg.k_TOV_min..cfg.k_TOV_max)
      - mm/kV "hipotético" (eixo ilustrativo de escoamento específico)

    Observação:
      - A margem PF é função de U_pf_corr e V_TOV (não depende do escoamento).
        Mantemos o eixo mm/kV como eixo de sensibilidade/visualização (padrão do template),
        mas a superfície ficará "constante" na direção mm/kV para um item.
    """
    if item is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Sem item para superfície 3D", ha="center", va="center")
        ax.axis("off")
        return _fig_to_base64(fig)

    # malha k_TOV
    k_min = float(cfg.k_TOV_min)
    k_max = float(cfg.k_TOV_max)
    if k_max < k_min:
        k_min, k_max = k_max, k_min

    ks = [
        k_min + (k_max - k_min) * i / max(cfg.n_points_3d_k - 1, 1)
        for i in range(cfg.n_points_3d_k)
    ]

    # eixo mm/kV (ilustrativo)
    creep_values = [
        10.0 + (45.0 - 10.0) * j / max(cfg.n_points_3d_creep - 1, 1)
        for j in range(cfg.n_points_3d_creep)
    ]

    # tensões
    Um = _safe_positive(cfg.Um_kV, _safe_positive(cfg.Vnom_kV, 0.0))
    V_phase_um_kV = Um / math.sqrt(3.0) if Um > 0 else 0.0

    # U_pf corrigida
    alt_m = float(item.altitude_m) if item.altitude_m is not None else float(cfg.altitude_m or 0.0)
    Ka = _altitude_correction_factor(alt_m)
    U_pf_corr_kV = float(item.U_pf_withstand_kV) / Ka if Ka > 0 else float(item.U_pf_withstand_kV)

    # matrizes para plot_surface (listas aninhadas)
    X, Y, Z = [], [], []
    for creep in creep_values:
        row_k, row_c, row_m = [], [], []
        for k in ks:
            V_TOV_kV = k * V_phase_um_kV
            if V_TOV_kV > 1e-9:
                margin = (U_pf_corr_kV / V_TOV_kV - 1.0) * 100.0
            else:
                margin = 0.0
            row_k.append(k)
            row_c.append(creep)
            row_m.append(margin)
        X.append(row_k)
        Y.append(row_c)
        Z.append(row_m)

    fig = plt.figure(figsize=(7.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    # matplotlib exige arrays com atributo 'ndim'
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0.25, edgecolor="k", antialiased=True)

    ax.set_xlabel("Fator de sobretensão temporária k_TOV")
    ax.set_ylabel("mm/kV (eixo ilustrativo)")
    ax.set_zlabel("Margem PF (%)")
    ax.set_title(f"Superfície de Margem PF – {item.name}")
    ax.view_init(elev=25, azim=-140)
    fig.colorbar(surf, shrink=0.6, aspect=12)

    return _fig_to_base64(fig)


# ======================= Relatório HTML + CSS ========================


def generate_html_report_vmax(project: Any, cfg: VmaxConfig, results: List[InsulationItemResult]) -> str:
    """
    Gera relatório HTML + CSS de Estudo de Isolamento à Tensão Máxima Operativa.
    Aceita 'project' como VmaxProjectInfo, objeto com atributos ou dict.
    """
    project_name = _get_project_field(project, "nome_projeto", "name", default="Projeto")
    project_client = _get_project_field(project, "cliente", "client", default="")
    project_number = _get_project_field(project, "numero_projeto", "project_number", default="")

    first_item = results[0].item if results else None

    img_margin_bar = _plot_margin_bar_chart(results, cfg)
    img_creep_bar = _plot_creepage_bar_chart(results)
    img_surface_3d = _plot_margin_surface_3d(cfg, first_item) if first_item is not None else ""

    # Resumo global
    n_total = len(results or [])
    n_ok_pf = sum(1 for r in (results or []) if r.meets_pf_margin)
    n_ok_creep = sum(1 for r in (results or []) if r.meets_creepage)

    css = """
    <style>
      body {
        font-family: "Segoe UI", Arial, sans-serif;
        background-color: #f5f7fa;
        color: #222;
        margin: 0;
        padding: 0;
      }
      .container {
        max-width: 1080px;
        margin: 0 auto;
        padding: 24px;
        background-color: #ffffff;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
      }
      h1, h2, h3 { color: #0b3c5d; }
      .header {
        border-bottom: 2px solid #e0e4ea;
        margin-bottom: 16px;
        padding-bottom: 8px;
      }
      .meta { font-size: 0.95rem; color: #555; }
      table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
        font-size: 0.9rem;
      }
      th, td {
        border: 1px solid #dde2eb;
        padding: 6px 8px;
        text-align: right;
      }
      th { background-color: #f0f3f9; font-weight: 600; }
      td.label { text-align: left; font-weight: 500; }
      .img-block { text-align: center; margin: 16px 0; }
      .img-block img {
        max-width: 100%;
        border-radius: 6px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.12);
      }
      .eq-block {
        background-color: #f8fafc;
        border-left: 4px solid #0b3c5d;
        padding: 10px 12px;
        font-family: "Consolas","Courier New", monospace;
        font-size: 0.85rem;
      }
      .small-note { font-size: 0.8rem; color: #777; }
      .status-ok { color: #0a7d00; font-weight: 600; }
      .status-nok { color: #c00000; font-weight: 600; }
      .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: #f0f3f9;
        color: #0b3c5d;
        font-size: 0.85rem;
        margin-right: 6px;
      }
    </style>
    """

    V_phase_nom = (cfg.Vnom_kV / math.sqrt(3.0)) if cfg.Vnom_kV > 0 else 0.0
    V_phase_um = (cfg.Um_kV / math.sqrt(3.0)) if cfg.Um_kV > 0 else 0.0
    V_tov_ref = (cfg.k_TOV_ref * V_phase_um) if (cfg.k_TOV_ref > 0 and V_phase_um > 0) else 0.0

    entrada_html = f"""
    <h2>Dados de Entrada</h2>
    <table>
      <tr><th class="label">Parâmetro</th><th>Valor</th><th>Unidade</th></tr>
      <tr><td class="label">Tensão nominal (V<sub>nom</sub>)</td><td>{cfg.Vnom_kV:.2f}</td><td>kV (L-L)</td></tr>
      <tr><td class="label">Maior tensão do sistema (U<sub>m</sub>)</td><td>{cfg.Um_kV:.2f}</td><td>kV (L-L)</td></tr>
      <tr><td class="label">Tensão de fase nominal</td><td>{V_phase_nom:.2f}</td><td>kV (fase-terra)</td></tr>
      <tr><td class="label">Tensão de fase baseada em U<sub>m</sub></td><td>{V_phase_um:.2f}</td><td>kV (fase-terra)</td></tr>
      <tr><td class="label">Fator TOV – mínimo</td><td>{cfg.k_TOV_min:.3f}</td><td>-</td></tr>
      <tr><td class="label">Fator TOV – máximo</td><td>{cfg.k_TOV_max:.3f}</td><td>-</td></tr>
      <tr><td class="label">Fator TOV – referência</td><td>{cfg.k_TOV_ref:.3f}</td><td>-</td></tr>
      <tr><td class="label">Tensão aplicada de referência (TOV)</td><td>{V_tov_ref:.2f}</td><td>kV rms (fase-terra)</td></tr>
      <tr><td class="label">Margem mínima recomendada (PF)</td><td>{cfg.min_safety_margin_percent:.1f}</td><td>%</td></tr>
      <tr><td class="label">Altitude do local (referência)</td><td>{cfg.altitude_m:.1f}</td><td>m</td></tr>
    </table>
    """

    resumo_global_html = f"""
    <h2>Resumo de Conformidade</h2>
    <p>
      <span class="pill">Itens avaliados: {n_total}</span>
      <span class="pill">PF (margem): {n_ok_pf}/{n_total}</span>
      <span class="pill">Escoamento: {n_ok_creep}/{n_total}</span>
    </p>
    """

    # Tabela item a item
    linhas = ""
    for r in (results or []):
        status_pf = '<span class="status-ok">ATENDE</span>' if r.meets_pf_margin else '<span class="status-nok">NÃO ATENDE</span>'
        status_creep = '<span class="status-ok">ATENDE</span>' if r.meets_creepage else '<span class="status-nok">NÃO ATENDE</span>'
        desc = r.item.description or "-"

        linhas += f"""
        <tr>
          <td class="label">{r.item.name}</td>
          <td>{r.V_phase_um_kV:.2f}</td>
          <td>{r.k_TOV_used:.3f}</td>
          <td>{r.V_TOV_kV:.2f}</td>
          <td>{r.item.U_pf_withstand_kV:.2f}</td>
          <td>{r.Ka:.3f}</td>
          <td>{r.U_pf_corr_kV:.2f}</td>
          <td>{r.margin_pf_percent:.1f} % ({status_pf})</td>
          <td>{r.item.creepage_mm:.1f}</td>
          <td>{r.creepage_required_mm:.1f}</td>
          <td>{r.creepage_mm_per_kV:.1f}</td>
          <td>{r.creepage_required_mm_per_kV:.1f} ({status_creep})</td>
          <td>{r.item.pollution_level}</td>
          <td>{r.altitude_m_used:.0f}</td>
          <td class="label">{desc}</td>
        </tr>
        """

    tabela_itens_html = f"""
    <h2>Resultados por Componente</h2>
    <table>
      <tr>
        <th class="label">Componente</th>
        <th>U<sub>m</sub>/√3</th>
        <th>k<sub>TOV</sub></th>
        <th>V<sub>TOV</sub></th>
        <th>U<sub>pf,ref</sub></th>
        <th>K<sub>a</sub></th>
        <th>U<sub>pf,corr</sub></th>
        <th>Margem PF</th>
        <th>L<sub>esco</sub> disp</th>
        <th>L<sub>esco</sub> req</th>
        <th>mm/kV disp</th>
        <th>mm/kV req</th>
        <th>Poluição</th>
        <th>Alt (m)</th>
        <th class="label">Descrição</th>
      </tr>
      {linhas}
    </table>
    <p class="small-note">
      Unidades: tensões em kV (fase-terra); L<sub>esco</sub> em mm; mm/kV referidos a U<sub>m</sub>/√3.
    </p>
    """

    eqs_html = f"""
    <div class="eq-block">
      <strong>Equações fundamentais utilizadas</strong><br/><br/>
      V<sub>fase,nom</sub> = V<sub>nom</sub> / √3  [kV rms, fase-terra]<br/>
      V<sub>fase,Um</sub> = U<sub>m</sub> / √3  [kV rms, fase-terra]<br/>
      V<sub>TOV</sub> = k<sub>TOV</sub> · V<sub>fase,Um</sub>  [kV rms]<br/><br/>
      K<sub>a</sub> ≈ 1 + 0,1 · (h/1000)  (aprox. conservadora)<br/>
      U<sub>pf,corr</sub> = U<sub>pf,ref</sub> / K<sub>a</sub><br/>
      Margem<sub>PF</sub>(%) = (U<sub>pf,corr</sub> / V<sub>TOV</sub> − 1) · 100<br/><br/>
      (mm/kV)<sub>disp</sub> = L<sub>esco</sub> / V<sub>fase,Um</sub><br/>
      (mm/kV)<sub>req</sub> = f(nível de poluição)  (valores típicos)<br/>
      L<sub>esco,req</sub> = (mm/kV)<sub>req</sub> · V<sub>fase,Um</sub><br/><br/>
      Valores típicos (IEC 60815 – fase-terra):<br/>
      &nbsp;&nbsp;Nível I ≈ 16 mm/kV<br/>
      &nbsp;&nbsp;Nível II ≈ {cfg.creepage_mm_per_kV_level_II:.1f} mm/kV<br/>
      &nbsp;&nbsp;Nível III ≈ {cfg.creepage_mm_per_kV_level_III:.1f} mm/kV<br/>
      &nbsp;&nbsp;Nível IV ≈ {cfg.creepage_mm_per_kV_level_IV:.1f} mm/kV<br/>
    </div>
    """

    metodologia_html = f"""
    <h2>Metodologia e Critérios</h2>
    <p>
      Este estudo avalia a coordenação do isolamento em frequência industrial considerando a
      <strong>maior tensão do sistema (U<sub>m</sub>)</strong> e um fator de sobretensão temporária
      <strong>k<sub>TOV</sub></strong>. A tensão aplicada de verificação é
      V<sub>TOV</sub> = k<sub>TOV</sub> · (U<sub>m</sub>/√3).
    </p>
    <p>
      A tensão suportável em frequência industrial (1 min) é corrigida por altitude via um fator K<sub>a</sub>
      (aproximação conservadora). A margem PF é comparada com uma margem mínima recomendada.
    </p>
    <p>
      O comprimento de escoamento é verificado em termos de mm/kV (fase-terra) conforme valores típicos por nível de
      poluição (IEC 60815), com base em U<sub>m</sub>/√3.
    </p>
    <p class="small-note">{cfg.norma_ref}</p>
    """

    conclusoes_html = """
    <h2>Discussão e Recomendações</h2>
    <p>
      Itens que <strong>não atendem</strong> à margem PF sugerem revisão da classe de ensaio, redução do estresse
      elétrico (por exemplo, controle de sobretensões temporárias) ou adoção de equipamentos/isolação com maior
      suportabilidade em frequência industrial.
    </p>
    <p>
      Itens que <strong>não atendem</strong> ao critério de escoamento indicam necessidade de aumento do comprimento
      de escoamento (cadeias/isoladores), revisão do nível de poluição assumido e/ou adoção de soluções específicas
      (isoladores anti-poluição, revestimentos, lavagens programadas, etc.).
    </p>
    """

    img3d_html = ""
    if img_surface_3d:
        img3d_html = f"""
        <div class="img-block">
          <h3>Superfície 3D – Margem PF vs k<sub>TOV</sub> e eixo mm/kV (ilustrativo)</h3>
          <img src="data:image/png;base64,{img_surface_3d}" alt="Superfície 3D de margem PF"/>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="utf-8"/>
      <title>Relatório – Isolamento à Tensão Máxima Operativa – {project_name}</title>
      {css}
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>BK_Estudos_Eletricos – Estudo de Isolamento à Tensão Máxima Operativa (Vmax/Um)</h1>
          <div class="meta">
            <strong>Projeto:</strong> {project_name}<br/>
            <strong>Cliente:</strong> {project_client}<br/>
            <strong>Nº Projeto:</strong> {project_number}<br/>
          </div>
        </div>

        {entrada_html}
        {resumo_global_html}
        {tabela_itens_html}

        <div class="img-block">
          <h3>Margem PF por Componente</h3>
          <img src="data:image/png;base64,{img_margin_bar}" alt="Margem PF por componente"/>
        </div>

        <div class="img-block">
          <h3>Comprimento de Escoamento – Disponível x Requerido</h3>
          <img src="data:image/png;base64,{img_creep_bar}" alt="Escoamento disponível x requerido"/>
        </div>

        {img3d_html}

        <h2>Metodologia e Equações</h2>
        {eqs_html}
        {metodologia_html}
        {conclusoes_html}

        <p class="small-note">
          Nota: parâmetros (mm/kV e margem mínima) são típicos de projeto. Ajuste conforme normas aplicáveis,
          especificações do cliente e condições reais do empreendimento.
        </p>
      </div>
    </body>
    </html>
    """

    return html


# ======================= Função de alto nível ========================


def run_vmax_insulation_study(project: Any, cfg: VmaxConfig, items: List[InsulationItem]) -> VmaxStudyResult:
    """Função de alto nível para o estudo de isolamento à tensão máxima operativa."""
    results = compute_all_items_insulation(cfg, items)
    html = generate_html_report_vmax(project, cfg, results)

    if isinstance(project, VmaxProjectInfo):
        proj_obj = project
    else:
        nome = _get_project_field(project, "nome_projeto", "name", default="Projeto")
        cli = _get_project_field(project, "cliente", "client", default="")
        num = _get_project_field(project, "numero_projeto", "project_number", default="")
        proj_obj = VmaxProjectInfo(nome_projeto=nome, cliente=cli, numero_projeto=num)

    return VmaxStudyResult(project=proj_obj, config=cfg, item_results=results, html_report=html)


# ======================= Teste rápido (opcional) =====================


if __name__ == "__main__":
    project = VmaxProjectInfo(
        nome_projeto="SE 138 kV – Exemplo Isolamento Vmax",
        cliente="BK Engenharia e Tecnologia",
        numero_projeto="2025-VMAX-001",
    )

    cfg = VmaxConfig(
        Vnom_kV=138.0,
        Um_kV=145.0,
        k_TOV_min=1.10,
        k_TOV_max=1.40,
        k_TOV_ref=1.20,
        min_safety_margin_percent=15.0,
        altitude_m=500.0,
    )

    items = [
        InsulationItem(
            name="Cadeia suspensão – fase A",
            U_pf_withstand_kV=325.0,
            U_impulse_withstand_kV=650.0,
            creepage_mm=4600.0,
            pollution_level=2,
            description="Cadeia de vidro 138 kV, poluição moderada",
        ),
        InsulationItem(
            name="Bucha de TR 138/13,8 kV",
            U_pf_withstand_kV=325.0,
            U_impulse_withstand_kV=650.0,
            creepage_mm=5200.0,
            pollution_level=3,
            description="Bucha lado AT de transformador de potência",
        ),
        InsulationItem(
            name="Disjuntor 138 kV – polo completo",
            U_pf_withstand_kV=325.0,
            U_impulse_withstand_kV=650.0,
            creepage_mm=5000.0,
            pollution_level=2,
            description="Disjuntor AT, isolamento externo",
        ),
    ]

    result = run_vmax_insulation_study(project, cfg, items)

    with open("relatorio_isolamento_vmax_exemplo.html", "w", encoding="utf-8") as f:
        f.write(result.html_report)

    print("Relatório de isolamento Vmax gerado: relatorio_isolamento_vmax_exemplo.html")
    for r in result.item_results:
        print(
            f"{r.item.name}: margem PF = {r.margin_pf_percent:.1f}% "
            f"(mm/kV {r.creepage_mm_per_kV:.1f}, req {r.creepage_required_mm_per_kV:.1f})"
        )

# bk_estudos_eletricos/core/ri_ra.py
# Estudo de RI – Rádio Interferência e RA – Ruído Audível
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import base64
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessário para projeção 3D)

# Import structs from line_params (ProjectInfo, LineGeometry, LineParamsResult, plot_geometry_2d)
try:
    from .line_params import (
        ProjectInfo,
        LineGeometry,
        LineParamsResult,
        plot_geometry_2d,
    )
except Exception:
    # Fallback stubs if line_params not present — mantém tipos para editores/static typing.
    ProjectInfo = Any
    LineGeometry = Any
    LineParamsResult = Any

    def plot_geometry_2d(*args, **kwargs):
        return ""


# ======================= Configuração / Resultados ====================


@dataclass
class RIRAConfig:
    """
    Configuração do estudo de Rádio Interferência (RI) e Ruído Audível (RA).

    Observações:
    - freq_MHz refere-se à frequência de avaliação de RI (tipicamente 0,5–1 MHz).
    - weather é uma string descritiva; a conformidade usa o caso de chuva como mais crítico.
    - V_LL_kV é opcional e serve apenas para exibição no relatório (não entra no modelo empírico).
    """

    # RI
    freq_MHz: float = 0.5
    weather: str = "seco"

    # Faixa lateral (m)
    distance_min_m: float = 0.0
    distance_max_m: float = 60.0
    n_points: int = 241
    observation_height_m: float = 1.5

    # Período para RA (NBR 10151)
    day_period: str = "diurno"

    # Apenas para cabeçalho do relatório (opcional)
    V_LL_kV: Optional[float] = None

    # Limites típicos (podem ser alterados)
    limit_RI_dBuV_m: float = 55.0  # dB(µV/m) @ 0,5–1 MHz (valor típico)
    limit_RA_dBA_day: float = 55.0
    limit_RA_dBA_night: float = 50.0


@dataclass
class RIRAProfiles:
    """
    Perfis calculados de RI e RA ao longo da faixa lateral.
    """

    cfg: RIRAConfig

    # Distâncias laterais (m)
    distances_m: np.ndarray

    # Níveis de rádio interferência – dB(µV/m)
    RI_seco_dBuV_m: np.ndarray
    RI_chuva_dBuV_m: np.ndarray

    # Níveis de ruído audível – dB(A)
    RA_seco_dBA: np.ndarray
    RA_chuva_dBA: np.ndarray

    # Gradiente superficial (do módulo de parâmetros)
    Ec_kV_cm: float

    # Valores na borda da faixa (último ponto) em chuva (cenário crítico)
    RI_edge_chuva_dBuV_m: float
    RA_edge_chuva_dBA: float

    # Flags de conformidade na borda (chuva)
    exceeds_RI_limit: bool
    exceeds_RA_limit: bool

    # Comentários de conformidade (texto pronto para relatório)
    comment_RI: str
    comment_RA: str


# ======================= Helpers =====================================


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


# ======================= Modelos empíricos RI / RA ====================


def _ri_base_from_Ec(Ec_kV_cm: float, freq_MHz: float, weather: str) -> float:
    """
    Modelo empírico simplificado para nível de RI (dB(µV/m)).

    Nota: modelo indicativo para estudos preliminares.
    """
    Ec = max(Ec_kV_cm, 0.0)
    fMHz = max(freq_MHz, 1e-6)

    if Ec <= 0:
        base = 20.0
    else:
        Ec_thr = 21.0
        base = 35.0 if Ec <= Ec_thr else (35.0 + 2.0 * (Ec - Ec_thr))

    # Ajuste (bem simplificado) com frequência: acima de ~1 MHz tende a reduzir um pouco
    if fMHz >= 1.0:
        base -= 5.0

    # Efeito chuva
    if (weather or "").strip().lower().startswith("chu"):
        base += 8.0

    return max(base, 10.0)


def _ra_base_from_Ec(Ec_kV_cm: float, weather: str) -> float:
    """
    Modelo empírico simplificado para nível de ruído audível (dB(A)).
    """
    Ec = max(Ec_kV_cm, 0.0)

    if Ec <= 0:
        base = 20.0
    else:
        Ec_thr = 21.0
        base = 35.0 if Ec <= Ec_thr else (35.0 + 1.5 * (Ec - Ec_thr))

    if (weather or "").strip().lower().startswith("chu"):
        base += 10.0

    return max(base, 20.0)


def _apply_distance_law(level_ref: float, d_ref_m: float, d_m: np.ndarray) -> np.ndarray:
    """
    Lei de decaimento 1/r em dB:
      L(d) = L_ref - 20 log10(d / d_ref)

    Protege distâncias muito pequenas.
    """
    d = np.maximum(d_m.astype(float), 1.0)
    dref = max(float(d_ref_m), 1.0)
    return float(level_ref) - 20.0 * np.log10(d / dref)


# ======================= Cálculo principal ============================


def compute_ri_ra_profiles(
    params: LineParamsResult,
    length_km: float,
    cfg: Optional[RIRAConfig] = None,
) -> RIRAProfiles:
    """
    Calcula perfis de RI e RA usando o gradiente superficial Ec (kV/cm)
    presente em LineParamsResult (campo params.Ec_kV_cm).

    Conformidade verificada na borda da faixa (último ponto), cenário chuva (mais crítico).
    """
    if cfg is None:
        cfg = RIRAConfig()

    if cfg.n_points < 2:
        raise ValueError("n_points deve ser pelo menos 2.")
    if cfg.distance_max_m < cfg.distance_min_m:
        raise ValueError("distance_max_m deve ser >= distance_min_m.")

    # Vetor de distâncias laterais
    x = np.linspace(cfg.distance_min_m, cfg.distance_max_m, cfg.n_points, dtype=float)

    # Extrai Ec com fallback
    Ec = _safe_float(getattr(params, "Ec_kV_cm", 0.0), 0.0)

    # RI: referência a 30 m
    RI_ref_seco = _ri_base_from_Ec(Ec, cfg.freq_MHz, "seco")
    RI_ref_chuva = _ri_base_from_Ec(Ec, cfg.freq_MHz, "chuva")
    RI_seco = _apply_distance_law(RI_ref_seco, 30.0, x)
    RI_chuva = _apply_distance_law(RI_ref_chuva, 30.0, x)

    # RA: referência a 25 m
    RA_ref_seco = _ra_base_from_Ec(Ec, "seco")
    RA_ref_chuva = _ra_base_from_Ec(Ec, "chuva")
    RA_seco = _apply_distance_law(RA_ref_seco, 25.0, x)
    RA_chuva = _apply_distance_law(RA_ref_chuva, 25.0, x)

    # Limite RA por período
    period = (cfg.day_period or "").strip().lower()
    limit_RA = cfg.limit_RA_dBA_day if period.startswith("diur") else cfg.limit_RA_dBA_night

    # Valores na borda (último ponto) - cenário chuva (mais crítico)
    RI_edge = float(RI_chuva[-1]) if RI_chuva.size else 0.0
    RA_edge = float(RA_chuva[-1]) if RA_chuva.size else 0.0

    exceeds_RI = RI_edge > cfg.limit_RI_dBuV_m
    exceeds_RA = RA_edge > limit_RA

    # Comentários
    if not exceeds_RI:
        comment_RI = (
            f"RI na borda da faixa (chuva) ≈ {RI_edge:.1f} dB(µV/m), "
            f"abaixo do limite adotado {cfg.limit_RI_dBuV_m:.1f} dB(µV/m). "
            "Condição considerada ATENDIDA em termos de rádio interferência."
        )
    else:
        comment_RI = (
            f"RI na borda da faixa (chuva) ≈ {RI_edge:.1f} dB(µV/m), "
            f"acima do limite adotado {cfg.limit_RI_dBuV_m:.1f} dB(µV/m). "
            "Recomenda-se revisar geometria, tipo de cabo ou medidas mitigadoras."
        )

    period_label = "diurno" if period.startswith("diur") else "noturno"
    if not exceeds_RA:
        comment_RA = (
            f"Ruído audível na borda da faixa (chuva) ≈ {RA_edge:.1f} dB(A), "
            f"abaixo do limite de {limit_RA:.1f} dB(A) ({period_label}, NBR 10151). "
            "Condição considerada ATENDIDA em termos de ruído ambiental."
        )
    else:
        comment_RA = (
            f"Ruído audível na borda da faixa (chuva) ≈ {RA_edge:.1f} dB(A), "
            f"acima do limite de {limit_RA:.1f} dB(A) ({period_label}, NBR 10151). "
            "Podem ser necessárias medidas adicionais (ajuste geométrico, mitigação de corona, etc.)."
        )

    return RIRAProfiles(
        cfg=cfg,
        distances_m=x,
        RI_seco_dBuV_m=RI_seco,
        RI_chuva_dBuV_m=RI_chuva,
        RA_seco_dBA=RA_seco,
        RA_chuva_dBA=RA_chuva,
        Ec_kV_cm=Ec,
        RI_edge_chuva_dBuV_m=RI_edge,
        RA_edge_chuva_dBA=RA_edge,
        exceeds_RI_limit=exceeds_RI,
        exceeds_RA_limit=exceeds_RA,
        comment_RI=comment_RI,
        comment_RA=comment_RA,
    )


# ======================= Geração de gráficos =========================


def _plot_ri_2d(profiles: RIRAProfiles) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(profiles.distances_m, profiles.RI_seco_dBuV_m, label="RI – tempo bom")
    ax.plot(profiles.distances_m, profiles.RI_chuva_dBuV_m, label="RI – chuva", linestyle="--")
    ax.axhline(profiles.cfg.limit_RI_dBuV_m, linestyle=":", linewidth=1.2, label=f"Limite = {profiles.cfg.limit_RI_dBuV_m:.1f} dB(µV/m)")
    ax.set_xlabel("Distância lateral (m)")
    ax.set_ylabel("RI (dB(µV/m))")
    ax.set_title("Perfil de Rádio Interferência ao Longo da Faixa")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    return _fig_to_base64(fig)


def _plot_ra_2d(profiles: RIRAProfiles) -> str:
    period = (profiles.cfg.day_period or "").strip().lower()
    limit_RA = profiles.cfg.limit_RA_dBA_day if period.startswith("diur") else profiles.cfg.limit_RA_dBA_night

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(profiles.distances_m, profiles.RA_seco_dBA, label="RA – tempo bom")
    ax.plot(profiles.distances_m, profiles.RA_chuva_dBA, label="RA – chuva", linestyle="--")
    ax.axhline(limit_RA, linestyle=":", linewidth=1.2, label=f"Limite ({'diurno' if period.startswith('diur') else 'noturno'}) = {limit_RA:.1f} dB(A)")
    ax.set_xlabel("Distância lateral (m)")
    ax.set_ylabel("Ruído audível (dB(A))")
    ax.set_title("Perfil de Ruído Audível ao Longo da Faixa")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    return _fig_to_base64(fig)


def _plot_ri_3d(profiles: RIRAProfiles) -> str:
    Ec0 = max(profiles.Ec_kV_cm, 1.0)
    Ec_vals = np.linspace(0.8 * Ec0, 1.2 * Ec0, 25)
    X, Y = np.meshgrid(profiles.distances_m, Ec_vals)
    Z = np.zeros_like(X)

    for i, Ec in enumerate(Ec_vals):
        RI_ref = _ri_base_from_Ec(float(Ec), profiles.cfg.freq_MHz, "chuva")
        Z[i, :] = _apply_distance_law(RI_ref, 30.0, profiles.distances_m)

    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("Distância lateral (m)")
    ax.set_ylabel("Ec (kV/cm)")
    ax.set_zlabel("RI (dB(µV/m))")
    ax.set_title("Superfície RI × Distância × Gradiente (chuva)")
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1)
    return _fig_to_base64(fig)


def _plot_ra_3d(profiles: RIRAProfiles) -> str:
    Ec0 = max(profiles.Ec_kV_cm, 1.0)
    Ec_vals = np.linspace(0.8 * Ec0, 1.2 * Ec0, 25)
    X, Y = np.meshgrid(profiles.distances_m, Ec_vals)
    Z = np.zeros_like(X)

    for i, Ec in enumerate(Ec_vals):
        RA_ref = _ra_base_from_Ec(float(Ec), "chuva")
        Z[i, :] = _apply_distance_law(RA_ref, 25.0, profiles.distances_m)

    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("Distância lateral (m)")
    ax.set_ylabel("Ec (kV/cm)")
    ax.set_zlabel("RA (dB(A))")
    ax.set_title("Superfície RA × Distância × Gradiente (chuva)")
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1)
    return _fig_to_base64(fig)


# ======================= Relatório HTML + CSS ========================


def generate_html_report_ri_ra(
    project: ProjectInfo,
    geom: LineGeometry,
    params: LineParamsResult,
    length_km: float,
    profiles: RIRAProfiles,
) -> str:
    """
    Gera relatório HTML + CSS para o estudo de RI/RA.
    """
    # Geometria (opcional)
    geom_img_b64 = ""
    try:
        geom_img_b64 = plot_geometry_2d(geom, "Geometria da Linha – RI/RA")
    except Exception:
        geom_img_b64 = ""

    ri_2d_b64 = _plot_ri_2d(profiles)
    ra_2d_b64 = _plot_ra_2d(profiles)
    ri_3d_b64 = _plot_ri_3d(profiles)
    ra_3d_b64 = _plot_ra_3d(profiles)

    cfg = profiles.cfg
    Ec = profiles.Ec_kV_cm

    # Safe extraction of some params for summary
    Zc = _safe_float(getattr(params, "Zc_ohm", 0.0), 0.0)
    SIL = _safe_float(getattr(params, "SIL_MW", 0.0), 0.0)

    # Limite RA por período
    period = (cfg.day_period or "").strip().lower()
    limit_RA = cfg.limit_RA_dBA_day if period.startswith("diur") else cfg.limit_RA_dBA_night
    period_label = "Diurno" if period.startswith("diur") else "Noturno"

    css = """
    <style>
      body { font-family: "Segoe UI", Arial, sans-serif; background-color:#f5f7fa; color:#222; margin:0; padding:0; }
      .container { max-width:1120px; margin:0 auto; padding:24px; background:#fff; box-shadow:0 4px 16px rgba(0,0,0,0.08); }
      h1,h2,h3 { color:#0b3c5d; }
      .header { border-bottom:2px solid #e0e4ea; margin-bottom:16px; padding-bottom:8px; }
      .meta { font-size:0.95rem; color:#555; }
      table { width:100%; border-collapse:collapse; margin:12px 0; font-size:0.9rem; }
      th,td { border:1px solid #dde2eb; padding:6px 8px; text-align:right; }
      th { background-color:#f0f3f9; font-weight:600; }
      td.label { text-align:left; font-weight:500; }
      .img-block { text-align:center; margin:16px 0; }
      .eq-block { background:#f8fafc; border-left:4px solid #0b3c5d; padding:8px 12px; font-family:Consolas,monospace; font-size:0.85rem; }
      .small-note { font-size:0.8rem; color:#777; }
      .ok { color:#0a7f3f; font-weight:600; }
      .warn { color:#c0392b; font-weight:600; }
    </style>
    """

    vll_line = f"<tr><td class='label'>Tensão nominal</td><td>{cfg.V_LL_kV:.1f}</td><td>kV (L-L)</td></tr>" if cfg.V_LL_kV is not None else ""

    resumo_tab = f"""
    <table>
      <tr><th class="label">Grandeza</th><th>Valor</th><th>Unidade</th></tr>
      {vll_line}
      <tr><td class="label">Frequência de avaliação (RI)</td><td>{cfg.freq_MHz:.3f}</td><td>MHz</td></tr>
      <tr><td class="label">Comprimento da linha</td><td>{length_km:.3f}</td><td>km</td></tr>
      <tr><td class="label">Zc</td><td>{Zc:.3f}</td><td>Ω</td></tr>
      <tr><td class="label">SIL</td><td>{SIL:.3f}</td><td>MW</td></tr>
      <tr><td class="label">Gradiente superficial Ec</td><td>{Ec:.3f}</td><td>kV/cm</td></tr>
      <tr><td class="label">Borda da faixa (chuva) – RI</td><td>{profiles.RI_edge_chuva_dBuV_m:.1f}</td><td>dB(µV/m)</td></tr>
      <tr><td class="label">Borda da faixa (chuva) – RA</td><td>{profiles.RA_edge_chuva_dBA:.1f}</td><td>dB(A)</td></tr>
    </table>
    """

    ri_class = "ok" if not profiles.exceeds_RI_limit else "warn"
    ra_class = "ok" if not profiles.exceeds_RA_limit else "warn"

    methodology_html = f"""
    <div class="eq-block">
      <strong>Metodologia – RI (Rádio Interferência)</strong><br/>
      • Modelo empírico correlacionando Ec com níveis de RI a 30 m (tempo bom/chuva).<br/>
      • Lei de decaimento adotada: L(d) = L_ref − 20 log10(d/d_ref).<br/>
      • Cenário de chuva considerado para verificação de conformidade (mais crítico).<br/>
      • Limite adotado (referência): {cfg.limit_RI_dBuV_m:.1f} dB(µV/m).<br/>
    </div>
    <br/>
    <div class="eq-block">
      <strong>Metodologia – RA (Ruído Audível)</strong><br/>
      • Modelo empírico correlacionando Ec com ruído de corona a 25 m (tempo bom/chuva).<br/>
      • Comparação com limite conforme período ({period_label}) na borda da faixa (condição de chuva).<br/>
      • Limite adotado: {limit_RA:.1f} dB(A) (NBR 10151 – referência de uso).<br/>
    </div>
    """

    geom_block = ""
    if geom_img_b64:
        geom_block = f"""
        <h2>Geometria da Linha</h2>
        <div class="img-block">
          <img src="data:image/png;base64,{geom_img_b64}" alt="Geometria da Linha – RI/RA"/>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="utf-8"/>
      <title>RI/RA – {project.nome_projeto}</title>
      {css}
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>BK_Estudos_Eletricos – Estudo de RI (Rádio Interferência) e RA (Ruído Audível)</h1>
          <div class="meta">
            <strong>Projeto:</strong> {project.nome_projeto}<br/>
            <strong>Cliente:</strong> {project.cliente}<br/>
            <strong>Nº Projeto:</strong> {project.numero_projeto}<br/>
            <strong>Comprimento da linha:</strong> {length_km:.3f} km<br/>
            <strong>Período avaliado (RA):</strong> {period_label}
          </div>
        </div>

        {geom_block}

        <h2>Resumo Elétrico / Parâmetros de Entrada</h2>
        {resumo_tab}

        <h2>Resultados – Rádio Interferência (RI)</h2>
        <p class="{ri_class}">{profiles.comment_RI}</p>
        <div class="img-block">
          <img src="data:image/png;base64,{ri_2d_b64}" alt="Perfil de RI 2D"/>
        </div>
        <div class="img-block">
          <img src="data:image/png;base64,{ri_3d_b64}" alt="Superfície RI 3D"/>
        </div>

        <h2>Resultados – Ruído Audível (RA)</h2>
        <p class="{ra_class}">{profiles.comment_RA}</p>
        <div class="img-block">
          <img src="data:image/png;base64,{ra_2d_b64}" alt="Perfil de RA 2D"/>
        </div>
        <div class="img-block">
          <img src="data:image/png;base64,{ra_3d_b64}" alt="Superfície RA 3D"/>
        </div>

        <h2>Metodologia e Equações</h2>
        {methodology_html}

        <p class="small-note">
          Observação: os modelos são empíricos e indicativos. Para conformidade final com normas e
          licenciamento, recomenda-se medições de campo e/ou modelos dedicados.
        </p>
      </div>
    </body>
    </html>
    """
    return html


# ======================= Teste rápido (opcional) =====================

if __name__ == "__main__":
    # Exemplo sintético para validação do módulo
    proj = ProjectInfo(
        nome_projeto="LT 138 kV – Estudo RI/RA",
        cliente="BK Engenharia e Tecnologia",
        numero_projeto="2025-XXX",
    )

    # Geometria deve ser criada conforme sua LineGeometry real; aqui deixamos vazio
    geom = LineGeometry(conductors=[])

    # Params de linha (exemplo)
    fake_params = LineParamsResult(
        circuit_index=1,
        R_ohm_km=0.05,
        X_ohm_km=0.35,
        B_S_km=3.5e-6,
        L_H_km=1.2e-3,
        C_F_km=12e-9,
        L_mH_km=1.2,
        C_nF_km=12.0,
        Zc_ohm=380.0,
        SIL_MW=50.0,
        lambda_m=3000.0,
        Q_mC_km_por_fase=0.5,
        Ec_kV_cm=22.0,
        GMD_m=10.0,
        GMR_eq_m=0.012,
        r_eq_m=0.015,
        Vs=None,
        Vr=None,
    )

    cfg = RIRAConfig(
        freq_MHz=0.5,
        weather="chuva",
        distance_min_m=0.0,
        distance_max_m=60.0,
        n_points=241,
        observation_height_m=1.5,
        day_period="diurno",
        V_LL_kV=138.0,
    )
    profiles = compute_ri_ra_profiles(params=fake_params, length_km=100.0, cfg=cfg)
    html = generate_html_report_ri_ra(project=proj, geom=geom, params=fake_params, length_km=100.0, profiles=profiles)

    with open("relatorio_ri_ra_exemplo.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Relatório RI/RA de exemplo gerado: relatorio_ri_ra_exemplo.html")

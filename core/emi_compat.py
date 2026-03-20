# bk_estudos_eletricos/core/emi_compat.py
# ======================================================================
# Compatibilidade Eletromagnética (EMI) – acoplamento linha–duto e linha–comunicação
#
# Implementação "BK": módulo leve, sem dependências externas de EMT.
# Objetivo: triagem preliminar e geração de relatório HTML com gráficos.
#
# API pública esperada pelo UI:
#   - EMIConfig
#   - run_emi_study(project, geom, cfg, include_pipeline=True, include_comm=True) -> EMIStudyResult
#   - generate_html_report_emi(project, geom, cfg, pipeline_res, comm_res) -> str
#
# Observação:
# - Este módulo NÃO tenta reproduzir normas detalhadas (CIGRÉ/IEEE/ITU) por completo.
# - Ele fornece estimativas conservadoras e coerentes dimensionalmente para comparação.
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict
import math
import base64
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------

@dataclass
class EMIConfig:
    # Sistema
    f_hz: float = 60.0

    # Paralelismo linha–duto / linha–via
    length_parallel_km: float = 10.0
    separation_m: float = 50.0  # distância lateral média linha–duto (m)

    # Corrente de referência (triagem)
    I_load_A: float = 600.0     # corrente RMS por fase (A)

    # Limites (valores típicos de triagem; ajuste conforme norma do cliente)
    pipeline_cont_limit_V_per_km: float = 60.0  # V/km (contínuo)
    pipeline_short_limit_V_per_km: float = 300.0  # V/km (evento curto)

    comm_E_limit_V_per_m: float = 5.0  # V/m (campo longitudinal equivalente)

    # Fatores empíricos (calibração)
    k_mutual: float = 1.0e-6  # fator "mutual" efetivo (H/m) para triagem
    k_screen: float = 1.0     # fator de blindagem (1 = sem blindagem)


# ----------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------

@dataclass
class PipelineResult:
    V_induced_cont_V_per_km: float
    V_induced_short_V_per_km: float
    V_limit_cont_V_per_km: float
    V_limit_short_V_per_km: float
    exceeds_cont_limit: bool
    exceeds_short_limit: bool
    notes: str = ""


@dataclass
class CommResult:
    E_longitudinal_V_per_m: float
    E_limit_V_per_m: float
    exceeds_E_limit: bool
    notes: str = ""


@dataclass
class EMIStudyResult:
    config: EMIConfig
    pipeline_result: Optional[PipelineResult] = None
    comm_result: Optional[CommResult] = None
    summary: str = ""


# ----------------------------------------------------------------------
# Núcleo de cálculo (triagem)
# ----------------------------------------------------------------------

def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _estimate_I_load_from_project(project: Any, cfg: EMIConfig) -> float:
    """
    Estima corrente a partir de ProjectInfo quando disponível:
      I ≈ S / (sqrt(3) * V_LL)
    Se não houver tensão/potência, retorna cfg.I_load_A.
    """
    S_MVA = _safe_float(getattr(project, "power_mva", None), cfg.I_load_A * 0.0)
    V_LL_kV = _safe_float(getattr(project, "voltage_kv", None), 0.0)

    if S_MVA > 0 and V_LL_kV > 0:
        I = (S_MVA * 1e6) / (math.sqrt(3.0) * V_LL_kV * 1e3)
        return float(I)
    return float(cfg.I_load_A)


def _pipeline_induced_voltage_per_km(cfg: EMIConfig, I_A: float) -> Dict[str, float]:
    """
    Modelo simplificado: V_ind ∝ ω * M_eff * I.
    - M_eff é um fator efetivo (H/m) ajustável (cfg.k_mutual) / separação.
    - ω = 2πf
    Retorna contínuo e evento curto (multiplicador típico).
    """
    f = max(cfg.f_hz, 1e-6)
    sep = max(cfg.separation_m, 1.0)
    omega = 2.0 * math.pi * f

    # M_eff decai com separação (triagem). k_mutual ~ ordem de µH/m para casos próximos.
    M_eff = (cfg.k_mutual / sep) * cfg.k_screen  # H/m

    # tensão induzida por metro (V/m): ω M I
    V_per_m = omega * M_eff * max(I_A, 0.0)
    V_per_km = V_per_m * 1000.0

    # Evento curto: multiplicador (ex.: faltas/curtos, assimetria). Valor de triagem.
    V_short_per_km = 5.0 * V_per_km

    return {"cont": float(V_per_km), "short": float(V_short_per_km)}


def _comm_longitudinal_field(cfg: EMIConfig, V_induced_V_per_km: float) -> float:
    """
    Aproxima um 'campo longitudinal equivalente' proporcional à tensão induzida ao longo de 1 km:
      E ≈ V/km / 1000  (V/m)
    e aplica um fator conservador.
    """
    return float(0.6 * (V_induced_V_per_km / 1000.0))


def run_emi_study(
    project: Any,
    geom: Any,
    cfg: Optional[EMIConfig] = None,
    include_pipeline: bool = True,
    include_comm: bool = True,
) -> EMIStudyResult:
    """
    Pipeline principal. 'geom' é aceito para compatibilidade com o restante do app,
    mas o modelo de triagem aqui não precisa da geometria detalhada.
    """
    cfg = cfg or EMIConfig()

    # Corrente
    I_A = _estimate_I_load_from_project(project, cfg)

    pipeline_res: Optional[PipelineResult] = None
    comm_res: Optional[CommResult] = None
    msgs = []

    if include_pipeline:
        vals = _pipeline_induced_voltage_per_km(cfg, I_A)
        Vc = vals["cont"]
        Vs = vals["short"]

        pipeline_res = PipelineResult(
            V_induced_cont_V_per_km=Vc,
            V_induced_short_V_per_km=Vs,
            V_limit_cont_V_per_km=float(cfg.pipeline_cont_limit_V_per_km),
            V_limit_short_V_per_km=float(cfg.pipeline_short_limit_V_per_km),
            exceeds_cont_limit=bool(Vc > cfg.pipeline_cont_limit_V_per_km),
            exceeds_short_limit=bool(Vs > cfg.pipeline_short_limit_V_per_km),
            notes=f"I_ref={I_A:.1f} A | sep={cfg.separation_m:.1f} m | Lpar={cfg.length_parallel_km:.2f} km",
        )
        msgs.append(
            f"Duto: V_ind(cont)={Vc:.1f} V/km (lim {cfg.pipeline_cont_limit_V_per_km:.1f}) | "
            f"V_ind(curto)={Vs:.1f} V/km (lim {cfg.pipeline_short_limit_V_per_km:.1f})"
        )

    if include_comm:
        # Se pipeline foi calculado, usa ele; senão calcula com a mesma base
        Vbase = pipeline_res.V_induced_cont_V_per_km if pipeline_res else _pipeline_induced_voltage_per_km(cfg, I_A)["cont"]
        E = _comm_longitudinal_field(cfg, Vbase)
        comm_res = CommResult(
            E_longitudinal_V_per_m=E,
            E_limit_V_per_m=float(cfg.comm_E_limit_V_per_m),
            exceeds_E_limit=bool(E > cfg.comm_E_limit_V_per_m),
            notes=f"Derivado de V_ind(cont)={Vbase:.1f} V/km",
        )
        msgs.append(f"Comunicação: E_long={E:.3f} V/m (lim {cfg.comm_E_limit_V_per_m:.3f})")

    summary = " | ".join(msgs) if msgs else "Nenhum subestudo selecionado."

    return EMIStudyResult(config=cfg, pipeline_result=pipeline_res, comm_result=comm_res, summary=summary)


# ----------------------------------------------------------------------
# Relatório HTML
# ----------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _plot_pipeline(p: PipelineResult) -> str:
    labels = ["Contínuo", "Curto (triagem)"]
    vals = [p.V_induced_cont_V_per_km, p.V_induced_short_V_per_km]
    limits = [p.V_limit_cont_V_per_km, p.V_limit_short_V_per_km]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(x, vals)
    ax.plot(x, limits, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("V_ind (V/km)")
    ax.set_title("Tensão induzida no duto – triagem")
    ax.grid(True, axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _plot_comm(c: CommResult) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar([0], [c.E_longitudinal_V_per_m])
    ax.axhline(c.E_limit_V_per_m, linestyle="--")
    ax.set_xticks([0])
    ax.set_xticklabels(["E_long"])
    ax.set_ylabel("V/m")
    ax.set_title("Campo longitudinal equivalente – comunicação (triagem)")
    ax.grid(True, axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def generate_html_report_emi(
    project: Any,
    geom: Any,
    cfg: EMIConfig,
    pipeline_res: Optional[PipelineResult],
    comm_res: Optional[CommResult],
) -> str:
    proj_name = getattr(project, "nome_projeto", getattr(project, "name", "Projeto"))
    cliente = getattr(project, "cliente", getattr(project, "client", ""))
    numero = getattr(project, "numero_projeto", getattr(project, "project_number", ""))

    img_pipe = _plot_pipeline(pipeline_res) if pipeline_res else None
    img_comm = _plot_comm(comm_res) if comm_res else None

    def ok_bad(flag: bool) -> str:
        return "NÃO ATENDE" if flag else "ATENDE"

    pipe_html = ""
    if pipeline_res:
        p = pipeline_res
        pipe_html = f"""
        <h2>Duto / Pipeline</h2>
        <p><b>V_ind contínuo:</b> {p.V_induced_cont_V_per_km:.1f} V/km (limite {p.V_limit_cont_V_per_km:.1f}) → <b>{ok_bad(p.exceeds_cont_limit)}</b></p>
        <p><b>V_ind curto (triagem):</b> {p.V_induced_short_V_per_km:.1f} V/km (limite {p.V_limit_short_V_per_km:.1f}) → <b>{ok_bad(p.exceeds_short_limit)}</b></p>
        <p class="note">{p.notes}</p>
        <div class="img"><img src="data:image/png;base64,{img_pipe}" alt="pipeline"/></div>
        """

    comm_html = ""
    if comm_res:
        c = comm_res
        comm_html = f"""
        <h2>Comunicação</h2>
        <p><b>E_long:</b> {c.E_longitudinal_V_per_m:.3f} V/m (limite {c.E_limit_V_per_m:.3f}) → <b>{ok_bad(c.exceeds_E_limit)}</b></p>
        <p class="note">{c.notes}</p>
        <div class="img"><img src="data:image/png;base64,{img_comm}" alt="comm"/></div>
        """

    css = """
    <style>
      body { font-family: "Segoe UI", Arial, sans-serif; background:#f5f7fa; color:#222; margin:0; }
      .container { max-width:1080px; margin:0 auto; padding:24px; background:#fff; box-shadow:0 4px 16px rgba(0,0,0,0.08); }
      h1,h2 { color:#0b3c5d; margin: 10px 0; }
      .meta { color:#555; font-size:0.95rem; }
      .img { text-align:center; margin:12px 0 18px; }
      img { max-width: 980px; width:100%; height:auto; border:1px solid #e5e9f2; }
      .note { font-size:0.9rem; color:#666; }
      .box { background:#f8fafc; border-left:4px solid #0b3c5d; padding:10px 14px; margin:12px 0; }
      table { width:100%; border-collapse:collapse; margin:12px 0; font-size:0.92rem; }
      th,td { border:1px solid #dde2eb; padding:6px 8px; text-align:right; }
      th { background:#f0f3f9; font-weight:600; }
      td.label { text-align:left; }
    </style>
    """

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <title>Relatório – EMI – {proj_name}</title>
  {css}
</head>
<body>
  <div class="container">
    <h1>BK_Estudos_Eletricos – Compatibilidade Eletromagnética (EMI)</h1>
    <div class="meta">
      <b>Projeto:</b> {proj_name}<br/>
      <b>Cliente:</b> {cliente}<br/>
      <b>Nº Projeto:</b> {numero}<br/>
      <b>f:</b> {cfg.f_hz:.1f} Hz | <b>L paralela:</b> {cfg.length_parallel_km:.2f} km | <b>Separação:</b> {cfg.separation_m:.1f} m | <b>I_ref:</b> {cfg.I_load_A:.1f} A
    </div>

    <div class="box">
      <b>Escopo:</b> triagem preliminar de acoplamento eletromagnético linha–duto e linha–comunicação.
      Ajuste limites e fatores empíricos conforme norma/cliente.
    </div>

    {pipe_html}
    {comm_html}

  </div>
</body>
</html>
"""
    return html


# ----------------------------------------------------------------------
# Teste rápido
# ----------------------------------------------------------------------
if __name__ == "__main__":
    class _Proj:
        nome_projeto = "Exemplo EMI"
        cliente = "BK Engenharia"
        numero_projeto = "2025-EMI-001"
        power_mva = 100.0
        voltage_kv = 138.0

    cfg = EMIConfig()
    res = run_emi_study(_Proj(), geom=None, cfg=cfg, include_pipeline=True, include_comm=True)
    html = generate_html_report_emi(_Proj(), geom=None, cfg=cfg, pipeline_res=res.pipeline_result, comm_res=res.comm_result)
    Path("relatorio_emi_exemplo.html").write_text(html, encoding="utf-8")
    print(res.summary)
    print("Relatório gerado: relatorio_emi_exemplo.html")

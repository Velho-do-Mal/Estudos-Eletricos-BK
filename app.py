"""
BK Estudos Eletricos - Linhas de Transmissao
Streamlit Edition - BK Engenharia e Tecnologia
13 modulos de estudo completos
"""
from __future__ import annotations
import sys, math, json, traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
 
from core.cables import Cable, default_cable_db, find_cable, calc_line_params_from_cable, EPS0, MU0
from core.geometry_model import ConductorInstance, LineGeometry, build_geometry_from_home
from core.line_params import (compute_all_circuits_params, ProjectInfo, VoltageSpec, LineParamsResult,
    compute_GMD_for_circuit, compute_surface_field_Ec, compute_line_charge_per_phase, approximate_lambda,
    solve_vr_pi)
from core.corona import CoronaConfig, CoronaCircuitResult, compute_corona_all_circuits
from core.ampacity_sag import AmpacitySagConfig, AmpacitySagSummary, compute_ampacity_sag_for_geometry
from core.shielding import ShieldingConfig, ShieldingResult, compute_shielding
from core.vmax_insulation import VmaxConfig, InsulationItem, InsulationItemResult, compute_all_items_insulation
from core.reclosing_tripolar import ReclosingConfig, ReclosingStudyResult, compute_reclosing_study
from core.emi_compat import EMIConfig, EMIStudyResult, run_emi_study
from core.coord_isol import CoordIsolConfig, CoordIsolResult, compute_coord_isolation
 
try:
    from core.field_em import FieldConfig, AneelLimits, FieldProfilesResult, compute_fields_profiles
    HAS_FIELDS = True
except Exception:
    HAS_FIELDS = False
try:
    from core.ri_ra import RIRAConfig, RIRAProfiles, compute_ri_ra_profiles
    HAS_RIRA = True
except Exception:
    HAS_RIRA = False
try:
    from core.power_flow import (Bus as PFBus, Branch as PFBranch, PowerFlowCase as PFCase,
        solve_power_flow_newton, generate_html_report_power_flow)
    HAS_PF = True
except Exception:
    HAS_PF = False
 
from theme import (apply_bk_theme, bk_header, bk_section, bk_kpi_row,
    BK_BLUE, BK_BLUE_LIGHT, BK_TEAL, BK_GREEN, BK_ORANGE, BK_RED, BK_PURPLE,
    BK_DARK, BK_GRAY, BK_COLORS, PLOTLY_LAYOUT)
 
# DB — persistência local + Neon
from db import (upsert_project, list_projects, get_project, delete_project,
    init_neon, upsert_project_neon, list_projects_neon, get_project_neon,
    save_study_neon, list_studies_neon, load_study_neon)
 
# Report engine (DOCX generation)
# report_bridge só depende de streamlit (sempre disponível)
try:
    import report_bridge as rb
except Exception:
    class _DummyRB:
        def __getattr__(self, name):
            return lambda *a, **kw: ({}, {})
    rb = _DummyRB()
 
try:
    from reports import generate_report, REPORT_TITLES
    HAS_DOCX = True
    REPORT_FILENAMES = {k: f"BK_Relatorio_{v.replace(' ','_')}.docx" for k, v in REPORT_TITLES.items()}
except Exception as _e:
    HAS_DOCX = False
    REPORT_FILENAMES = {}
    REPORT_TITLES = {}
    def generate_report(*a, **kw): return b""
 
st.set_page_config(page_title="BK Estudos Eletricos", page_icon="\u26a1", layout="wide", initial_sidebar_state="expanded")
apply_bk_theme()
 
# === SESSION STATE ===
# ─────────────────────────────────────────────────────────────────────────────
# ARQUITETURA DE PERSISTÊNCIA
#
# Problema: Streamlit limpa automaticamente do session_state as chaves de
# widgets (key=) que NÃO foram renderizadas no último ciclo de script. Isso
# faz com que os parâmetros da Home (tensão, geometria, cabos…) sejam perdidos
# ao navegar para uma página de estudo, pois os widgets da Home não renderizam.
#
# Solução: manter um dict `_proj` em session_state que NÃO é widget key e,
# portanto, NUNCA é limpo automaticamente pelo Streamlit. Ele é a "fonte da
# verdade" de todos os estudos. Ao renderizar a Home, sincronizamos widgets →
# _proj. Em cada _init_state(), restauramos as chaves de widget a partir de
# _proj antes de aplicar os defaults de fábrica.
# ─────────────────────────────────────────────────────────────────────────────
 
# Chaves que pertencem ao projeto e precisam persistir entre páginas
_PROJ_KEYS = [
    "proj_name", "client", "proj_number",
    "voltage_kv", "power_mva", "freq_hz", "pf_load", "altitude_m",
    "n_circuits", "geometry_type", "circuits_layout", "n_lines",
    "bundle_n", "bundle_ds", "phase_vert_spacing",
    "dx_B", "dx_C", "circuit_spacing",
    "h_phase_ref", "h_min_phase", "h_shield",
    "cable_phase_key", "cable_shield_key",
    "n_shield_wires", "shield_dx_m",
    "line_length_km", "temp_C", "Vs_mag", "Vs_ang",
]
 
# Valores padrão de fábrica (usados só quando não há projeto carregado)
_PROJ_DEFAULTS: dict = dict(
    proj_name="", client="", proj_number="",
    voltage_kv=138.0, power_mva=100.0, freq_hz=60.0,
    pf_load=1.0, altitude_m=0.0,
    n_circuits=1, geometry_type="horizontal", circuits_layout="side", n_lines=1,
    bundle_n=1, bundle_ds=0.4, phase_vert_spacing=4.0,
    dx_B=8.0, dx_C=16.0, circuit_spacing=20.0,
    h_phase_ref=15.0, h_min_phase=12.0, h_shield=20.0,
    cable_phase_key="ACSR_477", cable_shield_key="EHS_3_8in",
    n_shield_wires=1, shield_dx_m=4.0,
    line_length_km=100.0, temp_C=50.0,
    Vs_mag=138.0, Vs_ang=0.0,
)
 
 
def _ensure_proj() -> None:
    """Garante que _proj existe em session_state com todos os campos."""
    if "_proj" not in st.session_state:
        st.session_state["_proj"] = _PROJ_DEFAULTS.copy()
    else:
        # Preenche chaves que possam estar faltando (upgrade de versão)
        for k, v in _PROJ_DEFAULTS.items():
            st.session_state["_proj"].setdefault(k, v)
 
 
def _sync_proj() -> None:
    """Copia valores dos widgets da Home para _proj.
    Deve ser chamado ao FINAL de cada render da Home page."""
    _ensure_proj()
    for k in _PROJ_KEYS:
        if k in st.session_state:
            st.session_state["_proj"][k] = st.session_state[k]
 
 
def _restore_from_proj() -> None:
    """Restaura chaves de widget a partir de _proj.
    Chamado em _init_state() para reverter limpeza automática do Streamlit."""
    _ensure_proj()
    proj = st.session_state["_proj"]
    for k in _PROJ_KEYS:
        if k not in st.session_state:
            st.session_state[k] = proj.get(k, _PROJ_DEFAULTS.get(k))
 
 
def _populate_from_project(_pd: dict) -> None:
    """Popula session_state E _proj a partir de um dict de projeto (Neon/local).
    Usado pelo auto-load na inicialização e pelo botão 'Carregar'."""
    _ensure_proj()
    st.session_state.proj_name     = _pd.get("name", "")
    st.session_state.client        = _pd.get("client", "")
    st.session_state.proj_number   = _pd.get("project_number", "")
    st.session_state.voltage_kv    = float(_pd.get("voltage_kv")   or 138.0)
    st.session_state.power_mva     = float(_pd.get("power_mva")    or 100.0)
    st.session_state.freq_hz       = float(_pd.get("frequency_hz") or 60.0)
    st.session_state.n_circuits    = int(_pd.get("n_circuits")     or 1)
    st.session_state.geometry_type = _pd.get("geometry_type")      or "horizontal"
    st.session_state.n_lines       = int(_pd.get("n_lines")        or 1)
    st.session_state.Vs_mag        = float(_pd.get("voltage_kv")   or 138.0)
    _meta = _pd.get("meta") or {}
    # Validação contra valores corrompidos (mínimos de widget gravados por versões antigas)
    _meta_validators = {
        "line_length_km": lambda v: float(v) if v is not None and float(v) >= 1.0 else 100.0,
        "temp_C":         lambda v: float(v) if v is not None and float(v) >= 0.0 else 50.0,
        "bundle_n":       lambda v: int(v)   if v is not None and int(v) >= 1 else 1,
        "n_shield_wires": lambda v: int(v)   if v is not None and int(v) >= 0 else 1,
        "h_phase_ref":    lambda v: float(v) if v is not None and float(v) >= 0.5 else 15.0,
        "h_min_phase":    lambda v: float(v) if v is not None and float(v) >= 0.5 else 12.0,
        "h_shield":       lambda v: float(v) if v is not None and float(v) >= 0.5 else 20.0,
    }
    for _mk in ["pf_load", "altitude_m", "circuits_layout", "bundle_n", "bundle_ds",
                "phase_vert_spacing", "dx_B", "dx_C", "circuit_spacing", "h_phase_ref",
                "h_min_phase", "h_shield", "cable_phase_key", "cable_shield_key",
                "n_shield_wires", "shield_dx_m", "line_length_km", "temp_C", "Vs_ang"]:
        if _mk in _meta:
            _raw = _meta[_mk]
            _validator = _meta_validators.get(_mk)
            try:
                st.session_state[_mk] = _validator(_raw) if _validator else _raw
            except Exception:
                st.session_state[_mk] = _PROJ_DEFAULTS.get(_mk, _raw)
    # Invalida cache de resultados
    st.session_state.results = None
    st.session_state["computed_vr"] = None
    st.session_state["prev_voltage_kv"] = st.session_state.voltage_kv
    # ← Crítico: sincroniza _proj imediatamente para que estudos usem esses valores
    _sync_proj()
 
 
def _auto_load_latest_project() -> None:
    """Na primeira execução da sessão, carrega automaticamente o projeto
    mais recente do Neon DB. Evita que o app abra sempre com os padrões
    de fábrica após reload do browser."""
    try:
        init_neon()
        projs = list_projects_neon()
        if projs:
            pid = projs[0]["id"]   # lista já ordenada por data de atualização
            _pd = get_project_neon(pid)
            if _pd:
                _populate_from_project(_pd)
                st.session_state["neon_project_id"] = pid
    except Exception:
        pass  # falha silenciosa — app continua funcional sem BD
 
 
def _build_save_dict() -> dict:
    """Monta o dict do projeto para salvar. Lê de _proj (fonte da verdade)
    com fallback para session_state."""
    _ensure_proj()
    p = st.session_state["_proj"]
    return {
        "name": p.get("proj_name", "") or "Sem nome",
        "client": p.get("client", ""),
        "project_number": p.get("proj_number", ""),
        "voltage_kv": p.get("voltage_kv", 138.0),
        "power_mva": p.get("power_mva", 100.0),
        "frequency_hz": p.get("freq_hz", 60.0),
        "n_circuits": p.get("n_circuits", 1),
        "n_cables_per_phase": p.get("bundle_n", 1),
        "geometry_type": p.get("geometry_type", "horizontal"),
        "n_lines": p.get("n_lines", 1),
        "meta": {
            "pf_load": p.get("pf_load", 1.0),
            "altitude_m": p.get("altitude_m", 0.0),
            "circuits_layout": p.get("circuits_layout", "side"),
            "bundle_n": p.get("bundle_n", 1),
            "bundle_ds": p.get("bundle_ds", 0.4),
            "phase_vert_spacing": p.get("phase_vert_spacing", 4.0),
            "dx_B": p.get("dx_B", 8.0),
            "dx_C": p.get("dx_C", 16.0),
            "circuit_spacing": p.get("circuit_spacing", 20.0),
            "h_phase_ref": p.get("h_phase_ref", 15.0),
            "h_min_phase": p.get("h_min_phase", 12.0),
            "h_shield": p.get("h_shield", 20.0),
            "cable_phase_key": p.get("cable_phase_key", "ACSR_477"),
            "cable_shield_key": p.get("cable_shield_key", "EHS_3_8in"),
            "n_shield_wires": p.get("n_shield_wires", 1),
            "shield_dx_m": p.get("shield_dx_m", 4.0),
            "line_length_km": p.get("line_length_km", 100.0),
            "temp_C": p.get("temp_C", 50.0),
            "Vs_ang": p.get("Vs_ang", 0.0),
        },
    }
 
 
def _auto_save_home() -> None:
    """Salva automaticamente no Neon ao navegar da Home para outro estudo.
    Evita perda de alterações não confirmadas com 'Salvar Projeto'."""
    _npid = st.session_state.get("neon_project_id")
    if not _npid:
        return
    try:
        upsert_project_neon(_build_save_dict(), _npid)
    except Exception:
        pass  # falha silenciosa
 
 
def _init_state():
    # ── 1. Garante _proj (dict persistente — Streamlit NUNCA o limpa) ──────
    _ensure_proj()
 
    # ── 2. Restaura chaves de widget que o Streamlit possa ter limpado ──────
    # Isso é o coração do fix: se voltage_kv, n_circuits, etc. foram removidos
    # pelo Streamlit ao navegar para outra página, aqui os recuperamos de _proj.
    _restore_from_proj()
 
    # ── 3. Defaults para chaves NÃO relacionadas ao projeto ─────────────────
    _non_proj_defaults = dict(
        n_cables_phase=1,
        results=None,
        prev_voltage_kv=138.0,
        neon_project_id=None,
        computed_vr=None,
    )
    for k, v in _non_proj_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
 
    # ── 4. Banco de cabos ────────────────────────────────────────────────────
    if "cable_db" not in st.session_state:
        cables = default_cable_db()
        st.session_state.cable_db = pd.DataFrame([{
            "Codigo": c.key, "Material": c.material,
            "Area_kcmil": round(c.area_kcmil, 1),
            "Diametro_mm": round(c.diameter_mm, 2),
            "GMR_mm": round(c.gmr_mm, 2),
            "Rdc_ohm_km": round(c.rdc_ohm_km_20C, 6),
            "eps_r": c.eps_r_insulation,
            "Notas": c.notes,
        } for c in cables])
 
    # ── 5. Auto-load do último projeto na PRIMEIRA execução da sessão ────────
    # Em sessões novas (reload de browser), os dados do projeto são perdidos.
    # Aqui buscamos o projeto mais recente do Neon DB automaticamente.
    if not st.session_state.get("_session_init_done"):
        st.session_state["_session_init_done"] = True
        _auto_load_latest_project()
_init_state()
 
# ── Detecta mudança de tensão e invalida cache ──────────────
_cur_v = st.session_state.get("voltage_kv", 138.0)
_prev_v = st.session_state.get("prev_voltage_kv", 138.0)
if abs(_cur_v - _prev_v) > 0.01:
    st.session_state.results = None  # força recálculo
    st.session_state.Vs_mag = _cur_v  # sincroniza Vs com tensão nominal
    st.session_state["prev_voltage_kv"] = _cur_v
    # Limpa caches de todos os módulos
    for _ck in ["corona_result", "fields_result", "ampacity_result",
                "shielding_result", "vmax_result", "coord_isol_result",
                "reclosing_result", "emi_result", "ri_ra_result", "pf_result",
                "computed_vr"]:
        if _ck in st.session_state:
            del st.session_state[_ck]
 
# Invalidação de cache por mudança de cabo/geometria
_cur_cable = st.session_state.get("cable_phase_key", "")
_prev_cable = st.session_state.get("prev_cable_phase_key", _cur_cable)
_cur_geom   = st.session_state.get("geometry_type", "")
_prev_geom  = st.session_state.get("prev_geometry_type", _cur_geom)
_cur_bundle = st.session_state.get("bundle_n", 1)
_prev_bundle= st.session_state.get("prev_bundle_n", _cur_bundle)
if _cur_cable != _prev_cable or _cur_geom != _prev_geom or _cur_bundle != _prev_bundle:
    st.session_state.results = None
    for _ck in ["corona_result","fields_result","ampacity_result",
                "shielding_result","vmax_result","coord_isol_result",
                "reclosing_result","emi_result","ri_ra_result","pf_result","computed_vr"]:
        if _ck in st.session_state:
            del st.session_state[_ck]
st.session_state["prev_cable_phase_key"] = _cur_cable
st.session_state["prev_geometry_type"]   = _cur_geom
st.session_state["prev_bundle_n"]        = _cur_bundle
 
# === HELPERS ===
def _proj_info():
    s = st.session_state
    return s.get("proj_name",""), s.get("client",""), s.get("proj_number","")
 
def _report_btns(module_key, extract_func=None, result_key=None, case_obj=None):
    """Botões de relatório DOCX + salvar estudo no BD para cada módulo."""
    if not HAS_DOCX:
        st.caption("⚠️ Relatórios Word indisponíveis — instale: `pip install python-docx`")
        return
    st.divider()
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if st.button("📄 Gerar Relatório Word", key=f"rpt_{module_key}", use_container_width=True):
            try:
                with st.spinner("Gerando relatório Word..."):
                    raw_result = st.session_state.get(result_key) if result_key else None
                    if extract_func is not None and callable(extract_func):
                        fname = getattr(extract_func, "__name__", "")
                        if "power_flow" in fname:
                            results, cfg = extract_func(raw_result, case_obj)
                        else:
                            results, cfg = extract_func(raw_result)
                    else:
                        results, cfg = {}, {}
                    codigo = st.session_state.get("proj_number", "")
                    docx_bytes = generate_report(module_key, results, cfg, codigo_doc=codigo)
                    st.session_state[f"rpt_data_{module_key}"] = docx_bytes
                    st.success("Relatório gerado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao gerar relatório: {e}")
    with c2:
        data = st.session_state.get(f"rpt_data_{module_key}")
        if data:
            st.download_button("⬇️ Download DOCX", data,
                file_name=REPORT_FILENAMES.get(module_key, f"relatorio_{module_key}.docx"),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"dl_{module_key}", use_container_width=True)
    with c3:
        # Botão salvar estudo no Neon
        _npid = st.session_state.get("neon_project_id")
        if _npid:
            if st.button("💾 Salvar BD", key=f"save_s_{module_key}", use_container_width=True):
                try:
                    _inp = {"voltage_kv": st.session_state.voltage_kv,
                            "power_mva": st.session_state.power_mva,
                            "freq_hz": st.session_state.freq_hz,
                            "module": module_key}
                    _res = {"status": "calculado"}
                    # Tenta extrair dados para o report (reusa extract_func)
                    raw = st.session_state.get(result_key)
                    if extract_func and raw:
                        try:
                            fname = getattr(extract_func, "__name__", "")
                            if "power_flow" in fname:
                                _r, _c = extract_func(raw, case_obj)
                            else:
                                _r, _c = extract_func(raw)
                            _res.update(_r)
                            _inp.update(_c)
                        except Exception:
                            pass
                    from db import save_study_neon as _ssn
                    _label = REPORT_TITLES.get(module_key, module_key)
                    sid = _ssn(_npid, module_key, _label, _inp, _res)
                    if sid:
                        st.success(f"✅ Salvo (ID {sid})")
                    else:
                        st.warning("⚠️ Erro Neon")
                except Exception as _e:
                    st.error(f"Erro: {_e}")
        elif st.session_state.get(f"rpt_data_{module_key}"):
            st.caption("✅ Pronto")
def _df_to_cables(df):
    cables = []
    for _, row in df.iterrows():
        try:
            cables.append(Cable(
                key=str(row.get("Codigo", "")), material=str(row.get("Material", "Al")),
                area_kcmil=float(row.get("Area_kcmil", 0)),
                diameter_mm=float(row.get("Diametro_mm", 0)),
                gmr_mm=float(row.get("GMR_mm", 0)),
                rdc_ohm_km_20C=float(row.get("Rdc_ohm_km", 0)),
                eps_r_insulation=float(row.get("eps_r", 1.0)),
                notes=str(row.get("Notas", "")),
            ))
        except Exception:
            continue
    return cables
 
def _cable_keys():
    return st.session_state.cable_db["Codigo"].tolist()
 
def _find_cable(key):
    return find_cable(_df_to_cables(st.session_state.cable_db), key)
 
def _build_home_dict():
    s = st.session_state
    return dict(
        n_circuits=s.n_circuits, n_cables_per_phase=s.bundle_n, bundle_spacing_m=s.bundle_ds,
        geometry_type=s.geometry_type, circuits_layout=s.circuits_layout,
        ground_clearance_m=s.h_phase_ref, phase_B_dx_m=s.dx_B, phase_C_dx_m=s.dx_C,
        phase_vert_spacing_m=s.phase_vert_spacing, circuit_spacing_m=s.circuit_spacing,
        cable_phase_key=s.cable_phase_key, cable_shield_key=s.cable_shield_key,
        shield_present=True, shield_dy_m=s.h_shield - s.h_phase_ref,
        shield_dx_m=s.shield_dx_m, n_shield_wires=s.n_shield_wires)
 
def _get_geom():
    return build_geometry_from_home(_build_home_dict())
 
def _get_cables():
    return _df_to_cables(st.session_state.cable_db)
 
def _compute_params():
    s = st.session_state
    if s.results:
        rd = {r.circuit_index: r for r in s.results}
    else:
        geom = _get_geom()
        vs = VoltageSpec(s.voltage_kv, s.Vs_ang)
        vs_d = {i: vs for i in range(1, s.n_circuits + 1)}
        rd = compute_all_circuits_params(geom=geom, cable_db=_get_cables(), V_LL_kV=s.voltage_kv,
            f_hz=s.freq_hz, temp_C=s.temp_C, Vs_by_circuit=vs_d, Vr_by_circuit=None)
        s.results = [rd[k] for k in sorted(rd.keys())]
 
    # Calcula Vr para cada circuito via solve_vr_pi (modelo π nominal)
    existing_vr = st.session_state.get("computed_vr")
    if not existing_vr:
        vs = VoltageSpec(s.voltage_kv, s.Vs_ang)
        computed_vr = {}
        vr_errors = []
        for cidx, r in rd.items():
            try:
                S_per_circ = float(s.power_mva) / max(1, int(s.n_circuits))
                pf = float(s.pf_load) if s.pf_load else 0.92
                Vr_kV, Vr_ang, I_A, Ploss = solve_vr_pi(
                    vs, r, float(s.line_length_km), S_per_circ, pf)
                if Vr_kV is not None and Vr_kV > 0:
                    computed_vr[cidx] = (Vr_kV, Vr_ang, I_A, Ploss)
                else:
                    vr_errors.append(f"Circ {cidx}: solve_vr_pi retornou None")
            except Exception as e:
                vr_errors.append(f"Circ {cidx}: {e}")
        st.session_state["computed_vr"] = computed_vr
        if vr_errors:
            st.session_state["vr_errors"] = vr_errors
    return rd
 
def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:
        st.error(f"\u274c {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        return default
 
def _save_study_btn(module_key: str, module_label: str, input_data: dict, result_data: dict):
    """Botão genérico para salvar estudo no Neon."""
    _npid = st.session_state.get("neon_project_id")
    if not _npid:
        st.caption("💡 Salve o projeto na Home para habilitar persistência dos estudos.")
        return
    if st.button(f"💾 Salvar Estudo no BD", key=f"save_study_{module_key}", use_container_width=True):
        sid = save_study_neon(_npid, module_key, module_label, input_data, result_data)
        if sid:
            st.success(f"✅ {module_label} salvo (ID {sid})")
        else:
            st.warning("⚠️ Neon indisponível")
 
def _plot_tower(geom):
    fig = go.Figure()
    cc = {1: BK_BLUE, 2: BK_TEAL, 3: BK_ORANGE, 4: BK_RED, 5: BK_GREEN}
    for c in geom.conductors:
        if c.is_shield:
            fig.add_trace(go.Scatter(x=[c.x_m], y=[c.y_m], mode="markers+text",
                marker=dict(size=12, color="#78909C", symbol="diamond", line=dict(width=2, color="white")),
                text=["GW"], textposition="top center", textfont=dict(size=10, color="#78909C"),
                name="GW", hovertemplate=f"<b>GW</b><br>{c.cable_key}<br>({c.x_m:.1f}, {c.y_m:.1f})m<extra></extra>"))
        else:
            color = cc.get(c.circuit_index, BK_BLUE)
            fig.add_trace(go.Scatter(x=[c.x_m], y=[c.y_m], mode="markers+text",
                marker=dict(size=14, color=color, symbol="circle", line=dict(width=2, color="white")),
                text=[c.name], textposition="top center", textfont=dict(size=11, color=color),
                name=c.name, hovertemplate=f"<b>{c.name}</b><br>{c.cable_key}<br>({c.x_m:.1f}, {c.y_m:.1f})m<br>Bundle {c.bundle_n}x{c.ds_bundle_m:.2f}m<extra></extra>"))
    xs = [c.x_m for c in geom.conductors]
    fig.add_trace(go.Scatter(x=[min(xs)-5, max(xs)+5], y=[0,0], mode="lines",
        line=dict(color="#8D6E63", width=3, dash="dot"), name="Solo", hoverinfo="skip"))
    fig.update_layout(**PLOTLY_LAYOUT, title="Secao Transversal da Torre", height=420,
        xaxis_title="Distancia Horizontal (m)", yaxis_title="Altura (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig
 
# === REPORT BUTTONS ===
def _report_buttons(module_key: str, results, cfg: dict):
    """Adiciona botoes de download e preview do relatorio Word."""
    if results is None:
        return
    st.divider()
    titulo = REPORT_TITLES.get(module_key, module_key)
    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"\U0001f4c4 Gerar Relatório Word", key=f"rpt_gen_{module_key}", type="secondary", use_container_width=True):
            try:
                cod = st.session_state.get("proj_number", "")
                data = generate_report(module_key, results, cfg, codigo_doc=cod, revisao="0A")
                fname = f"BK_{module_key}_{cod or 'report'}.docx".replace(" ", "_")
                st.session_state[f"rpt_data_{module_key}"] = data
                st.session_state[f"rpt_fname_{module_key}"] = fname
                st.success(f"Relatório '{titulo}' gerado!")
            except Exception as e:
                st.error(f"Erro ao gerar relatório: {e}")
    with c2:
        rpt_data = st.session_state.get(f"rpt_data_{module_key}")
        rpt_fname = st.session_state.get(f"rpt_fname_{module_key}", f"BK_{module_key}.docx")
        if rpt_data:
            st.download_button(
                f"\u2b07\ufe0f Download {titulo}",
                data=rpt_data,
                file_name=rpt_fname,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"rpt_dl_{module_key}",
                use_container_width=True,
            )
        else:
            st.caption("Clique em 'Gerar Relatório' primeiro")
 
 
# === SIDEBAR ===
PAGES = [
    "🏠 Home",
    "📐 Parametros Eletricos",
    "📊 Banco de Cabos",
    "⛈️ Corona",
    "🔋 Campos EM",
    "🌡️ Ampacidade & Flecha",
    "📻 RI e RA",
    "🛡️ Blindagem",
    "⚡ Isolamento Vmax",
    "🔌 Coord. Isolamento",
    "🔄 Religamento Tripolar",
    "📡 Compat. Eletromagnetica",
    "🔀 Fluxo de Potencia",
    
]
 
# ── Navegação via session_state (evita loop React do st.radio) ──────
if "active_page" not in st.session_state:
    st.session_state["active_page"] = PAGES[0]
if st.session_state["active_page"] not in PAGES:
    st.session_state["active_page"] = PAGES[0]
 
with st.sidebar:
    # ── Cabeçalho ──────────────────────────────────────────────────
    st.markdown(
        "<div style='padding:18px 16px 4px 16px;'>"
        "<div style='font-size:1.15rem;font-weight:700;color:#E0E0E0;'>⚡ BK Estudos Elétricos</div>"
        "<div style='font-size:0.76rem;color:#90CAF9;margin-top:2px;'>Linhas de Transmissão e Subestações v3.0</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<hr style='border:none;border-top:1px solid #2a3a4a;margin:8px 4px;'/>",
        unsafe_allow_html=True,
    )
 
    # ── Botões de navegação ────────────────────────────────────────
    # Injeta CSS uma vez para todos os botões nav da sidebar
    _active_idx = PAGES.index(st.session_state["active_page"]) + 1
    st.markdown(
        f"""<style>
        /* Reseta todos os botões da sidebar para estilo nav */
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
            background: transparent !important;
            border: none !important;
            border-left: 3px solid transparent !important;
            border-radius: 0 8px 8px 0 !important;
            box-shadow: none !important;
            color: #B0BEC5 !important;
            font-size: 0.88rem !important;
            font-weight: 400 !important;
            text-align: left !important;
            padding: 9px 14px !important;
            margin: 1px 0 !important;
            width: 100% !important;
            justify-content: flex-start !important;
            transition: background 0.15s, color 0.15s !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {{
            background: rgba(255,255,255,0.07) !important;
            border-left-color: #5C9BD6 !important;
            color: #E0E0E0 !important;
            box-shadow: none !important;
            transform: none !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus {{
            outline: none !important;
            box-shadow: none !important;
        }}
        /* Destaca botão ativo usando nth-of-type */
        section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type({_active_idx}) > button {{
            background: rgba(21,101,192,0.28) !important;
            border-left: 3px solid #42A5F5 !important;
            color: #E3F2FD !important;
            font-weight: 700 !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )
 
    for _pg in PAGES:
        _btn_type = "primary" if _pg == st.session_state["active_page"] else "secondary"
        if st.button(_pg, key=f"_nav_{_pg}", use_container_width=True, type=_btn_type):
            # Auto-salva parâmetros da Home ao navegar para outro estudo
            # garante que alterações não salvas explicitamente sejam persistidas
            if st.session_state.get("active_page") == PAGES[0]:
                _auto_save_home()
            st.session_state["active_page"] = _pg
            st.rerun()
 
    # ── Info do projeto ────────────────────────────────────────────
    st.markdown(
        "<hr style='border:none;border-top:1px solid #2a3a4a;margin:8px 4px;'/>",
        unsafe_allow_html=True,
    )
    _pn = st.session_state.get("proj_name")   or "—"
    _pc = st.session_state.get("client")      or "—"
    _pd = st.session_state.get("proj_number") or "—"
    st.markdown(
        f"<div style='padding:4px 16px 16px 16px;font-size:0.79rem;color:#90CAF9;line-height:1.9;'>"
        f"<b>Projeto ativo</b><br>"
        f"📌 {_pn}<br>🏢 {_pc}<br>🔢 {_pd}</div>",
        unsafe_allow_html=True,
    )
 
page = st.session_state["active_page"]
 
 
# ===================================================================
# PAGE: HOME
# ===================================================================
if page == PAGES[0]:
    bk_header("Dados do Sistema de Transmissao", "Configure parametros basicos e geometria da torre")
 
    # ── Gerenciar Projeto (salvar/carregar do Neon) ──────────
    bk_section("Projeto — Banco de Dados Neon")
    _db_c1, _db_c2, _db_c3 = st.columns([2, 2, 1])
    with _db_c1:
        if st.button("💾 Salvar Projeto no BD", type="primary", use_container_width=True):
            try:
                init_neon()
                _npid = st.session_state.get("neon_project_id")
                _new_id = upsert_project_neon(_build_save_dict(), _npid)
                if _new_id:
                    st.session_state["neon_project_id"] = _new_id
                    st.success(f"✅ Projeto salvo no Neon (ID {_new_id})")
                else:
                    # Fallback local
                    _lid = upsert_project(_build_save_dict())
                    st.warning(f"⚠️ Neon indisponível — salvo localmente (ID {_lid})")
            except Exception as _e:
                st.error(f"Erro ao salvar: {_e}")
    with _db_c2:
        try:
            init_neon()
            _neon_projs = list_projects_neon()
        except Exception:
            _neon_projs = []
        if _neon_projs:
            _proj_opts = {f"{p['name']} ({p.get('project_number','')}) [{p['id']}]": p['id'] for p in _neon_projs}
            _sel = st.selectbox("Carregar projeto", ["(nenhum)"] + list(_proj_opts.keys()), key="_load_proj_sel")
            if _sel != "(nenhum)":
                _sel_id = _proj_opts[_sel]
                if st.button("📂 Carregar", key="load_proj_btn"):
                    _pd = get_project_neon(_sel_id)
                    if _pd:
                        _populate_from_project(_pd)
                        st.session_state["neon_project_id"] = _sel_id
                        st.success(f"✅ Projeto '{_pd.get('name')}' carregado — {st.session_state.voltage_kv:.0f} kV")
                        st.rerun()
        else:
            st.caption("Nenhum projeto salvo no Neon")
    with _db_c3:
        _npid = st.session_state.get("neon_project_id")
        if _npid:
            st.caption(f"📌 ID Neon: **{_npid}**")
            try:
                _studies = list_studies_neon(_npid)
                if _studies:
                    st.caption(f"📊 {len(_studies)} estudo(s) salvo(s)")
            except Exception:
                pass
        # Botão Excluir sempre visível quando há projetos no BD
        # (independente de qual projeto está ativo)
        if st.session_state.get("_confirm_delete"):
            _del_target = st.session_state.get("_del_target_id", _npid)
            st.warning(f"⚠️ Excluir projeto ID {_del_target}?")
            _cy, _cn = st.columns(2)
            with _cy:
                if st.button("✅ Sim", key="_del_yes", use_container_width=True):
                    try:
                        delete_project_neon(_del_target)
                    except Exception:
                        pass
                    if st.session_state.get("neon_project_id") == _del_target:
                        st.session_state["neon_project_id"] = None
                        # Limpa _proj para não referenciar projeto excluído
                        st.session_state["_proj"] = _PROJ_DEFAULTS.copy()
                    st.session_state["_confirm_delete"] = False
                    st.session_state["_del_target_id"] = None
                    st.rerun()
            with _cn:
                if st.button("❌ Nao", key="_del_no", use_container_width=True):
                    st.session_state["_confirm_delete"] = False
                    st.session_state["_del_target_id"] = None
                    st.rerun()
        else:
            # Mostra botão excluir se há projetos salvos OU projeto ativo
            _has_projects = bool(_neon_projs) if '_neon_projs' in dir() else bool(_npid)
            if _has_projects or _npid:
                # Se há projeto ativo, exclui ele direto; senão, usa seleção
                _del_id = _npid if _npid else (
                    _proj_opts.get(_sel) if '_sel' in dir() and _sel != "(nenhum)"
                    and '_proj_opts' in dir() else None)
                if _del_id:
                    if st.button("🗑️ Excluir", key="_del_proj", use_container_width=True,
                                 help="Excluir projeto ativo do banco de dados"):
                        st.session_state["_confirm_delete"] = True
                        st.session_state["_del_target_id"] = _del_id
                        st.rerun()
 
    bk_section("Identificacao do Projeto")
    p1, p2, p3 = st.columns(3)
    with p1: st.text_input("Nome do Projeto", key="proj_name", help="Nome ou titulo do estudo eletrico")
    with p2: st.text_input("Cliente", key="client", help="Nome do cliente ou contratante")
    with p3: st.text_input("No do Documento", key="proj_number", help="Codigo/numero do documento BK (ex: BK-EE-001-R0)")
    bk_section("Dados Eletricos Basicos")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.number_input("Tensao Nominal (kV L-L)", key="voltage_kv", min_value=1.0, step=1.0, format="%.1f", help="Tensao nominal L-L. Ex: 69, 138, 230, 500 kV")
    with c2: st.number_input("Potencia Total (MVA)", key="power_mva", min_value=0.1, step=10.0, format="%.1f", help="Potencia aparente total do sistema")
    with c3: st.number_input("Frequencia (Hz)", key="freq_hz", min_value=50.0, max_value=60.0, step=10.0, format="%.0f", help="60 Hz (Brasil) ou 50 Hz (Europa)")
    with c4: st.number_input("Fator de Potencia", key="pf_load", min_value=-1.0, max_value=1.0, step=0.01, format="%.2f", help="FP carga. Negativo=capacitivo. Ref: NBR 5422")
    c5, c6 = st.columns(2)
    with c5: st.number_input("Altitude (m)", key="altitude_m", min_value=0.0, step=100.0, format="%.0f", help="Afeta corona e isolamento (IEC 60071-2)")
    with c6: st.number_input("No Linhas (<=4)", key="n_lines", min_value=1, max_value=4, step=1, help="Linhas paralelas na faixa de servidao")
 
    bk_section("Topologia da Linha")
    t1, t2, t3 = st.columns(3)
    with t1: st.number_input("No Circuitos (<=5)", key="n_circuits", min_value=1, max_value=5, step=1, help="Circuitos trifasicos na mesma torre")
    with t2: st.selectbox("Geometria", ["horizontal", "vertical", "triangular"], key="geometry_type", help="Disposicao das fases: H, V ou delta")
    with t3: st.selectbox("Layout Circuitos", ["side", "stacked"], key="circuits_layout", help="side=lado a lado; stacked=empilhado")
 
    bk_section("Selecao de Cabos")
    ck = _cable_keys()
    cb1, cb2 = st.columns(2)
    with cb1:
        idx_p = ck.index(st.session_state.cable_phase_key) if st.session_state.cable_phase_key in ck else 0
        st.selectbox("Cabo de Fase", ck, index=idx_p, key="cable_phase_key", help="Edite na aba Banco de Cabos")
    with cb2:
        idx_s = ck.index(st.session_state.cable_shield_key) if st.session_state.cable_shield_key in ck else 0
        st.selectbox("Cabo-Guarda (GW)", ck, index=idx_s, key="cable_shield_key", help="Cabo para-raios/OPGW (NBR 5422 par.6)")
 
    # ── Configuração de cabos-guarda (1 ou 2) ──────────────────────────
    gw1, gw2 = st.columns(2)
    with gw1:
        st.number_input("Nº de cabos-guarda", key="n_shield_wires", min_value=1, max_value=2, step=1,
            help="1 GW: centralizado. 2 GW: simétricos ±d do eixo (NBR 5422 / IEEE 1243)")
    with gw2:
        if st.session_state.get("n_shield_wires", 1) == 2:
            st.number_input("Distância horizontal GW ao eixo (m)", key="shield_dx_m", min_value=0.5, step=0.5, format="%.1f",
                help="Distância horizontal de 1 GW ao centro das fases. O 2º GW fica no lado oposto à mesma distância.")
        else:
            st.number_input("Deslocamento horizontal GW (m)", key="shield_dx_m", step=0.5, format="%.1f",
                help="Deslocamento do GW único em relação ao centro das fases (0 = centralizado)")
 
    bk_section("Feixe de Subcondutores (Bundle)")
    b1, b2, b3 = st.columns(3)
    with b1: st.number_input("Subcondutores (n)", key="bundle_n", min_value=1, max_value=6, step=1, help="1=sem feixe; 2-4 para 230-765kV")
    with b2: st.number_input("Espacamento bundle (m)", key="bundle_ds", min_value=0.1, step=0.05, format="%.2f", help="Tipico: 0.30-0.45 m")
    with b3: st.number_input("Espac. vertical fases (m)", key="phase_vert_spacing", min_value=0.5, step=0.5, format="%.1f", help="Entre fases A-B e B-C (NBR 5422)")
 
    bk_section("Geometria da Torre")
    g1, g2 = st.columns(2)
    with g1:
        st.number_input("dx B rel A (m)", key="dx_B", step=0.5, format="%.1f", help="Deslocamento horizontal fase B")
        st.number_input("dx C rel A (m)", key="dx_C", step=0.5, format="%.1f", help="Deslocamento horizontal fase C")
        st.number_input("Espac. entre circuitos (m)", key="circuit_spacing", min_value=1.0, step=1.0, format="%.1f", help="H (side) ou V (stacked)")
    with g2:
        st.number_input("Altura fase A C1 (m)", key="h_phase_ref", min_value=5.0, step=0.5, format="%.1f", help="NBR 5422 Tab.4: min 7-8m")
        st.number_input("Altura fase mais baixa (m)", key="h_min_phase", min_value=3.0, step=0.5, format="%.1f", help="Verificar NBR 5422")
        st.number_input("Altura cabo-guarda (m)", key="h_shield", min_value=5.0, step=0.5, format="%.1f", help="Angulo blindagem <= 30deg (IEEE 1243)")
 
    bk_section("Visualizacao da Torre")
    try:
        geom = _get_geom()
        st.plotly_chart(_plot_tower(geom), use_container_width=True)
        cd = [{"Nome": c.name, "Cabo": c.cable_key, "x(m)": f"{c.x_m:.2f}", "y(m)": f"{c.y_m:.2f}",
               "Circ": c.circuit_index, "Fase": c.phase or "GW", "Bundle": f"{c.bundle_n}x{c.ds_bundle_m:.2f}m"}
              for c in geom.conductors]
        st.dataframe(pd.DataFrame(cd), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Erro geometria: {e}")
    cable = _find_cable(st.session_state.cable_phase_key)
    if cable:
        bk_kpi_row([("Tensao", f"{st.session_state.voltage_kv:.0f} kV", "blue"),
            ("Potencia", f"{st.session_state.power_mva:.0f} MVA", "teal"),
            ("Cabo Fase", cable.key, "green"),
            ("Circuitos", str(st.session_state.n_circuits), "orange"),
            ("Bundle", f"{st.session_state.bundle_n}x", "gray")])
 
    # ── CRÍTICO: sincroniza widgets → _proj ao final de cada render da Home ──
    # _proj persiste mesmo quando o Streamlit limpa as chaves de widget ao
    # navegar para outros estudos. Sem isso, os estudos usariam defaults.
    _sync_proj()
 
 
# ===================================================================
# PAGE: PARAMETROS ELETRICOS
# ===================================================================
elif page == PAGES[1]:
    bk_header("Parametros Eletricos da Linha", "R X B L C por km - Modelo pi - Zc - SIL")
    bk_section("Configuracao do Calculo")
    p1, p2 = st.columns(2)
    with p1:
        st.number_input("Comprimento (km)", key="line_length_km", min_value=0.1, step=10.0, format="%.1f", help="Para L>250km considerar pi equivalente (Stevenson par.5)")
        st.number_input("Temp. condutor (C)", key="temp_C", min_value=-10.0, max_value=150.0, step=5.0, format="%.0f", help="Afeta R_ac. Tipico: 50-75C (IEEE 738)")
    with p2:
        st.number_input("ang Vs (deg)", key="Vs_ang", step=0.5, format="%.1f", help="Angulo tensao envio (ref)")
        st.caption(f"⚡ Tensão de envio (Vs) = **{st.session_state.voltage_kv:.1f} kV** (definida na Home)")
        st.caption("📐 **Vr (tensão de chegada)** e **ângulo** serão **calculados** pelo modelo π nominal com base em S, fp, Z e Y da linha.")
    if st.button("Calcular Parametros", type="primary", use_container_width=True):
        st.session_state.results = None
        st.session_state["computed_vr"] = None
        st.session_state["vr_errors"] = None
        rd = _safe(_compute_params)
        if rd: st.success(f"OK {len(rd)} circuito(s)")
    results = st.session_state.results
    if results:
        # Garante que computed_vr está populado (solve_vr_pi)
        _safe(_compute_params)
        _cvr = st.session_state.get("computed_vr") or {}
        _vr_errs = st.session_state.get("vr_errors") or []
 
        r0 = results[0]
        # KPIs de impedância + Vr principal (se disponível)
        kpis = [("R (Ω/km)", f"{r0.R_ohm_km:.4f}", "blue"), ("X (Ω/km)", f"{r0.X_ohm_km:.4f}", "teal"),
            ("Zc (Ω)", f"{r0.Zc_ohm:.1f}", "green"), ("SIL (MW)", f"{r0.SIL_MW:.1f}", "orange"), ("GMD (m)", f"{r0.GMD_m:.2f}", "gray")]
        if _cvr:
            vr0 = list(_cvr.values())[0]
            kpis.append(("Vr (kV)", f"{vr0[0]:.2f}", "red"))
        bk_kpi_row(kpis)
 
        # Tabela principal — parâmetros + Vr
        rows = []
        for r in results:
            row = {"Circ": r.circuit_index, "R (Ω/km)": f"{r.R_ohm_km:.6f}", "X (Ω/km)": f"{r.X_ohm_km:.6f}",
                 "B (S/km)": f"{r.B_S_km:.6e}", "L (mH/km)": f"{r.L_mH_km:.4f}", "C (nF/km)": f"{r.C_nF_km:.4f}",
                 "Zc (Ω)": f"{r.Zc_ohm:.2f}", "SIL (MW)": f"{r.SIL_MW:.2f}", "Ec (kV/cm)": f"{r.Ec_kV_cm:.4f}"}
            vr_data = _cvr.get(r.circuit_index)
            if vr_data:
                vr_kv, vr_ang, i_a, ploss = vr_data
                row["Vr (kV)"] = f"{vr_kv:.3f}"
                row["∠Vr (°)"] = f"{vr_ang:.2f}"
                row["I (A)"] = f"{i_a:.1f}"
                vs_kv = st.session_state.voltage_kv
                reg = ((vs_kv - vr_kv) / vr_kv * 100.0) if vr_kv > 0 else 0
                row["Reg. (%)"] = f"{reg:.2f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
 
        # ── Seção detalhada de Vr ──────────────
        if _cvr:
            bk_section("Tensão de Chegada (Vr) — Calculada via Modelo π Nominal")
            st.caption("Ref.: Stevenson — Elements of Power System Analysis, Cap. 5; Zanetta — Fundamentos de SEP, Cap. 3")
            vr_rows = []
            for cidx, (vr_kv, vr_ang, i_a, ploss) in sorted(_cvr.items()):
                vs_kv = st.session_state.voltage_kv
                reg_pct = ((vs_kv - vr_kv) / vr_kv * 100.0) if vr_kv > 0 else 0.0
                vr_rows.append({
                    "Circuito": cidx,
                    "Vs (kV)": f"{vs_kv:.2f}",
                    "∠Vs (°)": f"{st.session_state.Vs_ang:.2f}",
                    "Vr (kV)": f"{vr_kv:.3f}",
                    "∠Vr (°)": f"{vr_ang:.2f}",
                    "I (A)": f"{i_a:.1f}",
                    "P_perdas (MW)": f"{ploss:.3f}",
                    "Regulação (%)": f"{reg_pct:.2f}",
                })
            st.dataframe(pd.DataFrame(vr_rows), use_container_width=True, hide_index=True)
            vr0 = list(_cvr.values())[0]
            reg0 = ((st.session_state.voltage_kv - vr0[0]) / vr0[0] * 100.0) if vr0[0] > 0 else 0
            bk_kpi_row([
                ("Vr", f"{vr0[0]:.2f} kV", "blue"),
                ("∠Vr", f"{vr0[1]:.2f}°", "teal"),
                ("I série", f"{vr0[2]:.1f} A", "green"),
                ("Perdas", f"{vr0[3]:.3f} MW", "orange"),
                ("Regulação", f"{reg0:.2f}%", "red" if abs(reg0) > 5 else "gray"),
            ])
        else:
            st.warning("⚠️ Vr não pôde ser calculado. Verifique S (MVA), fp e comprimento da linha.")
            if _vr_errs:
                with st.expander("Detalhes do erro"):
                    for e in _vr_errs:
                        st.code(e)
 
        tab_g, tab_pi = st.tabs(["Torre", "Modelo π"])
        with tab_g: st.plotly_chart(_plot_tower(_get_geom()), use_container_width=True)
        with tab_pi:
            for r in results:
                L = st.session_state.line_length_km
                Z = complex(r.R_ohm_km*L, r.X_ohm_km*L); Y = complex(0, r.B_S_km*L)
                st.markdown(f"**Circuito {r.circuit_index} — L={L}km**")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Z série (Ω)", f"{Z.real:.2f}+j{Z.imag:.2f}")
                mc2.metric("Y/2 shunt (S)", f"j{Y.imag/2:.6f}")
                mc3.metric("|Z| (Ω)", f"{abs(Z):.2f}")
                # Vr no modelo pi
                vr_data = _cvr.get(r.circuit_index)
                if vr_data:
                    mv1, mv2, mv3, mv4 = st.columns(4)
                    mv1.metric("Vs (kV)", f"{st.session_state.voltage_kv:.2f}")
                    mv2.metric("Vr (kV)", f"{vr_data[0]:.3f}")
                    mv3.metric("∠Vr (°)", f"{vr_data[1]:.2f}")
                    mv4.metric("I (A)", f"{vr_data[2]:.1f}")
        for r in results:
            with st.expander(f"Detalhes Circuito {r.circuit_index}"):
                d1, d2, d3 = st.columns(3)
                with d1: st.write(f"R={r.R_ohm_km:.6f} X={r.X_ohm_km:.6f} ohm/km")
                with d2: st.write(f"L={r.L_mH_km:.4f} mH/km C={r.C_nF_km:.4f} nF/km")
                with d3: st.write(f"Zc={r.Zc_ohm:.2f} SIL={r.SIL_MW:.2f} Ec={r.Ec_kV_cm:.4f} lambda={r.lambda_m:.0f}m")
 
        # Salva estudo no Neon
        _input_d = {"voltage_kv": st.session_state.voltage_kv, "line_length_km": st.session_state.line_length_km,
                     "temp_C": st.session_state.temp_C, "freq_hz": st.session_state.freq_hz}
        _result_d = {"circuits": [{"circ": r.circuit_index, "R": r.R_ohm_km, "X": r.X_ohm_km,
                     "Zc": r.Zc_ohm, "SIL": r.SIL_MW} for r in results]}
        if _cvr:
            _result_d["vr_computed"] = {str(k): {"Vr_kV": v[0], "ang": v[1], "I_A": v[2], "Ploss": v[3]}
                                         for k, v in _cvr.items()}
        _save_study_btn("params", "Parâmetros Elétricos", _input_d, _result_d)
 
        _report_btns("params", extract_func=rb.extract_params, result_key="results")
    else: st.info("Configure dados na Home e clique Calcular.")
 
# ===================================================================
# PAGE: BANCO DE CABOS
# ===================================================================
elif page == PAGES[2]:
    bk_header("Banco de Cabos Condutores", "Edite na tabela - Adicione novos - Dados catalogo")
    bk_section("Tabela de Cabos - Edicao tipo Excel")
    st.caption("Clique em qualquer celula para editar. Use + para adicionar.")
    col_cfg = {
        "Codigo": st.column_config.TextColumn("Codigo", help="ID unico do cabo (ex: ACSR_477)", width="medium"),
        "Material": st.column_config.SelectboxColumn("Material", help="Cu, Al, ACSR, Steel", options=["Cu", "Al", "ACSR", "Steel"], width="small"),
        "Area_kcmil": st.column_config.NumberColumn("Area (kcmil)", help="Secao transversal. 1 kcmil=0.5067mm2", format="%.1f", min_value=0.1),
        "Diametro_mm": st.column_config.NumberColumn("Diam (mm)", help="Diametro externo total", format="%.2f", min_value=0.1),
        "GMR_mm": st.column_config.NumberColumn("GMR (mm)", help="Raio Medio Geometrico. GMR~0.7788*r p/ solido (Stevenson par.4.5)", format="%.2f", min_value=0.01),
        "Rdc_ohm_km": st.column_config.NumberColumn("Rdc 20C (ohm/km)", help="Resistencia DC 20C. Corrigida p/ T via alfa", format="%.6f", min_value=0.0),
        "eps_r": st.column_config.NumberColumn("eps_r", help="1.0=nu, 2.3-3.5=XLPE/EPR", format="%.1f", min_value=1.0),
        "Notas": st.column_config.TextColumn("Notas", help="Obs: fabricante, stranding", width="large"),
    }
    edited = st.data_editor(st.session_state.cable_db, column_config=col_cfg, num_rows="dynamic", use_container_width=True, hide_index=True, key="cable_ed")
    st.session_state.cable_db = edited
    bk_section("Detalhes do Cabo Selecionado")
    sel = st.selectbox("Cabo", _cable_keys(), help="Selecione para ver detalhes")
    cable = _find_cable(sel)
    if cable:
        s = st.session_state
        R_ac = cable.ac_resistance_per_m(s.freq_hz, s.temp_C) * 1000
        GMR_eq, r_eq = cable.bundle_equivalents(s.bundle_n, s.bundle_ds)
        st.write(f"**{cable.key}** {cable.material} | diam={cable.diameter_mm:.2f}mm | GMR={cable.gmr_mm:.2f}mm | Rdc={cable.rdc_ohm_km_20C:.6f} ohm/km")
        st.write(f"R_ac({s.temp_C:.0f}C,{s.freq_hz:.0f}Hz)={R_ac:.6f} ohm/km | Bundle {s.bundle_n}x -> GMR_eq={GMR_eq*1000:.3f}mm")
    bk_kpi_row([("Total", str(len(edited)), "blue"), ("Materiais", str(edited["Material"].nunique()), "teal"),
        ("Area Min", f"{edited['Area_kcmil'].min():.0f}", "green"), ("Area Max", f"{edited['Area_kcmil'].max():.0f}", "orange")])
 
 
# ===================================================================
# PAGE: CORONA
# ===================================================================
elif page == PAGES[3]:
    bk_header("Estudo de Corona", "Tensao critica de Peek - Perdas - Campo superficial")
    bk_section("Configuracao")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        corona_temp = st.number_input("T ambiente (C)", value=25.0, step=5.0, help="Temperatura do ar para calculo de delta (densidade)")
        corona_press = st.number_input("Pressao (kPa)", value=101.3, step=1.0, help="Pressao barometrica. Se 0 estima pela altitude (IEC 60071-2)")
    with cc2:
        corona_weather = st.selectbox("Condicao condutor", ["normal", "brilhante", "limpo", "rugoso/envelhecido"], help="Fator m0 Peek: brilhante=1.0, normal=0.93, rugoso=0.85")
        corona_k = st.number_input("Fator k sobretensao", value=1.1, step=0.1, format="%.2f", help="Fator IEC sobretensao temporaria (1.0-1.5)")
    with cc3:
        corona_length = st.number_input("Comprimento (km)", value=st.session_state.line_length_km, step=10.0, help="Comprimento para perda total")
    if st.button("Calcular Corona", type="primary", use_container_width=True):
        def _run_corona():
            s = st.session_state; geom = _get_geom(); cfgs = {}
            for cidx in range(1, s.n_circuits+1):
                cfgs[cidx] = CoronaConfig(circuit_index=cidx, V_LL_kV=s.voltage_kv, f_hz=s.freq_hz,
                    temp_C=corona_temp, pressure_kPa=corona_press, altitude_m=s.altitude_m,
                    weather=corona_weather, k_factor=corona_k)
            return compute_corona_all_circuits(geom, cfgs, _get_cables(), length_km=corona_length)
        cr = _safe(_run_corona)
        if cr: st.session_state.corona_results = cr; st.success("OK Corona")
    cr = st.session_state.get("corona_results")
    if cr:
        for cidx, res in cr.items():
            bk_section(f"Circuito {cidx}")
            bk_kpi_row([("Vd critica (kV)", f"{res.Vd_LL_kV:.1f}", "blue"),
                ("Ec crit (kV/cm)", f"{res.Ec_crit_kV_cm:.2f}", "teal"),
                ("Ec superf (kV/cm)", f"{res.Esurface_kV_cm:.2f}", "green" if res.corona_ok else "red"),
                ("Perda (kW/km/fase)", f"{res.corona_loss_kW_km_phase:.3f}", "orange"),
                ("Status", "OK" if res.corona_ok else "CORONA", "green" if res.corona_ok else "red")])
            with st.expander(f"Detalhes C{cidx}"):
                st.write(f"V fase={res.V_phase_kV:.2f}kV | delta={res.delta_air:.4f} | m0={res.m0:.2f} | r_eq={res.r_eq_cm:.4f}cm")
                st.write(f"GMD={res.GMD_m:.4f}m | Margem Vd={res.margin_Vd_percent:.1f}% | Ve surto={res.Ve_surge_LL_kV:.1f}kV")
                st.info(res.corona_message)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Ec superficie", "Ec critico"], y=[res.Esurface_kV_cm, res.Ec_crit_kV_cm],
                marker_color=[BK_RED if not res.corona_ok else BK_GREEN, BK_BLUE],
                hovertemplate="<b>%{x}</b><br>%{y:.3f} kV/cm<extra></extra>"))
            fig.update_layout(**PLOTLY_LAYOUT, title="Campo Superficial vs Critico", height=350, yaxis_title="kV/cm")
            st.plotly_chart(fig, use_container_width=True)
        _report_btns("corona", extract_func=rb.extract_corona, result_key="corona_results")
    else: st.info("Configure e clique Calcular Corona.")
 
# ===================================================================
# PAGE: CAMPOS EM
# ===================================================================
elif page == PAGES[4]:
    bk_header("Campos Eletrico e Magnetico", "Perfis laterais |E| e |B| — Limites ANEEL RN 915/2021")
    if not HAS_FIELDS:
        st.warning("Modulo field_em nao disponivel.")
    else:
        bk_section("Configuracao — ANEEL RN 915/2021")
        st.caption(
            "Avaliação conforme **ANEEL Resolução Normativa nº 915/2021** "
            "(revoga nº 616/2014). Altura de avaliação: **1,5 m**. "
            "Distância lateral mínima recomendada: **±30 m** a partir do eixo da linha. "
            "Campo E: MSC com Método das Imagens. Campo B: Imagens Complexas de Deri."
        )
 
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            f_hobs = st.number_input("Altura obs (m)", value=1.5, step=0.5,
                help="1,5 m = tronco humano. ANEEL RN 915/2021, art. 4º")
            f_xmin = st.number_input("x min (m)", value=-30.0, step=5.0,
                help="Limite esquerdo. Mínimo recomendado: -30 m")
        with fc2:
            f_xmax = st.number_input("x max (m)", value=30.0, step=5.0,
                help="Limite direito. Mínimo recomendado: +30 m")
            f_npts = st.number_input("Nº pontos", value=301, min_value=10, step=50,
                help="Resolução do perfil lateral")
        with fc3:
            f_Elim_g = st.number_input("Lim. E público geral (kV/m)", value=4.17, step=0.1, format="%.2f",
                help="ANEEL RN 915/2021: 4,17 kV/m (público geral)")
            f_Blim_g = st.number_input("Lim. B público geral (µT)", value=200.0, step=10.0,
                help="ANEEL RN 915/2021: 200 µT (público geral)")
        with fc4:
            f_Elim_o = st.number_input("Lim. E ocupacional (kV/m)", value=8.33, step=0.1, format="%.2f",
                help="ANEEL RN 915/2021: 8,33 kV/m (acesso restrito)")
            f_Blim_o = st.number_input("Lim. B ocupacional (µT)", value=1000.0, step=50.0,
                help="ANEEL RN 915/2021: 1000 µT (acesso restrito)")
 
        bk_section("Dados do Solo (Campo B — Método de Deri)")
        sd1, sd2 = st.columns(2)
        with sd1:
            f_rho = st.number_input("Resistividade do solo ρ (Ω·m)", value=100.0, step=10.0,
                help="Típico: 100 Ω·m. Para ρ ≥ 50 Ω·m o resultado é equivalente ao solo perfeito (Vieira, UFSJ 2013)")
        with sd2:
            f_Ic = st.number_input("Corrente manual por circuito (A) — 0 = automático", value=0.0, step=50.0,
                help="Se 0, calcula automaticamente por S e V")
 
        if st.button("⚡ Calcular Campos", type="primary", use_container_width=True):
            def _run_fields():
                s   = st.session_state
                pd2 = _compute_params()
                cfg = FieldConfig(
                    h_obs_m   = f_hobs,
                    x_min_m   = f_xmin,
                    x_max_m   = f_xmax,
                    n_points  = int(f_npts),
                    Ic_manual = float(f_Ic) if f_Ic > 0 else None,
                    rho_solo  = f_rho,
                    freq_hz   = float(s.get("freq_hz", 60.0)),
                )
                lim = AneelLimits(
                    E_max_kV_m_geral = f_Elim_g,
                    B_max_uT_geral   = f_Blim_g,
                    E_max_kV_m_ocup  = f_Elim_o,
                    B_max_uT_ocup    = f_Blim_o,
                )
                return compute_fields_profiles(_get_geom(), pd2, cfg, lim, s.voltage_kv, s.power_mva)
            fr = _safe(_run_fields)
            if fr:
                st.session_state.fields_result = fr
                st.success("✅ Campos calculados com sucesso!")
 
        fr = st.session_state.get("fields_result")
        if fr:
            lim = fr.limits
            # KPIs
            bk_kpi_row([
                ("|E| máx (kV/m)",   f"{fr.E_max_kV_m:.4f}",        "blue"),
                ("x E_máx (m)",      f"{fr.x_E_max_m:.1f}",         "teal"),
                ("|B| máx (µT)",     f"{fr.B_max_uT:.4f}",          "green"),
                ("x B_máx (m)",      f"{fr.x_B_max_m:.1f}",         "orange"),
                ("h_obs (m)",        f"{fr.config.h_obs_m:.1f}",    "gray"),
                ("ρ_solo (Ω·m)",     f"{fr.config.rho_solo:.0f}",   "gray"),
            ])
 
            # Status de conformidade — dois limites
            col_e, col_b = st.columns(2)
            with col_e:
                ok_e_g = fr.E_max_kV_m <= lim.E_max_kV_m_geral
                ok_e_o = fr.E_max_kV_m <= lim.E_max_kV_m_ocup
                st.metric("Campo E — Público Geral",
                          f"{fr.E_max_kV_m:.3f} kV/m",
                          delta=f"{'✅ ATENDE' if ok_e_g else '❌ EXCEDE'} lim. {lim.E_max_kV_m_geral:.2f} kV/m",
                          delta_color="normal" if ok_e_g else "inverse")
                st.metric("Campo E — Ocupacional",
                          f"{fr.E_max_kV_m:.3f} kV/m",
                          delta=f"{'✅ ATENDE' if ok_e_o else '❌ EXCEDE'} lim. {lim.E_max_kV_m_ocup:.2f} kV/m",
                          delta_color="normal" if ok_e_o else "inverse")
            with col_b:
                ok_b_g = fr.B_max_uT <= lim.B_max_uT_geral
                ok_b_o = fr.B_max_uT <= lim.B_max_uT_ocup
                st.metric("Campo B — Público Geral",
                          f"{fr.B_max_uT:.3f} µT",
                          delta=f"{'✅ ATENDE' if ok_b_g else '❌ EXCEDE'} lim. {lim.B_max_uT_geral:.0f} µT",
                          delta_color="normal" if ok_b_g else "inverse")
                st.metric("Campo B — Ocupacional",
                          f"{fr.B_max_uT:.3f} µT",
                          delta=f"{'✅ ATENDE' if ok_b_o else '❌ EXCEDE'} lim. {lim.B_max_uT_ocup:.0f} µT",
                          delta_color="normal" if ok_b_o else "inverse")
 
            # ── Abas: 2D (perfis) + 3D interativo ──────────────────────
            tab_2d_e, tab_2d_b, tab_3d_e, tab_3d_b = st.tabs([
                "📈 Perfil E (2D)", "📈 Perfil B (2D)",
                "🌐 Mapa 3D |E|",   "🌐 Mapa 3D |B|",
            ])
 
            with tab_2d_e:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fr.x_m, y=fr.E_kV_m, mode="lines",
                    fill="tozeroy", fillcolor="rgba(21,101,192,0.12)",
                    name="|E|", line=dict(color=BK_BLUE, width=2.5),
                    hovertemplate="x=%{x:.1f} m<br>|E|=%{y:.4f} kV/m<extra></extra>"))
                # Limites — só mostra se dentro da faixa visível (evita reta dominando)
                _e_ymax = max(fr.E_kV_m) * 1.35 if max(fr.E_kV_m) > 0 else 1.0
                if lim.E_max_kV_m_geral <= _e_ymax * 2:
                    fig.add_hline(y=lim.E_max_kV_m_geral, line_dash="dash", line_color=BK_RED,
                        annotation_text=f"Lim. geral {lim.E_max_kV_m_geral:.2f} kV/m")
                if lim.E_max_kV_m_ocup <= _e_ymax * 2:
                    fig.add_hline(y=lim.E_max_kV_m_ocup, line_dash="dot", line_color=BK_ORANGE,
                        annotation_text=f"Lim. ocup. {lim.E_max_kV_m_ocup:.2f} kV/m")
                fig.add_vline(x=fr.x_E_max_m, line_dash="dashdot", line_color="#555",
                    annotation_text=f"x_Emáx={fr.x_E_max_m:.1f}m")
                # Marcador no ponto de máximo
                fig.add_trace(go.Scatter(x=[fr.x_E_max_m], y=[fr.E_max_kV_m], mode="markers+text",
                    marker=dict(size=10, color=BK_RED, symbol="diamond"),
                    text=[f"{fr.E_max_kV_m:.4f} kV/m"], textposition="top center",
                    textfont=dict(size=10, color=BK_RED), name="Emáx", showlegend=False))
                fig.update_layout(**PLOTLY_LAYOUT, title="Perfil Campo Elétrico — ANEEL RN 915/2021",
                    xaxis_title="Distância lateral (m)", yaxis_title="|E| (kV/m)", height=450,
                    yaxis=dict(rangemode="tozero", range=[0, _e_ymax]))
                st.plotly_chart(fig, use_container_width=True)
                st.info(fr.E_compliance_msg)
 
            with tab_2d_b:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fr.x_m, y=fr.B_uT, mode="lines",
                    fill="tozeroy", fillcolor="rgba(0,137,123,0.12)",
                    name="|B|", line=dict(color=BK_TEAL, width=2.5),
                    hovertemplate="x=%{x:.1f} m<br>|B|=%{y:.4f} µT<extra></extra>"))
                _b_ymax = max(fr.B_uT) * 1.35 if max(fr.B_uT) > 0 else 1.0
                if lim.B_max_uT_geral <= _b_ymax * 2:
                    fig.add_hline(y=lim.B_max_uT_geral, line_dash="dash", line_color=BK_RED,
                        annotation_text=f"Lim. geral {lim.B_max_uT_geral:.0f} µT")
                if lim.B_max_uT_ocup <= _b_ymax * 2:
                    fig.add_hline(y=lim.B_max_uT_ocup, line_dash="dot", line_color=BK_ORANGE,
                        annotation_text=f"Lim. ocup. {lim.B_max_uT_ocup:.0f} µT")
                fig.add_vline(x=fr.x_B_max_m, line_dash="dashdot", line_color="#555",
                    annotation_text=f"x_Bmáx={fr.x_B_max_m:.1f}m")
                fig.add_trace(go.Scatter(x=[fr.x_B_max_m], y=[fr.B_max_uT], mode="markers+text",
                    marker=dict(size=10, color=BK_RED, symbol="diamond"),
                    text=[f"{fr.B_max_uT:.4f} µT"], textposition="top center",
                    textfont=dict(size=10, color=BK_RED), name="Bmáx", showlegend=False))
                fig.update_layout(**PLOTLY_LAYOUT, title="Perfil Campo Magnético — ANEEL RN 915/2021",
                    xaxis_title="Distância lateral (m)", yaxis_title="|B| (µT)", height=450,
                    yaxis=dict(rangemode="tozero", range=[0, _b_ymax]))
                st.plotly_chart(fig, use_container_width=True)
                st.info(fr.B_compliance_msg)
 
            with tab_3d_e:
                st.caption("Superfície 3D interativa: |E|(x, z) — arraste para rotacionar, zoom com scroll.")
                with st.spinner("Calculando superfície 3D do campo elétrico..."):
                    try:
                        from core.field_em import compute_plotly_surface_data
                        import numpy as np
                        pd2_3d = _compute_params()
                        x3, z3, E3d, _ = compute_plotly_surface_data(_get_geom(), pd2_3d, fr, n_heights=25)
                        if x3 and z3 and E3d:
                            # Clip outliers perto dos condutores
                            E_arr = np.array(E3d, dtype=float)
                            p98 = np.percentile(E_arr[E_arr > 0], 98) if np.any(E_arr > 0) else 1.0
                            E_arr = np.clip(E_arr, 0, p98)
                            fig3d = go.Figure(data=[go.Surface(
                                x=x3, y=z3, z=E_arr.tolist(),
                                colorscale="Plasma", showscale=True,
                                colorbar=dict(title="|E| (kV/m)", thickness=18),
                                hovertemplate="x=%{x:.1f}m<br>z=%{y:.1f}m<br>|E|=%{z:.3f}kV/m<extra></extra>",
                                contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                            )])
                            fig3d.update_layout(
                                title="Mapa 3D — Campo Elétrico |E|(x, z)",
                                scene=dict(
                                    xaxis_title="Dist. lateral x (m)",
                                    yaxis_title="Altura z (m)",
                                    zaxis_title="|E| (kV/m)",
                                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                                    aspectratio=dict(x=1.5, y=1, z=0.7),
                                ),
                                height=600,
                                margin=dict(l=0, r=0, t=50, b=0),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig3d, use_container_width=True)
                        else:
                            st.warning("Não foi possível gerar a grade 3D para campo elétrico.")
                    except Exception as _e3d:
                        st.error(f"Erro ao gerar mapa 3D |E|: {_e3d}")
 
            with tab_3d_b:
                st.caption("Superfície 3D interativa: |B|(x, z) — arraste para rotacionar, zoom com scroll.")
                with st.spinner("Calculando superfície 3D do campo magnético..."):
                    try:
                        from core.field_em import compute_plotly_surface_data
                        import numpy as np
                        pd2_3d = _compute_params()
                        x3, z3, _, B3d = compute_plotly_surface_data(_get_geom(), pd2_3d, fr, n_heights=25)
                        if x3 and z3 and B3d:
                            B_arr = np.array(B3d, dtype=float)
                            p98 = np.percentile(B_arr[B_arr > 0], 98) if np.any(B_arr > 0) else 1.0
                            B_arr = np.clip(B_arr, 0, p98)
                            fig3d = go.Figure(data=[go.Surface(
                                x=x3, y=z3, z=B_arr.tolist(),
                                colorscale="Viridis", showscale=True,
                                colorbar=dict(title="|B| (µT)", thickness=18),
                                hovertemplate="x=%{x:.1f}m<br>z=%{y:.1f}m<br>|B|=%{z:.3f}µT<extra></extra>",
                                contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                            )])
                            fig3d.update_layout(
                                title="Mapa 3D — Campo Magnético |B|(x, z)",
                                scene=dict(
                                    xaxis_title="Dist. lateral x (m)",
                                    yaxis_title="Altura z (m)",
                                    zaxis_title="|B| (µT)",
                                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                                    aspectratio=dict(x=1.5, y=1, z=0.7),
                                ),
                                height=600,
                                margin=dict(l=0, r=0, t=50, b=0),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig3d, use_container_width=True)
                        else:
                            st.warning("Não foi possível gerar a grade 3D para campo magnético.")
                    except Exception as _e3d:
                        st.error(f"Erro ao gerar mapa 3D |B|: {_e3d}")
 
            _report_btns("campos_em", extract_func=rb.extract_fields, result_key="fields_result")
        else:
            st.info("Configure os parâmetros acima e clique **⚡ Calcular Campos**.")
            st.markdown("""
            **Norma de referência:** ANEEL Resolução Normativa nº 915/2021  
            **Limites:** E ≤ 4,17 kV/m e B ≤ 200 µT (público geral) | E ≤ 8,33 kV/m e B ≤ 1000 µT (ocupacional)  
            **Metodologia:** MSC com Imagens Elétricas (Campo E) + Imagens Complexas de Deri (Campo B)
            """)
 
# ===================================================================
# PAGE: AMPACIDADE & FLECHA
# ===================================================================
elif page == PAGES[5]:
    bk_header("Ampacidade e Flecha", "Balanco termico IEEE 738 - Modelo parabolico de flecha")
    bk_section("Condicoes Ambientais e Limites")
    a1, a2, a3 = st.columns(3)
    with a1:
        amp_tamb = st.number_input("T ambiente (C)", value=25.0, step=5.0, help="Temperatura ambiente (IEEE 738 par.4.4)")
        amp_tmax = st.number_input("T max condutor (C)", value=75.0, step=5.0, help="T max admissivel. ACSR: 75-90C")
        amp_wind = st.number_input("Vento (m/s)", value=0.6, step=0.1, format="%.1f", help="Perpendicular ao condutor. 0.6=conservador (IEEE 738)")
    with a2:
        amp_solar = st.number_input("Irradiancia (W/m2)", value=800.0, step=50.0, help="1000=sol pleno, 800=parcial")
        amp_absorp = st.number_input("Absortividade", value=0.5, step=0.05, format="%.2f", help="0.23 novo -> 0.9 envelhecido")
        amp_emiss = st.number_input("Emissividade", value=0.5, step=0.05, format="%.2f", help="0.23 novo -> 0.9 envelhecido")
    with a3:
        amp_ioper = st.number_input("I operacao (A)", value=600.0, step=50.0, help="Corrente para verificacao de temperatura")
    if st.button("Calcular Ampacidade", type="primary", use_container_width=True):
        def _run_amp():
            cfg = AmpacitySagConfig(frequency_hz=st.session_state.freq_hz, ambient_temp_C=amp_tamb,
                max_conductor_temp_C=amp_tmax, wind_speed_m_s=amp_wind, solar_irradiance_W_m2=amp_solar,
                absorptivity=amp_absorp, emissivity=amp_emiss, operating_current_A=amp_ioper)
            return compute_ampacity_sag_for_geometry(_get_geom(), st.session_state.voltage_kv, cfg, _get_cables())
        ar = _safe(_run_amp)
        if ar: st.session_state.amp_result = ar; st.success("OK Ampacidade")
    ar = st.session_state.get("amp_result")
    if ar:
        for cidx, rc in ar.ampacity_per_circuit.items():
            bk_section(f"Circuito {cidx} - {rc.cable_key}")
            bk_kpi_row([("I max (A)", f"{rc.I_max_A:.1f}", "blue"), ("I oper (A)", f"{rc.I_oper_A:.1f}", "teal"),
                ("T limite (C)", f"{rc.temp_limit_C:.0f}", "green" if rc.compliant_temp else "red"),
                ("Flecha (m)", f"{rc.sag_ref_m:.2f}", "orange"),
                ("Status", "OK" if rc.compliant_temp else "EXCEDE", "green" if rc.compliant_temp else "red")])
            with st.expander(f"Detalhes C{cidx}"):
                st.write(f"R_ac={rc.R_ac_ohm_km_at_Tmax:.6f} | q_conv={rc.q_conv_W_m:.3f} | q_rad={rc.q_rad_W_m:.3f} | q_solar={rc.q_solar_W_m:.3f}")
                st.write(f"Vao ref={rc.span_ref_m:.0f}m | H={rc.H_ref_N:.1f}N | w={rc.w_N_m:.4f}N/m")
        if ar.sag_surface and hasattr(ar.sag_surface, "span_lengths_m") and ar.sag_surface.span_lengths_m:
            ss = ar.sag_surface
            # Extract midpoint sag (max sag) for each span
            mid_sags = []
            for j, row in enumerate(ss.y_surface_m):
                mid_sags.append(abs(min(row)) if row else 0.0)
            fig = go.Figure(data=[go.Scatter(x=list(ss.span_lengths_m), y=mid_sags, mode="lines+markers",
                line=dict(color=BK_BLUE, width=2), hovertemplate="Vao=%{x:.0f}m<br>Flecha=%{y:.2f}m<extra></extra>")])
            fig.update_layout(**PLOTLY_LAYOUT, title="Flecha vs Vao", xaxis_title="Vao (m)", yaxis_title="Flecha (m)", height=400)
            st.plotly_chart(fig, use_container_width=True)
        _report_btns("ampacidade", extract_func=rb.extract_ampacity, result_key="amp_result")
    else: st.info("Clique Calcular Ampacidade.")
 
 
# ===================================================================
# PAGE: RI E RA
# ===================================================================
elif page == PAGES[6]:
    bk_header("Radio Interferencia e Ruido Audivel", "Perfis laterais RI (dBuV/m) e RA (dBA)")
    if not HAS_RIRA:
        st.warning("Modulo ri_ra nao disponivel.")
    else:
        bk_section("Configuracao")
        ri1, ri2 = st.columns(2)
        with ri1:
            ri_freq = st.number_input("Freq RI (MHz)", value=0.5, step=0.1, format="%.1f", help="Frequencia avaliacao RI: 0.5-1.0 MHz")
            ri_weather = st.selectbox("Condicao", ["seco", "chuva"], help="Chuva e mais critico para RI/RA")
            ri_dmax = st.number_input("Distancia max (m)", value=60.0, step=10.0, help="Borda da faixa de servidao")
        with ri2:
            ri_lim_ri = st.number_input("Limite RI (dBuV/m)", value=55.0, step=5.0, help="Limite RI na borda da faixa")
            ri_lim_ra_d = st.number_input("Limite RA diurno (dBA)", value=55.0, step=5.0, help="NBR 10151 diurno")
            ri_lim_ra_n = st.number_input("Limite RA noturno (dBA)", value=50.0, step=5.0, help="NBR 10151 noturno")
        if st.button("Calcular RI/RA", type="primary", use_container_width=True):
            def _run_rira():
                pd2 = _compute_params()
                cfg = RIRAConfig(freq_MHz=ri_freq, weather=ri_weather, distance_max_m=ri_dmax,
                    limit_RI_dBuV_m=ri_lim_ri, limit_RA_dBA_day=ri_lim_ra_d, limit_RA_dBA_night=ri_lim_ra_n,
                    V_LL_kV=st.session_state.voltage_kv)
                results = {}
                for cidx, params in pd2.items():
                    results[cidx] = compute_ri_ra_profiles(params, st.session_state.line_length_km, cfg)
                return results
            rira = _safe(_run_rira)
            if rira: st.session_state.rira_results = rira; st.success("OK RI/RA")
        rira = st.session_state.get("rira_results")
        if rira:
            for cidx, prof in rira.items():
                bk_section(f"Circuito {cidx}")
                bk_kpi_row([
                    ("RI borda chuva", f"{prof.RI_edge_chuva_dBuV_m:.1f} dBuV/m", "red" if prof.exceeds_RI_limit else "green"),
                    ("RA borda chuva", f"{prof.RA_edge_chuva_dBA:.1f} dBA", "red" if prof.exceeds_RA_limit else "green"),
                    ("Ec (kV/cm)", f"{prof.Ec_kV_cm:.4f}", "blue")])
                fig = make_subplots(rows=1, cols=2, subplot_titles=["RI (dBuV/m)", "RA (dBA)"])
                fig.add_trace(go.Scatter(x=prof.distances_m, y=prof.RI_seco_dBuV_m, name="RI seco", line=dict(color=BK_BLUE)), row=1, col=1)
                fig.add_trace(go.Scatter(x=prof.distances_m, y=prof.RI_chuva_dBuV_m, name="RI chuva", line=dict(color=BK_RED, dash="dash")), row=1, col=1)
                fig.add_trace(go.Scatter(x=prof.distances_m, y=prof.RA_seco_dBA, name="RA seco", line=dict(color=BK_TEAL)), row=1, col=2)
                fig.add_trace(go.Scatter(x=prof.distances_m, y=prof.RA_chuva_dBA, name="RA chuva", line=dict(color=BK_ORANGE, dash="dash")), row=1, col=2)
                fig.update_layout(**PLOTLY_LAYOUT, height=400, title=f"Perfis RI/RA - Circuito {cidx}")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"RI: {prof.comment_RI}")
                st.info(f"RA: {prof.comment_RA}")
            _report_btns("ri_ra", extract_func=rb.extract_ri_ra, result_key="rira_results")
        else: st.info("Calcule Parametros primeiro, depois RI/RA.")
 
# ===================================================================
# PAGE: BLINDAGEM
# ===================================================================
elif page == PAGES[7]:
    bk_header("Blindagem contra Descargas Atmosfericas", "Angulo de protecao - Aterramento - Backflashover")
    bk_section("Configuracao")
    sh1, sh2 = st.columns(2)
    with sh1:
        sh_theta = st.number_input("Angulo max admissivel (deg)", value=40.0, step=5.0, help="IEEE 1243: <= 30deg. Pratica BR: <= 45deg")
        sh_R = st.number_input("R aterramento (ohm)", value=10.0, step=1.0, help="Resistencia pe de torre. Tipico: 10-25 ohm")
        sh_BIL = st.number_input("BIL/NBI (kV)", value=650.0, step=50.0, help="138kV->650kV, 230kV->1050kV (IEC 60071-1 Tab.2)")
    with sh2:
        sh_Imin = st.number_input("I descarga min (kA)", value=5.0, step=1.0, help="Corrente minima de descarga")
        sh_Imax = st.number_input("I descarga max (kA)", value=50.0, step=5.0, help="CIGRE: mediana ~30 kA")
    if st.button("Calcular Blindagem", type="primary", use_container_width=True):
        def _run_shield():
            cfg = ShieldingConfig(V_LL_kV=st.session_state.voltage_kv, theta_max_deg=sh_theta,
                tower_footing_R_ohm=sh_R, BIL_kV=sh_BIL, I_kA_min=sh_Imin, I_kA_max=sh_Imax)
            return compute_shielding(_get_geom(), cfg)
        sr = _safe(_run_shield)
        if sr: st.session_state.shield_result = sr; st.success("OK Blindagem")
    sr = st.session_state.get("shield_result")
    if sr:
        bk_kpi_row([("Pior angulo (deg)", f"{sr.worst_theta_deg:.1f}", "red" if not sr.all_phases_protected else "green"),
            ("Todas protegidas?", "SIM" if sr.all_phases_protected else "NAO", "green" if sr.all_phases_protected else "red"),
            ("Backflash fracao", f"{sr.grounding.fraction_exceeds:.1%}", "orange")])
        bk_section("Angulo de Protecao por Fase")
        ph_data = [{"Circ": p.circuit_index, "Fase": p.phase, "theta (deg)": f"{p.theta_deg:.1f}",
            "GW": p.nearest_shield_name or "-", "dh (m)": f"{p.delta_h_m:.2f}",
            "d_horiz (m)": f"{p.horizontal_distance_m:.2f}",
            "Protegida": "SIM" if p.is_protected else "NAO"} for p in sr.per_phase]
        st.dataframe(pd.DataFrame(ph_data), use_container_width=True, hide_index=True)
        bk_section("V torre vs I descarga")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sr.grounding.I_kA, y=sr.grounding.V_tower_kV, mode="lines", name="V torre",
            line=dict(color=BK_BLUE, width=2), hovertemplate="I=%{x:.1f}kA<br>V=%{y:.0f}kV<extra></extra>"))
        fig.add_hline(y=sh_BIL, line_dash="dash", line_color=BK_RED, annotation_text=f"BIL={sh_BIL:.0f}kV")
        fig.update_layout(**PLOTLY_LAYOUT, title="V_torre(I) vs BIL", xaxis_title="I (kA)", yaxis_title="V (kV)", height=400)
        st.plotly_chart(fig, use_container_width=True)
        _report_btns("blindagem", extract_func=rb.extract_shielding, result_key="shield_result")
    else: st.info("Configure e clique Calcular Blindagem.")
 
# ===================================================================
# PAGE: ISOLAMENTO VMAX
# ===================================================================
elif page == PAGES[8]:
    bk_header("Isolamento Vmax", "Verificacao isolacao equipamentos - Margem PF - Escoamento - IEC 60071")
    bk_section("Configuracao do Sistema")
    vm1, vm2 = st.columns(2)
    with vm1:
        vm_vnom = st.number_input("Vnom (kV L-L)", value=st.session_state.voltage_kv, step=1.0, help="Tensao nominal L-L (kV)", key="vm_vnom")
        vm_um = st.number_input("Um (kV L-L)", value=145.0, step=1.0, help="Maior tensao do sistema conforme IEC 60071 (kV L-L)", key="vm_um")
        vm_ktov = st.number_input("k_TOV referencia", value=1.20, step=0.05, format="%.2f", help="Fator sobretensao temporaria", key="vm_ktov")
    with vm2:
        vm_alt = st.number_input("Altitude (m)", value=st.session_state.altitude_m, step=100.0, help="Correcao altitude IEC 60071-2", key="vm_alt")
        vm_margin = st.number_input("Margem min seguranca (%)", value=15.0, step=1.0, help="Margem minima recomendada", key="vm_margin")
    bk_section("Itens de Isolacao - Tabela Editavel")
    if "vmax_items" not in st.session_state:
        st.session_state.vmax_items = pd.DataFrame([
            {"Equipamento": "Cadeia suspensao", "U_pf_kV": 275.0, "U_impulso_kV": 650.0, "Escoamento_mm": 4000.0, "Poluicao": 2},
            {"Equipamento": "Bucha AT", "U_pf_kV": 280.0, "U_impulso_kV": 650.0, "Escoamento_mm": 4200.0, "Poluicao": 2},
            {"Equipamento": "Disjuntor", "U_pf_kV": 275.0, "U_impulso_kV": 650.0, "Escoamento_mm": 3500.0, "Poluicao": 2},
        ])
    vmcfg = {
        "Equipamento": st.column_config.TextColumn("Equipamento", help="Nome do equipamento/isolador"),
        "U_pf_kV": st.column_config.NumberColumn("U_pf (kV rms)", help="Tensao suportavel freq. industrial 1min (kV rms)", format="%.1f"),
        "U_impulso_kV": st.column_config.NumberColumn("U_impulso (kV crest)", help="BIL/NBI impulso atmosferico (kV pico)", format="%.1f"),
        "Escoamento_mm": st.column_config.NumberColumn("Escoamento (mm)", help="Comprimento de escoamento total (mm)", format="%.0f"),
        "Poluicao": st.column_config.NumberColumn("Poluicao (1-4)", help="IEC 60815: 1=Leve 2=Medio 3=Pesado 4=Muito Pesado", min_value=1, max_value=4),
    }
    ed_vmax = st.data_editor(st.session_state.vmax_items, column_config=vmcfg, num_rows="dynamic", use_container_width=True, hide_index=True, key="vmax_ed")
    st.session_state.vmax_items = ed_vmax
    if st.button("Verificar Isolamento", type="primary", use_container_width=True):
        def _run_vmax():
            cfg = VmaxConfig(Vnom_kV=vm_vnom, Um_kV=vm_um, k_TOV_ref=vm_ktov,
                min_safety_margin_percent=vm_margin, altitude_m=vm_alt)
            items = []
            for _, row in ed_vmax.iterrows():
                items.append(InsulationItem(name=str(row["Equipamento"]),
                    U_pf_withstand_kV=float(row["U_pf_kV"]),
                    U_impulse_withstand_kV=float(row["U_impulso_kV"]),
                    creepage_mm=float(row["Escoamento_mm"]),
                    pollution_level=int(row.get("Poluicao", 2))))
            return compute_all_items_insulation(cfg, items), cfg
        result = _safe(_run_vmax)
        if result:
            st.session_state.vmax_results = result[0]; st.success(f"OK {len(result[0])} itens verificados")
    vr = st.session_state.get("vmax_results")
    if vr:
        bk_section("Resultados")
        rows = []
        for r in vr:
            ok = r.meets_pf_margin and r.meets_creepage
            rows.append({"Equipamento": r.item.name,
                "V_TOV (kV)": f"{r.V_TOV_kV:.2f}", "U_pf corr (kV)": f"{r.U_pf_corr_kV:.1f}",
                "Margem PF (%)": f"{r.margin_pf_percent:.1f}", "Ka": f"{r.Ka:.4f}",
                "Escoa. req (mm)": f"{r.creepage_required_mm:.0f}", "Escoa. forn (mm)": f"{r.item.creepage_mm:.0f}",
                "Status": "ATENDE" if ok else "NAO ATENDE"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        fig = go.Figure()
        names = [r.item.name for r in vr]; margins = [r.margin_pf_percent for r in vr]
        colors = [BK_GREEN if m >= vm_margin else BK_RED for m in margins]
        fig.add_trace(go.Bar(x=names, y=margins, marker_color=colors, hovertemplate="<b>%{x}</b><br>Margem PF: %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=vm_margin, line_dash="dash", line_color=BK_ORANGE, annotation_text=f"Min {vm_margin}%")
        fig.update_layout(**PLOTLY_LAYOUT, title="Margem PF por Equipamento", yaxis_title="Margem (%)", height=380)
        st.plotly_chart(fig, use_container_width=True)
        _report_btns("vmax", extract_func=rb.extract_vmax, result_key="vmax_results")
    else: st.info("Preencha tabela e clique Verificar.")
 
 
# ===================================================================
# PAGE: COORD. ISOLAMENTO
# ===================================================================
elif page == PAGES[9]:
    bk_header("Coordenacao de Isolamento", "IEC 60071 - Impulso - Manobra - Para-raios - Isoladores")
    bk_section("Parametros do Estudo")
    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        ci_vnom = st.number_input("V nominal (kV)", value=st.session_state.voltage_kv, key="ci_vnom", help="Tensao L-L nominal")
        ci_bil = st.number_input("BIL/NBI (kV)", value=650.0, key="ci_bil", help="Nivel Basico Isolamento impulso (IEC 60071-1)")
        ci_k_imp = st.number_input("Fator k impulso", value=1.1, step=0.1, format="%.2f", key="ci_kimp", help="Fator IEC sobretensao temporaria")
    with ci2:
        ci_vpr = st.number_input("V para-raios ref (kV)", value=100.0, key="ci_vpr", help="Tensao referencia do para-raios ZnO (Vr)")
        ci_ipr = st.number_input("I para-raios ref (kA)", value=10.0, key="ci_ipr", help="Corrente nominal descarga (In)")
        ci_v0 = st.number_input("V0 impulso (kV)", value=650.0, key="ci_v0", help="Amplitude onda impulso padrao 1.2/50us")
    with ci3:
        ci_vdisc = st.number_input("V disco (kV)", value=18.0, key="ci_vdisc", help="Tensao suportavel por disco (freq industrial)")
        ci_vimp = st.number_input("V impulso/disco (kV)", value=50.0, key="ci_vimp", help="Tensao impulso suportavel por disco")
        ci_creep = st.number_input("Escoamento/disco (mm)", value=400.0, key="ci_creep", help="Distancia escoamento por disco (IEC 60815)")
    if st.button("Calcular Coord. Isolamento", type="primary", use_container_width=True):
        def _run_coord():
            s = st.session_state
            cfg = CoordIsolConfig(Vnom_kV=ci_vnom, Vbil_kV=ci_bil, k_impulse=ci_k_imp,
                V0_kV=ci_v0, Vpr_kV=ci_vpr, Ipr_kA=ci_ipr,
                V_disco_kV=ci_vdisc, V_impulso_disco_kV=ci_vimp,
                single_disc_creepage_mm=ci_creep,
                h_cg_m=s.h_shield, h_fase_m=s.h_phase_ref)
            return compute_coord_isolation(cfg)
        cr = _safe(_run_coord)
        if cr: st.session_state.coord_result = cr; st.success("OK Coord. Isolamento")
    cr = st.session_state.get("coord_result")
    if cr:
        bk_kpi_row([("V impulso max (kV)", f"{cr.Vmax_impulse_kV:.1f}", "blue"),
            ("Angulo protecao (deg)", f"{cr.shield.theta_deg:.1f}", "teal"),
            ("No discos (normal)", f"{cr.insulator.N_disc_normal}", "green"),
            ("No discos (poluido)", f"{cr.insulator.N_disc_polluted}", "orange"),
            ("Atende NBI", "SIM" if cr.insulator.atende_NBI else "NAO", "green" if cr.insulator.atende_NBI else "red")])
        tab_imp, tab_arr, tab_ins = st.tabs(["Impulso", "Para-raios", "Isoladores"])
        with tab_imp:
            iw = cr.impulse
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[t*1e6 for t in iw.t_s], y=list(iw.V_kV), mode="lines", name="V(t)",
                line=dict(color=BK_BLUE, width=2), hovertemplate="t=%{x:.1f}us<br>V=%{y:.0f}kV<extra></extra>"))
            fig.update_layout(**PLOTLY_LAYOUT, title="Onda de Impulso 1.2/50 us", xaxis_title="Tempo (us)", yaxis_title="Tensao (kV)", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with tab_arr:
            arr = cr.arrester
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(arr.I_kA), y=list(arr.V_kV), mode="lines", name="V(I)",
                line=dict(color=BK_TEAL, width=2), hovertemplate="I=%{x:.1f}kA<br>V=%{y:.0f}kV<extra></extra>"))
            fig.add_trace(go.Scatter(x=[arr.I_ref_kA], y=[arr.V_ref_kV], mode="markers", name="Referencia",
                marker=dict(size=12, color=BK_RED, symbol="star")))
            fig.update_layout(**PLOTLY_LAYOUT, title="Curva VxI Para-raios", xaxis_title="I (kA)", yaxis_title="V (kV)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Energia dissipada: **{arr.E_J:.1f} J** ({arr.E_J/1000:.3f} kJ)")
        with tab_ins:
            ins = cr.insulator
            ins_data = {"V operacao (kV)": f"{ins.V_operacao_kV:.1f}", "Escoamento (mm)": f"{ins.L_escoamento_mm:.0f}",
                "N discos normal": ins.N_disc_normal, "N discos poluido": ins.N_disc_polluted,
                "V impulso cadeia (kV)": f"{ins.V_impulso_cadeia_kV:.0f}", "Atende NBI": "SIM" if ins.atende_NBI else "NAO"}
            st.dataframe(pd.DataFrame([ins_data]), use_container_width=True, hide_index=True)
        with st.expander("Resumo Completo"):
            st.markdown(cr.resumo_coord)
        _report_btns("coord_isol", extract_func=rb.extract_coord_isol, result_key="coord_result")
    else: st.info("Clique Calcular Coord. Isolamento.")
 
# ===================================================================
# PAGE: RELIGAMENTO TRIPOLAR
# ===================================================================
elif page == PAGES[10]:
    bk_header("Religamento Tripolar", "Sobretensao transitoria - Janelas aceitaveis - Fator sobretensao")
    bk_section("Configuracao")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        rc_dead = st.number_input("Tempo morto (s)", value=0.3, step=0.05, format="%.2f", help="Dead time religamento. Tipico: 0.2-1.0s")
        rc_trap = st.number_input("Fator carga presa (pu)", value=1.0, step=0.1, format="%.1f", help="0=sem carga presa, 1=pior caso")
    with rc2:
        rc_damp = st.number_input("Amortecimento alpha (Np/s)", value=0.0, step=0.5, help="Fator amortecimento. 0=sem perdas")
        rc_limit = st.number_input("Limite sobretensao (pu)", value=2.0, step=0.1, format="%.1f", help="Limite FO aceitavel (tipicamente 2.0 pu)")
    with rc3:
        rc_tsim = st.number_input("Janela simulacao (s)", value=0.3, step=0.05, format="%.2f", help="Duracao total simulacao temporal")
    if st.button("Calcular Religamento", type="primary", use_container_width=True):
        def _run_recl():
            s = st.session_state; pd2 = _compute_params()
            cfg = ReclosingConfig(V_LL_kV=s.voltage_kv, f_hz=s.freq_hz, length_km=s.line_length_km,
                dead_time_s=rc_dead, trapped_kpu=rc_trap, damping_alpha=rc_damp,
                overvoltage_limit_pu=rc_limit, t_sim_s=rc_tsim)
            return compute_reclosing_study(pd2, cfg)
        rr = _safe(_run_recl)
        if rr: st.session_state.recl_result = rr; st.success("OK Religamento")
    rr = st.session_state.get("recl_result")
    if rr:
        for cidx, rc in rr.per_circuit.items():
            bk_section(f"Circuito {cidx}")
            bk_kpi_row([("f0 natural (Hz)", f"{rc.f0_hz:.2f}", "blue"),
                ("FO dead time (pu)", f"{rc.FO_dead_pu:.3f}", "teal"),
                ("FO max (pu)", f"{rc.FO_max_pu:.3f}", "red" if rc.FO_max_pu > rc_limit else "green"),
                ("Dead time OK", "SIM" if rc.is_dead_time_acceptable else "NAO", "green" if rc.is_dead_time_acceptable else "red"),
                ("Janelas", str(len(rc.acceptable_windows)), "orange")])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rc.t_s, y=rc.FO_pu, mode="lines", name="FO(t)",
                line=dict(color=BK_BLUE, width=2), hovertemplate="t=%{x:.4f}s<br>FO=%{y:.3f}pu<extra></extra>"))
            fig.add_hline(y=rc_limit, line_dash="dash", line_color=BK_RED, annotation_text=f"Limite {rc_limit} pu")
            fig.add_vline(x=rc_dead, line_dash="dot", line_color=BK_ORANGE, annotation_text=f"Dead time {rc_dead}s")
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Fator Sobretensao - Circuito {cidx}", xaxis_title="Tempo (s)", yaxis_title="FO (pu)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            if rc.acceptable_windows:
                st.markdown("**Janelas de religamento aceitaveis:**")
                w_data = [{"#": i+1, "t_inicio (s)": f"{w.t_start_s:.4f}", "t_fim (s)": f"{w.t_end_s:.4f}",
                    "Duracao (ms)": f"{(w.t_end_s-w.t_start_s)*1000:.1f}"} for i, w in enumerate(rc.acceptable_windows)]
                st.dataframe(pd.DataFrame(w_data), use_container_width=True, hide_index=True)
        _report_btns("religamento", extract_func=rb.extract_reclosing, result_key="recl_result")
    else: st.info("Calcule Parametros primeiro, depois Religamento.")
 
# ===================================================================
# PAGE: COMPAT. ELETROMAGNETICA
# ===================================================================
elif page == PAGES[11]:
    bk_header("Compatibilidade Eletromagnetica", "Tensao induzida em dutos - Campo em linhas comunicacao")
    bk_section("Configuracao")
    em1, em2 = st.columns(2)
    with em1:
        em_len = st.number_input("Comprimento paralelo (km)", value=10.0, step=1.0, help="Extensao paralelismo LT-duto/cabo comunicacao")
        em_sep = st.number_input("Separacao lateral (m)", value=50.0, step=10.0, help="Distancia horizontal media LT-infraestrutura")
        em_I = st.number_input("Corrente carga (A)", value=600.0, step=50.0, help="Corrente RMS por fase")
    with em2:
        em_lim_cont = st.number_input("Limite V cont (V/km)", value=60.0, step=10.0, help="Limite tensao induzida continua")
        em_lim_short = st.number_input("Limite V curto (V/km)", value=300.0, step=50.0, help="Limite tensao durante falta")
        em_lim_E = st.number_input("Limite E longit (V/m)", value=5.0, step=1.0, help="Limite campo eletrico longitudinal cabos comunicacao")
    if st.button("Calcular EMI", type="primary", use_container_width=True):
        def _run_emi():
            cfg = EMIConfig(f_hz=st.session_state.freq_hz, length_parallel_km=em_len, separation_m=em_sep,
                I_load_A=em_I, pipeline_cont_limit_V_per_km=em_lim_cont,
                pipeline_short_limit_V_per_km=em_lim_short, comm_E_limit_V_per_m=em_lim_E)
            proj = {"voltage_kv": st.session_state.voltage_kv, "power_mva": st.session_state.power_mva}
            return run_emi_study(proj, _get_geom(), cfg)
        er = _safe(_run_emi)
        if er: st.session_state.emi_result = er; st.success("OK EMI")
    er = st.session_state.get("emi_result")
    if er:
        if er.pipeline_result:
            p = er.pipeline_result
            bk_section("Dutos / Pipelines")
            bk_kpi_row([("V continua (V/km)", f"{p.V_induced_cont_V_per_km:.2f}", "red" if p.exceeds_cont_limit else "green"),
                ("V curto-circ (V/km)", f"{p.V_induced_short_V_per_km:.2f}", "red" if p.exceeds_short_limit else "green")])
            if p.notes: st.info(p.notes)
        if er.comm_result:
            c = er.comm_result
            bk_section("Linhas de Comunicacao")
            bk_kpi_row([("E longitudinal (V/m)", f"{c.E_longitudinal_V_per_m:.3f}", "red" if c.exceeds_E_limit else "green")])
            if c.notes: st.info(c.notes)
        if er.summary:
            with st.expander("Resumo"): st.markdown(er.summary)
        _report_btns("emi", extract_func=rb.extract_emi, result_key="emi_result")
    else: st.info("Clique Calcular EMI.")
 
# ===================================================================
# PAGE: FLUXO DE POTENCIA
# ===================================================================
elif page == PAGES[12]:
    bk_header("Fluxo de Potencia", "Newton-Raphson multibarras - PQ/PV/Slack")
    if not HAS_PF:
        st.warning("Modulo power_flow nao disponivel. Verifique dependencias.")
    else:
        import cmath
        bk_section("Dados do Sistema (base)")
        pb1, pb2 = st.columns(2)
        with pb1:
            pf_base_mva = st.number_input("Base MVA", value=100.0, step=10.0, help="Potencia base para conversao pu")
        with pb2:
            pf_base_kv = st.number_input("Base kV (L-L)", value=st.session_state.voltage_kv, step=1.0, help="Tensao base L-L")
        bk_section("Barras - Tabela Editavel")
        if "pf_buses" not in st.session_state:
            st.session_state.pf_buses = pd.DataFrame([
                {"Bus": 1, "Tipo": "SLACK", "Vm_kV": pf_base_kv, "Va_deg": 0.0, "Pg_MW": 0.0, "Qg_Mvar": 0.0, "Pl_MW": 0.0, "Ql_Mvar": 0.0},
                {"Bus": 2, "Tipo": "PQ", "Vm_kV": pf_base_kv, "Va_deg": 0.0, "Pg_MW": 0.0, "Qg_Mvar": 0.0, "Pl_MW": 50.0, "Ql_Mvar": 20.0},
            ])
        pf_bcfg = {
            "Bus": st.column_config.NumberColumn("Bus", help="Numero da barra (inteiro unico)", min_value=1),
            "Tipo": st.column_config.SelectboxColumn("Tipo", help="SLACK, PV ou PQ", options=["SLACK", "PV", "PQ"]),
            "Vm_kV": st.column_config.NumberColumn("Vm (kV)", help="Modulo tensao (kV L-L) ou Vset para PV", format="%.1f"),
            "Va_deg": st.column_config.NumberColumn("Va (deg)", help="Angulo inicial (graus)", format="%.1f"),
            "Pg_MW": st.column_config.NumberColumn("Pg (MW)", help="Geracao ativa", format="%.1f"),
            "Qg_Mvar": st.column_config.NumberColumn("Qg (Mvar)", help="Geracao reativa", format="%.1f"),
            "Pl_MW": st.column_config.NumberColumn("Pl (MW)", help="Carga ativa", format="%.1f"),
            "Ql_Mvar": st.column_config.NumberColumn("Ql (Mvar)", help="Carga reativa", format="%.1f"),
        }
        ed_bus = st.data_editor(st.session_state.pf_buses, column_config=pf_bcfg, num_rows="dynamic", use_container_width=True, hide_index=True, key="pf_bus_ed")
        st.session_state.pf_buses = ed_bus
 
        bk_section("Ramos - Tabela Editavel")
        if "pf_branches" not in st.session_state:
            st.session_state.pf_branches = pd.DataFrame([
                {"De": 1, "Para": 2, "L_km": 100.0, "R_ohm_km": 0.286, "X_ohm_km": 0.516, "B_S_km": 3.88e-6, "Tap": 1.0},
            ])
        pf_brcfg = {
            "De": st.column_config.NumberColumn("De", help="Barra origem", min_value=1, format="%d"),
            "Para": st.column_config.NumberColumn("Para", help="Barra destino", min_value=1, format="%d"),
            "L_km": st.column_config.NumberColumn("L (km)", help="Comprimento do ramo", format="%.1f"),
            "R_ohm_km": st.column_config.NumberColumn("R (ohm/km)", help="Resistencia serie", format="%.6f"),
            "X_ohm_km": st.column_config.NumberColumn("X (ohm/km)", help="Reatancia serie", format="%.6f"),
            "B_S_km": st.column_config.NumberColumn("B (S/km)", help="Susceptancia shunt total", format="%.2e"),
            "Tap": st.column_config.NumberColumn("Tap", help="Tap trafo (1.0=sem trafo)", format="%.3f"),
        }
        ed_br = st.data_editor(st.session_state.pf_branches, column_config=pf_brcfg, num_rows="dynamic", use_container_width=True, hide_index=True, key="pf_br_ed")
        st.session_state.pf_branches = ed_br
 
        if st.button("Executar Fluxo", type="primary", use_container_width=True):
            def _run_pf():
                buses = []
                for _, row in ed_bus.iterrows():
                    buses.append(PFBus(bus=int(row["Bus"]), type=str(row["Tipo"]),
                        vm_kv=float(row["Vm_kV"]), va_deg=float(row["Va_deg"]),
                        pg_mw=float(row["Pg_MW"]), qg_mvar=float(row["Qg_Mvar"]),
                        pl_mw=float(row["Pl_MW"]), ql_mvar=float(row["Ql_Mvar"])))
                branches = []
                for _, row in ed_br.iterrows():
                    branches.append(PFBranch(frm=int(row["De"]), to=int(row["Para"]),
                        length_km=float(row["L_km"]),
                        r_ohm_km=float(row["R_ohm_km"]), x_ohm_km=float(row["X_ohm_km"]),
                        b_s_km=float(row["B_S_km"]), tap=float(row.get("Tap", 1.0))))
                case = PFCase(buses=buses, branches=branches, base_mva=pf_base_mva, base_kv_ll=pf_base_kv)
                st.session_state.pf_case = case
                return solve_power_flow_newton(case)
            pfr = _safe(_run_pf)
            if pfr:
                st.session_state.pf_result = pfr
                if pfr.converged:
                    st.success(f"OK Convergiu em {pfr.iters} iteracoes (mismatch: {pfr.max_mismatch_pu:.2e} pu)")
                else:
                    st.warning(f"Nao convergiu em {pfr.iters} iteracoes")
        pfr = st.session_state.get("pf_result")
        if pfr:
            bk_section("Resultados das Barras")
            bus_rows = []
            for bid, v_pu in sorted(pfr.v_pu.items()):
                vm = abs(v_pu); va = cmath.phase(v_pu) * 180 / math.pi
                vm_kv = vm * pf_base_kv
                p = pfr.p_calc_pu.get(bid, 0) * pf_base_mva
                q = pfr.q_calc_pu.get(bid, 0) * pf_base_mva
                bus_rows.append({"Barra": bid, "|V| (pu)": f"{vm:.4f}", "|V| (kV)": f"{vm_kv:.2f}",
                    "Ang (deg)": f"{va:.2f}", "P (MW)": f"{p:.2f}", "Q (Mvar)": f"{q:.2f}"})
            st.dataframe(pd.DataFrame(bus_rows), use_container_width=True, hide_index=True)
            bk_kpi_row([("Slack P", f"{pfr.slack_p_mw:.2f} MW", "blue"),
                ("Slack Q", f"{pfr.slack_q_mvar:.2f} Mvar", "teal"),
                ("Iteracoes", str(pfr.iters), "green"),
                ("Mismatch", f"{pfr.max_mismatch_pu:.2e} pu", "orange")])
            if pfr.branch_flows:
                bk_section("Fluxos nos Ramos")
                bf_rows = [{"De->Para": f"{bf.frm}->{bf.to}", "P (MW)": f"{bf.p_mw:.2f}",
                    "Q (Mvar)": f"{bf.q_mvar:.2f}", "Perdas P (MW)": f"{bf.p_loss_mw:.4f}",
                    "Perdas Q (Mvar)": f"{bf.q_loss_mvar:.4f}"} for bf in pfr.branch_flows]
                st.dataframe(pd.DataFrame(bf_rows), use_container_width=True, hide_index=True)
            fig = go.Figure()
            bus_ids = sorted(pfr.v_pu.keys())
            vm_vals = [abs(pfr.v_pu[b]) for b in bus_ids]
            fig.add_trace(go.Bar(x=[str(b) for b in bus_ids], y=vm_vals,
                marker_color=[BK_BLUE if v >= 0.95 else BK_RED for v in vm_vals],
                hovertemplate="<b>Barra %{x}</b><br>V=%{y:.4f} pu<extra></extra>"))
            fig.add_hline(y=0.95, line_dash="dash", line_color=BK_RED, annotation_text="0.95 pu")
            fig.add_hline(y=1.05, line_dash="dash", line_color=BK_RED, annotation_text="1.05 pu")
            fig.update_layout(**PLOTLY_LAYOUT, title="Perfil de Tensao", yaxis_title="V (pu)", height=380)
            st.plotly_chart(fig, use_container_width=True)
            _report_btns("fluxo", extract_func=rb.extract_power_flow, result_key="pf_result", case_obj=st.session_state.get("pf_case"))
 
# ===================================================================
# FOOTER
# ===================================================================
st.divider()
st.caption("BK Estudos Eletricos v2.0 - BK Engenharia e Tecnologia - Streamlit Edition")

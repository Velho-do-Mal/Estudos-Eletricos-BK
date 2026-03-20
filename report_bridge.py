# report_bridge.py
# ====================================================================
# Ponte entre o session_state do Streamlit e os geradores de relatorio
# Cada funcao extrai os dados necessarios do session_state e dos
# resultados calculados, formatando para o gerador de relatorio.
# ====================================================================

from __future__ import annotations
from typing import Any, Dict, Optional
import streamlit as st


def _ss(key: str, default=None):
    """Busca valor no session_state."""
    return st.session_state.get(key, default)


def _base_cfg() -> Dict[str, Any]:
    """Configuracao base comum a todos os modulos."""
    return {
        "voltage_kv": _ss("voltage_kv", 138),
        "power_mva": _ss("power_mva", 100),
        "freq_hz": _ss("freq_hz", 60),
        "altitude_m": _ss("altitude_m", 0),
        "line_length_km": _ss("line_length_km", 100),
        "temp_C": _ss("temp_C", 50),
    }


def extract_params(result_obj) -> Dict[str, Any]:
    """Extrai resultados de Parametros Eletricos."""
    cfg = _base_cfg()
    cfg["pf_load"] = _ss("pf_load", 1.0)
    cfg["n_circuits"] = _ss("n_circuits", 1)
    cfg["Vs_ang"] = _ss("Vs_ang", 0.0)
    results = {"circuits": []}

    if result_obj is None:
        return results, cfg

    # result_obj pode ser dict {circuit_idx: LineParamsResult} ou lista [LineParamsResult]
    items = []
    if isinstance(result_obj, dict):
        items = [result_obj[k] for k in sorted(result_obj.keys())]
    elif isinstance(result_obj, (list, tuple)):
        items = list(result_obj)

    for r in items:
        circ = {"label": f"Circuito {getattr(r, 'circuit_index', 0)}"}
        for attr in ["R_ohm_km", "X_ohm_km", "B_S_km", "L_H_km", "C_F_km",
                     "GMR_eq_m", "r_eq_m", "Zc_ohm", "SIL_MW", "Ec_kV_cm",
                     "L_mH_km", "C_nF_km", "lambda_m", "Q_mC_km_por_fase", "GMD_m"]:
            circ[attr] = getattr(r, attr, 0)
        results["circuits"].append(circ)

    # ── Vr calculado (solve_vr_pi) ──────────────────────────
    _cvr = _ss("computed_vr") or {}
    if _cvr:
        vr_list = []
        vs_kv = _ss("voltage_kv", 138.0)
        vs_ang = _ss("Vs_ang", 0.0)
        for cidx in sorted(_cvr.keys()):
            vr_kv, vr_ang, i_a, ploss = _cvr[cidx]
            reg = ((vs_kv - vr_kv) / vr_kv * 100.0) if vr_kv > 0 else 0.0
            vr_list.append({
                "circuit": cidx,
                "Vs_kV": vs_kv, "Vs_ang": vs_ang,
                "Vr_kV": vr_kv, "Vr_ang": vr_ang,
                "I_A": i_a, "Ploss_MW": ploss,
                "reg_pct": reg,
            })
        results["vr_computed"] = vr_list

    return results, cfg


def extract_corona(result_list) -> Dict[str, Any]:
    """Extrai resultados de Corona (dict ou lista de CoronaCircuitResult)."""
    cfg = _base_cfg()
    cfg["weather"] = _ss("corona_weather", "seco")
    results = {"circuits": []}

    if result_list is None:
        return results, cfg

    items = result_list.values() if isinstance(result_list, dict) else result_list
    for r in items:
        circ = {"label": f"Circuito {getattr(r, 'circuit_index', 0)}"}
        for attr in ["Ec_crit_kV_cm", "Esurface_kV_cm", "Vd_LL_kV", "corona_ok",
                     "corona_loss_kW_km_phase", "margin_Vd_percent", "V_phase_kV",
                     "delta_air", "m0", "r_eq_cm", "GMD_m", "Ve_surge_LL_kV"]:
            circ[attr] = getattr(r, attr, 0)
        results["circuits"].append(circ)

    return results, cfg


def extract_fields(result_obj) -> Dict[str, Any]:
    """Extrai resultados de Campos EM — ANEEL RN 915/2021."""
    cfg = _base_cfg()
    # Limites do RN 915/2021
    cfg["E_limit_geral"] = _ss("f_Elim_g", 4.17)
    cfg["B_limit_geral"] = _ss("f_Blim_g", 200.0)
    cfg["E_limit_ocup"]  = _ss("f_Elim_o", 8.33)
    cfg["B_limit_ocup"]  = _ss("f_Blim_o", 1000.0)
    # Aliases legados
    cfg["E_limit"] = cfg["E_limit_geral"]
    cfg["B_limit"] = cfg["B_limit_geral"]
    results = {}

    if result_obj is None:
        return results, cfg

    results["E_max_kV_m"] = getattr(result_obj, "E_max_kV_m", 0)
    results["B_max_uT"]   = getattr(result_obj, "B_max_uT",   0)
    results["x_E_max_m"]  = getattr(result_obj, "x_E_max_m",  0)
    results["x_B_max_m"]  = getattr(result_obj, "x_B_max_m",  0)

    # Metadados da configuração
    _config = getattr(result_obj, "config", None)
    if _config:
        cfg["h_obs_m"]  = getattr(_config, "h_obs_m",  1.5)
        cfg["rho_solo"] = getattr(_config, "rho_solo",  100.0)
        cfg["freq_hz"]  = getattr(_config, "freq_hz",   60.0)
        cfg["x_min_m"]  = getattr(_config, "x_min_m",  -30.0)
        cfg["x_max_m"]  = getattr(_config, "x_max_m",   30.0)
        results["x_min_m"] = cfg["x_min_m"]
        results["x_max_m"] = cfg["x_max_m"]

    # Limites do objeto resultado (sobrescreve se disponível)
    _limits = getattr(result_obj, "limits", None)
    if _limits:
        cfg["E_limit_geral"] = getattr(_limits, "E_max_kV_m_geral", cfg["E_limit_geral"])
        cfg["B_limit_geral"] = getattr(_limits, "B_max_uT_geral",   cfg["B_limit_geral"])
        cfg["E_limit_ocup"]  = getattr(_limits, "E_max_kV_m_ocup",  cfg["E_limit_ocup"])
        cfg["B_limit_ocup"]  = getattr(_limits, "B_max_uT_ocup",    cfg["B_limit_ocup"])
        cfg["E_limit"] = cfg["E_limit_geral"]
        cfg["B_limit"] = cfg["B_limit_geral"]

    # Arrays de perfil
    x = getattr(result_obj, "x_m", None)
    E = getattr(result_obj, "E_kV_m", None)
    B = getattr(result_obj, "B_uT", None)
    if x is not None:
        results["x_m"]    = list(x) if hasattr(x, "__iter__") else []
        results["E_kV_m"] = list(E) if E is not None and hasattr(E, "__iter__") else []
        results["B_uT"]   = list(B) if B is not None and hasattr(B, "__iter__") else []

    return results, cfg


def extract_ampacity(result_obj) -> Dict[str, Any]:
    """Extrai resultados de Ampacidade."""
    cfg = _base_cfg()
    cfg["ambient_temp_C"] = _ss("amp_temp_amb", 35)
    cfg["max_conductor_temp_C"] = _ss("amp_temp_max", 75)
    cfg["wind_speed_m_s"] = _ss("amp_wind", 0.6)
    cfg["solar_irradiance"] = _ss("amp_solar", 1000)
    cfg["cable_phase_key"] = _ss("cable_phase_key", "ACSR_477")
    cfg["design_tension_ratio"] = _ss("amp_tension_ratio", 0.25)
    results = {"circuits": []}

    if result_obj is None:
        return results, cfg

    amp_by_circ = getattr(result_obj, "ampacity_per_circuit", {})
    for idx in sorted(amp_by_circ.keys()):
        a = amp_by_circ[idx]
        results["circuits"].append({
            "label": f"Circuito {idx}",
            "I_max_A": getattr(a, "I_max_A", 0),
            "I_oper_A": getattr(a, "I_oper_A", 0),
            "T_lim_C": getattr(a, "temp_limit_C", 0),
            "sag_m": getattr(a, "sag_ref_m", 0),
            "H_ref_N": getattr(a, "H_ref_N", 0),
            "w_N_m": getattr(a, "w_N_m", 0),
            "cable_key": getattr(a, "cable_key", ""),
            "status": "ATENDE" if getattr(a, "compliant_temp", True) else "NÃO ATENDE",
        })

    return results, cfg


def extract_ri_ra(result_list) -> Dict[str, Any]:
    """Extrai resultados de RI e RA."""
    cfg = _base_cfg()
    results = {"circuits": []}

    if result_list is None:
        return results, cfg

    items = result_list.values() if isinstance(result_list, dict) else (result_list if isinstance(result_list, list) else [result_list])
    for r in items:
        circ = {
            "RI_edge_chuva": getattr(r, "RI_edge_chuva_dBuV_m", 0),
            "RA_edge_chuva": getattr(r, "RA_edge_chuva_dBA", 0),
            "Ec_kV_cm": getattr(r, "Ec_kV_cm", 0),
            "exceeds_RI": getattr(r, "exceeds_RI_limit", False),
            "exceeds_RA": getattr(r, "exceeds_RA_limit", False),
        }
        # Arrays
        for attr_pair in [("distances_m", "distances_m"),
                          ("RI_seco", "RI_seco_dBuV_m"),
                          ("RI_chuva", "RI_chuva_dBuV_m"),
                          ("RA_seco", "RA_seco_dBA"),
                          ("RA_chuva", "RA_chuva_dBA")]:
            arr = getattr(r, attr_pair[1], None)
            if arr is not None and hasattr(arr, '__iter__'):
                circ[attr_pair[0]] = list(arr)
        results["circuits"].append(circ)

    return results, cfg


def extract_shielding(result_obj) -> Dict[str, Any]:
    """Extrai resultados de Blindagem."""
    cfg = _base_cfg()
    cfg["BIL_kV"] = _ss("shield_BIL", 650)
    results = {}

    if result_obj is None:
        return results, cfg

    results["worst_theta_deg"] = getattr(result_obj, "worst_theta_deg", 0)
    results["all_phases_protected"] = getattr(result_obj, "all_phases_protected", False)

    gnd = getattr(result_obj, "grounding", None)
    if gnd:
        results["backflash_fraction"] = getattr(gnd, "fraction_exceeds", 0)
        I_arr = getattr(gnd, "I_kA", None)
        V_arr = getattr(gnd, "V_tower_kV", None)
        if I_arr is not None and hasattr(I_arr, '__iter__'):
            results["I_kA"] = list(I_arr)
            results["V_tower_kV"] = list(V_arr) if V_arr is not None else []

    per_phase = getattr(result_obj, "per_phase", [])
    results["per_phase"] = []
    for ph in per_phase:
        results["per_phase"].append({
            "circuit": getattr(ph, "circuit_index", 0),
            "phase": getattr(ph, "phase", ""),
            "theta_deg": getattr(ph, "theta_deg", 0),
            "gw_name": getattr(ph, "nearest_shield_name", ""),
            "delta_h": getattr(ph, "delta_h_m", 0),
            "d_horiz": getattr(ph, "horizontal_distance_m", 0),
            "protected": getattr(ph, "is_protected", False),
        })

    return results, cfg


def extract_vmax(result_list) -> Dict[str, Any]:
    """Extrai resultados de Vmax Insulation."""
    cfg = _base_cfg()
    cfg["min_margin"] = _ss("vmax_margin", 15)
    results = {"items": []}

    if result_list is None:
        return results, cfg

    for r in result_list:
        item = getattr(r, "item", None)
        results["items"].append({
            "name": getattr(item, "name", "") if item else "",
            "V_TOV_kV": getattr(r, "V_TOV_kV", 0),
            "U_pf_corr_kV": getattr(r, "U_pf_corr_kV", 0),
            "margin_pf_percent": getattr(r, "margin_pf_percent", 0),
            "Ka": getattr(r, "Ka", 1),
            "creepage_req_mm": getattr(r, "creepage_required_mm", 0),
            "creepage_forn_mm": getattr(item, "creepage_mm", 0) if item else 0,
            "meets_pf": getattr(r, "meets_pf_margin", True),
            "meets_creepage": getattr(r, "meets_creepage", True),
        })

    return results, cfg


def extract_coord_isol(result_obj) -> Dict[str, Any]:
    """Extrai resultados de Coordenacao de Isolamento."""
    cfg = _base_cfg()
    results = {}

    if result_obj is None:
        return results, cfg

    results["V_impulse_max_kV"] = getattr(result_obj, "Vmax_impulse_kV", 0)

    shield = getattr(result_obj, "shield", None)
    results["theta_deg"] = getattr(shield, "theta_deg", 0) if shield else 0

    insul = getattr(result_obj, "insulator", None)
    if insul:
        results["N_disc_normal"] = getattr(insul, "N_disc_normal", 0)
        results["N_disc_polluted"] = getattr(insul, "N_disc_polluted", 0)
        results["atende_NBI"] = getattr(insul, "atende_NBI", False)
        results["insulator_table"] = {
            "N_normal": results["N_disc_normal"],
            "N_polluted": results["N_disc_polluted"],
            "V_imp_cadeia": getattr(insul, "V_impulso_cadeia_kV", 0),
            "atende_NBI": results["atende_NBI"],
        }

    impulse = getattr(result_obj, "impulse", None)
    if impulse:
        t = getattr(impulse, "t_s", None)
        V = getattr(impulse, "V_kV", None)
        if t is not None and hasattr(t, '__iter__'):
            results["impulse_t"] = list(t)
            results["impulse_V"] = list(V) if V is not None else []

    arrester = getattr(result_obj, "arrester", None)
    if arrester:
        I = getattr(arrester, "I_kA", None)
        V = getattr(arrester, "V_kV", None)
        if I is not None and hasattr(I, '__iter__'):
            results["arrester_I"] = list(I)
            results["arrester_V"] = list(V) if V is not None else []
        results["arrester_I_ref"] = getattr(arrester, "I_ref_kA", 0)
        results["arrester_V_ref"] = getattr(arrester, "V_ref_kV", 0)

    return results, cfg


def extract_reclosing(result_obj) -> Dict[str, Any]:
    """Extrai resultados de Religamento."""
    cfg = _base_cfg()
    cfg["overvoltage_limit_pu"] = _ss("recl_ov_limit", 2.0)
    cfg["dead_time_s"] = _ss("recl_dead_time", 0.5)
    results = {"circuits": []}

    if result_obj is None:
        return results, cfg

    per_circuit = getattr(result_obj, "per_circuit", {})
    items = per_circuit.values() if isinstance(per_circuit, dict) else per_circuit
    for r in items:
        circ = {
            "f0_hz": getattr(r, "f0_hz", 0),
            "FO_dead_pu": getattr(r, "FO_dead_pu", 0),
            "FO_max_pu": getattr(r, "FO_max_pu", 0),
            "dead_time_ok": getattr(r, "is_dead_time_acceptable", False),
        }
        # Time series
        t = getattr(r, "t_s", None)
        fo = getattr(r, "FO_pu", None)
        if t is not None and hasattr(t, '__iter__'):
            circ["t_s"] = list(t)
            circ["FO_pu"] = list(fo) if fo is not None else []

        # Windows
        windows = getattr(r, "acceptable_windows", [])
        circ["n_windows"] = len(windows)
        circ["windows"] = []
        for w in windows:
            circ["windows"].append({
                "t_start": getattr(w, "t_start_s", 0),
                "t_end": getattr(w, "t_end_s", 0),
            })
        results["circuits"].append(circ)

    return results, cfg


def extract_emi(result_obj) -> Dict[str, Any]:
    """Extrai resultados de EMI."""
    cfg = _base_cfg()
    results = {}

    if result_obj is None:
        return results, cfg

    pipe = getattr(result_obj, "pipeline_result", None)
    if pipe:
        results["pipeline"] = {
            "V_cont": getattr(pipe, "V_induced_cont_V_per_km", 0),
            "V_short": getattr(pipe, "V_induced_short_V_per_km", 0),
            "lim_cont": getattr(pipe, "V_limit_cont_V_per_km", 60),
            "lim_short": getattr(pipe, "V_limit_short_V_per_km", 300),
            "exceeds_cont": getattr(pipe, "exceeds_cont_limit", False),
        }
    comm = getattr(result_obj, "comm_result", None)
    if comm:
        results["comm"] = {
            "E_long": getattr(comm, "E_longitudinal_V_per_m", 0),
            "lim_E": getattr(comm, "E_limit_V_per_m", 5),
            "exceeds_E": getattr(comm, "exceeds_E_limit", False),
        }

    return results, cfg


def extract_power_flow(result_obj, case_obj=None) -> Dict[str, Any]:
    """Extrai resultados de Fluxo de Potencia."""
    cfg = _base_cfg()
    results = {}

    if result_obj is None:
        return results, cfg

    results["converged"] = getattr(result_obj, "converged", False)
    results["iters"] = getattr(result_obj, "iters", 0)
    results["max_mismatch"] = getattr(result_obj, "max_mismatch_pu", 0)
    results["slack_p_mw"] = getattr(result_obj, "slack_p_mw", 0)
    results["slack_q_mvar"] = getattr(result_obj, "slack_q_mvar", 0)

    # Bus results
    v_pu_dict = getattr(result_obj, "v_pu", {})
    p_dict = getattr(result_obj, "p_calc_pu", {})
    q_dict = getattr(result_obj, "q_calc_pu", {})
    base_kv = _ss("pf_base_kv", 138)
    base_mva = _ss("pf_base_mva", 100)

    results["buses"] = []
    if case_obj is not None:
        for b in sorted(getattr(case_obj, "buses", []), key=lambda x: x.bus):
            v = v_pu_dict.get(b.bus, 1 + 0j)
            import cmath
            results["buses"].append({
                "bus": b.bus,
                "type": b.type,
                "V_pu": abs(v),
                "V_kV": abs(v) * base_kv,
                "angle_deg": cmath.phase(v) * 180 / 3.14159,
                "P_MW": p_dict.get(b.bus, 0) * base_mva,
                "Q_Mvar": q_dict.get(b.bus, 0) * base_mva,
            })

    # Branch flows
    results["branches"] = []
    for bf in getattr(result_obj, "branch_flows", []):
        results["branches"].append({
            "frm": getattr(bf, "frm", 0),
            "to": getattr(bf, "to", 0),
            "p_mw": getattr(bf, "p_mw", 0),
            "q_mvar": getattr(bf, "q_mvar", 0),
            "p_loss": getattr(bf, "p_loss_mw", 0),
            "q_loss": getattr(bf, "q_loss_mvar", 0),
        })

    return results, cfg

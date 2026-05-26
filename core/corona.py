# bk_estudos_eletricos/core/corona.py
# ======================================================================
# CORRECOES BK_Fixes_v1 (2026-05-26):
#   Fix #1 - peek_corona_loss_kW_km_phase: formula Peek corrigida.
#             ANTES: Pc = 241e-5*(f+25)*((V-Vd)/(m0*delta*r))^2*sqrt(r/D)
#             APOS:  Pc = 241e-5*(f+25)/delta * sqrt(r/D) * (V-Vd)^2
#             Ref.: Peek (1929); Glover,Sarma&Overbye 5a ed. eq.4-98.
#   Fix #2 - campo corona_loss_kW_km_total -> corona_loss_kW_total
#             (armazena kW total do trecho, NAO e kW/km)
# ======================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import math, base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from .cables import Cable, default_cable_db, find_cable
from .line_params import LineGeometry, compute_GMD_for_circuit, ProjectInfo

@dataclass
class CoronaConfig:
    circuit_index: int
    V_LL_kV: float
    f_hz: float
    temp_C: float
    pressure_kPa: float
    altitude_m: float
    weather: str = "normal"
    k_factor: float = 1.1

@dataclass
class CoronaCircuitResult:
    circuit_index: int
    V_phase_kV: float
    Vd_phase_kV: float
    Vd_LL_kV: float
    Ec_crit_kV_cm: float
    Esurface_kV_cm: float
    corona_loss_kW_km_phase: float
    corona_loss_kW_total: float  # FIX #2: era corona_loss_kW_km_total (nome errado)
    delta_air: float
    m0: float
    GMD_m: float
    r_eq_m: float
    r_eq_cm: float
    Ve_surge_LL_kV: float
    margin_Vd_percent: float
    corona_ok: bool
    corona_message: str

def weather_to_m0(weather: str) -> float:
    w = (weather or "").strip().lower()
    if "bril" in w: return 1.0
    if "limp" in w: return 0.98
    if "rug" in w or "velho" in w or "envelhec" in w: return 0.85
    return 0.96

def air_density_factor(temp_C: float, pressure_kPa: float, altitude_m: float) -> float:
    if pressure_kPa is None or pressure_kPa <= 0:
        pressure_kPa = 101.3 * math.exp(-altitude_m / 8150.0)
    p_atm = pressure_kPa / 101.3
    delta = p_atm * 293.0 / (273.0 + temp_C)
    return max(0.3, min(delta, 1.3))

def peek_critical_voltage_phase_kV(r_eq_cm, GMD_cm, m0, delta):
    if r_eq_cm <= 0 or GMD_cm <= r_eq_cm: return 0.0, 0.0
    base_ln = math.log(GMD_cm / r_eq_cm)
    if base_ln <= 0: return 0.0, 0.0
    aux = 1.0 + 0.301 / math.sqrt(max(1e-12, delta * r_eq_cm))
    Vd_phase_kV = 21.1 * m0 * delta * r_eq_cm * aux * base_ln
    Ec_kV_cm = Vd_phase_kV / (r_eq_cm * base_ln) if base_ln > 0 else 0.0
    return Vd_phase_kV, Ec_kV_cm

def peek_corona_loss_kW_km_phase(
    f_hz, V_phase_kV, Vd_phase_kV, r_eq_cm, GMD_cm, m0, delta):
    """
    FIX #1 - Formula de Peek corrigida:
        Pc = 241e-5*(f+25)/delta * sqrt(r/D) * (V-Vd)^2  [kW/km/fase]
    Ref.: Peek (1929); Glover, Sarma & Overbye, 5a ed., eq. 4-98.
    """
    if V_phase_kV <= Vd_phase_kV or r_eq_cm <= 0 or GMD_cm <= 0 or delta <= 0:
        return 0.0
    dV = V_phase_kV - Vd_phase_kV
    Pc = 241e-5 * (f_hz + 25.0) / delta * math.sqrt(r_eq_cm / GMD_cm) * (dV ** 2)
    return max(0.0, Pc)

def _select_phase_A_cable(geom, circuit_index, cable_db):
    phases = geom.phases_of_circuit(circuit_index)
    if not phases: raise ValueError(f"Sem condutor para circuito {circuit_index}.")
    phase_A = phases.get("A") or list(phases.values())[0]
    cable = find_cable(cable_db, phase_A.cable_key)
    if cable is None: raise ValueError(f"Cabo '{phase_A.cable_key}' nao encontrado.")
    GMD_m = compute_GMD_for_circuit(geom, circuit_index)
    try:
        _, r_eq_m = cable.bundle_equivalents(
            n_bundle=max(1, getattr(phase_A, "bundle_n", 1)),
            ds_m=getattr(phase_A, "ds_bundle_m", 0.0))
    except Exception:
        r_eq_m = getattr(cable, "radius_m", 0.0)
    if not r_eq_m or r_eq_m <= 0:
        r_eq_m = getattr(cable, "radius_m", 0.0)
    if r_eq_m <= 0: raise ValueError("Raio invalido.")
    return cable, r_eq_m, GMD_m

def compute_corona_for_circuit(geom, circuit_index, config, cable_db=None, length_km=1.0):
    if cable_db is None: cable_db = default_cable_db()
    _, r_eq_m, GMD_m = _select_phase_A_cable(geom, circuit_index, cable_db)
    if r_eq_m <= 0 or GMD_m <= 0: raise ValueError("Geometria invalida.")
    r_eq_cm, GMD_cm = r_eq_m * 100.0, GMD_m * 100.0
    delta = air_density_factor(config.temp_C, config.pressure_kPa, config.altitude_m)
    m0 = weather_to_m0(config.weather)
    V_phase_kV = config.V_LL_kV / math.sqrt(3.0)
    Vd_phase_kV, Ec_crit_kV_cm = peek_critical_voltage_phase_kV(r_eq_cm, GMD_cm, m0, delta)
    Vd_LL_kV = Vd_phase_kV * math.sqrt(3.0)
    base_ln = math.log(GMD_cm / r_eq_cm) if (GMD_cm > r_eq_cm and r_eq_cm > 0) else 0.0
    Esurface_kV_cm = V_phase_kV / (r_eq_cm * base_ln) if base_ln > 0 else 0.0
    loss_kW_km_phase = peek_corona_loss_kW_km_phase(
        config.f_hz, V_phase_kV, Vd_phase_kV, r_eq_cm, GMD_cm, m0, delta)
    loss_total_kW = loss_kW_km_phase * 3.0 * max(length_km, 0.0)
    Ve_surge_LL_kV = config.k_factor * config.V_LL_kV
    margin_Vd_percent = ((Vd_LL_kV - Ve_surge_LL_kV) / Vd_LL_kV * 100.0) if Vd_LL_kV > 0 else -999.0
    corona_ok = Vd_LL_kV >= Ve_surge_LL_kV
    return CoronaCircuitResult(
        circuit_index=circuit_index,
        V_phase_kV=V_phase_kV, Vd_phase_kV=Vd_phase_kV,
        Vd_LL_kV=Vd_LL_kV, Ec_crit_kV_cm=Ec_crit_kV_cm,
        Esurface_kV_cm=Esurface_kV_cm,
        corona_loss_kW_km_phase=loss_kW_km_phase,
        corona_loss_kW_total=loss_total_kW,
        delta_air=delta, m0=m0, GMD_m=GMD_m, r_eq_m=r_eq_m, r_eq_cm=r_eq_cm,
        Ve_surge_LL_kV=Ve_surge_LL_kV, margin_Vd_percent=margin_Vd_percent,
        corona_ok=corona_ok,
        corona_message="OK" if corona_ok else "FALHA: Vd < Ve_surge.",
    )

def compute_corona_all_circuits(geom, config_by_circuit, cable_db=None, length_km=1.0):
    return {c: compute_corona_for_circuit(geom, c, cfg, cable_db, length_km)
            for c, cfg in config_by_circuit.items()}

if __name__ == "__main__":
    print("corona.py BK_Fixes_v1 OK")

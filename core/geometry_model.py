# bk_estudos_eletricos/core/geometry_model.py
# ==========================================================
# Construção da geometria física da linha de transmissão
# Compatível com IEC / IEEE / práticas ANEEL
# Suporta 1 ou 2 cabos-guarda (para-raios / OPGW)
# ==========================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


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

    def phases_of_circuit(self, circuit_index: int):
        return {
            c.phase: c
            for c in self.conductors
            if c.circuit_index == circuit_index and c.phase in ("A", "B", "C")
        }

    def shields(self) -> List[ConductorInstance]:
        return [c for c in self.conductors if c.is_shield]


def _bool(v) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return v != 0
    if isinstance(v, str): return v.strip().lower() in ("1","true","sim","yes","y")
    return False


def build_geometry_from_home(home: Dict[str, Any]) -> LineGeometry:
    n_circuits  = max(1, int(home.get("n_circuits", 1)))
    bundle_n    = max(1, int(home.get("n_cables_per_phase", 1)))
    bundle_ds   = float(home.get("bundle_spacing_m", 0.4))
    geom_type   = str(home.get("geometry_type", "horizontal")).lower()
    layout      = str(home.get("circuits_layout", "side")).lower()
    h_ref       = float(home.get("ground_clearance_m", 15.0))
    dx_B        = float(home.get("phase_B_dx_m", 8.0))
    dx_C        = float(home.get("phase_C_dx_m", 16.0))
    dv          = float(home.get("phase_vert_spacing_m", 3.0))
    dc          = float(home.get("circuit_spacing_m", 20.0))
    phase_cable  = home.get("cable_phase_key")
    shield_cable = home.get("cable_shield_key")
    shield_present = _bool(home.get("shield_present", True))

    # ── Número de cabos-guarda: 1 ou 2 ────────────────────────────
    n_shield = max(1, min(2, int(home.get("n_shield_wires", 1))))

    # shield_dx_m = deslocamento horizontal de 1 GW em relação ao centro
    # Com 2 GW: eles ficam em x_centro ± shield_dx_m (simétricos)
    shield_dx_offset = float(home.get("shield_dx_m", 0.0) or 0.0)

    conductors: List[ConductorInstance] = []
    prev_c_y = None

    # === FASES ===
    for cidx in range(1, n_circuits + 1):
        if layout == "side":
            base_x   = (cidx - 1) * dc
            base_y_A = h_ref
        else:
            base_x = 0.0
            if cidx == 1:
                base_y_A = h_ref
            else:
                if prev_c_y is None:
                    prev_c_y = h_ref - 2 * dv
                base_y_A = prev_c_y - dc

        if geom_type == "horizontal":
            pos = {"A": (base_x, base_y_A), "B": (base_x + dx_B, base_y_A), "C": (base_x + dx_C, base_y_A)}
        elif geom_type == "vertical":
            pos = {"A": (base_x, base_y_A), "B": (base_x, base_y_A - dv), "C": (base_x, base_y_A - 2*dv)}
        else:  # triangular
            pos = {"A": (base_x, base_y_A + dv), "B": (base_x + dx_B, base_y_A), "C": (base_x - dx_C, base_y_A)}

        prev_c_y = pos["C"][1]

        for ph, (x, y) in pos.items():
            conductors.append(ConductorInstance(
                name=f"C{cidx}_{ph}", cable_key=phase_cable,
                x_m=float(x), y_m=float(y), circuit_index=cidx,
                phase=ph, bundle_n=bundle_n, ds_bundle_m=bundle_ds, is_shield=False,
            ))

    # === CABO(S)-GUARDA ===
    if shield_present and shield_cable:
        phase_pts = [(c.x_m, c.y_m) for c in conductors if not c.is_shield and c.phase in ("A","B","C")]
        x_center  = sum(p[0] for p in phase_pts) / len(phase_pts) if phase_pts else 0.0
        y_max     = max(p[1] for p in phase_pts) if phase_pts else h_ref
        y_gw      = max(h_ref + float(home.get("shield_dy_m", 5.0)), y_max + 0.5)

        if n_shield == 1:
            # 1 GW: centralizado + eventual deslocamento manual
            conductors.append(ConductorInstance(
                name="GW1", cable_key=shield_cable,
                x_m=float(x_center + shield_dx_offset), y_m=float(y_gw),
                circuit_index=1, phase=None, bundle_n=1, is_shield=True,
            ))
        else:
            # 2 GW: simétricos ao centro
            # shield_dx_offset = distância de UM lado ao centro (mínimo 1 m)
            half = max(abs(shield_dx_offset), 1.0)
            for i, sinal in enumerate((-1.0, +1.0), start=1):
                conductors.append(ConductorInstance(
                    name=f"GW{i}", cable_key=shield_cable,
                    x_m=float(x_center + sinal * half), y_m=float(y_gw),
                    circuit_index=1, phase=None, bundle_n=1, is_shield=True,
                ))

    return LineGeometry(conductors)

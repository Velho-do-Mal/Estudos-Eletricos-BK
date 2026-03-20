# bk_estudos_eletricos/core/power_flow.py
# =====================================================================
# Fluxo de Potência AC (Newton-Raphson) – rede multibarras
# - Entrada em ohm/km + km (conversão automática para pu)
# - Base única global (Vbase_LL kV, Sbase MVA)
# - Barras: Slack, PV, PQ (com suporte opcional a limites de Q)
# =====================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import cmath


@dataclass
class Bus:
    bus: int
    type: str  # "SLACK", "PV", "PQ"
    vm_kv: float = 0.0   # magnitude inicial (kV LL) ou Vset (PV)
    va_deg: float = 0.0  # ângulo inicial
    pg_mw: float = 0.0
    qg_mvar: float = 0.0
    pl_mw: float = 0.0
    ql_mvar: float = 0.0
    qmin_mvar: Optional[float] = None
    qmax_mvar: Optional[float] = None

    def net_p_mw(self) -> float:
        return float(self.pg_mw) - float(self.pl_mw)

    def net_q_mvar(self) -> float:
        return float(self.qg_mvar) - float(self.ql_mvar)


@dataclass
class Branch:
    frm: int
    to: int
    length_km: float
    r_ohm_km: float
    x_ohm_km: float
    b_s_km: float = 0.0  # susceptância shunt total por km (S/km) – opcional
    tap: float = 1.0     # tap real no lado "from" (se não houver, 1.0)

    def z_ohm(self) -> complex:
        return complex(self.r_ohm_km, self.x_ohm_km) * float(self.length_km)

    def b_s(self) -> float:
        return float(self.b_s_km) * float(self.length_km)


@dataclass
class PowerFlowCase:
    base_mva: float
    base_kv_ll: float
    buses: List[Bus]
    branches: List[Branch]


@dataclass
class BranchFlow:
    frm: int
    to: int
    p_mw: float
    q_mvar: float
    p_loss_mw: float
    q_loss_mvar: float


@dataclass
class PowerFlowResult:
    converged: bool
    iters: int
    max_mismatch_pu: float
    v_pu: Dict[int, complex]
    p_calc_pu: Dict[int, float]
    q_calc_pu: Dict[int, float]
    slack_p_mw: float
    slack_q_mvar: float
    branch_flows: List[BranchFlow]


def _zbase_ohm(base_kv_ll: float, base_mva: float) -> float:
    v = float(base_kv_ll)
    s = float(base_mva)
    if v <= 0 or s <= 0:
        return 1.0
    return (v * v) / s


def _ybase_s(base_kv_ll: float, base_mva: float) -> float:
    z = _zbase_ohm(base_kv_ll, base_mva)
    return 1.0 / z if z != 0 else 1.0


def build_ybus(case: PowerFlowCase) -> Tuple[List[int], List[List[complex]]]:
    bus_ids = sorted({b.bus for b in case.buses})
    n = len(bus_ids)
    idx = {bus_id: i for i, bus_id in enumerate(bus_ids)}

    Y = [[0j for _ in range(n)] for _ in range(n)]

    zbase = _zbase_ohm(case.base_kv_ll, case.base_mva)
    ybase = _ybase_s(case.base_kv_ll, case.base_mva)

    for br in case.branches:
        if br.frm not in idx or br.to not in idx:
            continue

        z_ohm = br.z_ohm()
        if abs(z_ohm) <= 0:
            continue
        z_pu = z_ohm / zbase
        y_series = 1.0 / z_pu

        b_total_s = br.b_s()
        b_total_pu = (b_total_s / ybase) if ybase != 0 else 0.0
        y_shunt = 1j * b_total_pu / 2.0

        tap = float(br.tap) if br.tap else 1.0
        if tap == 0:
            tap = 1.0

        i = idx[br.frm]
        k = idx[br.to]

        Y[i][i] += (y_series + y_shunt) / (tap * tap)
        Y[k][k] += (y_series + y_shunt)
        Y[i][k] -= y_series / tap
        Y[k][i] -= y_series / tap

    return bus_ids, Y


def _calc_power_injections(bus_ids: List[int], Y: List[List[complex]], V: List[complex]) -> Tuple[List[float], List[float]]:
    n = len(bus_ids)
    P = [0.0] * n
    Q = [0.0] * n
    for i in range(n):
        Ii = 0j
        for k in range(n):
            Ii += Y[i][k] * V[k]
        Si = V[i] * (Ii.conjugate())
        P[i] = Si.real
        Q[i] = Si.imag
    return P, Q


def _solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    if n == 0:
        return []
    M = [row[:] for row in A]
    x = b[:]

    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[piv][k]) < 1e-14:
            return [0.0] * n
        if piv != k:
            M[k], M[piv] = M[piv], M[k]
            x[k], x[piv] = x[piv], x[k]

        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            if f == 0:
                continue
            x[i] -= f * x[k]
            for j in range(k, n):
                M[i][j] -= f * M[k][j]

    sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = x[i]
        for j in range(i + 1, n):
            s -= M[i][j] * sol[j]
        sol[i] = s / M[i][i]
    return sol


def solve_power_flow_newton(
    case: PowerFlowCase,
    tol_pu: float = 1e-7,
    max_iter: int = 30,
    enforce_q_limits: bool = True,
) -> PowerFlowResult:
    bus_ids, Y = build_ybus(case)
    n = len(bus_ids)
    buses_by_id: Dict[int, Bus] = {b.bus: b for b in case.buses}
    types = [buses_by_id[i].type.strip().upper() for i in bus_ids]

    pv = [i for i, t in enumerate(types) if t == "PV"]
    pq = [i for i, t in enumerate(types) if t == "PQ"]
    slack = [i for i, t in enumerate(types) if t == "SLACK"]
    if not slack:
        slack = [0]
        types[0] = "SLACK"
        pv = [i for i, t in enumerate(types) if t == "PV"]
        pq = [i for i, t in enumerate(types) if t == "PQ"]

    V: List[complex] = []
    for bid in bus_ids:
        b = buses_by_id[bid]
        vm_pu = (b.vm_kv / case.base_kv_ll) if case.base_kv_ll else 1.0
        if vm_pu <= 0:
            vm_pu = 1.0
        va = math.radians(float(b.va_deg))
        V.append(vm_pu * cmath.exp(1j * va))

    P_spec = [0.0] * n
    Q_spec = [0.0] * n
    for i, bid in enumerate(bus_ids):
        b = buses_by_id[bid]
        P_spec[i] = b.net_p_mw() / case.base_mva
        Q_spec[i] = b.net_q_mvar() / case.base_mva

    angle_vars = [i for i in range(n) if i not in slack]
    vm_vars = pq[:]

    def mismatch(P_calc: List[float], Q_calc: List[float]) -> List[float]:
        mis = []
        for i in angle_vars:
            mis.append(P_spec[i] - P_calc[i])
        for i in vm_vars:
            mis.append(Q_spec[i] - Q_calc[i])
        return mis

    converged = False
    iters = 0
    max_mis = 1e9

    for it in range(1, max_iter + 1):
        iters = it
        P_calc, Q_calc = _calc_power_injections(bus_ids, Y, V)
        mis = mismatch(P_calc, Q_calc)
        max_mis = max(abs(x) for x in mis) if mis else 0.0
        if max_mis < tol_pu:
            converged = True
            break

        Vm = [abs(v) for v in V]
        Va = [cmath.phase(v) for v in V]
        G = [[Y[i][k].real for k in range(n)] for i in range(n)]
        B = [[Y[i][k].imag for k in range(n)] for i in range(n)]

        na = len(angle_vars)
        nv = len(vm_vars)
        dim = na + nv
        J = [[0.0 for _ in range(dim)] for _ in range(dim)]
        a_pos = {bus_i: pos for pos, bus_i in enumerate(angle_vars)}
        v_pos = {bus_i: pos for pos, bus_i in enumerate(vm_vars)}

        # H e N
        for i in angle_vars:
            for k in angle_vars:
                if i == k:
                    s = 0.0
                    for m in range(n):
                        s += Vm[i] * Vm[m] * (-G[i][m] * math.sin(Va[i] - Va[m]) + B[i][m] * math.cos(Va[i] - Va[m]))
                    J[a_pos[i]][a_pos[k]] = s - (Vm[i] ** 2) * B[i][i]
                else:
                    J[a_pos[i]][a_pos[k]] = Vm[i] * Vm[k] * (G[i][k] * math.sin(Va[i] - Va[k]) - B[i][k] * math.cos(Va[i] - Va[k]))

            for k in vm_vars:
                col = na + v_pos[k]
                if i == k:
                    s = 0.0
                    for m in range(n):
                        s += Vm[m] * (G[i][m] * math.cos(Va[i] - Va[m]) + B[i][m] * math.sin(Va[i] - Va[m]))
                    J[a_pos[i]][col] = s + G[i][i] * Vm[i]
                else:
                    J[a_pos[i]][col] = Vm[i] * (G[i][k] * math.cos(Va[i] - Va[k]) + B[i][k] * math.sin(Va[i] - Va[k]))

        # M e L
        for i in vm_vars:
            row = na + v_pos[i]
            for k in angle_vars:
                if i == k:
                    s = 0.0
                    for m in range(n):
                        s += Vm[i] * Vm[m] * (G[i][m] * math.cos(Va[i] - Va[m]) + B[i][m] * math.sin(Va[i] - Va[m]))
                    J[row][a_pos[k]] = -s + (Vm[i] ** 2) * G[i][i]
                else:
                    J[row][a_pos[k]] = -Vm[i] * Vm[k] * (G[i][k] * math.cos(Va[i] - Va[k]) + B[i][k] * math.sin(Va[i] - Va[k]))

            for k in vm_vars:
                col = na + v_pos[k]
                if i == k:
                    s = 0.0
                    for m in range(n):
                        s += Vm[m] * (G[i][m] * math.sin(Va[i] - Va[m]) - B[i][m] * math.cos(Va[i] - Va[m]))
                    J[row][col] = s - B[i][i] * Vm[i]
                else:
                    J[row][col] = Vm[i] * (G[i][k] * math.sin(Va[i] - Va[k]) - B[i][k] * math.cos(Va[i] - Va[k]))

        dx = _solve_linear(J, mis)

        for pos, i in enumerate(angle_vars):
            Va_i = cmath.phase(V[i]) + dx[pos]
            V[i] = abs(V[i]) * cmath.exp(1j * Va_i)

        for pos, i in enumerate(vm_vars):
            Vm_i = abs(V[i]) + dx[na + pos]
            Vm_i = max(0.2, Vm_i)
            V[i] = Vm_i * cmath.exp(1j * cmath.phase(V[i]))

        for i in pv:
            b = buses_by_id[bus_ids[i]]
            vset_pu = (b.vm_kv / case.base_kv_ll) if case.base_kv_ll else 1.0
            V[i] = max(0.2, vset_pu) * cmath.exp(1j * cmath.phase(V[i]))

        if enforce_q_limits and pv:
            P_calc, Q_calc = _calc_power_injections(bus_ids, Y, V)
            changed = False
            for i in list(pv):
                b = buses_by_id[bus_ids[i]]
                if b.qmin_mvar is None and b.qmax_mvar is None:
                    continue
                q_mvar = Q_calc[i] * case.base_mva
                if b.qmax_mvar is not None and q_mvar > b.qmax_mvar + 1e-6:
                    Q_spec[i] = b.qmax_mvar / case.base_mva
                    types[i] = "PQ"
                    changed = True
                elif b.qmin_mvar is not None and q_mvar < b.qmin_mvar - 1e-6:
                    Q_spec[i] = b.qmin_mvar / case.base_mva
                    types[i] = "PQ"
                    changed = True
            if changed:
                pv = [i for i, t in enumerate(types) if t == "PV"]
                pq = [i for i, t in enumerate(types) if t == "PQ"]
                angle_vars = [i for i in range(n) if i not in slack]
                vm_vars = pq[:]

    P_calc, Q_calc = _calc_power_injections(bus_ids, Y, V)
    v_map = {bus_ids[i]: V[i] for i in range(n)}
    p_map = {bus_ids[i]: P_calc[i] for i in range(n)}
    q_map = {bus_ids[i]: Q_calc[i] for i in range(n)}

    slack_idx = slack[0]
    slack_p = P_calc[slack_idx] * case.base_mva
    slack_q = Q_calc[slack_idx] * case.base_mva

    flows = compute_branch_flows(case, bus_ids, V)

    return PowerFlowResult(
        converged=converged,
        iters=iters,
        max_mismatch_pu=float(max_mis),
        v_pu=v_map,
        p_calc_pu=p_map,
        q_calc_pu=q_map,
        slack_p_mw=float(slack_p),
        slack_q_mvar=float(slack_q),
        branch_flows=flows,
    )


def compute_branch_flows(case: PowerFlowCase, bus_ids: List[int], V: List[complex]) -> List[BranchFlow]:
    idx = {bid: i for i, bid in enumerate(bus_ids)}
    zbase = _zbase_ohm(case.base_kv_ll, case.base_mva)
    ybase = _ybase_s(case.base_kv_ll, case.base_mva)

    flows: List[BranchFlow] = []
    for br in case.branches:
        if br.frm not in idx or br.to not in idx:
            continue
        i = idx[br.frm]
        k = idx[br.to]

        z_ohm = br.z_ohm()
        if abs(z_ohm) <= 0:
            continue
        z_pu = z_ohm / zbase
        y = 1.0 / z_pu

        b_total_s = br.b_s()
        b_total_pu = (b_total_s / ybase) if ybase != 0 else 0.0
        ysh = 1j * b_total_pu / 2.0

        tap = float(br.tap) if br.tap else 1.0
        if tap == 0:
            tap = 1.0

        Vi = V[i]
        Vk = V[k]

        Iik = (Vi / tap - Vk) * y + (Vi / tap) * ysh
        Iki = (Vk - Vi / tap) * y + Vk * ysh

        Sik = Vi * Iik.conjugate()
        Ski = Vk * Iki.conjugate()

        p_ik = Sik.real * case.base_mva
        q_ik = Sik.imag * case.base_mva
        p_ki = Ski.real * case.base_mva
        q_ki = Ski.imag * case.base_mva

        flows.append(
            BranchFlow(
                frm=br.frm,
                to=br.to,
                p_mw=float(p_ik),
                q_mvar=float(q_ik),
                p_loss_mw=float(p_ik + p_ki),
                q_loss_mvar=float(q_ik + q_ki),
            )
        )
    return flows


def generate_html_report_power_flow(project_name: str, case: PowerFlowCase, result: PowerFlowResult) -> str:
    rows_bus = []
    for b in sorted(case.buses, key=lambda x: x.bus):
        v = result.v_pu.get(b.bus, 1 + 0j)
        vm = abs(v) * case.base_kv_ll
        va = math.degrees(cmath.phase(v))
        p = result.p_calc_pu.get(b.bus, 0.0) * case.base_mva
        q = result.q_calc_pu.get(b.bus, 0.0) * case.base_mva
        rows_bus.append(
            f"<tr><td>{b.bus}</td><td>{b.type}</td><td>{vm:.3f}</td><td>{va:.3f}</td>"
            f"<td>{p:.3f}</td><td>{q:.3f}</td><td>{b.net_p_mw():.3f}</td><td>{b.net_q_mvar():.3f}</td></tr>"
        )

    rows_br = []
    for f in result.branch_flows:
        rows_br.append(
            f"<tr><td>{f.frm}</td><td>{f.to}</td><td>{f.p_mw:.3f}</td><td>{f.q_mvar:.3f}</td>"
            f"<td>{f.p_loss_mw:.6f}</td><td>{f.q_loss_mvar:.6f}</td></tr>"
        )

    status = "CONVERGIU" if result.converged else "NÃO CONVERGIU"
    return f"""
    <h2>Fluxo de Potência – Rede Multibarras</h2>
    <p><b>Projeto:</b> {project_name}</p>
    <p><b>Base:</b> {case.base_mva:.3f} MVA &nbsp;&nbsp; | &nbsp;&nbsp; {case.base_kv_ll:.3f} kV (L-L)</p>
    <p><b>Status:</b> {status} &nbsp;&nbsp; | &nbsp;&nbsp; Iterações: {result.iters} &nbsp;&nbsp; | &nbsp;&nbsp; Máx. mismatch (pu): {result.max_mismatch_pu:.3e}</p>
    <p><b>Slack:</b> P = {result.slack_p_mw:.3f} MW &nbsp;&nbsp; Q = {result.slack_q_mvar:.3f} MVAr</p>

    <h3>Resultados por Barra</h3>
    <table>
      <tr><th>Barra</th><th>Tipo</th><th>V (kV)</th><th>Ângulo (°)</th><th>P calc (MW)</th><th>Q calc (MVAr)</th><th>P esp (MW)</th><th>Q esp (MVAr)</th></tr>
      {''.join(rows_bus)}
    </table>

    <h3>Fluxos por Ramo (sentido from→to)</h3>
    <table>
      <tr><th>De</th><th>Para</th><th>P (MW)</th><th>Q (MVAr)</th><th>Perda P (MW)</th><th>Perda Q (MVAr)</th></tr>
      {''.join(rows_br)}
    </table>
    """

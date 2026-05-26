# bk_estudos_eletricos/core/power_flow.py
# =====================================================================
# CORREÇÕES BK_Fixes_v1:
#   Fix #5 – build_ybus: susceptância shunt (b_s) do modelo pi da linha
#             era dividida por tap² junto com a série, o que é incorreto.
#             Para transformadores com tap != 1, apenas a admitância série
#             é afetada pelo tap; a carga shunt conecta-se diretamente ao terminal.
#             ANTES: Y[i][i] += (y_series + y_shunt) / tap²
#             APÓS:  Y[i][i] += y_series/tap²  +  y_shunt
#             (Para linhas sem transformador, tap=1 e não há diferença.)
#
#   Fix #6 – _solve_linear: retornava [0,0,...,0] silenciosamente para
#             Jacobiano singular (ilhamento, topologia incorreta).
#             Agora lança RuntimeWarning e retorna lista de NaN, o que
#             faz o Newton-Raphson encerrar sem convergência com mensagem clara.
#
#   Fix #7 – solve_power_flow_newton: barras PV convertidas a PQ por
#             violação de limite Q nunca revertiam para PV quando Q
#             retornava ao intervalo [Qmin, Qmax]. Adicionado rastreamento
#             de barras originalmente PV e lógica de reversão.
# =====================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import math, cmath, warnings

@dataclass
class Bus:
    bus: int
    type: str   # "SLACK", "PV", "PQ"
    vm_kv: float = 0.0
    va_deg: float = 0.0
    pg_mw: float = 0.0
    qg_mvar: float = 0.0
    pl_mw: float = 0.0
    ql_mvar: float = 0.0
    qmin_mvar: Optional[float] = None
    qmax_mvar: Optional[float] = None

    def net_p_mw(self)   -> float: return float(self.pg_mw)  - float(self.pl_mw)
    def net_q_mvar(self) -> float: return float(self.qg_mvar) - float(self.ql_mvar)

@dataclass
class Branch:
    frm: int
    to: int
    length_km: float
    r_ohm_km: float
    x_ohm_km: float
    b_s_km: float = 0.0
    tap: float = 1.0

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
    frm: int; to: int
    p_mw: float; q_mvar: float
    p_loss_mw: float; q_loss_mvar: float

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

def _zbase_ohm(base_kv_ll, base_mva):
    v = float(base_kv_ll); s = float(base_mva)
    return (v*v)/s if v>0 and s>0 else 1.0

def _ybase_s(base_kv_ll, base_mva):
    z = _zbase_ohm(base_kv_ll, base_mva)
    return 1.0/z if z != 0 else 1.0

def build_ybus(case: PowerFlowCase) -> Tuple[List[int], List[List[complex]]]:
    bus_ids = sorted({b.bus for b in case.buses})
    n       = len(bus_ids)
    idx     = {bid: i for i, bid in enumerate(bus_ids)}
    Y       = [[0j]*n for _ in range(n)]
    zbase   = _zbase_ohm(case.base_kv_ll, case.base_mva)
    ybase   = _ybase_s(case.base_kv_ll, case.base_mva)

    for br in case.branches:
        if br.frm not in idx or br.to not in idx: continue
        z_ohm = br.z_ohm()
        if abs(z_ohm) <= 0: continue
        z_pu     = z_ohm / zbase
        y_series = 1.0 / z_pu

        b_total_s  = br.b_s()
        b_total_pu = (b_total_s / ybase) if ybase != 0 else 0.0
        y_shunt    = 1j * b_total_pu / 2.0   # metade em cada terminal (modelo π)

        tap = float(br.tap) if br.tap else 1.0
        if tap == 0: tap = 1.0

        i = idx[br.frm]; k = idx[br.to]

        # FIX #5: y_shunt NAO é dividido por tap².
        # Modelo pi do transformador com tap no lado "from":
        #   Y_ii += y_series/tap²  +  y_shunt   (shunt no terminal "from")
        #   Y_kk += y_series       +  y_shunt   (shunt no terminal "to")
        #   Y_ik -= y_series/tap
        #   Y_ki -= y_series/tap   (assumindo tap real; para tap complexo usar conj(tap))
        Y[i][i] += y_series / (tap * tap) + y_shunt  # FIX #5
        Y[k][k] += y_series               + y_shunt  # FIX #5
        Y[i][k] -= y_series / tap
        Y[k][i] -= y_series / tap

    return bus_ids, Y

def _calc_power_injections(bus_ids, Y, V):
    n  = len(bus_ids)
    P  = [0.0]*n; Q = [0.0]*n
    for i in range(n):
        Ii = sum(Y[i][k]*V[k] for k in range(n))
        Si = V[i] * Ii.conjugate()
        P[i] = Si.real; Q[i] = Si.imag
    return P, Q

def _solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Eliminação gaussiana com pivotamento parcial.

    FIX #6: Jacobiano singular (pivot < 1e-14) agora emite RuntimeWarning
    e retorna lista de NaN, em vez de retornar zeros silenciosamente.
    O Newton-Raphson detecta NaN no vetor de correção, encerra as iterações
    e retorna converged=False com mensagem informativa.
    """
    n = len(b)
    if n == 0: return []
    M = [row[:] for row in A]
    x = b[:]

    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[piv][k]) < 1e-14:
            warnings.warn(
                f"Jacobiano singular detectado na coluna {k} do solver linear "
                f"(|pivot| = {abs(M[piv][k]):.2e}). Verifique topologia da rede "
                f"(ilhamento, barra isolada ou tipo de barra incorreto).",
                RuntimeWarning, stacklevel=2,
            )
            return [float("nan")] * n   # FIX #6
        if piv != k:
            M[k], M[piv] = M[piv], M[k]
            x[k], x[piv] = x[piv], x[k]
        for i in range(k+1, n):
            f = M[i][k] / M[k][k]
            if f == 0: continue
            x[i] -= f*x[k]
            for j in range(k, n):
                M[i][j] -= f*M[k][j]

    sol = [0.0]*n
    for i in range(n-1, -1, -1):
        s = x[i]
        for j in range(i+1, n):
            s -= M[i][j]*sol[j]
        sol[i] = s / M[i][i]
    return sol

def solve_power_flow_newton(
    case: PowerFlowCase,
    tol_pu: float = 1e-7,
    max_iter: int = 30,
    enforce_q_limits: bool = True,
) -> PowerFlowResult:
    bus_ids, Y    = build_ybus(case)
    n             = len(bus_ids)
    buses_by_id   = {b.bus: b for b in case.buses}
    types         = [buses_by_id[i].type.strip().upper() for i in bus_ids]

    slack = [i for i, t in enumerate(types) if t == "SLACK"]
    if not slack:
        slack    = [0]; types[0] = "SLACK"
    pv   = [i for i, t in enumerate(types) if t == "PV"]
    pq   = [i for i, t in enumerate(types) if t == "PQ"]

    # FIX #7: rastrear quais índices eram originalmente PV
    original_pv: Set[int] = set(pv)

    V: List[complex] = []
    for bid in bus_ids:
        b    = buses_by_id[bid]
        vm   = (b.vm_kv / case.base_kv_ll) if case.base_kv_ll else 1.0
        if vm <= 0: vm = 1.0
        va   = math.radians(float(b.va_deg))
        V.append(vm * cmath.exp(1j*va))

    P_spec = [buses_by_id[bid].net_p_mw()   / case.base_mva for bid in bus_ids]
    Q_spec = [buses_by_id[bid].net_q_mvar() / case.base_mva for bid in bus_ids]

    angle_vars = [i for i in range(n) if i not in slack]
    vm_vars    = pq[:]

    def mismatch(P_calc, Q_calc):
        return ([P_spec[i]-P_calc[i] for i in angle_vars] +
                [Q_spec[i]-Q_calc[i] for i in vm_vars])

    converged = False; iters = 0; max_mis = 1e9

    for it in range(1, max_iter+1):
        iters   = it
        P_calc, Q_calc = _calc_power_injections(bus_ids, Y, V)
        mis     = mismatch(P_calc, Q_calc)
        max_mis = max(abs(x) for x in mis) if mis else 0.0
        if max_mis < tol_pu:
            converged = True; break

        Vm  = [abs(v) for v in V]
        Va  = [cmath.phase(v) for v in V]
        G   = [[Y[i][k].real for k in range(n)] for i in range(n)]
        B   = [[Y[i][k].imag for k in range(n)] for i in range(n)]

        na     = len(angle_vars); nv = len(vm_vars)
        dim    = na + nv
        J      = [[0.0]*dim for _ in range(dim)]
        a_pos  = {bi: p for p, bi in enumerate(angle_vars)}
        v_pos  = {bi: p for p, bi in enumerate(vm_vars)}

        # Submatriz H (dP/dθ) e N (dP/d|V|)
        for i in angle_vars:
            for k in angle_vars:
                if i == k:
                    s = sum(Vm[i]*Vm[m]*(-G[i][m]*math.sin(Va[i]-Va[m])
                            + B[i][m]*math.cos(Va[i]-Va[m])) for m in range(n))
                    J[a_pos[i]][a_pos[k]] = s - (Vm[i]**2)*B[i][i]
                else:
                    J[a_pos[i]][a_pos[k]] = Vm[i]*Vm[k]*(
                        G[i][k]*math.sin(Va[i]-Va[k]) - B[i][k]*math.cos(Va[i]-Va[k]))
            for k in vm_vars:
                col = na + v_pos[k]
                if i == k:
                    s = sum(Vm[m]*(G[i][m]*math.cos(Va[i]-Va[m])
                            + B[i][m]*math.sin(Va[i]-Va[m])) for m in range(n))
                    J[a_pos[i]][col] = s + G[i][i]*Vm[i]
                else:
                    J[a_pos[i]][col] = Vm[i]*(
                        G[i][k]*math.cos(Va[i]-Va[k]) + B[i][k]*math.sin(Va[i]-Va[k]))

        # Submatriz M (dQ/dθ) e L (dQ/d|V|)
        for i in vm_vars:
            row = na + v_pos[i]
            for k in angle_vars:
                if i == k:
                    s = sum(Vm[i]*Vm[m]*(G[i][m]*math.cos(Va[i]-Va[m])
                            + B[i][m]*math.sin(Va[i]-Va[m])) for m in range(n))
                    J[row][a_pos[k]] = -s + (Vm[i]**2)*G[i][i]
                else:
                    J[row][a_pos[k]] = -Vm[i]*Vm[k]*(
                        G[i][k]*math.cos(Va[i]-Va[k]) + B[i][k]*math.sin(Va[i]-Va[k]))
            for k in vm_vars:
                col = na + v_pos[k]
                if i == k:
                    s = sum(Vm[m]*(G[i][m]*math.sin(Va[i]-Va[m])
                            - B[i][m]*math.cos(Va[i]-Va[m])) for m in range(n))
                    J[row][col] = s - B[i][i]*Vm[i]
                else:
                    J[row][col] = Vm[i]*(
                        G[i][k]*math.sin(Va[i]-Va[k]) - B[i][k]*math.cos(Va[i]-Va[k]))

        dx = _solve_linear(J, mis)

        # FIX #6: abortar se solver retornou NaN (Jacobiano singular)
        if any(math.isnan(v) for v in dx):
            break

        for pos, i in enumerate(angle_vars):
            Va_i  = cmath.phase(V[i]) + dx[pos]
            V[i]  = abs(V[i]) * cmath.exp(1j*Va_i)
        for pos, i in enumerate(vm_vars):
            Vm_i  = max(0.2, abs(V[i]) + dx[na+pos])
            V[i]  = Vm_i * cmath.exp(1j*cmath.phase(V[i]))

        # Enforçar tensão nas barras PV (regulação de tensão)
        for i in pv:
            b    = buses_by_id[bus_ids[i]]
            vset = max(0.2, (b.vm_kv / case.base_kv_ll) if case.base_kv_ll else 1.0)
            V[i] = vset * cmath.exp(1j*cmath.phase(V[i]))

        # FIX #7 – Controle de limites Q com reversão PQ → PV
        if enforce_q_limits:
            P_calc, Q_calc = _calc_power_injections(bus_ids, Y, V)
            changed = False

            # Conversão PV → PQ por violação de limite
            for i in list(pv):
                b = buses_by_id[bus_ids[i]]
                if b.qmin_mvar is None and b.qmax_mvar is None: continue
                q_mvar = Q_calc[i] * case.base_mva
                if b.qmax_mvar is not None and q_mvar > b.qmax_mvar + 1e-6:
                    Q_spec[i] = b.qmax_mvar / case.base_mva
                    types[i]  = "PQ"; changed = True
                elif b.qmin_mvar is not None and q_mvar < b.qmin_mvar - 1e-6:
                    Q_spec[i] = b.qmin_mvar / case.base_mva
                    types[i]  = "PQ"; changed = True

            # FIX #7 – Reversão PQ → PV quando Q retornar ao intervalo permitido
            for i in original_pv:
                if types[i] != "PQ": continue   # já é PV ou SLACK
                b = buses_by_id[bus_ids[i]]
                if b.qmin_mvar is None and b.qmax_mvar is None: continue
                q_mvar = Q_calc[i] * case.base_mva
                qmin   = b.qmin_mvar if b.qmin_mvar is not None else -1e9
                qmax   = b.qmax_mvar if b.qmax_mvar is not None else  1e9
                if qmin - 1e-6 <= q_mvar <= qmax + 1e-6:
                    types[i]  = "PV"
                    Q_spec[i] = b.net_q_mvar() / case.base_mva
                    changed   = True

            if changed:
                pv         = [i for i, t in enumerate(types) if t == "PV"]
                pq         = [i for i, t in enumerate(types) if t == "PQ"]
                angle_vars = [i for i in range(n) if i not in slack]
                vm_vars    = pq[:]

    P_calc, Q_calc = _calc_power_injections(bus_ids, Y, V)
    slack_idx  = slack[0]
    flows      = compute_branch_flows(case, bus_ids, V)

    return PowerFlowResult(
        converged=converged, iters=iters,
        max_mismatch_pu=float(max_mis),
        v_pu={bus_ids[i]: V[i] for i in range(n)},
        p_calc_pu={bus_ids[i]: P_calc[i] for i in range(n)},
        q_calc_pu={bus_ids[i]: Q_calc[i] for i in range(n)},
        slack_p_mw=float(P_calc[slack_idx]*case.base_mva),
        slack_q_mvar=float(Q_calc[slack_idx]*case.base_mva),
        branch_flows=flows,
    )

def compute_branch_flows(case: PowerFlowCase, bus_ids: List[int], V: List[complex]) -> List[BranchFlow]:
    idx   = {bid: i for i, bid in enumerate(bus_ids)}
    zbase = _zbase_ohm(case.base_kv_ll, case.base_mva)
    ybase = _ybase_s(case.base_kv_ll, case.base_mva)
    flows: List[BranchFlow] = []
    for br in case.branches:
        if br.frm not in idx or br.to not in idx: continue
        z_ohm = br.z_ohm()
        if abs(z_ohm) <= 0: continue
        z_pu    = z_ohm / zbase
        y       = 1.0 / z_pu
        b_pu    = (br.b_s() / ybase) if ybase != 0 else 0.0
        ysh     = 1j * b_pu / 2.0
        tap     = float(br.tap) if br.tap else 1.0
        if tap == 0: tap = 1.0
        i = idx[br.frm]; k = idx[br.to]
        Vi = V[i]; Vk = V[k]
        # FIX #5 aplicado também ao cálculo de fluxo
        Iik = (Vi/tap - Vk)*y + (Vi/tap)*ysh
        Iki = (Vk - Vi/tap)*y + Vk*ysh
        Sik = Vi * Iik.conjugate()
        Ski = Vk * Iki.conjugate()
        flows.append(BranchFlow(
            frm=br.frm, to=br.to,
            p_mw=float(Sik.real*case.base_mva),
            q_mvar=float(Sik.imag*case.base_mva),
            p_loss_mw=float((Sik.real+Ski.real)*case.base_mva),
            q_loss_mvar=float((Sik.imag+Ski.imag)*case.base_mva),
        ))
    return flows

def generate_html_report_power_flow(project_name: str, case: PowerFlowCase, result: PowerFlowResult) -> str:
    rows_bus = []
    for b in sorted(case.buses, key=lambda x: x.bus):
        v   = result.v_pu.get(b.bus, 1+0j)
        vm  = abs(v)*case.base_kv_ll
        va  = math.degrees(cmath.phase(v))
        p   = result.p_calc_pu.get(b.bus, 0.0)*case.base_mva
        q   = result.q_calc_pu.get(b.bus, 0.0)*case.base_mva
        rows_bus.append(
            f"<tr><td>{b.bus}</td><td>{b.type}</td><td>{vm:.3f}</td><td>{va:.3f}</td>"
            f"<td>{p:.3f}</td><td>{q:.3f}</td><td>{b.net_p_mw():.3f}</td><td>{b.net_q_mvar():.3f}</td></tr>")
    rows_br = []
    for f in result.branch_flows:
        rows_br.append(
            f"<tr><td>{f.frm}</td><td>{f.to}</td><td>{f.p_mw:.3f}</td><td>{f.q_mvar:.3f}</td>"
            f"<td>{f.p_loss_mw:.6f}</td><td>{f.q_loss_mvar:.6f}</td></tr>")
    status = "CONVERGIU" if result.converged else "NAO CONVERGIU"
    return f"""<h2>Fluxo de Potência – Rede Multibarras</h2>
<p><b>Projeto:</b> {project_name}</p>
<p><b>Base:</b> {case.base_mva:.3f} MVA | {case.base_kv_ll:.3f} kV (L-L)</p>
<p><b>Status:</b> {status} | Iterações: {result.iters} | Max mismatch: {result.max_mismatch_pu:.3e} pu</p>
<p><b>Slack:</b> P = {result.slack_p_mw:.3f} MW  Q = {result.slack_q_mvar:.3f} MVAr</p>
<h3>Barras</h3><table>
<tr><th>Barra</th><th>Tipo</th><th>V (kV)</th><th>Ang (°)</th>
    <th>P calc (MW)</th><th>Q calc (MVAr)</th><th>P esp (MW)</th><th>Q esp (MVAr)</th></tr>
{''.join(rows_bus)}</table>
<h3>Ramos (from→to)</h3><table>
<tr><th>De</th><th>Para</th><th>P (MW)</th><th>Q (MVAr)</th>
    <th>Perda P (MW)</th><th>Perda Q (MVAr)</th></tr>
{''.join(rows_br)}</table>"""

if __name__ == "__main__":
    print("power_flow_fixed.py OK — importe como módulo para uso.")

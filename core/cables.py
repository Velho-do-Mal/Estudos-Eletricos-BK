# cables.py
# Módulo de cabos para o BK_Estudos_Eletricos
# - Banco de cabos (JSON ou embutido)
# - Cálculo de resistência AC, indutância e capacitância por unidade de comprimento
# - Suporte a feixes (bundle)

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import json
import math

# ======================= Constantes físicas ==========================
EPS0 = 8.854187817e-12   # F/m
MU0  = 4 * math.pi * 1e-7  # H/m

# ======================= Funções auxiliares =========================
def mm_to_m(x: float) -> float:
    return x / 1000.0

def m_to_mm(x: float) -> float:
    return x * 1000.0

def temp_resistivity(rho20: float, T: float, alpha: float = 0.00393) -> float:
    """
    Calcula a resistividade em função da temperatura:
        ρ(T) = ρ(20°C) * [1 + α*(T - 20)]
    alpha ~ 0.00393 para cobre / alumínio (aprox. genérica).
    """
    return rho20 * (1.0 + alpha * (T - 20.0))

def skin_depth(f_hz: float, mu: float, sigma: float) -> float:
    """
    Profundidade de penetração (skin depth) em condutor cilíndrico:
        δ = sqrt(2 / (ω μ σ))
    """
    w = 2 * math.pi * f_hz
    if w <= 0 or mu <= 0 or sigma <= 0:
        return float("inf")
    return math.sqrt(2.0 / (w * mu * sigma))

def ac_resistance_from_skin(
    Rdc_per_m: float,
    radius_m: float,
    depth_m: float
) -> float:
    """
    Ajuste simples de resistência AC a partir da DC e skin depth.
    Modelo simplificado:
        R_ac ≈ R_dc * (1 + r / (2 δ)), limitado em >= R_dc.
    """
    if depth_m <= 0 or radius_m <= 0:
        return Rdc_per_m
    Rac = Rdc_per_m * (1.0 + max(0.0, radius_m / (2.0 * depth_m)))
    return max(Rdc_per_m, Rac)

def bundle_equivalents(
    GMR_m: float,
    r_m: float,
    n: int,
    ds_m: float
) -> Tuple[float, float]:
    """
    Equivalente de feixe (bundle):
      - GMR_eq
      - raio equivalente r_eq
    Para n=1, devolve o próprio GMR e raio.
    Fórmulas clássicas:
       GMR_eq = (GMR * ds^(n-1))^(1/n)
       r_eq   = (r   * ds^(n-1))^(1/n)
    """
    if n <= 1:
        return GMR_m, r_m
    # proteção contra ds_m == 0
    ds_m_safe = max(ds_m, 1e-6)
    GMR_eq = (GMR_m * (ds_m_safe ** (n - 1))) ** (1.0 / n)
    r_eq   = (r_m   * (ds_m_safe ** (n - 1))) ** (1.0 / n)
    return GMR_eq, r_eq

# ======================= Materiais / Resistividade ===================
def rho_copper_20C() -> float:
    # Valor típico ~ 1.724e-8 Ω·m a 20°C
    return 1.724e-8

def rho_al_20C() -> float:
    # Valor típico ~ 2.826e-8 Ω·m a 20°C
    return 2.826e-8

def rho_acsr_20C() -> float:
    # ACSR é composto Al + Aço. Valor efetivo aproximado.
    return 2.1e-8

def rho_steel_20C() -> float:
    # Valor aproximado para aços comuns (~1.43e-7 Ω·m)
    return 1.43e-7

def guess_rho_from_material(material: str) -> float:
    material = (material or "").strip().lower()
    if "cu" in material:
        return rho_copper_20C()
    if "al" in material and "acsr" not in material:
        return rho_al_20C()
    if "acsr" in material:
        return rho_acsr_20C()
    if "steel" in material or "european high strength" in material or "ehs" in material:
        return rho_steel_20C()
    # fallback genérico
    return rho_copper_20C()

# ======================= Modelo de Cabo ==============================
@dataclass
class Cable:
    """
    Representa um tipo de cabo (geralmente nu, para linha aérea).
    Todos os valores geométricos em mm / kcmil conforme catálogo.
    Resistência DC em Ω/km a 20°C.
    """
    key: str
    material: str
    area_kcmil: float
    diameter_mm: float
    gmr_mm: float
    rdc_ohm_km_20C: float
    notes: str = ""
    # Para cabos isolados / subterrâneos, se necessário:
    eps_r_insulation: float = 1.0  # ~1.0 para cabo nu em ar

    @property
    def radius_m(self) -> float:
        return mm_to_m(self.diameter_mm) / 2.0

    @property
    def gmr_m(self) -> float:
        return mm_to_m(self.gmr_mm)

    @property
    def rdc_ohm_per_m_20C(self) -> float:
        return self.rdc_ohm_km_20C / 1000.0

    def rho_20C(self) -> float:
        """
        Resistividade equivalente do material (aprox).
        Útil para estimar skin depth.
        """
        return guess_rho_from_material(self.material)

    # ------------------- Elétrica por unidade de comprimento ---------
    def ac_resistance_per_m(
        self,
        f_hz: float,
        temp_C: float
    ) -> float:
        """
        Calcula R_ac por metro, considerando efeito de temperatura e skin.
        """
        rho20 = self.rho_20C()
        rhoT  = temp_resistivity(rho20, temp_C)
        sigma = 1.0 / rhoT if rhoT > 0 else 0.0

        # Escalonamento da resistência DC de catálogo para a temperatura T
        if rho20 > 0:
            fator_T = rhoT / rho20
        else:
            fator_T = 1.0
        fator_T = max(fator_T, 0.0)

        Rdc_T = self.rdc_ohm_per_m_20C * fator_T

        # Skin depth com ρ(T)
        delta = skin_depth(f_hz, MU0, sigma)

        return ac_resistance_from_skin(
            Rdc_T, self.radius_m, delta
        )

    def bundle_equivalents(
        self,
        n_bundle: int,
        ds_m: float
    ) -> Tuple[float, float]:
        """
        Devolve (GMR_eq_m, r_eq_m) para o feixe (bundle).
        """
        return bundle_equivalents(self.gmr_m, self.radius_m,
                                  n_bundle, ds_m)

    # ------------------- Parâmetros L', C' a partir de GMD ----------
    def line_LC_from_GMD(
        self,
        GMD_m: float,
        f_hz: float,
        temp_C: float,
        n_bundle: int = 1,
        ds_bundle_m: float = 0.4,
        eps_r_ambiente: float = 1.0
    ) -> Dict[str, float]:
        """
        Calcula L', C', R', X' e B' a partir de:
          - GMD_m: Geometric Mean Distance da terna [m]
          - Dados do cabo + bundle (n_bundle, ds_bundle)
          - Frequência f_hz
          - Temperatura do condutor temp_C
        """
        if GMD_m <= 0:
            raise ValueError("GMD deve ser > 0.")

        # Equivalentes geométricos do feixe
        GMR_eq_m, r_eq_m = self.bundle_equivalents(n_bundle, ds_bundle_m)

        # Indutância e Capacitância por metro
        L_per_m = MU0 / (2 * math.pi) * math.log(GMD_m / max(1e-12, GMR_eq_m))
        C_per_m = (2 * math.pi * EPS0 * eps_r_ambiente) / \
                  math.log(GMD_m / max(1e-12, r_eq_m))

        # Resistência AC por metro de um subcondutor
        R_per_m_single = self.ac_resistance_per_m(f_hz, temp_C)
        # Feixe de n subcondutores em paralelo → resistência por fase
        n_eff = max(1, int(n_bundle))
        R_per_m = R_per_m_single / n_eff

        X_per_m = 2 * math.pi * f_hz * L_per_m
        B_per_m = 2 * math.pi * f_hz * C_per_m

        # Converte para por km, que é o mais usado em estudos de linha
        scale = 1000.0
        R_pk = R_per_m * scale
        X_pk = X_per_m * scale
        B_pk = B_per_m * scale
        L_pk = L_per_m * scale
        C_pk = C_per_m * scale

        return {
            "R_per_m": R_per_m,
            "X_per_m": X_per_m,
            "B_per_m": B_per_m,
            "L_per_m": L_per_m,
            "C_per_m": C_per_m,
            "R_ohm_km": R_pk,
            "X_ohm_km": X_pk,
            "B_S_km": B_pk,
            "L_H_km": L_pk,
            "C_F_km": C_pk,
            "GMR_eq_m": GMR_eq_m,
            "r_eq_m": r_eq_m,
        }

# ======================= Banco de cabos ==============================
def default_cable_db() -> List[Cable]:
    """
    Banco inicial de cabos (você pode ampliar até 1000 kcmil).
    Esses dados foram baseados nos que você já utilizava e com acréscimo
    de cabos guarda OPGW e EHS.
    """
    # Observação: diameters/gmr/rdc são estimativas; ajuste com dados de catálogo quando disponíveis.
    data = [
        {"key": "CU_4/0",    "material": "Cu",
         "area_kcmil": 212.0, "diameter_mm": 11.68,
         "gmr_mm": 4.1, "rdc_ohm_km_20C": 0.160,
         "notes": "Cobre 4/0 AWG (indicativo)"},
        {"key": "AL_4/0",    "material": "Al",
         "area_kcmil": 212.0, "diameter_mm": 12.8,
         "gmr_mm": 4.5, "rdc_ohm_km_20C": 0.275,
         "notes": "Alumínio 4/0 AWG (indicativo)"},
        {"key": "ACSR_266.8", "material": "ACSR",
         "area_kcmil": 266.8, "diameter_mm": 18.3,
         "gmr_mm": 7.1, "rdc_ohm_km_20C": 0.272,
         "notes": "ACSR 266.8"},
        {"key": "ACSR_336.4", "material": "ACSR",
         "area_kcmil": 336.4, "diameter_mm": 20.4,
         "gmr_mm": 8.3, "rdc_ohm_km_20C": 0.221,
         "notes": "ACSR 336.4"},
        {"key": "ACSR_397.5", "material": "ACSR",
         "area_kcmil": 397.5, "diameter_mm": 22.1,
         "gmr_mm": 9.3, "rdc_ohm_km_20C": 0.189,
         "notes": "ACSR 397.5"},
        {"key": "ACSR_477",   "material": "ACSR",
         "area_kcmil": 477.0, "diameter_mm": 24.3,
         "gmr_mm": 10.7, "rdc_ohm_km_20C": 0.159,
         "notes": "ACSR 477"},
        {"key": "ACSR_556.5", "material": "ACSR",
         "area_kcmil": 556.5, "diameter_mm": 26.2,
         "gmr_mm": 11.7, "rdc_ohm_km_20C": 0.140,
         "notes": "ACSR 556.5"},
        {"key": "ACSR_636",   "material": "ACSR",
         "area_kcmil": 636.0, "diameter_mm": 28.1,
         "gmr_mm": 12.9, "rdc_ohm_km_20C": 0.124,
         "notes": "ACSR 636"},
        {"key": "ACSR_795",   "material": "ACSR",
         "area_kcmil": 795.0, "diameter_mm": 31.8,
         "gmr_mm": 14.7, "rdc_ohm_km_20C": 0.098,
         "notes": "ACSR 795"},
        {"key": "ACSR_954",   "material": "ACSR",
         "area_kcmil": 954.0, "diameter_mm": 34.6,
         "gmr_mm": 16.2, "rdc_ohm_km_20C": 0.084,
         "notes": "ACSR 954"},

        # ----------------- CABOS GUARDA (OPGW / EHS) -----------------
        # As áreas em mm² informadas pelo usuário foram convertidas para diâmetros estimados.
        # Valores Rdc calculados por área/ resistividade do alumínio (aprox).
        {"key": "OPGW_10_2mm2",  "material": "Al",  "area_kcmil": 10.2/0.5067,  # aproximação kcmil
         "diameter_mm":  ( (4.0 * 10.2 / math.pi) ** 0.5 ),  # estimativa: area -> diam
         "gmr_mm": 2.8, "rdc_ohm_km_20C": round(28.26 / 10.2, 6),
         "notes": "OPGW approx 10.2 mm² (estimativa)"},
        {"key": "OPGW_13_3mm2",  "material": "Al",  "area_kcmil": 13.3/0.5067,
         "diameter_mm": ( (4.0 * 13.3 / math.pi) ** 0.5 ),
         "gmr_mm": 3.2, "rdc_ohm_km_20C": round(28.26 / 13.3, 6),
         "notes": "OPGW approx 13.3 mm² (estimativa)"},
        {"key": "OPGW_14_1mm2",  "material": "Al",  "area_kcmil": 14.1/0.5067,
         "diameter_mm": ( (4.0 * 14.1 / math.pi) ** 0.5 ),
         "gmr_mm": 3.3, "rdc_ohm_km_20C": round(28.26 / 14.1, 6),
         "notes": "OPGW approx 14.1 mm² (estimativa)"},
        {"key": "OPGW_15_5mm2",  "material": "Al",  "area_kcmil": 15.5/0.5067,
         "diameter_mm": ( (4.0 * 15.5 / math.pi) ** 0.5 ),
         "gmr_mm": 3.5, "rdc_ohm_km_20C": round(28.26 / 15.5, 6),
         "notes": "OPGW approx 15.5 mm² (estimativa)"},
        # 3/8'' EHS (EHS - Earth/High Strength steel) — diâmetro 3/8" ≈ 9.525 mm
        {"key": "EHS_3_8in",    "material": "Steel",
         "area_kcmil": (math.pi * (9.525 ** 2) / 4) / 0.5067,  # aproximação kcmil
         "diameter_mm": 9.525,
         "gmr_mm": 4.0, "rdc_ohm_km_20C": round((1.43e2) / (math.pi * (9.525 ** 2) * 1e-6), 6),
         "notes": "Cabos guarda 3/8'' EHS (estimativa)"},
    ]
    # Constrói objetos Cable a partir da lista de dicionários
    return [Cable(**row) for row in data]

def cable_list_to_json(cables: List[Cable]) -> str:
    return json.dumps(
        [asdict(c) for c in cables],
        ensure_ascii=False,
        indent=2
    )

def cable_list_from_json(s: str) -> List[Cable]:
    data = json.loads(s)
    return [Cable(**row) for row in data]

def load_cable_db(path: str) -> List[Cable]:
    with open(path, "r", encoding="utf-8") as f:
        return cable_list_from_json(f.read())

def save_cable_db(path: str, cables: List[Cable]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(cable_list_to_json(cables))

def find_cable(cables: List[Cable], key: str) -> Optional[Cable]:
    if not key:
        return None
    key = key.strip()
    for c in cables:
        if c.key == key:
            return c
    return None

# ======================= Função de alto nível ========================
def calc_line_params_from_cable(
    cable: Cable,
    GMD_m: float,
    f_hz: float,
    temp_C: float,
    n_bundle: int = 1,
    ds_bundle_m: float = 0.4,
    eps_r_ambiente: float = 1.0
) -> Dict[str, float]:
    """
    Função de alto nível para o módulo de Parâmetros Elétricos.
    Retorna um dicionário com tudo o que a aba de parâmetros precisa:
        - R_ohm_km, X_ohm_km, B_S_km, L_H_km, C_F_km
        - GMR_eq_m, r_eq_m
    """
    return cable.line_LC_from_GMD(
        GMD_m=GMD_m,
        f_hz=f_hz,
        temp_C=temp_C,
        n_bundle=n_bundle,
        ds_bundle_m=ds_bundle_m,
        eps_r_ambiente=eps_r_ambiente,
    )

# ======================= Teste rápido (opcional) =====================
if __name__ == "__main__":
    # Pequeno teste de sanidade
    cables = default_cable_db()
    cabo = find_cable(cables, "ACSR_477")
    if cabo:
        params = calc_line_params_from_cable(
            cabo,
            GMD_m=8.0,        # exemplo
            f_hz=60.0,
            temp_C=50.0,
            n_bundle=1,
            ds_bundle_m=0.4,
        )
        print("Parâmetros por km (exemplo ACSR 477, GMD=8 m):")
        for k, v in params.items():
            print(f"  {k}: {v}")

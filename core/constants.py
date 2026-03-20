# bk_estudos_eletricos/core/constants.py
import math

EPS0 = 8.854187817e-12   # F/m
MU0  = 4 * math.pi * 1e-7 # H/m

def mm_to_m(x: float) -> float:
    return x / 1000.0

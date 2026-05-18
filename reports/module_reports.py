# reports/module_reports.py
# ====================================================================
# Geradores de conteudo para relatorios de cada modulo
# Cada funcao recebe um BKReport e os resultados calculados
# ====================================================================

from __future__ import annotations
from typing import Any, Dict, List, Optional
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bk_docx import BKReport
from . import omml

# ====================================================================
# Texto padrao de rodape dos relatorios
# ====================================================================
SOFTWARE_NOTE = (
    "Este relatório foi gerado automaticamente pelo software BK Estudos Elétricos, "
    "desenvolvido pela BK Engenharia e Tecnologia. Os cálculos seguem metodologias "
    "consagradas na literatura técnica e normas nacionais e internacionais. "
    "Recomenda-se verificação independente dos resultados para fins de projeto executivo."
)

# ====================================================================
# REFERENCIAS BIBLIOGRAFICAS POR MODULO
# ====================================================================

REFS_PARAMS = [
    "STEVENSON, W. D. Elements of Power System Analysis. McGraw-Hill, 4ª ed., 1982.",
    "GLOVER, J. D.; SARMA, M. S.; OVERBYE, T. J. Power Systems Analysis and Design. Cengage, 6ª ed., 2017.",
    "ABNT NBR 5422:1985 – Projeto de linhas aéreas de transmissão de energia elétrica.",
    "IEEE Std 738-2012 – IEEE Standard for Calculating the Current-Temperature Relationship of Bare Overhead Conductors.",
    "EPRI – Transmission Line Reference Book: 345 kV and Above. EPRI, 2ª ed., 1982.",
    "FUCHS, R. D. Transmissão de Energia Elétrica: Linhas Aéreas. LTC, 2ª ed., Vol. 1, 1979.",
    "ZANETTA JR., L. C. Transitórios Eletromagnéticos em Sistemas de Potência. EDUSP, 2003.",
]

REFS_CORONA = [
    "PEEK, F. W. Dielectric Phenomena in High Voltage Engineering. McGraw-Hill, 3ª ed., 1929.",
    "EPRI – Transmission Line Reference Book: 345 kV and Above. EPRI, 2ª ed., 1982.",
    "ABNT NBR 5422:1985 – Projeto de linhas aéreas de transmissão de energia elétrica.",
    "IEEE Std 539-2005 – IEEE Standard Definitions of Terms Relating to Corona and Field Effects of Overhead Power Lines.",
    "CIGRÉ TB 61 – Interferences Produced by Corona Effect of Electric Systems. 1974.",
    "FUCHS, R. D. Transmissão de Energia Elétrica: Linhas Aéreas. LTC, 2ª ed., Vol. 1, 1979.",
]

REFS_FIELDS = [
    "ANEEL – Resolução Normativa nº 398/2010 (limites de campo EM em faixa de servidão).",
    "ICNIRP Guidelines – Guidelines for Limiting Exposure to Time-Varying Electric, Magnetic, and Electromagnetic Fields (up to 300 GHz). Health Physics, 1998.",
    "IEEE Std C95.6-2002 – IEEE Standard for Safety Levels with Respect to Human Exposure to Electromagnetic Fields, 0–3 kHz.",
    "EPRI – Transmission Line Reference Book: 345 kV and Above. EPRI, 2ª ed., 1982.",
    "ABNT NBR 25415:2016 – Linhas de transmissão – Campos elétricos e magnéticos – Limites de exposição.",
    "FUCHS, R. D. Transmissão de Energia Elétrica: Linhas Aéreas. LTC, 2ª ed., Vol. 1, 1979.",
]

REFS_AMPACITY = [
    "IEEE Std 738-2012 – IEEE Standard for Calculating the Current-Temperature Relationship of Bare Overhead Conductors.",
    "CIGRÉ TB 207 – Thermal Behaviour of Overhead Conductors. 2002.",
    "ABNT NBR 5422:1985 – Projeto de linhas aéreas de transmissão de energia elétrica.",
    "EPRI – Transmission Line Reference Book: 345 kV and Above. EPRI, 2ª ed., 1982.",
    "KIESSLING, F. et al. Overhead Power Lines: Planning, Design, Construction. Springer, 2003.",
    "FUCHS, R. D. Transmissão de Energia Elétrica: Linhas Aéreas. LTC, 2ª ed., Vol. 1, 1979.",
]

REFS_RIRA = [
    "CIGRÉ TB 61 – Interferences Produced by Corona Effect of Electric Systems. 1974.",
    "IEEE Std 430-2017 – IEEE Standard Procedures for the Measurement of Radio Noise from Overhead Power Lines and Substations.",
    "EPRI – Transmission Line Reference Book: 345 kV and Above. EPRI, 2ª ed., 1982.",
    "ABNT NBR 5422:1985 – Projeto de linhas aéreas de transmissão de energia elétrica.",
    "IEC 62236 – Railway applications – Electromagnetic compatibility.",
    "FUCHS, R. D. Transmissão de Energia Elétrica: Linhas Aéreas. LTC, 2ª ed., Vol. 1, 1979.",
]

REFS_SHIELDING = [
    "IEEE Std 1243-1997 – IEEE Guide for Improving the Lightning Performance of Transmission Lines.",
    "IEEE Std 998-2012 – IEEE Guide for Direct Lightning Stroke Shielding of Substations.",
    "CIGRÉ TB 63 – Guide to Procedures for Estimating the Lightning Performance of Transmission Lines. 1991.",
    "ABNT NBR 5419:2015 – Proteção contra descargas atmosféricas.",
    "ABNT NBR 5422:1985 – Projeto de linhas aéreas de transmissão de energia elétrica.",
    "HILEMAN, A. R. Insulation Coordination for Power Systems. CRC Press, 1999.",
]

REFS_VMAX = [
    "IEC 60071-1:2019 – Insulation co-ordination – Part 1: Definitions, principles and rules.",
    "IEC 60071-2:2018 – Insulation co-ordination – Part 2: Application guidelines.",
    "ABNT NBR 6939:2000 – Coordenação de isolamento – Procedimento.",
    "ABNT NBR 5422:1985 – Projeto de linhas aéreas de transmissão de energia elétrica.",
    "HILEMAN, A. R. Insulation Coordination for Power Systems. CRC Press, 1999.",
    "EPRI – Transmission Line Reference Book: 345 kV and Above. EPRI, 2ª ed., 1982.",
]

REFS_COORD = [
    "IEC 60071-1:2019 – Insulation co-ordination – Part 1: Definitions, principles and rules.",
    "IEC 60071-2:2018 – Insulation co-ordination – Part 2: Application guidelines.",
    "IEC 60099-4:2014 – Surge arresters – Part 4: Metal-oxide surge arresters without gaps for a.c. systems.",
    "ABNT NBR 6939:2000 – Coordenação de isolamento – Procedimento.",
    "ABNT NBR 5422:1985 – Projeto de linhas aéreas de transmissão de energia elétrica.",
    "HILEMAN, A. R. Insulation Coordination for Power Systems. CRC Press, 1999.",
    "ZANETTA JR., L. C. Transitórios Eletromagnéticos em Sistemas de Potência. EDUSP, 2003.",
]

REFS_RECLOSING = [
    "IEEE Std C37.104-2012 – IEEE Guide for Automatic Reclosing on AC Distribution and Transmission Lines.",
    "CIGRÉ TB 311 – Application Guide for the Selection of Clearing Time for High Voltage Surge Arresters. 2007.",
    "IEC 62271-100:2021 – High-voltage switchgear and controlgear – Part 100: AC circuit-breakers.",
    "ZANETTA JR., L. C. Transitórios Eletromagnéticos em Sistemas de Potência. EDUSP, 2003.",
    "GREENWOOD, A. Electrical Transients in Power Systems. Wiley, 2ª ed., 1991.",
]

REFS_EMI = [
    "ITU-T K.68 – Operator responsibilities in maintaining lines in the vicinity of power lines.",
    "CIGRÉ TB 373 – Mitigation Techniques of Power Frequency Magnetic Fields Originated from Electric Power Systems. 2009.",
    "IEEE Std 776-1992 – IEEE Recommended Practice for Inductive Coordination of Electric Supply and Communication Lines.",
    "ABNT NBR 15415:2006 – Métodos de medição e níveis de referência para exposição a campos elétricos e magnéticos.",
    "DNIT – Normas de proteção de dutos em faixas de domínio.",
]

REFS_PF = [
    "MONTICELLI, A. Fluxo de Carga em Redes de Energia Elétrica. Edgard Blücher, 1983.",
    "GLOVER, J. D.; SARMA, M. S.; OVERBYE, T. J. Power Systems Analysis and Design. Cengage, 6ª ed., 2017.",
    "STEVENSON, W. D. Elements of Power System Analysis. McGraw-Hill, 4ª ed., 1982.",
    "IEEE Task Force – Power Flow Study Methods. IEEE Transactions on Power Systems.",
    "ONS – Procedimentos de Rede – Módulo 23: Estudos de Fluxo de Potência.",
]


# ====================================================================
# 1. PARAMETROS ELETRICOS
# ====================================================================

def report_parametros(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    """Gera relatorio de Parametros Eletricos da LT."""

    rpt.add_cover("Estudo de Parâmetros Elétricos", "Linha de Transmissão")
    rpt.add_page_break()

    # Sumario
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    # 1. Introducao
    rpt.add_heading1("Introdução")
    rpt.add_body(
        "O estudo de parâmetros elétricos de linhas de transmissão constitui etapa fundamental "
        "no planejamento e projeto de sistemas de potência. A determinação precisa das impedâncias "
        "série (resistência e reatância) e admitâncias shunt (susceptância capacitiva) por unidade "
        "de comprimento é essencial para a modelagem adequada da linha em estudos de fluxo de "
        "potência, estabilidade, transitórios eletromagnéticos e proteção."
    )
    rpt.add_body(
        "Os parâmetros elétricos são calculados a partir da geometria física da torre "
        "(arranjo de fases, alturas dos condutores, espaçamentos e configuração de feixes), "
        "das características dos cabos condutores (seção transversal, GMR, resistência DC) e "
        "das condições operativas (frequência, temperatura). A metodologia utiliza conceitos "
        "de distância média geométrica (GMD) e raio médio geométrico equivalente (GMR_eq) "
        "para feixes, conforme estabelecido na literatura clássica de sistemas de potência."
    )
    rpt.add_body(
        "Este relatório apresenta os resultados do cálculo dos parâmetros elétricos conforme "
        "as normas ABNT NBR 5422, IEEE Std 738 e metodologias consagradas nos livros de "
        "Stevenson, Glover/Sarma/Overbye e Fuchs, produzido pelo Software BK Estudos Elétricos "
        "da BK Engenharia e Tecnologia."
    )

    # 2. Objetivo
    rpt.add_heading1("Objetivo")
    rpt.add_body(
        "Determinar os parâmetros elétricos por unidade de comprimento da linha de transmissão "
        "em estudo, incluindo: resistência série R' (Ω/km), reatância indutiva X' (Ω/km), "
        "susceptância capacitiva B' (S/km), indutância L' (H/km) e capacitância C' (F/km), "
        "bem como a impedância característica Zc (Ω) e a potência natural SIL (MW). "
        "O modelo equivalente π da linha também é apresentado para uso em estudos de regime permanente."
    )

    # 3. Metodologia
    rpt.add_heading1("3. Metodologia")
    rpt.add_body(
        "A metodologia empregada baseia-se no cálculo clássico de parâmetros de linhas de "
        "transmissão, conforme Stevenson (1982), Glover/Sarma (2017) e Fuchs (1979), "
        "seguindo as etapas descritas a seguir."
    )

    rpt.add_heading2("3.1 Impedância série")
    rpt.add_body(
        "A impedância série por unidade de comprimento combina resistência e reatância indutiva. "
        "É o parâmetro que governa as perdas ôhmicas e o carregamento reativo da linha — "
        "sua determinação precisa é essencial para cálculo de fluxo de potência e estabilidade:"
    )
    rpt.add_equation(omml.eq_impedance_series(), "Impedância série por km")

    rpt.add_heading2("3.2 Indutância e GMR equivalente do feixe")
    rpt.add_body(
        "A indutância depende do raio médio geométrico (GMR) do condutor — ou do GMR "
        "equivalente do feixe, que captura o efeito da distribuição de n subcondutores sobre "
        "a indutividade. Usa-se a formulação com GMR_eq porque condutores em feixe "
        "concentram parte do fluxo magnético entre os subcondutores, reduzindo a reatância "
        "série em relação a um único condutor de mesma seção:"
    )
    rpt.add_equation(omml.eq_gmr_bundle(), "GMR equivalente do feixe (n subcondutores)")
    rpt.add_body(
        "onde d_s é a distância entre subcondutores adjacentes e n é o número de subcondutores. "
        "Com o GMR_eq determinado:"
    )
    rpt.add_equation(omml.eq_inductance(), "Indutância por km")
    rpt.add_body(
        "A distância média geométrica (GMD) entre fases é necessária porque a reatância "
        "indutiva depende do acoplamento magnético mútuo entre os condutores de cada fase — "
        "o GMD é a média geométrica das distâncias entre todos os pares de condutores."
    )

    rpt.add_heading2("3.3 Capacitância")
    rpt.add_body(
        "A capacitância shunt determina a corrente de carga (charging current) da linha, "
        "decisiva em linhas longas (>300 km) para o perfil de tensão e geração de reativo. "
        "Utiliza-se o raio equivalente do feixe r_eq (ao invés do GMR) pois a capacitância "
        "depende da distribuição de cargas elétricas na superfície do condutor:"
    )
    rpt.add_equation(omml.eq_capacitance(), "Capacitância por km")

    rpt.add_heading2("3.4 Resistência AC e efeito pelicular")
    rpt.add_body(
        "Em corrente alternada, a corrente tende a concentrar-se na superfície do condutor "
        "(efeito pelicular), aumentando a resistência efetiva em relação ao valor DC. "
        "Além disso, a resistência aumenta com a temperatura. A correção é feita por:"
    )
    rpt.add_equation(omml.eq_rac(), "Resistência AC com skin effect")
    rpt.add_body(
        "onde δ_skin = √(2ρ/ωμ) é a profundidade de penetração. Para condutores ACSR "
        "em 60 Hz, o aumento de R_ac sobre R_dc varia de ~2% (cabo pequeno) a ~10% (cabo grande)."
    )

    rpt.add_heading2("3.5 Impedância característica e potência natural (SIL)")
    rpt.add_body(
        "A impedância característica Zc e a potência natural SIL são parâmetros que permitem "
        "avaliar rapidamente o regime de operação da linha: quando P_transmitida > SIL, a linha "
        "consome reativo (regime indutivo); quando P < SIL, a linha gera reativo (regime capacitivo). "
        "Essa informação é crucial para decidir a necessidade de compensação reativa:"
    )
    rpt.add_equation(omml.eq_characteristic_impedance(), "Impedância característica (Ω)")
    rpt.add_equation(omml.eq_sil(), "Potência natural — SIL (MW)")

    rpt.add_heading2("3.6 Modelo π equivalente")
    rpt.add_body(
        "O modelo π concentrado é a representação padrão para estudos de regime permanente. "
        "Ele concentra os parâmetros distribuídos em três elementos: impedância série Z_serie "
        "e dois elementos shunt Y_shunt/2 em cada extremidade. É exato para comprimentos curtos "
        "(<150 km) e adequado para comprimentos médios com correção hiperbólica:"
    )
    rpt.add_equation(omml.eq_pi_model(), "Modelo π equivalente da linha")

    # 4. Resultados
    rpt.add_heading1("Resultados Obtidos")

    V = cfg.get("voltage_kv", 0)
    S = cfg.get("power_mva", 0)
    f = cfg.get("freq_hz", 60)
    T = cfg.get("temp_C", 50)
    L = cfg.get("line_length_km", 100)

    rpt.add_heading2("Dados de Entrada")
    pf = cfg.get("pf_load", 1.0)
    nc = cfg.get("n_circuits", 1)
    vs_ang = cfg.get("Vs_ang", 0.0)
    rpt.add_kpi_table([
        ("Tensão Nominal (Vs)", f"{V:.1f}", "kV"),
        ("∠Vs", f"{vs_ang:.1f}", "°"),
        ("Potência", f"{S:.1f}", "MVA"),
        ("Fator de Potência", f"{pf:.2f}", "—"),
        ("Frequência", f"{f:.0f}", "Hz"),
        ("Temperatura", f"{T:.0f}", "°C"),
        ("Comprimento", f"{L:.1f}", "km"),
        ("Nº Circuitos", f"{nc}", "—"),
    ])

    rpt.add_heading2("Parâmetros Calculados por Circuito")

    if "circuits" in results:
        headers = ["Parâmetro", "Unidade"] + [f"Circuito {i+1}" for i in range(len(results["circuits"]))]
        params_names = [
            ("R'", "Ω/km", "R_ohm_km"),
            ("X'", "Ω/km", "X_ohm_km"),
            ("B'", "µS/km", "B_S_km"),
            ("L'", "mH/km", "L_H_km"),
            ("C'", "nF/km", "C_F_km"),
            ("GMR_eq", "m", "GMR_eq_m"),
            ("r_eq", "m", "r_eq_m"),
            ("Zc", "Ω", "Zc_ohm"),
            ("SIL", "MW", "SIL_MW"),
        ]
        rows = []
        for name, unit, key in params_names:
            row = [name, unit]
            for circ in results["circuits"]:
                val = circ.get(key, 0)
                if "nF" in unit:
                    row.append(f"{val * 1e9:.4f}")
                elif "mH" in unit:
                    row.append(f"{val * 1e3:.4f}")
                elif "µS" in unit:
                    row.append(f"{val * 1e6:.4f}")
                else:
                    row.append(f"{val:.4f}")
            rows.append(row)
        rpt.add_result_table(headers, rows, "Parâmetros elétricos por circuito")

    # ── Tensão de Chegada (Vr) — calculada via modelo π ──────
    vr_computed = results.get("vr_computed", [])
    if vr_computed:
        rpt.add_heading2("Tensão de Chegada na Carga (Vr)")
        rpt.add_body(
            "A tensão no terminal receptor (Vr) é calculada pelo modelo π nominal da linha, "
            "aplicando o método de Newton-Raphson para resolver o circuito equivalente com carga "
            "S e fator de potência fp especificados. A regulação de tensão é definida como "
            "Reg(%) = (Vs − Vr)/Vr × 100%. Referências: Stevenson — Elements of Power System Analysis, "
            "Cap. 5; Zanetta — Fundamentos de SEP, Cap. 3; Fuchs — Transmissão de Energia Elétrica."
        )

        vr_headers = ["Circuito", "Vs (kV)", "∠Vs (°)", "Vr (kV)", "∠Vr (°)",
                       "I (A)", "P_perdas (MW)", "Regulação (%)"]
        vr_rows = []
        for vr in vr_computed:
            vr_rows.append([
                str(vr.get("circuit", 1)),
                f"{vr.get('Vs_kV', 0):.2f}",
                f"{vr.get('Vs_ang', 0):.2f}",
                f"{vr.get('Vr_kV', 0):.3f}",
                f"{vr.get('Vr_ang', 0):.2f}",
                f"{vr.get('I_A', 0):.1f}",
                f"{vr.get('Ploss_MW', 0):.3f}",
                f"{vr.get('reg_pct', 0):.2f}",
            ])
        rpt.add_result_table(vr_headers, vr_rows, "Tensão de chegada e regulação por circuito")

        # Resumo textual
        vr0 = vr_computed[0]
        reg0 = vr0.get("reg_pct", 0)
        rpt.add_body(
            f"Para o circuito 1, a tensão de chegada calculada é Vr = {vr0.get('Vr_kV',0):.3f} kV "
            f"com ângulo ∠Vr = {vr0.get('Vr_ang',0):.2f}°, "
            f"resultando em regulação de tensão de {reg0:.2f}%. "
            f"A corrente na linha série é de {vr0.get('I_A',0):.1f} A e as perdas ativas "
            f"são de {vr0.get('Ploss_MW',0):.3f} MW."
        )
        if abs(reg0) > 5:
            rpt.add_body(
                f"A regulação de {reg0:.2f}% excede 5%, indicando queda de tensão significativa. "
                "Recomenda-se avaliar compensação reativa ou alteração do tap do transformador."
            )
        elif abs(reg0) > 3:
            rpt.add_body(
                f"A regulação de {reg0:.2f}% está na faixa aceitável, porém recomenda-se monitorar "
                "o perfil de tensão em condições de carga máxima."
            )
        else:
            rpt.add_body(
                f"A regulação de {reg0:.2f}% indica perfil de tensão adequado."
            )

    # Circuito pi (diagrama)
    rpt.add_heading2("Modelo π Equivalente")
    if "circuits" in results and len(results["circuits"]) > 0:
        c0 = results["circuits"][0]
        fig = _plot_pi_model(
            R=c0.get("R_ohm_km", 0) * L,
            X=c0.get("X_ohm_km", 0) * L,
            B=c0.get("B_S_km", 0) * L if "B_S_km" in c0 else 0,
            L_km=L
        )
        rpt.add_figure_from_matplotlib(fig, "Modelo π equivalente da linha – Circuito 1")

    # Grafico de comparacao multi-circuito
    if "circuits" in results and len(results["circuits"]) > 1:
        fig2 = _plot_params_comparison(results["circuits"])
        rpt.add_figure_from_matplotlib(fig2, "Comparação de parâmetros entre circuitos")

    # 5. Conclusao
    rpt.add_heading1("Conclusão")
    if "circuits" in results and len(results["circuits"]) > 0:
        c0 = results["circuits"][0]
        zc = c0.get("Zc_ohm", 0)
        sil = c0.get("SIL_MW", 0)
        rpt.add_body(
            f"Os parâmetros elétricos foram calculados com sucesso para a linha de transmissão "
            f"de {V:.0f} kV em estudo. A impedância característica obtida é de {zc:.1f} Ω e a "
            f"potência natural (SIL) é de {sil:.1f} MW. "
        )

        # Conclusão sobre Vr
        vr_c = results.get("vr_computed", [])
        if vr_c:
            vr0 = vr_c[0]
            rpt.add_body(
                f"A tensão de chegada calculada é Vr = {vr0.get('Vr_kV',0):.3f} kV ∠ {vr0.get('Vr_ang',0):.2f}°, "
                f"com corrente de {vr0.get('I_A',0):.1f} A, perdas de {vr0.get('Ploss_MW',0):.3f} MW "
                f"e regulação de tensão de {vr0.get('reg_pct',0):.2f}%."
            )

        if S > 0 and sil > 0:
            ratio = S / sil
            if ratio > 1.5:
                rpt.add_body(
                    f"A potência transmitida ({S:.0f} MVA) é significativamente superior ao SIL "
                    f"({sil:.1f} MW, razão {ratio:.2f}), indicando que a linha opera em regime "
                    f"predominantemente indutivo. Recomenda-se avaliar a necessidade de compensação "
                    f"reativa shunt para manter os níveis de tensão dentro dos limites operativos."
                )
            elif ratio < 0.5:
                rpt.add_body(
                    f"A potência transmitida ({S:.0f} MVA) é significativamente inferior ao SIL "
                    f"({sil:.1f} MW, razão {ratio:.2f}), indicando regime predominantemente capacitivo. "
                    f"Recomenda-se avaliar a necessidade de compensação reativa série ou reatores shunt."
                )
            else:
                rpt.add_body(
                    f"A razão entre a potência transmitida e o SIL é de {ratio:.2f}, indicando "
                    f"operação próxima da condição natural, o que é favorável para o perfil de tensão."
                )

    rpt.add_body(SOFTWARE_NOTE)

    # 6. Referencias
    rpt.add_references_section(REFS_PARAMS)


# ====================================================================
# 2. CORONA
# ====================================================================

def report_corona(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    rpt.add_cover("Estudo de Efeito Corona", "Linha de Transmissão")
    rpt.add_page_break()
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    # Introducao
    rpt.add_heading1("Introdução")
    rpt.add_body(
        "O efeito corona é um fenômeno de descarga parcial que ocorre na superfície de "
        "condutores de alta tensão quando o campo elétrico superficial ultrapassa o valor "
        "crítico de ionização do ar. Este fenômeno é uma das principais preocupações no "
        "projeto de linhas de transmissão, pois resulta em perdas de energia, geração de "
        "ruído audível, interferência radioelétrica e produção de ozônio."
    )
    rpt.add_body(
        "A avaliação do efeito corona é realizada pela comparação entre o gradiente elétrico "
        "superficial do condutor (ou feixe) e o gradiente crítico de corona, determinado pela "
        "fórmula empírica de Peek (1929). A metodologia é amplamente aceita pela indústria "
        "e normatizada em publicações do IEEE, CIGRÉ e EPRI."
    )
    rpt.add_body(
        "Este relatório apresenta os resultados da avaliação de corona, produzido pelo "
        "Software BK Estudos Elétricos da BK Engenharia e Tecnologia."
    )

    # Objetivo
    rpt.add_heading1("Objetivo")
    rpt.add_body(
        "Verificar se o gradiente elétrico superficial dos condutores da linha de transmissão "
        "em estudo permanece abaixo do gradiente crítico de corona nas condições ambientais de "
        "projeto, quantificando a margem de segurança e estimando as perdas por corona quando aplicável."
    )

    # Metodologia
    rpt.add_heading1("3. Metodologia")
    rpt.add_body(
        "O efeito corona inicia quando o campo elétrico superficial do condutor supera o campo "
        "crítico de ionização do ar. A metodologia adota a fórmula empírica de Peek (1929), "
        "validada em décadas de medições de campo e consagrada pelo EPRI, CIGRÉ e IEEE."
    )

    rpt.add_heading2("3.1 Densidade relativa do ar")
    rpt.add_body(
        "O campo crítico de Peek é calibrado para condições-padrão de 25 °C e 1 atm. "
        "Para corrigir os efeitos de altitude e temperatura — que reduzem a densidade do ar "
        "e, portanto, o campo necessário para ionização — utiliza-se o fator δ:"
    )
    rpt.add_equation(omml.eq_air_density(), "Fator de densidade relativa do ar δ")
    rpt.add_body(
        "onde p é a pressão atmosférica em cmHg e T é a temperatura em °C. "
        "Em altitudes elevadas (ex.: 1 500 m), δ cai para ~0,82, reduzindo o gradiente crítico "
        "em ~18% — o que pode inviabilizar projetos dimensionados para nível do mar."
    )

    rpt.add_heading2("3.2 Gradiente crítico de corona (Peek)")
    rpt.add_body(
        "O campo elétrico superficial crítico acima do qual a ionização do ar se inicia "
        "é dado pela fórmula de Peek. O expoente 0,301/√(δ·r) captura a influência da "
        "curvatura superficial — condutores finos têm campo mais concentrado na superfície "
        "e coronam com gradiente menor:"
    )
    rpt.add_equation(omml.eq_peek_corona(), "Gradiente crítico de Peek — Ec (kV/cm)")
    rpt.add_body(
        "O fator m₀ penaliza condições adversas: m₀ = 1,0 (condutor polido, seco), "
        "m₀ ≈ 0,87 (condutor encordoado, seco) e m₀ ≈ 0,72–0,80 (tempo chuvoso). "
        "O projeto deve satisfazer E_superficial < Ec para m₀ da condição mais desfavorável."
    )

    rpt.add_heading2("3.3 Tensão crítica visual de corona")
    rpt.add_body(
        "A tensão de fase correspondente ao início de corona é obtida integrando o campo "
        "elétrico desde a superfície do condutor até a distância equivalente à GMD entre fases. "
        "Esta tensão define a margem de segurança em relação à tensão nominal da linha:"
    )
    rpt.add_equation(omml.eq_corona_voltage(), "Tensão crítica visual de corona — Vd (kV)")
    rpt.add_body(
        "A margem é calculada como: margem (%) = (Vd − V_fase) / V_fase × 100%. "
        "Valores positivos indicam ausência de corona; valores negativos indicam corona ativa. "
        "A prática de projeto adota margem mínima de 10% para condição de chuva."
    )

    # Resultados
    rpt.add_heading1("Resultados Obtidos")
    rpt.add_heading2("Dados de Entrada")
    V = cfg.get("voltage_kv", 0)
    rpt.add_kpi_table([
        ("Tensão", f"{V:.1f}", "kV"),
        ("Temperatura", f"{cfg.get('temp_C', 25):.0f}", "°C"),
        ("Altitude", f"{cfg.get('altitude_m', 0):.0f}", "m"),
        ("Condição", cfg.get("weather", "seco"), ""),
    ])

    if "circuits" in results:
        rpt.add_heading2("Resultados por Circuito")
        for i, circ in enumerate(results["circuits"]):
            rpt.add_heading3(f"Circuito {i+1}")
            ec_crit = circ.get("Ec_crit_kV_cm", 0)
            ec_surf = circ.get("Esurface_kV_cm", 0)
            vd = circ.get("Vd_LL_kV", 0)
            ok = circ.get("corona_ok", False)
            loss = circ.get("corona_loss_kW_km_phase", 0)
            margin = circ.get("margin_Vd_percent", 0)

            rpt.add_kpi_table([
                ("Ec crítico", f"{ec_crit:.2f}", "kV/cm"),
                ("Ec superficial", f"{ec_surf:.2f}", "kV/cm"),
                ("Vd crítica", f"{vd:.1f}", "kV"),
                ("Margem Vd", f"{margin:.1f}", "%"),
                ("Perdas", f"{loss:.3f}", "kW/km/fase"),
                ("Status", "ATENDE" if ok else "NÃO ATENDE", ""),
            ])

            # Grafico barras: Ec superficial vs Ec critico
            fig, ax = plt.subplots(figsize=(7, 3.5))
            bars = ax.bar(["Ec superficial", "Ec crítico"], [ec_surf, ec_crit],
                         color=["#E53935" if not ok else "#43A047", "#1565C0"])
            ax.set_ylabel("kV/cm")
            ax.set_title(f"Campo Elétrico Superficial vs Crítico – Circuito {i+1}")
            ax.axhline(ec_crit, color="#1565C0", linestyle="--", alpha=0.5, label="Limite Peek")
            ax.legend()
            ax.grid(True, alpha=0.3)
            rpt.add_figure_from_matplotlib(fig, f"Gradiente superficial vs crítico – Circuito {i+1}")

    # Conclusao
    rpt.add_heading1("Conclusão")
    all_ok = all(c.get("corona_ok", False) for c in results.get("circuits", [{}]))
    if all_ok:
        rpt.add_body(
            "Os resultados indicam que o gradiente elétrico superficial dos condutores permanece "
            "abaixo do gradiente crítico de Peek em todos os circuitos avaliados, nas condições "
            "ambientais consideradas. Portanto, não se espera ocorrência significativa de corona "
            "em condições normais de operação."
        )
    else:
        rpt.add_body(
            "Os resultados indicam que o gradiente elétrico superficial excede o gradiente crítico "
            "de Peek em pelo menos um circuito, nas condições ambientais consideradas. "
            "Recomenda-se avaliar as seguintes medidas mitigadoras:",
        )
        rpt.add_body_list([
            "Aumento do diâmetro equivalente do feixe (adicionar subcondutores ou aumentar espaçamento);",
            "Utilização de condutores com maior diâmetro ou seção transversal;",
            "Revisão dos espaçamentos entre fases para reduzir o gradiente superficial;",
            "Instalação de anéis de corona nas ferragens e conexões.",
        ])
    rpt.add_body(SOFTWARE_NOTE)
    rpt.add_references_section(REFS_CORONA)


# ====================================================================
# 3. CAMPOS ELETROMAGNETICOS
# ====================================================================

def report_campos_em(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    """
    Relatório completo de Campos Elétrico e Magnético.
    Norma: ANEEL Resolução Normativa nº 915/2021 (revoga nº 616/2014 e 398/2010).
    Método: MSC com Imagens Elétricas (Campo E) + Imagens Complexas de Deri (Campo B).
    """
    rpt.add_cover("Estudo de Campos Elétrico e Magnético", "Linha de Transmissão")
    rpt.add_page_break()
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    # ── 1. Introdução ──────────────────────────────────────────────────
    rpt.add_heading1("1. Introdução")
    rpt.add_body(
        "A avaliação dos campos elétrico e magnético gerados por linhas de transmissão de "
        "alta tensão é requisito obrigatório para o licenciamento ambiental e operação do "
        "sistema elétrico no Brasil, conforme a Resolução Normativa ANEEL nº 915 de "
        "23/02/2021, que revoga as anteriores nº 398/2010, nº 413/2010 e nº 616/2014."
    )
    rpt.add_body(
        "O campo elétrico (E) é proporcional à tensão aplicada e depende da geometria "
        "da torre e das alturas dos condutores. O campo magnético (B) é proporcional às "
        "correntes que circulam nos condutores. Ambos são calculados pelo Método da "
        "Simulação de Cargas (MSC), com a inclusão das imagens elétricas (Campo E) e "
        "das imagens complexas de Deri (Campo B), conforme procedimento descrito neste relatório."
    )
    rpt.add_body(
        "Este relatório foi produzido pelo software BK Estudos Elétricos da "
        "BK Engenharia e Tecnologia."
    )

    # ── 2. Objetivo ────────────────────────────────────────────────────
    rpt.add_heading1("2. Objetivo")
    rpt.add_body(
        "Determinar os perfis laterais de campo elétrico (kV/m) e campo magnético (µT) "
        "a 1,5 m de altura acima do nível do solo, verificando o atendimento aos limites "
        "regulatórios estabelecidos pela ANEEL RN 915/2021 e às recomendações internacionais "
        "da ICNIRP 2010 para exposição a campos eletromagnéticos na frequência de 60 Hz."
    )

    # ── 3. Documentos de Referência ────────────────────────────────────
    rpt.add_heading1("3. Documentos de Referência")
    refs_list = [
        "[1] ANEEL – Resolução Normativa nº 915, de 23/02/2021 (revoga nº 616/2014, 413/2010 e 398/2010)",
        "[2] ICNIRP – Guidelines for Limiting Exposure to Electric and Magnetic Fields "
            "(1 Hz–100 kHz). December 2010",
        "[3] PERRO, B.D.S. – Estudo dos Campos Eletromagnéticos em LTs à Frequência Industrial. "
            "Rio de Janeiro, 2007",
        "[4] PINHO, A.C. – Cálculo do Campo Elétrico 2D em LTs e o Efeito em Seres Vivos. "
            "UFSC, 1994",
        "[5] VELAME, M.R. – Cálculo dos Campos EM: MSC vs. Imagens. UFRB, 2019",
        "[6] EPRI – AC Transmission Line Reference Book 200 kV and above, 3rd ed., 2005",
        "[7] DERI, A. et al. – The Complex Ground Return Plane. IEEE Trans. PAS, ago. 1981",
        "[8] VIEIRA, H.R. – Acoplamento Magnético em LTs e Dutos Metálicos. UFSJ, 2013",
    ]
    for ref in refs_list:
        rpt.add_body(ref)

    # ── 4. Limites Normativos ──────────────────────────────────────────
    rpt.add_heading1("4. Limites Normativos — ANEEL RN 915/2021")
    rpt.add_body(
        "Conforme a Tabela 1 da ANEEL RN 915/2021, os limites recomendados para os valores "
        "dos campos elétrico e magnético variantes no tempo na frequência de 60 Hz são:"
    )
    rpt.add_kpi_table([
        ("Público em geral — Campo E", "4,17", "kV/m"),
        ("Público em geral — Campo B", "200",  "µT"),
        ("Ocupacional (acesso restrito) — Campo E", "8,33", "kV/m"),
        ("Ocupacional (acesso restrito) — Campo B", "1.000", "µT"),
    ])
    rpt.add_body(
        "Na análise, os limites para público em geral são aplicados ao entorno da instalação "
        "(perímetro externo / borda da faixa de servidão). Os limites ocupacionais são aplicados "
        "a áreas de acesso restrito a funcionários. "
        "Os cálculos são realizados para pontos a 1,5 m de altura acima do nível do solo, "
        "conforme art. 4º da RN 915/2021."
    )

    # ── 5. Metodologia de Cálculo ──────────────────────────────────────
    rpt.add_heading1("5. Metodologia de Cálculo")
    rpt.add_heading2("5.1 Campo Elétrico — MSC com Método das Imagens")
    rpt.add_body(
        "O Método da Simulação de Cargas (MSC) calcula o campo elétrico convertendo o potencial "
        "complexo de cada fase em cargas lineares equivalentes. O método é escolhido porque, "
        "diferentemente de métodos de elementos finitos, é analítico e computacionalmente "
        "eficiente para a geometria 2D de linhas de transmissão longas:"
    )
    rpt.add_equation(omml.eq_msc_charges(), "MSC — cargas equivalentes por fase [C/m]")
    rpt.add_body(
        "onde [Q̇] é a matriz das cargas complexas (C/m), [C] é a matriz de capacitâncias "
        "próprias e mútuas da linha (F/km) e [V̇] é a matriz das tensões fasoriais por fase. "
        "O Método das Imagens é aplicado para impor a condição de contorno de potencial zero "
        "no plano do solo: cada condutor real tem uma imagem espelhada em y_img = −yᵢ com "
        "carga de sinal oposto. A componente horizontal do campo em (x,y) resulta em:"
    )
    rpt.add_equation(omml.eq_msc_electric_field_x(), "Campo elétrico horizontal — MSC com imagens")
    rpt.add_body(
        "O segundo somatório representa a contribuição das imagens elétricas. "
        "O campo resultante é calculado como: |E| = √(|Ėxt|² + |Ėyt|²) [V/m], "
        "convertido para kV/m no relatório."
    )
    rpt.add_heading2("5.2 Campo Magnético — Imagens Complexas de Deri")
    rpt.add_body(
        "Para o campo magnético adota-se o Método das Imagens Complexas de Deri "
        "(DERI et al., IEEE Trans. PAS, 1981). Este método supera o modelo de solo perfeito "
        "porque inclui as correntes de retorno pelo solo, que percorrem uma profundidade "
        "finita determinada pela resistividade do solo — efeito relevante quando ρs < 100 Ω·m. "
        "A profundidade complexa de retorno é:"
    )
    rpt.add_equation(omml.eq_deri_depth(), "Profundidade complexa de retorno (Deri)")
    rpt.add_body(
        "onde ρs é a resistividade do solo (Ω·m), ω = 2πf é a frequência angular e "
        "μ₀ = 4π×10⁻⁷ H/m. A coordenada da imagem complexa do condutor i passa a ser:"
    )
    rpt.add_equation(omml.eq_deri_image(), "Coordenada da imagem complexa de Deri")
    rpt.add_body(
        "O campo magnético resultante — somando condutores reais e suas imagens — é:"
    )
    rpt.add_equation(omml.eq_magnetic_field_resultant(), "Campo magnético resultante |B| (T)")
    rpt.add_body(
        "Conforme VIEIRA (UFSJ, 2013), para ρs ≥ 50 Ω·m os resultados são praticamente "
        "equivalentes ao modelo de solo perfeito; para ρs < 50 Ω·m a correção de Deri "
        "pode aumentar |B| em até 15%."
    )
    rpt.add_heading2("5.3 Considerações Gerais")
    rpt.add_body(
        "a) Os efeitos dos cabos para-raios são desprezados no cálculo do campo elétrico, "
        "adotando condições mais desfavoráveis (conservador)."
    )
    rpt.add_body(
        "b) A superfície do solo é tratada como plana e equipotencial (potencial nulo). "
        "As estruturas de suporte não são consideradas no modelo 2D."
    )
    rpt.add_body(
        "c) As cargas dos condutores são uniformemente distribuídas (admitindo cabos "
        "homogêneos, de superfície lisa, comprimento infinito e sem influência de objetos próximos)."
    )

    # ── 6. Dados do Sistema ────────────────────────────────────────────
    rpt.add_heading1("6. Dados do Sistema Elétrico")
    E_max   = results.get("E_max_kV_m", 0)
    B_max   = results.get("B_max_uT", 0)
    E_lim_g = cfg.get("E_limit_geral",  cfg.get("E_limit", 4.17))
    B_lim_g = cfg.get("B_limit_geral",  cfg.get("B_limit", 200.0))
    E_lim_o = cfg.get("E_limit_ocup",   8.33)
    B_lim_o = cfg.get("B_limit_ocup",   1000.0)
    h_obs   = cfg.get("h_obs_m",  1.5)
    rho_s   = cfg.get("rho_solo", 100.0)
    freq_hz = cfg.get("freq_hz",  60.0)
    x_min   = results.get("x_min_m", cfg.get("x_min_m", -30.0))
    x_max   = results.get("x_max_m", cfg.get("x_max_m",  30.0))

    rpt.add_kpi_table([
        ("Tensão nominal (L-L)", f"{cfg.get('voltage_kv', 0):.1f}", "kV"),
        ("Frequência", f"{freq_hz:.0f}", "Hz"),
        ("Altura de avaliação (h_obs)", f"{h_obs:.2f}", "m"),
        ("Resistividade do solo (ρs)", f"{rho_s:.0f}", "Ω·m"),
        ("Faixa lateral avaliada", f"{x_min:.0f} a {x_max:.0f}", "m"),
    ])

    # ── 7. Resultados ──────────────────────────────────────────────────
    rpt.add_heading1("7. Resultados Obtidos")
    rpt.add_body(
        f"Os valores máximos de campo elétrico e magnético obtidos a {h_obs:.1f} m "
        f"de altura acima do nível do solo são apresentados na tabela abaixo, comparados "
        f"com os limites da ANEEL RN 915/2021:"
    )
    rpt.add_kpi_table([
        ("|E| máximo calculado",     f"{E_max:.4f}",  "kV/m"),
        ("Limite E — Público geral", f"{E_lim_g:.2f}", "kV/m"),
        ("Limite E — Ocupacional",   f"{E_lim_o:.2f}", "kV/m"),
        ("|B| máximo calculado",     f"{B_max:.4f}",  "µT"),
        ("Limite B — Público geral", f"{B_lim_g:.1f}", "µT"),
        ("Limite B — Ocupacional",   f"{B_lim_o:.0f}", "µT"),
        ("x de |E|máx",              f"{results.get('x_E_max_m', 0):.1f}", "m"),
        ("x de |B|máx",              f"{results.get('x_B_max_m', 0):.1f}", "m"),
    ])

    # Gráficos: Campo E e Campo B em figuras separadas
    if "x_m" in results and "E_kV_m" in results and "B_uT" in results:
        x  = results["x_m"]
        E  = results["E_kV_m"]
        B  = results["B_uT"]

        # Campo Elétrico
        fig_e, ax_e = plt.subplots(figsize=(13, 5))
        ax_e.plot(x, E, color="#1565C0", linewidth=2.0, label="|E|(x)")
        ax_e.axhline(E_lim_g, color="#E53935",  linestyle="--", linewidth=1.5,
                     label=f"Lim. geral = {E_lim_g:.2f} kV/m")
        ax_e.axhline(E_lim_o, color="#FB8C00", linestyle=":",  linewidth=1.5,
                     label=f"Lim. ocup. = {E_lim_o:.2f} kV/m")
        ax_e.axvline(results.get("x_E_max_m", 0), color="#546E7A", linestyle="-.",
                     linewidth=1.0, label=f"x_Emáx = {results.get('x_E_max_m',0):.1f} m")
        ax_e.set_xlabel("Distância lateral (m)", fontsize=11)
        ax_e.set_ylabel("|E| (kV/m)", fontsize=11)
        ax_e.set_title(f"Perfil lateral de campo elétrico — h_obs = {h_obs:.1f} m (ANEEL RN 915/2021)",
                       fontsize=11)
        ax_e.legend(fontsize=9); ax_e.grid(True, alpha=0.35)
        rpt.add_figure_from_matplotlib(fig_e,
            f"Figura 1 — Perfil de campo elétrico |E|(x) a {h_obs:.1f} m de altura")

        # Campo Magnético
        fig_b, ax_b = plt.subplots(figsize=(13, 5))
        ax_b.plot(x, B, color="#00897B", linewidth=2.0, label="|B|(x)")
        ax_b.axhline(B_lim_g, color="#E53935",  linestyle="--", linewidth=1.5,
                     label=f"Lim. geral = {B_lim_g:.1f} µT")
        ax_b.axhline(B_lim_o, color="#FB8C00", linestyle=":",  linewidth=1.5,
                     label=f"Lim. ocup. = {B_lim_o:.0f} µT")
        ax_b.axvline(results.get("x_B_max_m", 0), color="#546E7A", linestyle="-.",
                     linewidth=1.0, label=f"x_Bmáx = {results.get('x_B_max_m',0):.1f} m")
        ax_b.set_xlabel("Distância lateral (m)", fontsize=11)
        ax_b.set_ylabel("|B| (µT)", fontsize=11)
        ax_b.set_title(f"Perfil lateral de campo magnético — h_obs = {h_obs:.1f} m (ANEEL RN 915/2021)",
                       fontsize=11)
        ax_b.legend(fontsize=9); ax_b.grid(True, alpha=0.35)
        rpt.add_figure_from_matplotlib(fig_b,
            f"Figura 2 — Perfil de campo magnético |B|(x) a {h_obs:.1f} m de altura")

    # ── 8. Conclusão ───────────────────────────────────────────────────
    rpt.add_heading1("8. Conclusão")
    E_ok_g = E_max <= E_lim_g
    E_ok_o = E_max <= E_lim_o
    B_ok_g = B_max <= B_lim_g
    B_ok_o = B_max <= B_lim_o

    status_E = "ATENDE" if E_ok_g else ("ATENDE (ocup.)" if E_ok_o else "NÃO ATENDE")
    status_B = "ATENDE" if B_ok_g else ("ATENDE (ocup.)" if B_ok_o else "NÃO ATENDE")

    rpt.add_kpi_table([
        ("Campo E — Status geral",      status_E, ""),
        ("Campo B — Status geral",      status_B, ""),
    ])

    if E_ok_g and B_ok_g:
        rpt.add_body(
            f"Os campos elétrico ({E_max:.4f} kV/m) e magnético ({B_max:.4f} µT) calculados "
            f"ATENDEM aos limites para público em geral estabelecidos pela ANEEL RN 915/2021 "
            f"(E ≤ {E_lim_g:.2f} kV/m e B ≤ {B_lim_g:.0f} µT). "
            f"A linha de transmissão pode operar sem restrições adicionais de campos "
            f"eletromagnéticos na faixa de servidão."
        )
    else:
        if not E_ok_g:
            txt_e = (
                f"O campo elétrico máximo ({E_max:.4f} kV/m) EXCEDE o limite para público em "
                f"geral de {E_lim_g:.2f} kV/m. "
            )
            if E_ok_o:
                txt_e += (
                    f"Entretanto, ATENDE ao limite ocupacional de {E_lim_o:.2f} kV/m, "
                    f"sendo aceitável em área de acesso restrito a funcionários. "
                )
            txt_e += (
                "Recomenda-se avaliar: aumento da altura dos condutores, compactação da "
                "geometria ou ampliação da faixa de servidão."
            )
            rpt.add_body(txt_e)
        if not B_ok_g:
            txt_b = (
                f"O campo magnético máximo ({B_max:.4f} µT) EXCEDE o limite para público em "
                f"geral de {B_lim_g:.0f} µT. "
            )
            if B_ok_o:
                txt_b += (
                    f"Entretanto, ATENDE ao limite ocupacional de {B_lim_o:.0f} µT. "
                )
            txt_b += (
                "Recomenda-se avaliar: otimização do arranjo de fases (cancelamento magnético), "
                "aumento da altura dos condutores ou redução da corrente operativa."
            )
            rpt.add_body(txt_b)

    rpt.add_body(SOFTWARE_NOTE)
    rpt.add_references_section(REFS_FIELDS)


# ====================================================================
# 4. AMPACIDADE E FLECHA
# ====================================================================

def _add_sag_tension_tables(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    """
    Gera tabelas de Flecha e Tracionamento no relatório Word.
    Temperaturas: 0°C a 75°C (passo 5°C).
    Vãos: 30m a 1500m (passo 30m).
    Metodologia: equação de mudança de estado (NBR 5422 / Fuchs Cap. 4).
    Tabelas em landscape para caberem.
    """
    try:
        from core.ampacity_sag import (
            compute_sag_tension_table, plot_sag_temperature_span,
            AmpacitySagConfig, find_cable, default_cable_db,
        )
    except ImportError:
        try:
            from ..core.ampacity_sag import (
                compute_sag_tension_table, plot_sag_temperature_span,
                AmpacitySagConfig, find_cable, default_cable_db,
            )
        except ImportError:
            rpt.add_body("(Tabela de flechas indisponível — módulo ampacity_sag não encontrado)")
            return

    # Identifica o cabo a partir dos resultados
    cable_key = cfg.get("cable_phase_key", "ACSR_477")
    circs = results.get("circuits", [])
    if circs and circs[0].get("cable_key"):
        cable_key = circs[0]["cable_key"]

    cable_db = default_cable_db()
    cable = find_cable(cable_db, cable_key)
    if cable is None:
        rpt.add_body(f"(Cabo '{cable_key}' não encontrado para tabela de flechas)")
        return

    # Monta config mínima necessária
    amp_cfg = AmpacitySagConfig(
        design_tension_ratio=cfg.get("design_tension_ratio", 0.25),
        max_conductor_temp_C=cfg.get("max_conductor_temp_C", 75),
    )

    # Calcula tabela completa
    table = compute_sag_tension_table(
        cable=cable,
        config=amp_cfg,
        temp_min_C=0.0,
        temp_max_C=75.0,
        temp_step_C=5.0,
        span_min_m=30.0,
        span_max_m=1500.0,
        span_step_m=30.0,
        theta_ref_C=20.0,
    )

    temps = table.temperatures_C   # [0, 5, 10, ..., 75]  = 16 cols
    spans = table.spans_m          # [30, 60, ..., 1500]   = 50 rows

    # ── Seção de Flechas e Tracionamento ──────────────────────
    rpt.add_heading1("Tabela de Flechas e Tracionamento")
    rpt.add_body(
        f"As tabelas a seguir apresentam os valores de flecha (m) e tração horizontal (kN) "
        f"para o condutor {cable_key} ({cable.material}), calculados pela equação de mudança "
        f"de estado conforme NBR 5422:1985 (item 5.2) e Fuchs — Transmissão de Energia "
        f"Elétrica, Capítulo 4. O estado de referência adotado é θ_ref = {table.theta_ref_C:.0f}°C "
        f"com tração H_ref = {table.H_ref_N/1000:.2f} kN "
        f"({cfg.get('design_tension_ratio', 0.25)*100:.0f}% da carga de ruptura)."
    )
    rpt.add_body(
        f"Dados do condutor: peso próprio w = {table.w_N_m:.4f} N/m, "
        f"EA = {table.EA/1e6:.1f} MN, "
        f"α = {table.alpha*1e6:.1f} × 10⁻⁶ /°C. "
        f"Modelo parabólico: f = wL² / (8H)."
    )

    # ── Gráfico matplotlib (antes das tabelas, em portrait) ───
    try:
        b64_chart = plot_sag_temperature_span(table)
        rpt.add_figure_from_base64(b64_chart, "Flecha e Tração × Vão para diversas temperaturas", 16.0)
    except Exception:
        pass

    # Divide temperaturas em 2 faixas para caber em landscape
    mid = len(temps) // 2
    temp_groups = [
        (temps[:mid], "Faixa 1"),     # 0–30°C (ou ~metade)
        (temps[mid:], "Faixa 2"),     # 35–75°C
    ]

    # ── TABELA DE FLECHAS (landscape) ─────────────────────────
    rpt.start_landscape_section()
    rpt.add_heading2("Tabela de Flechas (m)")

    for tg_temps, tg_label in temp_groups:
        tg_idxs = [temps.index(t) for t in tg_temps]
        headers = ["Vão (m)"] + [f"{t:.0f}°C" for t in tg_temps]
        rows = []
        for j, span in enumerate(spans):
            row = [f"{span:.0f}"]
            for ti in tg_idxs:
                row.append(f"{table.sag_m[ti][j]:.2f}")
            rows.append(row)
        caption = f"Flecha (m) — {tg_temps[0]:.0f}°C a {tg_temps[-1]:.0f}°C — Cabo {cable_key}"
        rpt.add_compact_table(headers, rows, caption, font_size=6)
        rpt.add_page_break()

    # ── TABELA DE TRAÇÕES (landscape) ─────────────────────────
    rpt.add_heading2("Tabela de Tração Horizontal (kN)")

    for tg_temps, tg_label in temp_groups:
        tg_idxs = [temps.index(t) for t in tg_temps]
        headers = ["Vão (m)"] + [f"{t:.0f}°C" for t in tg_temps]
        rows = []
        for j, span in enumerate(spans):
            row = [f"{span:.0f}"]
            for ti in tg_idxs:
                row.append(f"{table.tension_N[ti][j]/1000:.2f}")
            rows.append(row)
        caption = f"Tração (kN) — {tg_temps[0]:.0f}°C a {tg_temps[-1]:.0f}°C — Cabo {cable_key}"
        rpt.add_compact_table(headers, rows, caption, font_size=6)
        if tg_label != temp_groups[-1][1]:
            rpt.add_page_break()

    rpt.end_landscape_section()


def report_ampacidade(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    rpt.add_cover("Estudo de Ampacidade e Flecha", "Linha de Transmissão")
    rpt.add_page_break()
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    rpt.add_heading1("Introdução")
    rpt.add_body(
        "A ampacidade de um condutor é a máxima corrente contínua que pode ser transportada "
        "sem que a temperatura do condutor exceda o limite térmico especificado, considerando "
        "as condições ambientais de projeto. O estudo é fundamental para garantir a segurança "
        "operacional e evitar danos ao condutor por sobreaquecimento."
    )
    rpt.add_body(
        "A metodologia segue a norma IEEE Std 738-2012, que estabelece o balanço térmico entre "
        "os ganhos de calor (efeito Joule e radiação solar) e as perdas (convecção natural e "
        "forçada, radiação térmica). A flecha do condutor é calculada pela teoria da catenária "
        "simplificada (parábola), considerando a temperatura de operação."
    )
    rpt.add_body(
        "Relatório produzido pelo Software BK Estudos Elétricos da BK Engenharia e Tecnologia."
    )

    rpt.add_heading1("Objetivo")
    rpt.add_body(
        "Determinar a corrente máxima admissível (ampacidade) do condutor da linha de transmissão "
        "nas condições ambientais de projeto, bem como a flecha correspondente à temperatura máxima "
        "de operação, verificando o atendimento aos critérios de segurança e distâncias mínimas."
    )

    rpt.add_heading1("3. Metodologia")

    rpt.add_heading2("3.1 Balanço térmico em regime permanente (IEEE 738)")
    rpt.add_body(
        "A ampacidade é determinada pela temperatura máxima admissível do condutor "
        "(tipicamente 75 °C para ACSR operação normal, ou 100 °C para emergência). "
        "A norma IEEE Std 738-2012 é adotada porque estabelece um modelo físico completo "
        "do balanço de calor, explicitando cada mecanismo de troca térmica:"
    )
    rpt.add_equation(omml.eq_ieee738_thermal(), "Balanço térmico em regime permanente (IEEE 738)")
    rpt.add_body(
        "Onde: q_c é o calor dissipado por convecção natural e forçada (W/m) — "
        "principal mecanismo de resfriamento; q_r é o calor dissipado por radiação "
        "infravermelha do condutor aquecido (W/m); q_s é o ganho por absorção de "
        "radiação solar (W/m); I² · R(Tc) é a geração de calor por efeito Joule (W/m). "
        "O calor de convecção forçada depende da velocidade do vento — razão pela qual "
        "o projeto adota condição conservadora (vento mínimo de 0,6 m/s)."
    )
    rpt.add_body(
        "A resistência AC na temperatura Tc é corrigida linearmente a partir do valor "
        "tabelado a 25 °C, usando o coeficiente de temperatura do material conductor. "
        "Para ampacidade, isola-se I da equação de balanço (q_c + q_r − q_s = I²·R(Tc))."
    )

    rpt.add_heading2("3.2 Flecha e tracionamento (parábola)")
    rpt.add_body(
        "A flecha do condutor cresce com a temperatura porque o aumento térmico dilata o "
        "comprimento do cabo e reduz a tensão mecânica. O cálculo de flecha é obrigatório "
        "para verificar as distâncias mínimas de segurança à vegetação e ao solo "
        "(NBR 5422, Tabela 6). A aproximação parabólica é adequada quando flecha < vão/8:"
    )
    rpt.add_equation(omml.eq_sag(), "Flecha parabólica (approximação para f < L/8)")
    rpt.add_body(
        "onde w é o peso linear total do condutor (N/m) — incluindo gelo e vento se aplicável —, "
        "L é o vão (m) e T é a tração horizontal (N). A tração é obtida pela equação de "
        "mudança de estado (EDS — Every Day Stress), que relaciona a variação de temperatura "
        "com a variação de comprimento e carga mecânica no condutor."
    )

    # Resultados
    rpt.add_heading1("Resultados Obtidos")
    rpt.add_heading2("Dados de Entrada")
    rpt.add_kpi_table([
        ("T ambiente", f"{cfg.get('ambient_temp_C', 35):.0f}", "°C"),
        ("T máx condutor", f"{cfg.get('max_conductor_temp_C', 75):.0f}", "°C"),
        ("Vento", f"{cfg.get('wind_speed_m_s', 0.6):.1f}", "m/s"),
        ("Irradiância", f"{cfg.get('solar_irradiance', 1000):.0f}", "W/m²"),
    ])

    if "circuits" in results:
        for i, circ in enumerate(results["circuits"]):
            rpt.add_heading2(f"Circuito {i+1}")
            rpt.add_kpi_table([
                ("Ampacidade", f"{circ.get('I_max_A', 0):.1f}", "A"),
                ("I operação", f"{circ.get('I_oper_A', 0):.1f}", "A"),
                ("T limite", f"{circ.get('T_lim_C', 0):.1f}", "°C"),
                ("Flecha", f"{circ.get('sag_m', 0):.2f}", "m"),
                ("Status", circ.get("status", "OK"), ""),
            ])

    # ── Tabela de Flechas e Tracionamento (0°C a 75°C, vãos até 1500m) ──
    _add_sag_tension_tables(rpt, results, cfg)

    rpt.add_heading1("Conclusão")
    rpt.add_body(
        "A ampacidade e a flecha foram determinadas conforme a norma IEEE Std 738-2012. "
        "Os valores obtidos devem ser verificados em relação aos critérios de distâncias "
        "mínimas de segurança conforme NBR 5422 e regulamentações específicas do projeto."
    )
    rpt.add_body(SOFTWARE_NOTE)
    rpt.add_references_section(REFS_AMPACITY)


# ====================================================================
# 5. RI E RA
# ====================================================================

def report_ri_ra(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    rpt.add_cover("Estudo de Interferência Radioelétrica e Ruído Audível", "Linha de Transmissão")
    rpt.add_page_break()
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    rpt.add_heading1("Introdução")
    rpt.add_body(
        "A interferência radioelétrica (RI) e o ruído audível (RA) são efeitos indesejados "
        "associados ao fenômeno de corona em linhas de transmissão de alta tensão. A RI pode "
        "afetar a recepção de sinais de radiodifusão AM na vizinhança da linha, enquanto o RA "
        "pode causar incômodo às comunidades próximas, especialmente em condições de chuva."
    )
    rpt.add_body(
        "A avaliação segue metodologias consagradas do EPRI, CIGRÉ e IEEE, utilizando modelos "
        "empíricos calibrados com medições de campo. Os limites são definidos por normas "
        "nacionais e internacionais para garantir a compatibilidade eletromagnética."
    )
    rpt.add_body(
        "Relatório produzido pelo Software BK Estudos Elétricos da BK Engenharia e Tecnologia."
    )

    rpt.add_heading1("Objetivo")
    rpt.add_body(
        "Estimar os níveis de interferência radioelétrica (dBµV/m) e ruído audível (dBA) "
        "na borda da faixa de servidão, verificando o atendimento aos limites regulatórios "
        "em condições de tempo seco e chuvoso."
    )

    rpt.add_heading1("3. Metodologia")

    rpt.add_heading2("3.1 Interferência radioelétrica (RI)")
    rpt.add_body(
        "A RI é causada pelas descargas parciais de corona que geram pulsos de corrente "
        "impulsiva nos condutores, irradiando energia eletromagnética na faixa de AM "
        "(500 kHz a 1,6 MHz). O nível de RI depende diretamente do gradiente superficial "
        "do condutor — razão pela qual o cálculo de corona é um pré-requisito. "
        "As fórmulas empíricas do EPRI são calibradas com medições de campo em centenas "
        "de linhas e são as referências adotadas internacionalmente (IEEE 430, CIGRÉ TB 61):"
    )
    rpt.add_equation(omml.eq_ri_epri(), "Nível de RI no feixe (EPRI) — dBµV/m")
    rpt.add_body(
        "onde Ec é o gradiente superficial máximo (kV/cm), r é o raio do subcondutor (cm), "
        "n é o número de subcondutores e os coeficientes k₁–k₄ são calibrados pelo EPRI "
        "separadamente para condição seca e chuvosa. A atenuação lateral com a distância D "
        "é somada ao nível base com o termo −10·log(D/D₀)."
    )

    rpt.add_heading2("3.2 Ruído audível (RA)")
    rpt.add_body(
        "O RA é gerado pelas micro-explosões das bolhas d'água nas superfícies dos condutores "
        "durante chuva — fenômeno que amplia o corona e produz pressão sonora audível "
        "(frequências de 100 Hz a 10 kHz, ponderação A). A condição crítica é chuva moderada, "
        "não a seca, ao contrário da RI. O modelo EPRI é:"
    )
    rpt.add_equation(omml.eq_ra_epri(), "Nível de RA (EPRI) — dBA")
    rpt.add_body(
        "O termo −10·log(D/D₀) representa a atenuação com a distância D à linha. "
        "O nível medido na borda da faixa de servidão é o critério de verificação. "
        "Limites típicos adotados no Brasil: RI < 46 dBµV/m, RA < 50 dBA (chuva)."
    )

    rpt.add_heading1("Resultados Obtidos")
    if "circuits" in results:
        for i, circ in enumerate(results["circuits"]):
            rpt.add_heading2(f"Circuito {i+1}")
            rpt.add_kpi_table([
                ("RI borda (chuva)", f"{circ.get('RI_edge_chuva', 0):.1f}", "dBµV/m"),
                ("RA borda (chuva)", f"{circ.get('RA_edge_chuva', 0):.1f}", "dBA"),
                ("Ec superficial", f"{circ.get('Ec_kV_cm', 0):.2f}", "kV/cm"),
                ("Atende RI", "SIM" if not circ.get("exceeds_RI", False) else "NÃO", ""),
                ("Atende RA", "SIM" if not circ.get("exceeds_RA", False) else "NÃO", ""),
            ])

            if "distances_m" in circ and "RI_chuva" in circ:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                d = circ["distances_m"]
                ax1.plot(d, circ.get("RI_seco", []), label="RI seco", color="#1565C0")
                ax1.plot(d, circ["RI_chuva"], label="RI chuva", color="#E53935")
                ax1.set_xlabel("Distância (m)")
                ax1.set_ylabel("RI (dBµV/m)")
                ax1.set_title(f"Interferência Radioelétrica – Circ. {i+1}")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2.plot(d, circ.get("RA_seco", []), label="RA seco", color="#1565C0")
                ax2.plot(d, circ.get("RA_chuva", []), label="RA chuva", color="#E53935")
                ax2.set_xlabel("Distância (m)")
                ax2.set_ylabel("RA (dBA)")
                ax2.set_title(f"Ruído Audível – Circ. {i+1}")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                rpt.add_figure_from_matplotlib(fig, f"Perfis de RI e RA – Circuito {i+1}")

    rpt.add_heading1("Conclusão")
    rpt.add_body(
        "Os níveis de RI e RA foram estimados conforme metodologia EPRI/CIGRÉ. "
        "Caso algum limite seja excedido, recomenda-se avaliar o aumento do diâmetro equivalente "
        "do feixe, aumento do número de subcondutores ou melhoria das ferragens."
    )
    rpt.add_body(SOFTWARE_NOTE)
    rpt.add_references_section(REFS_RIRA)


# ====================================================================
# Graficos auxiliares
# ====================================================================

def _plot_pi_model(R: float, X: float, B: float, L_km: float):
    """Desenha diagrama do circuito pi equivalente."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-2, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Barra esquerda (envio)
    ax.plot([0, 0], [-1, 3], color="#084C89", linewidth=3)
    ax.text(0, 3.3, "Vs", ha="center", fontsize=12, fontweight="bold", color="#084C89")

    # Barra direita (recebimento)
    ax.plot([10, 10], [-1, 3], color="#084C89", linewidth=3)
    ax.text(10, 3.3, "Vr", ha="center", fontsize=12, fontweight="bold", color="#084C89")

    # Impedancia serie (retangulo)
    rect = plt.Rectangle((3.5, 0.7), 3, 1.6, fill=True, facecolor="#E8F0FE",
                          edgecolor="#084C89", linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 1.5, f"Z = {R:.2f} + j{X:.2f} Ω", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#084C89")
    ax.text(5, 1.0, f"(L = {L_km:.0f} km)", ha="center", va="center",
            fontsize=8, color="#666666")

    # Linhas horizontais
    ax.plot([0, 3.5], [1.5, 1.5], color="#333333", linewidth=2)
    ax.plot([6.5, 10], [1.5, 1.5], color="#333333", linewidth=2)

    # Y/2 esquerdo
    ax.plot([1.5, 1.5], [1.5, -0.5], color="#333333", linewidth=1.5)
    ax.plot([1.5, 1.5], [-0.5, -1], color="#333333", linewidth=1.5)
    rect_y1 = plt.Rectangle((1.0, -0.5), 1.0, 0.8, fill=True, facecolor="#FFF3E0",
                             edgecolor="#FB8C00", linewidth=1.5)
    ax.add_patch(rect_y1)
    B_half = B / 2 * 1e6  # µS
    ax.text(1.5, -0.1, f"jB'L/2\n{B_half:.2f} µS", ha="center", va="center",
            fontsize=7, color="#E65100")

    # Y/2 direito
    ax.plot([8.5, 8.5], [1.5, -0.5], color="#333333", linewidth=1.5)
    ax.plot([8.5, 8.5], [-0.5, -1], color="#333333", linewidth=1.5)
    rect_y2 = plt.Rectangle((8.0, -0.5), 1.0, 0.8, fill=True, facecolor="#FFF3E0",
                             edgecolor="#FB8C00", linewidth=1.5)
    ax.add_patch(rect_y2)
    ax.text(8.5, -0.1, f"jB'L/2\n{B_half:.2f} µS", ha="center", va="center",
            fontsize=7, color="#E65100")

    # Terra
    ax.plot([0, 10], [-1, -1], color="#333333", linewidth=1, linestyle="--")
    ax.text(5, -1.5, "Modelo π equivalente", ha="center", fontsize=11,
            fontweight="bold", color="#084C89")

    return fig


def _plot_params_comparison(circuits: list):
    """Grafico de barras comparando parametros entre circuitos."""
    n = len(circuits)
    labels = [f"Circ. {i+1}" for i in range(n)]
    params = {
        "R' (Ω/km)": [c.get("R_ohm_km", 0) for c in circuits],
        "X' (Ω/km)": [c.get("X_ohm_km", 0) for c in circuits],
        "Zc (Ω)": [c.get("Zc_ohm", 0) for c in circuits],
        "SIL (MW)": [c.get("SIL_MW", 0) for c in circuits],
    }
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ["#1565C0", "#00897B", "#FB8C00", "#7B1FA2"]
    for ax, (name, vals), color in zip(axes, params.items(), colors):
        ax.bar(labels, vals, color=color, alpha=0.85)
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Comparação de Parâmetros entre Circuitos", fontsize=12, fontweight="bold")
    return fig

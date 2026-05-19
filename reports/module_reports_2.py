# reports/module_reports_2.py
# ====================================================================
# Geradores de conteudo — Modulos 6 a 13
# ====================================================================

from __future__ import annotations
from typing import Any, Dict, List
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bk_docx import BKReport
from . import omml
from .module_reports import (
    SOFTWARE_NOTE,
    REFS_SHIELDING, REFS_VMAX, REFS_COORD, REFS_RECLOSING, REFS_EMI, REFS_PF,
)


# ====================================================================
# 6. BLINDAGEM (SHIELDING)
# ====================================================================

def report_blindagem(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    rpt.add_cover("Estudo de Blindagem contra Descargas Atmosféricas", "Linha de Transmissão")
    rpt.add_page_break()
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    # ── Verificação: linha sem cabo-guarda ─────────────────────────────
    if not cfg.get("cabo_guarda", True):
        rpt.add_heading1("1. Aplicabilidade")
        rpt.add_body(
            "A linha em estudo não possui cabos-guarda (para-raios aéreos). "
            "Nesse caso, a análise de blindagem pelo Modelo Eletrogeométrico (EGM) e a avaliação "
            "de backflashover não se aplicam, pois ambas pressupõem a existência de cabo-guarda "
            "instalado no topo da torre."
        )
        rpt.add_body(
            "Para linhas sem cabo-guarda, a proteção contra descargas atmosféricas deve ser "
            "avaliada por metodologia alternativa (ex.: blindagem por objeto adjacente, SPDA "
            "dedicado por torre, ou aceitação de risco por critério estatístico de saídas por "
            "raio conforme CIGRÉ TB 63). A decisão de suprimir o cabo-guarda deve ser "
            "fundamentada em estudo específico de relação custo-benefício e nível ceráunico local."
        )
        rpt.add_body(
            "Referências: IEEE Std 1243-1997 §4; CIGRÉ TB 63 §3.2; ABNT NBR 5419:2015 Parte 2."
        )
        return  # Encerra o relatório de blindagem aqui

    rpt.add_heading1("Introdução")
    rpt.add_body(
        "A blindagem contra descargas atmosféricas é um requisito fundamental no projeto de "
        "linhas de transmissão de alta tensão, visando proteger os condutores fase contra "
        "incidência direta de raios e minimizar as taxas de desligamento por backflashover. "
        "A eficácia da blindagem depende do posicionamento dos cabos-guarda em relação às fases "
        "e da resistência de aterramento das torres."
    )
    rpt.add_body(
        "A metodologia segue o modelo eletrogeométrico (EGM) conforme IEEE Std 1243-1997 "
        "e IEEE Std 998-2012, complementado pela análise de backflashover conforme publicações "
        "do CIGRÉ. Os critérios de ângulo de blindagem e taxa de flashover são avaliados para "
        "garantir o desempenho desejado da linha."
    )
    rpt.add_body(
        "Relatório produzido pelo Software BK Estudos Elétricos da BK Engenharia e Tecnologia."
    )

    rpt.add_heading1("Objetivo")
    rpt.add_body(
        "Verificar a eficácia da blindagem dos cabos-guarda na proteção dos condutores fase "
        "contra incidência direta de descargas atmosféricas, avaliando os ângulos de proteção "
        "e a taxa de backflashover."
    )

    rpt.add_heading1("3. Metodologia")

    rpt.add_heading2("3.1 Modelo eletrogeométrico (EGM) — IEEE 1243")
    rpt.add_body(
        "O Modelo Eletrogeométrico (EGM) é adotado porque correlaciona a geometria do "
        "relâmpago com a da torre para determinar se o líder descendente atingirá a fase "
        "ou o cabo-guarda. O EGM é o método padrão do IEEE Std 1243-1997 para linhas de "
        "transmissão porque: (i) é físico (baseado na teoria do líder escalonado), "
        "(ii) foi calibrado com dados de centenas de ocorrências reais de raio, e "
        "(iii) é computacionalmente simples. O ângulo de blindagem para cada fase é:"
    )
    rpt.add_equation(omml.eq_shielding_angle(), "Ângulo de blindagem θ (°)")
    rpt.add_body(
        "onde d_h é a distância horizontal entre o cabo-guarda e o condutor fase, e Δh é a "
        "diferença de altura. Ângulos θ < 0° indicam blindagem negativa (over-shielding). "
        "O IEEE 1243 estabelece limites de θ em função da tensão nominal: "
        "≤ 20° para 69–230 kV; ≤ 10–15° para 345–765 kV; ≤ 0° para ≥ 500 kV."
    )

    rpt.add_heading2("3.2 Backflashover — sobretensão na torre")
    rpt.add_body(
        "Mesmo com blindagem eficaz, uma descarga que atinge o cabo-guarda pode causar "
        "flashover inverso (backflashover) da torre para o condutor fase, se a tensão "
        "desenvolvida na torre superar o nível básico de isolamento (NBI) da cadeia. "
        "A tensão na torre é determinada pela impedância de surto e aterramento da torre:"
    )
    rpt.add_equation(omml.eq_vtower(), "Tensão na torre durante impacto de raio (backflashover)")
    rpt.add_body(
        "onde I_desc é a corrente de pico do raio (kA), R_pe é a resistência de aterramento "
        "da torre (Ω) e L·dI/dt é a componente indutiva durante a frente de onda (td ≈ 2 µs). "
        "O backflashover ocorre quando V_torre > NBI. Por isso a resistência de aterramento "
        "deve ser minimizada — tipicamente ≤ 10–20 Ω (IEEE 998, ABNT NBR 5419)."
    )

    # Resultados
    rpt.add_heading1("Resultados Obtidos")
    worst = results.get("worst_theta_deg", 0)
    all_prot = results.get("all_phases_protected", False)
    bf = results.get("backflash_fraction", 0)

    rpt.add_kpi_table([
        ("Pior ângulo", f"{worst:.1f}", "°"),
        ("Todas protegidas", "SIM" if all_prot else "NÃO", ""),
        ("Fração backflash", f"{bf:.4f}", ""),
    ])

    # Tabela por fase
    if "per_phase" in results:
        headers = ["Circuito", "Fase", "θ (°)", "Cabo GW", "Δh (m)", "d_horiz (m)", "Protegida"]
        rows = []
        for ph in results["per_phase"]:
            rows.append([
                str(ph.get("circuit", "")),
                ph.get("phase", ""),
                f"{ph.get('theta_deg', 0):.1f}",
                ph.get("gw_name", ""),
                f"{ph.get('delta_h', 0):.2f}",
                f"{ph.get('d_horiz', 0):.2f}",
                "SIM" if ph.get("protected", False) else "NÃO",
            ])
        rpt.add_result_table(headers, rows, "Ângulos de blindagem por fase")

    # Grafico V_torre x I_descarga
    if "I_kA" in results and "V_tower_kV" in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(results["I_kA"], results["V_tower_kV"], color="#1565C0", linewidth=2,
                label="V torre")
        bil = cfg.get("BIL_kV", 0)
        if bil > 0:
            ax.axhline(bil, color="#E53935", linestyle="--", linewidth=1.5, label=f"BIL = {bil:.0f} kV")
        ax.set_xlabel("Corrente de descarga (kA)")
        ax.set_ylabel("Tensão na torre (kV)")
        ax.set_title("Tensão na Torre vs Corrente de Descarga")
        ax.legend()
        ax.grid(True, alpha=0.3)
        rpt.add_figure_from_matplotlib(fig, "Curva V torre × I descarga com limite BIL")

    rpt.add_heading1("Conclusão")
    if all_prot:
        rpt.add_body(
            f"Todas as fases estão adequadamente protegidas pelos cabos-guarda. O pior ângulo "
            f"de blindagem é de {worst:.1f}°, dentro do limite aceitável. "
        )
    else:
        rpt.add_body(
            f"Existem fases com ângulo de blindagem acima do limite aceitável (pior caso: {worst:.1f}°). "
            f"Recomenda-se reposicionar os cabos-guarda ou adicionar um segundo cabo-guarda para "
            f"garantir proteção adequada."
        )
    rpt.add_body(SOFTWARE_NOTE)
    rpt.add_references_section(REFS_SHIELDING)


# ====================================================================
# 7. ISOLAMENTO Vmax
# ====================================================================

def report_vmax_insulation(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    rpt.add_cover("Estudo de Isolamento – Sobretensões Temporárias", "Linha de Transmissão")
    rpt.add_page_break()
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    rpt.add_heading1("Introdução")
    rpt.add_body(
        "A verificação do isolamento frente a sobretensões temporárias (TOV) é parte essencial "
        "da coordenação de isolamento conforme IEC 60071-1/2 e ABNT NBR 6939. As sobretensões "
        "temporárias podem ser causadas por rejeição de carga, faltas à terra, ferrorressonância "
        "e outras perturbações do sistema."
    )
    rpt.add_body(
        "A avaliação verifica se a suportabilidade à frequência industrial (power frequency) dos "
        "equipamentos instalados na linha é adequada considerando o fator de sobretensão k_TOV, "
        "a correção atmosférica por altitude (Ka) e as distâncias de escoamento (creepage) para "
        "diferentes níveis de poluição."
    )
    rpt.add_body(
        "Relatório produzido pelo Software BK Estudos Elétricos da BK Engenharia e Tecnologia."
    )

    rpt.add_heading1("Objetivo")
    rpt.add_body(
        "Verificar a adequação do isolamento dos equipamentos da linha de transmissão frente "
        "a sobretensões temporárias (TOV), incluindo margem de segurança à frequência industrial "
        "e distância de escoamento para o nível de poluição aplicável."
    )

    rpt.add_heading1("3. Metodologia")

    rpt.add_heading2("3.1 Sobretensão temporária (TOV)")
    rpt.add_body(
        "As sobretensões temporárias (TOV — Temporary Overvoltages) são elevações de tensão "
        "de frequência industrial com duração de milissegundos a segundos. As principais causas "
        "em linhas de transmissão são: rejeição de carga (ferroressonância, efeito Ferranti) "
        "e faltas fase-terra em redes com neutro isolado ou impedante. "
        "O dimensionamento usa k_TOV como fator de amplificação da tensão fase-terra, "
        "conforme tabelas do IEC 60071-2 §2.3 em função do esquema de aterramento do neutro:"
    )
    rpt.add_equation(omml.eq_vmax_tov(), "Tensão de sobretensão temporária — V_TOV (kV)")
    rpt.add_body(
        "A suportabilidade do equipamento à frequência industrial (U_pf) deve superar V_TOV "
        "com margem de segurança mínima de 15% (IEC 60071-2 §4.3.3). "
        "Equipamentos que não atendem são propensos a flashover em regime de falta prolongada."
    )

    rpt.add_heading2("3.2 Correção atmosférica por altitude")
    rpt.add_body(
        "A rigidez dielétrica do ar diminui exponencialmente com a altitude porque a "
        "densidade do ar — e consequentemente a concentração de moléculas capazes de "
        "absorver a energia dos elétrons acelerados — é menor. O IEC 60071-2 §4.3.4 "
        "estabelece a correção pelo fator Ka, calculado para altitude H em metros:"
    )
    rpt.add_equation(omml.eq_ka_altitude(), "Fator de correção por altitude — Ka (IEC 60071-2)")
    rpt.add_body("O isolamento efetivo é reduzido pela relação:")
    rpt.add_equation(omml.eq_volt_corr_altitude(), "Tensão corrigida por altitude")
    rpt.add_body(
        "Por exemplo, a 1500 m Ka ≈ 1,20, reduzindo a suportabilidade em ~17%. "
        "Este fator é crítico em projetos de linhas nas regiões Sul e Centro-Oeste do Brasil."
    )

    rpt.add_heading2("3.3 Distância de escoamento e poluição")
    rpt.add_body(
        "A distância de escoamento (creepage) é o comprimento da trajetória superficial "
        "do isolador entre os terminais energizado e aterrado. A poluição deposita sais "
        "condutores na superfície do isolador, criando um caminho resistivo que pode "
        "resultar em flashover a tensões inferiores ao NBI. A distância mínima requerida "
        "é proporcional à tensão máxima e ao nível de poluição ambiental (SPS — Specific "
        "Creepage Distance, em mm/kV, conforme IEC 60815):"
    )
    rpt.add_equation(omml.eq_creepage(), "Distância de escoamento mínima (IEC 60815)")

    # Resultados
    rpt.add_heading1("Resultados Obtidos")
    if "items" in results:
        headers = ["Equipamento", "V_TOV (kV)", "U_pf corr (kV)", "Margem PF (%)",
                   "Ka", "Creep. req (mm)", "Creep. forn (mm)", "Status"]
        rows = []
        for it in results["items"]:
            rows.append([
                it.get("name", ""),
                f"{it.get('V_TOV_kV', 0):.1f}",
                f"{it.get('U_pf_corr_kV', 0):.1f}",
                f"{it.get('margin_pf_percent', 0):.1f}",
                f"{it.get('Ka', 1):.4f}",
                f"{it.get('creepage_req_mm', 0):.0f}",
                f"{it.get('creepage_forn_mm', 0):.0f}",
                "ATENDE" if it.get("meets_pf", True) and it.get("meets_creepage", True) else "NÃO ATENDE",
            ])
        rpt.add_result_table(headers, rows, "Verificação de isolamento por equipamento")

        # Grafico de margem
        if results["items"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            names = [it.get("name", f"Eq {i}") for i, it in enumerate(results["items"])]
            margins = [it.get("margin_pf_percent", 0) for it in results["items"]]
            min_margin = cfg.get("min_margin", 15)
            colors = ["#43A047" if m >= min_margin else "#E53935" for m in margins]
            ax.barh(names, margins, color=colors)
            ax.axvline(min_margin, color="#E53935", linestyle="--", label=f"Mín {min_margin}%")
            ax.set_xlabel("Margem PF (%)")
            ax.set_title("Margem de Isolamento à Frequência Industrial")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="x")
            rpt.add_figure_from_matplotlib(fig, "Margem de isolamento por equipamento")

    rpt.add_heading1("Conclusão")
    all_ok = all(
        it.get("meets_pf", True) and it.get("meets_creepage", True)
        for it in results.get("items", [{}])
    )
    if all_ok:
        rpt.add_body(
            "Todos os equipamentos avaliados atendem aos critérios de isolamento à frequência "
            "industrial e distância de escoamento para as condições de altitude e poluição consideradas."
        )
    else:
        rpt.add_body(
            "Existem equipamentos que não atendem aos critérios de isolamento. Recomenda-se "
            "substituir os isoladores por modelos com maior suportabilidade ou distância de "
            "escoamento, ou instalar para-raios adequados para limitar as sobretensões."
        )
    rpt.add_body(SOFTWARE_NOTE)
    rpt.add_references_section(REFS_VMAX)


# ====================================================================
# 8. COORDENACAO DE ISOLAMENTO
# ====================================================================

def report_coord_isolamento(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    rpt.add_cover("Estudo de Coordenação de Isolamento", "Linha de Transmissão")
    rpt.add_page_break()
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    rpt.add_heading1("Introdução")
    rpt.add_body(
        "A coordenação de isolamento é o processo de seleção dos níveis de isolamento dos "
        "equipamentos elétricos em relação às tensões que podem ocorrer no sistema, levando "
        "em conta os dispositivos de proteção contra sobretensões (para-raios). É regida pela "
        "norma IEC 60071-1/2 e ABNT NBR 6939."
    )
    rpt.add_body(
        "O estudo envolve a determinação dos níveis de impulso atmosférico (BIL/NBI), "
        "verificação da cadeia de isoladores, especificação de para-raios e análise das "
        "formas de onda padronizadas (1.2/50 µs para impulso atmosférico)."
    )
    rpt.add_body(
        "Relatório produzido pelo Software BK Estudos Elétricos da BK Engenharia e Tecnologia."
    )

    rpt.add_heading1("Objetivo")
    rpt.add_body(
        "Determinar e verificar os níveis de coordenação de isolamento da linha de transmissão, "
        "incluindo o número de isoladores necessários na cadeia, verificação do NBI (BIL) e "
        "especificação de para-raios de óxido metálico."
    )

    rpt.add_heading1("3. Metodologia")

    rpt.add_heading2("3.1 Níveis de isolamento (BIL/NBI)")
    rpt.add_body(
        "A coordenação de isolamento estabelece os níveis de suportabilidade dos equipamentos "
        "de forma que haja proteção em cascata: primeiro o para-raios limita a sobretensão, "
        "depois o isolador suporta o remanescente. O NBI (Nível Básico de Impulso, ou BIL) "
        "é o valor de pico da onda de impulso atmosférico padronizada (1,2/50 µs) que o "
        "equipamento deve suportar sem flashover. O critério de coordenação é:"
    )
    rpt.add_equation(omml.eq_coord_nbi(), "Critério de coordenação — V_impulso ≥ NBI (kV)")
    rpt.add_body(
        "A onda 1,2/50 µs é o padrão do IEC 60060-1 porque representa a envoltória "
        "estatística das ondas de raio medidas em campo, com frente de 1,2 µs e cauda "
        "de 50 µs. A verificação exige que a tensão suportável da cadeia de isoladores "
        "(N_discos × V_impulso_disco) supere o NBI especificado pelo IEC 60071-1."
    )

    rpt.add_heading2("3.2 Para-raios de óxido metálico (MOV)")
    rpt.add_body(
        "Os para-raios de óxido metálico (ZnO) são escolhidos em vez dos carburundum "
        "porque apresentam característica V-I altamente não-linear: conduzem praticamente "
        "sem corrente em tensão nominal e limitam a tensão residual em surtos. "
        "A especificação envolve três parâmetros críticos: "
        "(1) tensão de operação contínua (MCOV); "
        "(2) tensão residual a 10 kA (V_res), que define a proteção oferecida; "
        "(3) energia absorvida por evento (kJ/kV). "
        "A margem de proteção é: MP = (NBI − V_res) / NBI × 100% ≥ 20% (IEC 60099-4)."
    )

    rpt.add_heading2("3.3 Número de isoladores na cadeia")
    rpt.add_body(
        "O número de discos tipo pino é determinado para duas condições de poluição "
        "(conforme IEC 60815 e ABNT NBR 7326): condição normal (SPS ≥ 16 mm/kV) e "
        "condição de poluição severa (SPS ≥ 25 mm/kV). O critério de poluição governa "
        "o projeto em regiões costeiras ou industriais."
    )

    rpt.add_heading1("Resultados Obtidos")
    V_imp_max = results.get("V_impulse_max_kV", 0)
    theta = results.get("theta_deg", 0)
    n_disc_n = results.get("N_disc_normal", 0)
    n_disc_p = results.get("N_disc_polluted", 0)
    atende = results.get("atende_NBI", False)

    rpt.add_kpi_table([
        ("V impulso máx", f"{V_imp_max:.1f}", "kV"),
        ("Ângulo proteção", f"{theta:.1f}", "°"),
        ("N° discos (normal)", f"{n_disc_n}", "unid."),
        ("N° discos (poluído)", f"{n_disc_p}", "unid."),
        ("Atende NBI", "SIM" if atende else "NÃO", ""),
    ])

    # Onda de impulso
    if "impulse_t" in results and "impulse_V" in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.array(results["impulse_t"]) * 1e6, results["impulse_V"],
                color="#1565C0", linewidth=2, label="Onda 1.2/50 µs")
        ax.set_xlabel("Tempo (µs)")
        ax.set_ylabel("Tensão (kV)")
        ax.set_title("Forma de Onda de Impulso Atmosférico Normalizado")
        ax.legend()
        ax.grid(True, alpha=0.3)
        rpt.add_figure_from_matplotlib(fig, "Onda de impulso atmosférico 1.2/50 µs")

    # Curva V-I do para-raios
    if "arrester_I" in results and "arrester_V" in results:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(results["arrester_I"], results["arrester_V"],
                 color="#FB8C00", linewidth=2, label="V(I) para-raios")
        if "arrester_I_ref" in results:
            ax2.plot(results["arrester_I_ref"], results["arrester_V_ref"],
                     "ro", markersize=10, label="Ponto de referência")
        ax2.set_xlabel("Corrente (kA)")
        ax2.set_ylabel("Tensão residual (kV)")
        ax2.set_title("Característica V-I do Para-raios")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        rpt.add_figure_from_matplotlib(fig2, "Curva V-I do para-raios de óxido metálico")

    # Tabela de isoladores
    if "insulator_table" in results:
        rpt.add_heading2("Cadeia de Isoladores")
        ins_table = results["insulator_table"]
        if isinstance(ins_table, list) and len(ins_table) > 0:
            headers = ["Tipo", "N° Normal", "N° Poluição", "NBI req (kV)", "NBI forn (kV)", "Atende"]
            rows = []
            for ins in ins_table:
                rows.append([
                    str(ins.get("tipo", "-")),
                    str(ins.get("N_normal", 0)),
                    str(ins.get("N_poluicao", ins.get("N_polluted", 0))),
                    f"{ins.get('NBI_req_kV', 0):.0f}",
                    f"{ins.get('NBI_forn_kV', 0):.0f}",
                    "SIM" if ins.get("atende", ins.get("atende_NBI", False)) else "NÃO",
                ])
            rpt.add_result_table(headers, rows, "Especificação da cadeia de isoladores")
        elif isinstance(ins_table, dict):
            ins = ins_table
            headers = ["Parâmetro", "Normal", "Poluído"]
            rows = [
                ["N° de discos", str(ins.get("N_normal", 0)), str(ins.get("N_polluted", 0))],
                ["V impulso cadeia (kV)", f"{ins.get('V_imp_cadeia', 0):.1f}", "-"],
                ["Atende NBI", "SIM" if ins.get("atende_NBI", False) else "NÃO", "-"],
            ]
            rpt.add_result_table(headers, rows, "Especificação da cadeia de isoladores")

    rpt.add_heading1("Conclusão")
    if atende:
        rpt.add_body(
            f"A coordenação de isolamento atende ao NBI especificado. A cadeia requer "
            f"{n_disc_n} discos em condição normal e {n_disc_p} discos em condição de poluição. "
            f"O para-raios de óxido metálico especificado garante proteção adequada contra "
            f"sobretensões atmosféricas."
        )
    else:
        rpt.add_body(
            f"A coordenação de isolamento NÃO atende ao NBI especificado. Recomenda-se "
            f"aumentar o número de isoladores na cadeia (mínimo {n_disc_p} discos para "
            f"condição de poluição) ou revisar a especificação do para-raios."
        )
    rpt.add_body(SOFTWARE_NOTE)
    rpt.add_references_section(REFS_COORD)


# ====================================================================
# 9. RELIGAMENTO TRIPOLAR
# ====================================================================

def report_religamento(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    rpt.add_cover("Estudo de Religamento Tripolar", "Linha de Transmissão")
    rpt.add_page_break()
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    rpt.add_heading1("Introdução")
    rpt.add_body(
        "O religamento automático tripolar é uma prática comum em linhas de transmissão para "
        "restaurar a continuidade do serviço após faltas transitórias (tipicamente descargas "
        "atmosféricas). O tempo morto deve ser dimensionado para permitir a extinção do arco "
        "e a redução das sobretensões transitórias a níveis aceitáveis."
    )
    rpt.add_body(
        "A análise avalia a sobretensão transitória durante o religamento, considerando a "
        "carga aprisionada na linha, frequência natural de oscilação e amortecimento, conforme "
        "metodologias do IEEE e CIGRÉ."
    )
    rpt.add_body(
        "Relatório produzido pelo Software BK Estudos Elétricos da BK Engenharia e Tecnologia."
    )

    rpt.add_heading1("Objetivo")
    rpt.add_body(
        "Determinar as janelas de tempo aceitáveis para o religamento tripolar automático, "
        "garantindo que as sobretensões transitórias não excedam os limites de isolamento."
    )

    rpt.add_heading1("3. Metodologia")

    rpt.add_heading2("3.1 Sobretensão de religamento e carga aprisionada")
    rpt.add_body(
        "Quando a linha é desenergizada por abertura do disjuntor, a carga capacitiva "
        "retém uma tensão aprisionada (trapped charge). No momento do religamento, "
        "a diferença entre a tensão da fonte e a tensão aprisionada pode gerar um "
        "transitório de alta amplitude — potencialmente o dobro da tensão nominal. "
        "O modelo exponencialmente amortecido captura a oscilação natural da linha:"
    )
    rpt.add_equation(omml.eq_reclosing_fo(), "Fator de sobretensão transitória de religamento")
    rpt.add_body(
        "onde V_trap é a tensão aprisionada (pu). "
        "Os parâmetros do circuito de religamento são:"
    )
    rpt.add_equation(omml.eq_reclosing_params(), "Parâmetros do circuito de religamento — α e f₀")
    rpt.add_body(
        "O religamento deve ocorrer em uma janela de tempo em que o FO "
        "instantâneo seja menor que o limite de isolamento (tipicamente 1,5–2,0 pu)."
    )

    rpt.add_heading2("3.2 Determinação das janelas de religamento")
    rpt.add_body(
        "A análise identifica os intervalos de tempo em que o FO(t) cai abaixo do "
        "limite admissível. O tempo morto padrão (typically 300–500 ms) deve ser "
        "ajustado para cair dentro de uma janela aceitável. "
        "Para linhas longas ou com alta indutância, as janelas podem ser muito estreitas, "
        "exigindo uso de resistores de pré-inserção (pre-insertion resistors) para "
        "amortecer o transitório — neste caso o estudo deve ser refeito com o circuito "
        "de amortecimento modelado (IEEE C37.04 §5.4)."
    )

    rpt.add_heading1("Resultados Obtidos")
    if "circuits" in results:
        for i, circ in enumerate(results["circuits"]):
            rpt.add_heading2(f"Circuito {i+1}")
            rpt.add_kpi_table([
                ("f₀ natural", f"{circ.get('f0_hz', 0):.2f}", "Hz"),
                ("FO tempo morto", f"{circ.get('FO_dead_pu', 0):.3f}", "pu"),
                ("FO máximo", f"{circ.get('FO_max_pu', 0):.3f}", "pu"),
                ("Tempo morto OK", "SIM" if circ.get("dead_time_ok", False) else "NÃO", ""),
                ("N° janelas", str(circ.get("n_windows", 0)), ""),
            ])

            if "t_s" in circ and "FO_pu" in circ:
                fig, ax = plt.subplots(figsize=(10, 5))
                t = np.array(circ["t_s"]) * 1000  # ms
                ax.plot(t, circ["FO_pu"], color="#1565C0", linewidth=1.5, label="FO(t)")
                limit = cfg.get("overvoltage_limit_pu", 2.0)
                ax.axhline(limit, color="#E53935", linestyle="--", label=f"Limite {limit:.1f} pu")
                ax.axhline(-limit, color="#E53935", linestyle="--")
                dead = cfg.get("dead_time_s", 0.5) * 1000
                ax.axvline(dead, color="#FB8C00", linestyle=":", linewidth=2, label=f"Tempo morto {dead:.0f} ms")
                ax.set_xlabel("Tempo (ms)")
                ax.set_ylabel("FO (pu)")
                ax.set_title(f"Sobretensão de Religamento – Circuito {i+1}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                rpt.add_figure_from_matplotlib(fig, f"Evolução temporal do fator de sobretensão – Circuito {i+1}")

            if "windows" in circ:
                headers = ["Janela", "t início (ms)", "t fim (ms)", "Duração (ms)"]
                rows = []
                for j, w in enumerate(circ["windows"]):
                    rows.append([
                        str(j + 1),
                        f"{w.get('t_start', 0) * 1000:.1f}",
                        f"{w.get('t_end', 0) * 1000:.1f}",
                        f"{(w.get('t_end', 0) - w.get('t_start', 0)) * 1000:.1f}",
                    ])
                rpt.add_result_table(headers, rows, f"Janelas de religamento aceitáveis – Circuito {i+1}")

    rpt.add_heading1("Conclusão")
    rpt.add_body(
        "As janelas de religamento foram determinadas para garantir que as sobretensões "
        "transitórias não excedam os limites de isolamento. O tempo morto deve ser ajustado "
        "dentro de uma das janelas aceitáveis identificadas."
    )
    rpt.add_body(SOFTWARE_NOTE)
    rpt.add_references_section(REFS_RECLOSING)


# ====================================================================
# 10. COMPATIBILIDADE ELETROMAGNETICA
# ====================================================================

def report_emi(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    rpt.add_cover("Estudo de Compatibilidade Eletromagnética", "Linha de Transmissão")
    rpt.add_page_break()
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    rpt.add_heading1("Introdução")
    rpt.add_body(
        "O estudo de compatibilidade eletromagnética (EMC/EMI) avalia o acoplamento indutivo "
        "entre a linha de transmissão e infraestruturas paralelas, como dutos metálicos "
        "(gasodutos, oleodutos) e linhas de comunicação. A indução de tensões pode representar "
        "riscos à segurança pessoal e à integridade dos equipamentos."
    )
    rpt.add_body(
        "A metodologia segue recomendações da ITU-T K.68, IEEE 776 e CIGRÉ, utilizando modelos "
        "de impedância mútua para estimativa das tensões induzidas em regime permanente e "
        "durante faltas."
    )
    rpt.add_body(
        "Relatório produzido pelo Software BK Estudos Elétricos da BK Engenharia e Tecnologia."
    )

    rpt.add_heading1("Objetivo")
    rpt.add_body(
        "Estimar as tensões induzidas por acoplamento eletromagnético em dutos e linhas de "
        "comunicação paralelos à linha de transmissão, verificando o atendimento aos limites "
        "de segurança."
    )

    rpt.add_heading1("3. Metodologia")

    rpt.add_heading2("3.1 Acoplamento magnético indutivo")
    rpt.add_body(
        "A tensão induzida num duto ou linha de comunicação paralela é causada pelo fluxo "
        "magnético variável gerado pela corrente da linha de transmissão. A grandeza "
        "determinante é a impedância mútua M (H/m), que depende da separação lateral, "
        "da profundidade dos condutores e da resistividade do solo. O uso de M complexa "
        "em vez de M real é fundamental para incluir o retorno de corrente pelo solo "
        "(método de Carson-Clem), que pode aumentar M em até 50% para solos resistivos:"
    )
    rpt.add_equation(omml.eq_emi_induced(), "Tensão induzida por acoplamento magnético (V/km)")
    rpt.add_body(
        "onde ω = 2πf, M é a impedância mútua efetiva (H/m), I é a corrente de sequência "
        "zero ou de desequilíbrio da linha (A) e L é o comprimento de paralelismo (km). "
        "Em regime normal, o somatório vetorial das correntes das três fases tende a zero, "
        "reduzindo a indução. O valor máximo ocorre durante falta fase-terra (corrente 3I₀)."
    )

    rpt.add_heading2("3.2 Critérios de segurança")
    rpt.add_body(
        "Os limites de segurança são estabelecidos conforme o tipo de infraestrutura: "
        "(a) dutos e tubulações: V_contínua ≤ 60 V/km (segurança pessoal, DNIT); "
        "V_falta ≤ 300 V/km (CIGRÉ TB 95); "
        "(b) linhas de comunicação: campo longitudinal E ≤ 1–5 V/m (ITU-T K.68, "
        "dependendo do tipo de circuito). Ultrapassar esses limites exige estudos "
        "detalhados de mitigação (blindagem, aterramento adicional, separação)."
    )

    rpt.add_heading1("Resultados Obtidos")
    pipe = results.get("pipeline", {})
    comm = results.get("comm", {})

    if pipe:
        rpt.add_heading2("Duto / Pipeline")
        rpt.add_kpi_table([
            ("V cont", f"{pipe.get('V_cont', 0):.2f}", "V/km"),
            ("Limite cont", f"{pipe.get('lim_cont', 60):.0f}", "V/km"),
            ("V curto", f"{pipe.get('V_short', 0):.2f}", "V/km"),
            ("Limite curto", f"{pipe.get('lim_short', 300):.0f}", "V/km"),
            ("Atende", "SIM" if not pipe.get("exceeds_cont", False) else "NÃO", ""),
        ])

    if comm:
        rpt.add_heading2("Comunicação")
        rpt.add_kpi_table([
            ("E longitudinal", f"{comm.get('E_long', 0):.4f}", "V/m"),
            ("Limite", f"{comm.get('lim_E', 5):.3f}", "V/m"),
            ("Atende", "SIM" if not comm.get("exceeds_E", False) else "NÃO", ""),
        ])

    rpt.add_heading1("Conclusão")
    rpt.add_body(
        "Os resultados de triagem de compatibilidade eletromagnética foram apresentados. "
        "Caso os limites sejam excedidos, recomenda-se: aumento da separação, blindagem "
        "do duto, instalação de dispositivos de proteção catódica ou estudo detalhado "
        "com modelagem específica."
    )
    rpt.add_body(SOFTWARE_NOTE)
    rpt.add_references_section(REFS_EMI)


# ====================================================================
# 11. FLUXO DE POTENCIA
# ====================================================================

def report_fluxo_potencia(rpt: BKReport, results: Dict[str, Any], cfg: Dict[str, Any]):
    rpt.add_cover("Estudo de Fluxo de Potência", "Rede de Transmissão")
    rpt.add_page_break()
    rpt.add_heading1("Sumário")
    rpt.add_toc()
    rpt.add_page_break()

    rpt.add_heading1("Introdução")
    rpt.add_body(
        "O estudo de fluxo de potência (load flow) é a ferramenta fundamental para a análise "
        "de sistemas elétricos de potência em regime permanente. Permite determinar as tensões "
        "em todas as barras, os fluxos de potência ativa e reativa nos ramos, e as perdas "
        "totais do sistema."
    )
    rpt.add_body(
        "O método de Newton-Raphson completo é utilizado para resolver o sistema de equações "
        "não-lineares da rede, conforme descrito na literatura clássica de Monticelli, "
        "Stevenson e Glover. O método apresenta convergência quadrática e é o mais utilizado "
        "na indústria."
    )
    rpt.add_body(
        "Relatório produzido pelo Software BK Estudos Elétricos da BK Engenharia e Tecnologia."
    )

    rpt.add_heading1("Objetivo")
    rpt.add_body(
        "Determinar o perfil de tensão, fluxos de potência e perdas na rede de transmissão "
        "em estudo, verificando o atendimento aos critérios operativos de tensão (0.95–1.05 pu)."
    )

    rpt.add_heading1("3. Metodologia")

    rpt.add_heading2("3.1 Equações de balanço de potência")
    rpt.add_body(
        "O fluxo de potência em regime permanente é governado pelas equações de balanço "
        "em cada barra k da rede, que expressam que a potência injetada (geração − carga) "
        "deve ser igual à potência que flui para a rede através das admitâncias. "
        "Para uma rede de n barras:"
    )
    rpt.add_equation(omml.eq_power_balance(), "Balanço de potência ativa por barra")
    rpt.add_body(
        "onde Vᵢ e Vₖ são as magnitudes de tensão, θᵢₖ = θᵢ − θₖ é a diferença angular "
        "entre as barras i e k, Gᵢₖ e Bᵢₖ são os elementos da matriz admitância nodal Ybus. "
        "Este sistema não-linear possui 2(n−1) equações com 2(n−1) incógnitas (θ e |V|)."
    )

    rpt.add_heading2("3.2 Método de Newton-Raphson — convergência quadrática")
    rpt.add_body(
        "Newton-Raphson é o método padrão da indústria (ONS, ENERCASE, PSS/E) porque "
        "converge em poucos ciclos para problemas bem condicionados. A convergência quadrática "
        "significa que o número de algarismos corretos dobra a cada iteração — em contraste "
        "com a convergência linear do Gauss-Seidel. A iteração corrige Δθ e Δ|V| "
        "simultaneamente pela resolução do sistema linear:"
    )
    rpt.add_equation(omml.eq_newton_raphson_full(), "Sistema linear de Newton-Raphson")
    rpt.add_body(
        "onde H, N, M, L são as submatrizes da Jacobiana ∂P/∂θ, ∂P/∂V, ∂Q/∂θ, ∂Q/∂V. "
        "As correções ΔP e ΔQ são os mismatches (desvios entre potência calculada e especificada). "
        "O critério de convergência é |ΔP|, |ΔQ| < ε = 10⁻⁶ pu. "
        "Caso o método não convirja, pode indicar infactibilidade da solução (colapso de tensão)."
    )

    rpt.add_heading2("3.3 Tipos de barras")
    rpt.add_body(
        "Para fechar o sistema de equações, cada barra recebe um tipo que define quais "
        "variáveis são especificadas e quais são calculadas: "
        "(1) barra SLACK (referência angular θ = 0, |V| especificado) — fornece o balanço "
        "global de potência; "
        "(2) barra PV (P e |V| especificados, θ e Q calculados) — representa geradores; "
        "(3) barra PQ (P e Q especificados, |V| e θ calculados) — representa cargas e barras de carga."
    )

    rpt.add_heading1("Resultados Obtidos")
    conv = results.get("converged", False)
    iters = results.get("iters", 0)
    mismatch = results.get("max_mismatch", 0)
    slack_p = results.get("slack_p_mw", 0)
    slack_q = results.get("slack_q_mvar", 0)

    rpt.add_kpi_table([
        ("Convergiu", "SIM" if conv else "NÃO", ""),
        ("Iterações", str(iters), ""),
        ("Mismatch", f"{mismatch:.2e}", "pu"),
        ("Slack P", f"{slack_p:.3f}", "MW"),
        ("Slack Q", f"{slack_q:.3f}", "Mvar"),
    ])

    # Tabela barras
    if "buses" in results:
        headers = ["Barra", "Tipo", "|V| (kV)", "Ângulo (°)", "P (MW)", "Q (Mvar)"]
        rows = []
        for b in results["buses"]:
            rows.append([
                str(b.get("bus", "")),
                b.get("type", ""),
                f"{b.get('V_kV', 0):.3f}",
                f"{b.get('angle_deg', 0):.3f}",
                f"{b.get('P_MW', 0):.3f}",
                f"{b.get('Q_Mvar', 0):.3f}",
            ])
        rpt.add_result_table(headers, rows, "Resultados por barra")

    # Tabela fluxos
    if "branches" in results:
        headers = ["De", "Para", "P (MW)", "Q (Mvar)", "Perda P (MW)", "Perda Q (Mvar)"]
        rows = []
        for br in results["branches"]:
            rows.append([
                str(br.get("frm", "")),
                str(br.get("to", "")),
                f"{br.get('p_mw', 0):.3f}",
                f"{br.get('q_mvar', 0):.3f}",
                f"{br.get('p_loss', 0):.6f}",
                f"{br.get('q_loss', 0):.6f}",
            ])
        rpt.add_result_table(headers, rows, "Fluxos por ramo")

    # Grafico perfil de tensao
    if "buses" in results:
        fig, ax = plt.subplots(figsize=(10, 5))
        bus_ids = [b.get("bus", i) for i, b in enumerate(results["buses"])]
        v_pu = [b.get("V_pu", 1.0) for b in results["buses"]]
        ax.bar([str(b) for b in bus_ids], v_pu, color="#1565C0", alpha=0.85)
        ax.axhline(1.05, color="#E53935", linestyle="--", label="1.05 pu")
        ax.axhline(0.95, color="#E53935", linestyle="--", label="0.95 pu")
        ax.axhline(1.0, color="#333333", linestyle="-", alpha=0.3)
        ax.set_xlabel("Barra")
        ax.set_ylabel("|V| (pu)")
        ax.set_title("Perfil de Tensão nas Barras")
        ax.set_ylim(0.9, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
# reports/__init__.py
# ====================================================================
# API principal de geracao de relatorios
# ====================================================================

from .bk_docx import BKReport
from .module_reports import (
    report_parametros,
    report_corona,
    report_campos_em,
    report_ampacidade,
    report_ri_ra,
)
from .module_reports_2 import (
    report_blindagem,
    report_vmax_insulation,
    report_coord_isolamento,
    report_religamento,
    report_emi,
    report_fluxo_potencia,
)

# Mapa modulo -> funcao geradora
REPORT_GENERATORS = {
    "params":       report_parametros,
    "corona":       report_corona,
    "campos_em":    report_campos_em,
    "ampacidade":   report_ampacidade,
    "ri_ra":        report_ri_ra,
    "blindagem":    report_blindagem,
    "vmax":         report_vmax_insulation,
    "coord_isol":   report_coord_isolamento,
    "religamento":  report_religamento,
    "emi":          report_emi,
    "fluxo":        report_fluxo_potencia,
}

# Titulos dos estudos para o cabecalho
REPORT_TITLES = {
    "params":       "Parâmetros Elétricos",
    "corona":       "Efeito Corona",
    "campos_em":    "Campos Elétrico e Magnético",
    "ampacidade":   "Ampacidade e Flecha",
    "ri_ra":        "RI e RA",
    "blindagem":    "Blindagem Atmosférica",
    "vmax":         "Isolamento Vmax",
    "coord_isol":   "Coordenação de Isolamento",
    "religamento":  "Religamento Tripolar",
    "emi":          "Compatibilidade Eletromagnética",
    "fluxo":        "Fluxo de Potência",
}


def generate_report(
    module_key: str,
    results: dict,
    cfg: dict,
    codigo_doc: str = "",
    revisao: str = "0A",
) -> bytes:
    """
    Gera relatorio Word para o modulo especificado.
    Retorna bytes do .docx para download.

    Args:
        module_key: Chave do modulo (ex: 'params', 'corona', etc.)
        results: Dicionario com resultados calculados
        cfg: Dicionario com configuracoes/dados de entrada
        codigo_doc: Codigo do documento BK
        revisao: Revisao do documento
    """
    if module_key not in REPORT_GENERATORS:
        raise ValueError(f"Módulo '{module_key}' não possui gerador de relatório.")

    titulo = REPORT_TITLES.get(module_key, module_key)
    rpt = BKReport(
        titulo_estudo=titulo,
        codigo_doc=codigo_doc,
        revisao=revisao,
    )

    gen_func = REPORT_GENERATORS[module_key]
    gen_func(rpt, results, cfg)

    return rpt.to_bytes()

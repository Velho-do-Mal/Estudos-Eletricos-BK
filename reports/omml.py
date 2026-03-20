# reports/omml.py
# ====================================================================
# Construtor de equacoes OMML (Office Math Markup Language) para python-docx
# Permite inserir equacoes nativas do Word em documentos .docx
# ====================================================================

from lxml import etree
from docx.oxml.ns import qn

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

def _m(tag: str) -> str:
    """Retorna QName no namespace math."""
    return f"{{{MATH_NS}}}{tag}"

def _w(tag: str) -> str:
    return f"{{{WORD_NS}}}{tag}"

# ====================================================================
# Elementos basicos
# ====================================================================

def math_run(text: str, italic: bool = True, bold: bool = False) -> etree._Element:
    """Cria um m:r com texto."""
    r = etree.SubElement(etree.Element("dummy"), _m("r"))
    # run properties
    rpr = etree.SubElement(r, _m("rPr"))
    sty = etree.SubElement(rpr, _m("sty"))
    if italic and bold:
        sty.set(_m("val"), "bi")
    elif italic:
        sty.set(_m("val"), "p")
    elif bold:
        sty.set(_m("val"), "b")
    else:
        sty.set(_m("val"), "p")
    t = etree.SubElement(r, _m("t"))
    t.text = text
    return r

def math_text(text: str) -> etree._Element:
    """Texto simples em equacao (italico por padrao)."""
    return math_run(text, italic=True, bold=False)

def math_num(text: str) -> etree._Element:
    """Numero em equacao (nao-italico)."""
    return math_run(text, italic=False, bold=False)

def math_op(text: str) -> etree._Element:
    """Operador em equacao."""
    return math_run(text, italic=False, bold=False)

# ====================================================================
# Estruturas OMML
# ====================================================================

def fraction(num_elements: list, den_elements: list) -> etree._Element:
    """Fracao: num/den."""
    f = etree.Element(_m("f"))
    fpr = etree.SubElement(f, _m("fPr"))
    num = etree.SubElement(f, _m("num"))
    den = etree.SubElement(f, _m("den"))
    for el in num_elements:
        num.append(el)
    for el in den_elements:
        den.append(el)
    return f

def subscript(base_elements: list, sub_elements: list) -> etree._Element:
    """Subscrito: base_sub."""
    s = etree.Element(_m("sSub"))
    e = etree.SubElement(s, _m("e"))
    sub = etree.SubElement(s, _m("sub"))
    for el in base_elements:
        e.append(el)
    for el in sub_elements:
        sub.append(el)
    return s

def superscript(base_elements: list, sup_elements: list) -> etree._Element:
    """Sobrescrito: base^sup."""
    s = etree.Element(_m("sSup"))
    e = etree.SubElement(s, _m("e"))
    sup = etree.SubElement(s, _m("sup"))
    for el in base_elements:
        e.append(el)
    for el in sup_elements:
        sup.append(el)
    return s

def sub_sup(base_elements: list, sub_elements: list, sup_elements: list) -> etree._Element:
    """Subscrito e sobrescrito: base_sub^sup."""
    s = etree.Element(_m("sSubSup"))
    e = etree.SubElement(s, _m("e"))
    sub = etree.SubElement(s, _m("sub"))
    sup = etree.SubElement(s, _m("sup"))
    for el in base_elements:
        e.append(el)
    for el in sub_elements:
        sub.append(el)
    for el in sup_elements:
        sup.append(el)
    return s

def radical(deg_elements: list | None, base_elements: list) -> etree._Element:
    """Radical (raiz). Se deg_elements=None, raiz quadrada."""
    rad = etree.Element(_m("rad"))
    radpr = etree.SubElement(rad, _m("radPr"))
    if deg_elements is None:
        deghide = etree.SubElement(radpr, _m("degHide"))
        deghide.set(_m("val"), "1")
    deg = etree.SubElement(rad, _m("deg"))
    if deg_elements:
        for el in deg_elements:
            deg.append(el)
    e = etree.SubElement(rad, _m("e"))
    for el in base_elements:
        e.append(el)
    return rad

def delimiters(elements: list, open_char: str = "(", close_char: str = ")") -> etree._Element:
    """Delimitadores: (conteudo), [conteudo], etc."""
    d = etree.Element(_m("d"))
    dpr = etree.SubElement(d, _m("dPr"))
    bc = etree.SubElement(dpr, _m("begChr"))
    bc.set(_m("val"), open_char)
    ec = etree.SubElement(dpr, _m("endChr"))
    ec.set(_m("val"), close_char)
    e = etree.SubElement(d, _m("e"))
    for el in elements:
        e.append(el)
    return d

def bar_accent(elements: list, char: str = "\u0305") -> etree._Element:
    """Barra sobre o elemento (ex: Z barra)."""
    acc = etree.Element(_m("acc"))
    accpr = etree.SubElement(acc, _m("accPr"))
    c = etree.SubElement(accpr, _m("chr"))
    c.set(_m("val"), char)
    e = etree.SubElement(acc, _m("e"))
    for el in elements:
        e.append(el)
    return acc

def func(name: str, arg_elements: list) -> etree._Element:
    """Funcao: ln(x), sin(x), etc."""
    f = etree.Element(_m("func"))
    fpr = etree.SubElement(f, _m("funcPr"))
    fname = etree.SubElement(f, _m("fName"))
    fname.append(math_run(name, italic=False))
    e = etree.SubElement(f, _m("e"))
    for el in arg_elements:
        e.append(el)
    return e

def nary(symbol: str, sub_els: list, sup_els: list, body_els: list) -> etree._Element:
    """N-ario: sumatorio, integral, etc."""
    n = etree.Element(_m("nary"))
    npr = etree.SubElement(n, _m("naryPr"))
    ch = etree.SubElement(npr, _m("chr"))
    ch.set(_m("val"), symbol)
    sub = etree.SubElement(n, _m("sub"))
    sup = etree.SubElement(n, _m("sup"))
    e = etree.SubElement(n, _m("e"))
    for el in sub_els:
        sub.append(el)
    for el in sup_els:
        sup.append(el)
    for el in body_els:
        e.append(el)
    return n

def matrix(rows: list[list[list]]) -> etree._Element:
    """Matriz: rows = [[col1_elements, col2_elements], ...]."""
    m = etree.Element(_m("m"))
    mpr = etree.SubElement(m, _m("mPr"))
    for row_els in rows:
        mr = etree.SubElement(m, _m("mr"))
        for col_els in row_els:
            e = etree.SubElement(mr, _m("e"))
            for el in col_els:
                e.append(el)
    return m

# ====================================================================
# Construtor de paragrafo de equacao (centrado)
# ====================================================================

def build_omath_para(*elements) -> etree._Element:
    """Constroi m:oMathPara (equacao centrada, bloco)."""
    para = etree.Element(_m("oMathPara"))
    omath = etree.SubElement(para, _m("oMath"))
    for el in elements:
        omath.append(el)
    return para

def build_omath(*elements) -> etree._Element:
    """Constroi m:oMath (equacao inline)."""
    omath = etree.Element(_m("oMath"))
    for el in elements:
        omath.append(el)
    return omath

# ====================================================================
# Equacoes prontas para os modulos do BK Estudos Eletricos
# ====================================================================

def eq_impedance_series():
    """Z' = R' + jX'  [Ohm/km]"""
    return build_omath_para(
        subscript([math_text("Z")], [math_text("'")]),
        math_op(" = "),
        subscript([math_text("R")], [math_text("'")]),
        math_op(" + "),
        math_text("j"),
        subscript([math_text("X")], [math_text("'")]),
    )

def eq_inductance():
    """L' = (mu0 / 2pi) * ln(GMD / GMR_eq)"""
    return build_omath_para(
        subscript([math_text("L")], [math_text("'")]),
        math_op(" = "),
        fraction(
            [subscript([math_text("\u03bc")], [math_num("0")])],
            [math_num("2"), math_text("\u03c0")]
        ),
        math_op(" \u00b7 "),
        func("ln", [
            delimiters([
                fraction(
                    [math_text("GMD")],
                    [subscript([math_text("GMR")], [math_text("eq")])]
                )
            ])
        ]),
    )

def eq_capacitance():
    """C' = 2*pi*eps0 / ln(GMD / r_eq)"""
    return build_omath_para(
        subscript([math_text("C")], [math_text("'")]),
        math_op(" = "),
        fraction(
            [math_num("2"), math_text("\u03c0"), subscript([math_text("\u03b5")], [math_num("0")])],
            [func("ln", [
                delimiters([
                    fraction(
                        [math_text("GMD")],
                        [subscript([math_text("r")], [math_text("eq")])]
                    )
                ])
            ])]
        ),
    )

def eq_characteristic_impedance():
    """Zc = sqrt(Z'/Y')"""
    return build_omath_para(
        subscript([math_text("Z")], [math_text("c")]),
        math_op(" = "),
        radical(None, [
            fraction(
                [subscript([math_text("Z")], [math_text("'")])],
                [subscript([math_text("Y")], [math_text("'")])]
            )
        ]),
    )

def eq_sil():
    """SIL = V^2 / Zc"""
    return build_omath_para(
        math_text("SIL"),
        math_op(" = "),
        fraction(
            [superscript([math_text("V")], [math_num("2")])],
            [subscript([math_text("Z")], [math_text("c")])]
        ),
    )

def eq_gmr_bundle():
    """GMR_eq = (GMR * d_s^(n-1))^(1/n)"""
    return build_omath_para(
        subscript([math_text("GMR")], [math_text("eq")]),
        math_op(" = "),
        superscript(
            [delimiters([
                math_text("GMR"),
                math_op(" \u00b7 "),
                superscript(
                    [subscript([math_text("d")], [math_text("s")])],
                    [math_text("n"), math_op("\u2212"), math_num("1")]
                )
            ])],
            [fraction([math_num("1")], [math_text("n")])]
        ),
    )

def eq_peek_corona():
    """Ec = 30.3 * delta * m0 * (1 + 0.301/sqrt(delta*r))"""
    return build_omath_para(
        subscript([math_text("E")], [math_text("c")]),
        math_op(" = "),
        math_num("30.3"),
        math_op(" \u00b7 "),
        math_text("\u03b4"),
        math_op(" \u00b7 "),
        subscript([math_text("m")], [math_num("0")]),
        math_op(" \u00b7 "),
        delimiters([
            math_num("1"),
            math_op(" + "),
            fraction(
                [math_num("0.301")],
                [radical(None, [
                    math_text("\u03b4"),
                    math_op(" \u00b7 "),
                    math_text("r")
                ])]
            )
        ]),
    )

def eq_air_density():
    """delta = 3.92*p / (273 + T)"""
    return build_omath_para(
        math_text("\u03b4"),
        math_op(" = "),
        fraction(
            [math_num("3.92"), math_op(" \u00b7 "), math_text("p")],
            [math_num("273"), math_op(" + "), math_text("T")]
        ),
    )

def eq_corona_voltage():
    """Vd = Ec * r * ln(GMD/r)"""
    return build_omath_para(
        subscript([math_text("V")], [math_text("d")]),
        math_op(" = "),
        subscript([math_text("E")], [math_text("c")]),
        math_op(" \u00b7 "),
        math_text("r"),
        math_op(" \u00b7 "),
        func("ln", [
            delimiters([
                fraction(
                    [math_text("GMD")],
                    [math_text("r")]
                )
            ])
        ]),
    )

def eq_electric_field():
    """E(x,y) = -grad(V) = sum(qi / 2*pi*eps0 * ...)"""
    return build_omath_para(
        math_text("E"),
        delimiters([math_text("x"), math_op(","), math_text("y")]),
        math_op(" = "),
        math_op("\u2212"),
        math_text("\u2207"),
        math_text("V"),
        math_op(" = "),
        fraction(
            [math_num("1")],
            [math_num("2"), math_text("\u03c0"), subscript([math_text("\u03b5")], [math_num("0")])]
        ),
        nary("\u2211", [math_text("i")], [math_text("n")], [
            fraction(
                [subscript([math_text("q")], [math_text("i")])],
                [subscript([math_text("r")], [math_text("i")])]
            )
        ]),
    )

def eq_magnetic_field():
    """B(x,y) = mu0/(2*pi) * sum(Ii/ri)"""
    return build_omath_para(
        math_text("B"),
        delimiters([math_text("x"), math_op(","), math_text("y")]),
        math_op(" = "),
        fraction(
            [subscript([math_text("\u03bc")], [math_num("0")])],
            [math_num("2"), math_text("\u03c0")]
        ),
        nary("\u2211", [math_text("i")], [math_text("n")], [
            fraction(
                [subscript([math_text("I")], [math_text("i")])],
                [subscript([math_text("r")], [math_text("i")])]
            )
        ]),
    )

def eq_ieee738_thermal():
    """qc + qr = qs + I^2 * R(Tc)"""
    return build_omath_para(
        subscript([math_text("q")], [math_text("c")]),
        math_op(" + "),
        subscript([math_text("q")], [math_text("r")]),
        math_op(" = "),
        subscript([math_text("q")], [math_text("s")]),
        math_op(" + "),
        superscript([math_text("I")], [math_num("2")]),
        math_op(" \u00b7 "),
        math_text("R"),
        delimiters([subscript([math_text("T")], [math_text("c")])]),
    )

def eq_sag():
    """f = w*L^2 / (8*T)"""
    return build_omath_para(
        math_text("f"),
        math_op(" = "),
        fraction(
            [math_text("w"), math_op(" \u00b7 "), superscript([math_text("L")], [math_num("2")])],
            [math_num("8"), math_op(" \u00b7 "), math_text("T")]
        ),
    )

def eq_rac():
    """R_ac = R_dc * (1 + r/(2*delta_skin))"""
    return build_omath_para(
        subscript([math_text("R")], [math_text("ac")]),
        math_op(" = "),
        subscript([math_text("R")], [math_text("dc")]),
        math_op(" \u00b7 "),
        delimiters([
            math_num("1"),
            math_op(" + "),
            fraction(
                [math_text("r")],
                [math_num("2"), subscript([math_text("\u03b4")], [math_text("skin")])]
            )
        ]),
    )

def eq_shielding_angle():
    """theta = arctan(d_h / delta_h)"""
    return build_omath_para(
        math_text("\u03b8"),
        math_op(" = "),
        func("arctan", [
            delimiters([
                fraction(
                    [subscript([math_text("d")], [math_text("h")])],
                    [math_text("\u0394"), math_text("h")]
                )
            ])
        ]),
    )

def eq_vtower():
    """V_torre = I_desc * R_pe + L * dI/dt"""
    return build_omath_para(
        subscript([math_text("V")], [math_text("torre")]),
        math_op(" = "),
        subscript([math_text("I")], [math_text("desc")]),
        math_op(" \u00b7 "),
        subscript([math_text("R")], [math_text("pe")]),
        math_op(" + "),
        math_text("L"),
        math_op(" \u00b7 "),
        fraction(
            [math_text("d"), math_text("I")],
            [math_text("d"), math_text("t")]
        ),
    )

def eq_vmax_tov():
    """V_TOV = V_nom * k_TOV / sqrt(3)"""
    return build_omath_para(
        subscript([math_text("V")], [math_text("TOV")]),
        math_op(" = "),
        subscript([math_text("V")], [math_text("nom")]),
        math_op(" \u00b7 "),
        subscript([math_text("k")], [math_text("TOV")]),
        math_op(" / "),
        radical(None, [math_num("3")]),
    )

def eq_ka_altitude():
    """Ka = e^(H/8150)"""
    return build_omath_para(
        subscript([math_text("K")], [math_text("a")]),
        math_op(" = "),
        superscript(
            [math_text("e")],
            [math_text("H"), math_op("/"), math_num("8150")]
        ),
    )

def eq_creepage():
    """d_esc >= V_max * k_poluicao"""
    return build_omath_para(
        subscript([math_text("d")], [math_text("esc")]),
        math_op(" \u2265 "),
        subscript([math_text("V")], [math_text("max")]),
        math_op(" \u00b7 "),
        subscript([math_text("k")], [math_text("pol")]),
    )

def eq_coord_nbi():
    """V_impulso_cadeia >= NBI (BIL)"""
    return build_omath_para(
        subscript([math_text("V")], [math_text("impulso\u2006cadeia")]),
        math_op(" \u2265 "),
        math_text("NBI"),
    )

def eq_reclosing_fo():
    """FO(t) = V_trapped * e^(-alpha*t) * cos(2*pi*f0*t)"""
    return build_omath_para(
        math_text("FO"),
        delimiters([math_text("t")]),
        math_op(" = "),
        subscript([math_text("V")], [math_text("trap")]),
        math_op(" \u00b7 "),
        superscript(
            [math_text("e")],
            [math_op("\u2212"), math_text("\u03b1"), math_text("t")]
        ),
        math_op(" \u00b7 "),
        func("cos", [
            delimiters([
                math_num("2"), math_text("\u03c0"),
                subscript([math_text("f")], [math_num("0")]),
                math_text("t"),
            ])
        ]),
    )

def eq_emi_induced():
    """V_ind = omega * M * I * L"""
    return build_omath_para(
        subscript([math_text("V")], [math_text("ind")]),
        math_op(" = "),
        math_text("\u03c9"),
        math_op(" \u00b7 "),
        math_text("M"),
        math_op(" \u00b7 "),
        math_text("I"),
        math_op(" \u00b7 "),
        math_text("L"),
    )

def eq_power_flow_newton():
    """[delta_theta; delta_V] = -J^(-1) * [delta_P; delta_Q]"""
    return build_omath_para(
        matrix([
            [[math_text("\u0394\u03b8")]],
            [[math_text("\u0394V")]],
        ]),
        math_op(" = "),
        math_op("\u2212"),
        superscript([math_text("J")], [math_op("\u22121")]),
        math_op(" \u00b7 "),
        matrix([
            [[math_text("\u0394P")]],
            [[math_text("\u0394Q")]],
        ]),
    )

def eq_pi_model():
    """Pi model: Z_serie = (R + jX)*L, Y_shunt = jB*L/2"""
    return build_omath_para(
        subscript([math_text("Z")], [math_text("serie")]),
        math_op(" = "),
        delimiters([math_text("R'"), math_op("+"), math_text("jX'")]),
        math_op("\u00b7"),
        math_text("L"),
        math_op("  ;  "),
        subscript([math_text("Y")], [math_text("shunt")]),
        math_op(" = "),
        math_text("j"),
        math_text("B'"),
        math_op("\u00b7"),
        fraction([math_text("L")], [math_num("2")]),
    )

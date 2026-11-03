"""PDF-Export des aktuellen Plans (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from krb_evaluation import Settings, de, omega_text, pct, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "Ω": "Omega", "ω": "omega", "λ": "lambda", "α": "alpha", "μ": "mu", "θ": "theta", "κ": "kappa", "δ": "delta", "Φ": "Phi",
            "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "„": '"', "“": '"', "ᵀ": "^T", "‖": "|", "ā": "a", "√": "sqrt", "≈": "~"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, m: dict) -> bytes:
    inst, nom, rob = m["inst"], m["nominal"], m["robust"]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Robust als Kegel: ellipsoidale Unsicherheit als SOCP"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{inst.name}, Radius Omega = {omega_text(s.omega)}" + (", sichere Mindestzeile c^T x >= 0,6" if s.safe == "an" else "") + ". Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    rows = [("Zielwert des nominalen LP", de(nom.value, 4)), ("Zielwert des robusten Plans", de(rob.value, 4) if m["feasible"] else "unzulässig (Gebiet leer)"),
            ("Preis gegen nominal", pct(m["price"]) if m["feasible"] else "-"), ("aktive unsichere Zeilen", f"{m['n_active']} von {inst.m}"),
            ("Anteil verletzter gezogener Zeilen: nominaler Plan (irgendeine Zeile)", pct(m["viol_nominal"]["any"])), ("Anteil verletzter gezogener Zeilen: robuster Plan", pct(m["viol_robust"]["any"])),
            ("Omega_max (sichere Mindestzeile)", de(m["omega_max"], 3) if m["omega_max"] is not None else "-"), ("Lücke Primal/Dual des Lösers", sci(abs(rob.gap)))]
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(130, 6, "Kennzahl", border=1)
    pdf.cell(70, 6, "Wert", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for k, val in rows:
        pdf.cell(130, 5.5, _s(k), border=1)
        pdf.cell(70, 5.5, _s(val), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kegel-robust-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())

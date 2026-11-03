"""Robust als Kegel: ellipsoidale Unsicherheit als SOCP - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 14 der Reihe „Konvexe und konische Optimierung“: der Lorentz-Kegel aus Stück 2 bis 6 modelliert Unsicherheit. Eine LP-Zeile, deren Koeffizienten in einem Ellipsoid liegen, wird exakt zu einer Kegelzeile; die Demo zeigt das robuste Gebiet,
warum die Umformung exakt ist, was der Radius kostet, was er als Wahrscheinlichkeit bedeutet und wie das mit der Dimension wächst.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import krb_constants as C
from krb_evaluation import (
    Settings,
    cut_view,
    de,
    dim_view,
    main_view,
    ms,
    omega_text,
    pct,
    price_view,
    prob_view,
    row_view,
    sci,
    verdict_cut,
    verdict_dim,
    verdict_main,
    verdict_price,
    verdict_prob,
    verdict_row,
    verdict_rows,
)
from krb_layout import dunkles_design, kopf_css
from krb_pdf_export import build_pdf
from krb_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from krb_visualization import build_cuts, build_dim_price, build_dim_prob, build_price, build_prob, build_region, build_row_space, build_shapes

st.set_page_config(page_title="Robust als Kegel – Sebastian Hanisch", layout="wide")

VIOL_TEXT = f"{C.VIOL_N:,}".replace(",", ".")        # 50.000: Tausenderpunkt wie auf der Website
MC_TEXT = f"{C.MC_N:,}".replace(",", ".")


@st.cache_data(show_spinner=False)
def _main(seed, omega, safe):
    return main_view(Settings(seed=seed, omega=omega, safe=safe))


@st.cache_data(show_spinner=False)
def _row(seed, omega, row, safe):
    return row_view(Settings(seed=seed, omega=omega, row=row, safe=safe), _main(seed, omega, safe))


@st.cache_data(show_spinner=False)
def _cut(seed, omega, safe):
    return cut_view(Settings(seed=seed, omega=omega, safe=safe), _main(seed, omega, safe))


@st.cache_data(show_spinner=False)
def _price(seed, safe, delta):
    return price_view(Settings(seed=seed, safe=safe, delta=delta))


@st.cache_data(show_spinner=False)
def _prob(seed):
    return prob_view(Settings(seed=seed))


@st.cache_data(show_spinner=False)
def _dim(seed):
    return dim_view(Settings(seed=seed))


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in krb_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🛡️ Robust als Kegel")
st.markdown(
    """
**Unsicherheit in einer Nebenbedingung kostet genau eine Kegelzeile.** Liegen die Koeffizienten einer LP-Zeile `aᵀx ≤ b` in einem Ellipsoid `a = ā + Ω P u`, `‖u‖ ≤ 1`, so gilt die Zeile für **alle** möglichen `a` genau dann, wenn
`āᵀx + Ω‖Pᵀx‖ ≤ b`: ein **Lorentz-Kegel**, der Kegel aus Stück 2 bis 6, diesmal als Modell der Unsicherheit. Die Umformung ist **exakt**, keine Näherung: die unendlich vielen Halbräume der Menge sind in einer Zeile zusammengefasst. Gemessen wird,
wie das robuste Gebiet schrumpft, was der Radius `Ω` kostet, **was `Ω` als Wahrscheinlichkeit bedeutet** (bei gaußschen Zeilen ist die Verletzung einer aktiven Zeile genau `1 − Φ(Ω)`, aber das Ellipsoid enthält in hohen Dimensionen kaum Masse) und
wie der Preis mit der Dimension wächst. Stück 14 der Reihe „Konvexe und konische Optimierung“; ein Anwendungsstück ohne neuen Löser.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Die Aufgabe:** ein Polygon aus sechs Halbräumen `aᵢᵀx ≤ bᵢ` in der Ebene, maximiere `cᵀx`. Jede Zeile `aᵢ` ist unsicher: sie liegt in einem Ellipsoid um `āᵢ`.
2. **Die Umformung:** `max_{‖u‖ ≤ 1} (Ω P u)ᵀx = Ω‖Pᵀx‖` (Cauchy–Schwarz), angenommen bei `u* = Pᵀx/‖Pᵀx‖`. Die robuste Zeile ist `(bᵢ − āᵢᵀx ; −Ω Pᵢᵀx) ∈ Lorentz-Kegel`: das Programm ist ein **SOCP**.
3. **Der Löser:** der primal-duale innere Punkt aus Stück 6 (7 bis 18 Iterationen), je unsichere Zeile ein Lorentz-Kegel der Größe `1 + d`.
4. **Der Radius `Ω`:** größer heißt sicherer und teurer. Bei gaußschen Zeilen ist `1 − Φ(Ω)` die Verletzung einer aktiven Zeile.
5. **Die Gegenprobe:** Schnittebenen holen die Halbräume Runde für Runde nach und enden beim selben Wert; cvxpy rechnet dasselbe in den Tests.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Polygon (Ergebnis, Experiment 1 bis 4)**")
    seed = st.number_input("Zufalls-Seed des Polygons", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Polygon", width="stretch", on_click=randomize_seed)
    omega = st.select_slider("Radius Ω", options=C.OMEGAS, key="omega_select", format_func=omega_text, help="Radius der Unsicherheits-Ellipsoide: a = ā + Ω P u mit ‖u‖ ≤ 1. Ω = 0 ist das nominale LP.")
    safe = st.selectbox("Sichere Mindestzeile cᵀx ≥ 0,6", C.SAFE_CHOICES, key="safe_select", help="Eine Zeile ohne Unsicherheit, die das Gebiet nach unten begrenzt: ab einem Radius Ω_max ist das robuste Gebiet leer. Ohne sie ist x = 0 immer zulässig.")
    st.markdown("**Experiment 2 und 3**")
    row = st.select_slider("Zeile (Experiment 2)", options=C.ROW_CHOICES, key="row_select", help="Die Zeile, deren Ellipsoid im a-Raum gezeigt wird.")
    delta = st.select_slider("Abstand δ je Koordinate (Experiment 3)", options=C.DELTA_CHOICES, key="delta_select", format_func=lambda v: de(v, 1),
                             help="Vergleich der Mengenformen: Kugel mit Radius δ (innen), Box ±δ je Koordinate (LP), Kugel mit Radius δ√2 (außen).")

sync_query_params({"seed_input": int(seed), "omega_select": float(omega), "row_select": int(row), "safe_select": safe, "delta_select": float(delta)})

settings = Settings(int(seed), float(omega), int(row), safe, float(delta))
main = _main(settings.seed, settings.omega, settings.safe)
inst = main["inst"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m4, m5 = st.columns(2)
m1.metric("Nominales LP", de(main["nominal"].value, 4), help="Zielwert max cᵀx ohne Unsicherheit (Ω = 0).")
m2.metric("Robuster Plan", de(main["robust"].value, 4) if main["feasible"] else "leer", help="Zielwert, wenn jede Zeile für alle a im Ellipsoid gelten muss.")
m3.metric("Preis", pct(main["price"]) if main["feasible"] else "–", help="Robuster gegen nominalen Zielwert.")
m4.metric("Aktive Zeilen", f"{main['n_active']} von {inst.m}" if main["feasible"] else "–", help="Zeilen, deren Ellipsoid die Grenze berührt (Abstand 0).")
m5.metric("Verletzt (nominaler Plan)", pct(main["viol_nominal"]["any"]), help=f"Anteil der gleichverteilt im Ellipsoid gezogenen Zeilen (je {VIOL_TEXT} pro Zeile), die der nominale Plan in irgendeiner Zeile verletzt; der robuste Plan: {pct(main['viol_robust']['any'])}.")
st.plotly_chart(build_region(main, settings.omega), width="stretch", key="region_chart")
st.caption("Das nominale Gebiet (grau) und das robuste Gebiet (grün): mit wachsendem Ω werden die geraden Ränder zu gekrümmten, der Optimalpunkt wandert nach innen. Die gestrichelte Linie ist die Niveaulinie der Zielfunktion durch den robusten Plan.")

if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, main))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="robust-als-kegel.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was kostet ellipsoidale Unsicherheit, und wo ist die Umformung exakt?")
verdict(verdict_main(settings, main))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Der Plan gegen gezogene Zeilen")
st.caption(f"Je Zeile der Abstand des robusten Plans zur Grenze und der Anteil der {VIOL_TEXT} im Ellipsoid gezogenen Zeilen, die der nominale beziehungsweise der robuste Plan verletzt.")
if main["feasible"]:
    st.dataframe(pd.DataFrame([{"Zeile": i + 1, "Abstand zur Grenze": sci(max(float(main["slack"][i]), 0.0)), "aktiv": "ja" if i in main["active"] else "nein",
                                "nominaler Plan verletzt": pct(main["viol_nominal"]["per_row"][i]), "robuster Plan verletzt": pct(main["viol_robust"]["per_row"][i])} for i in range(inst.m)]), hide_index=True, width="stretch")
verdict(verdict_rows(settings, main))

st.markdown("---")

st.subheader("🔬 Experiment 2: Warum die Umformung exakt ist")
rv = _row(settings.seed, settings.omega, settings.row, settings.safe)
cv = _cut(settings.seed, settings.omega, settings.safe)
st.caption("Links der a-Raum der gewählten Zeile: das Ellipsoid, gezogene Zeilen, der Maximierer a* und die Gerade aᵀx = b des Plans; hält die Zeile, liegt das ganze Ellipsoid auf einer Seite. Rechts die Schnittebenen: das LP mit den Halbräumen, die Runde für Runde nachgeholt werden, fällt auf den SOCP-Wert.")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_row_space(rv, settings.omega), width="stretch", key="row_chart")
with c2:
    st.plotly_chart(build_cuts(cv), width="stretch", key="cut_chart")
verdict(verdict_row(settings, rv))
verdict(verdict_cut(settings, cv))

st.markdown("---")

st.subheader("🔬 Experiment 3: Der Preis der Robustheit und die Form der Menge")
pv = _price(settings.seed, settings.safe, settings.delta)
st.caption("Links der Zielwert gegen den Radius (mit der sicheren Mindestzeile endet die Kurve bei Ω_max). Rechts bei gleichem Abstand δ je Koordinate der Preis dreier Mengen: eingeschriebene Kugel, Box (ein LP) und umschreibende Kugel.")
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(build_price(pv), width="stretch", key="price_chart")
with c4:
    st.plotly_chart(build_shapes(pv), width="stretch", key="shapes_chart")
verdict(verdict_price(settings, pv))

st.markdown("---")

st.subheader("🔬 Experiment 4: Was Ω als Wahrscheinlichkeit bedeutet")
prv = _prob(settings.seed)
st.caption(f"Gaußsche Zeilen `aᵢ ~ N(āᵢ, Pᵢ Pᵢᵀ)`, unabhängig, je {MC_TEXT} Ziehungen (ohne sichere Mindestzeile): für den robusten Plan bei jedem Ω die Verletzung der aktiven Zeile gegen `1 − Φ(Ω)`, die gemeinsame Verletzung und der Anteil der Ziehungen, die im Ellipsoid liegen.")
st.plotly_chart(build_prob(prv), width="stretch", key="prob_chart")
st.dataframe(pd.DataFrame([{"Ω": omega_text(r["omega"]), "Zielwert": de(r["value"], 4), "Preis": pct(r["price"]), "aktive Zeile verletzt": pct(r["freq"], 2), "1 − Φ(Ω)": pct(r["one_minus_phi"], 2),
                            "irgendeine Zeile verletzt": pct(r["joint"], 1), "Summe der Zeilen": pct(r["sum_rows"], 1), "im Ellipsoid": pct(r["inside"], 1)} for r in prv["rows"]]), hide_index=True, width="stretch")
verdict(verdict_prob(settings, prv))

st.markdown("---")

st.subheader("🔬 Experiment 5: Die Dimension")
dv = _dim(settings.seed)
st.caption("Zufalls-LP mit `m = 2d` unsicheren Zeilen und `d` zwischen 2 und 10 (je `d` eine andere Zufallsinstanz): der Preis gegen den Radius je `d`, und bei Ω = 2 der Anteil der Ziehungen im Ellipsoid gegen die (von `d` unabhängige) Verletzung der aktiven Zeile.")
c5, c6 = st.columns(2)
with c5:
    st.plotly_chart(build_dim_price(dv), width="stretch", key="dim_price_chart")
with c6:
    st.plotly_chart(build_dim_prob(dv), width="stretch", key="dim_prob_chart")
st.dataframe(pd.DataFrame([{"d": r["d"], "Zeilen m": r["m"], "Preis Ω = 1": pct(r["prices"][1], 1), "Preis Ω = 2": pct(r["prices"][2], 1), "Preis Ω = 3": pct(r["prices"][3], 1), "Iterationen": f"{min(r['iterations'])} bis {max(r['iterations'])}",
                            "Zeit je Lösung": f"{ms(min(r['seconds']))} bis {ms(max(r['seconds']))}"} for r in dv]), hide_index=True, width="stretch")
verdict(verdict_dim(dv))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Zeilen sind zeilenweise unsicher** | Für harte Nebenbedingungen (alle möglichen `a`) zählt jede Zeile für sich: „für alle `u` und alle Zeilen“ ist dasselbe wie „für alle Zeilen und alle `u`“, eine gemeinsame Unsicherheit über Zeilen ändert das robuste Gebiet nicht (Ben-Tal und Nemirovski: zeilenweise Unsicherheit, im Gegensatz zur spaltenweisen von Soyster, die ein extrem konservatives LP ergibt). Die Wahrscheinlichkeit ändert sich dagegen sehr wohl (Experiment 4). | – |
| **Das Ellipsoid beschreibt die Daten** | Die Garantie gilt für `a` in der Menge, außerhalb gibt es keine Zusage. In hohen Dimensionen enthält das Ellipsoid fast keine Masse einer Gauß-Verteilung (bei Ω = 2: 86 % in `d = 2`, 5 % in `d = 10`, praktisch 0 in `d = 20`); der Radius sagt dort etwas über die Verletzung einzelner Zeilen, nicht über die Menge. | – |
| **`Ω` ist eine Wahrscheinlichkeit** | Nur für **unabhängig gaußsche** Zeilen und **je Zeile**: die aktive Zeile wird genau mit `1 − Φ(Ω)` verletzt, inaktive seltener. Gemeinsam über `m` Zeilen gilt höchstens die Summe (Bonferroni, vorsichtig: 1,7 % gemessen bei 5 % Ziel); andere Verteilungen sind nicht gemessen, eine verteilungsfreie Schranke wird nicht gezeigt. | – |
| **Die Umformung ist exakt** | Für lineare Nebenbedingungen und ein Ellipsoid ja. Unsichere quadratische oder Kegel-Nebenbedingungen und Entscheidungen, die nach der Beobachtung fallen (Zwei-Stufen-Pläne), brauchen andere Umformungen oder sind schwer; nicht gebaut. Eine unsichere Zielfunktion ist über die Epigraphform exakt (`min t` mit `cᵀx ≤ t`). | `robust-opt-demo` (Kapazitätsvorhaltung mit Zukauf) |
| **Die Box ist das Gleiche in anderer Form** | Eine Box `±δ` je Koordinate ist ein LP (`‖x‖₁`), teurer als die eingeschriebene und kaum billiger als die umschreibende Kugel (in 2D); das Budget Γ von Bertsimas und Sim liegt dazwischen und steht in der Schwester-Demo. | `robust-opt-demo` |
| **Schnittebenen genügen** | Sie holen die Halbräume nach und enden meist nach wenigen Runden; es gibt Fälle, in denen sie sich über Hunderte Runden schleppen (Phase-0-Messung), die App deckelt auf 30 Runden. Der Kegel braucht keine Runden. | – |
| **Der Löser skaliert** | Das dichte KKT-System wächst mit `m (1 + d)`; bei `d = 10` rund 0,1 s, bei `d = 20` etwa 2 s (Vormessung, nicht live). Zeiten hängen vom Rechner ab und werden nicht getestet. | – |
| **Distributionelle Robustheit** | Für lineare Verluste ist der Aufschlag auf den Mittelwert ein Normterm in `x` (Mohajerin Esfahani und Kuhn), derselbe Baustein wie hier; die Aussage wurde am Text nicht geprüft, die Demo rechnet sie nicht. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Die Aufgabe.** $\max c^\top x$ unter $a_i^\top x \le b_i$ für alle $a_i \in \mathcal{E}_i = \{\bar a_i + \Omega P_i u : \|u\|_2 \le 1\}$, $i = 1, \dots, m$, dazu sichere Zeilen $|x_j| \le \text{bound}$ und optional $c^\top x \ge \text{target}$.

**Die Umformung.** Für festes $x$ ist $\max_{a \in \mathcal{E}_i} a^\top x = \bar a_i^\top x + \Omega \max_{\|u\| \le 1} (P_i u)^\top x = \bar a_i^\top x + \Omega \|P_i^\top x\|_2$ (Cauchy–Schwarz, Gleichheit bei $u^* = P_i^\top x / \|P_i^\top x\|_2$). Also
$$\bar a_i^\top x + \Omega \|P_i^\top x\|_2 \le b_i \iff s_i = \begin{pmatrix} b_i - \bar a_i^\top x \\ -\Omega P_i^\top x \end{pmatrix} \in \mathcal{Q}^{1+d}.$$
Im Format des Kerns ($\min c^\top x$, $s = b - A x \in K$) ist das je unsichere Zeile ein Lorentz-Kegel; $\Omega = 0$ gibt wieder die Orthant-Zeile.

**Schnittebenen.** Das LP mit den Halbräumen $(\bar a_i + \Omega P_i u_k)^\top x \le b_i$; in jeder Runde kommt $u = P_i^\top x/\|P_i^\top x\|$ für jede verletzte Zeile hinzu. Endet es ohne Verletzung, ist $x$ robust zulässig und der Wert gleich dem SOCP-Wert.

**Die Box.** $a \in \bar a + [-\delta, \delta]^d$: $\max a^\top x = \bar a^\top x + \delta \|x\|_1$, ein LP mit Hilfsvariablen $t \ge |x|$. Es gilt $\|x\|_2 \le \|x\|_1 \le \sqrt d \|x\|_2$: die Box liegt zwischen der eingeschriebenen Kugel (Radius $\delta$) und der umschreibenden (Radius $\delta\sqrt d$).

**Gauß.** Ist $a \sim \mathcal{N}(\bar a, P P^\top)$, so ist $a^\top x \sim \mathcal{N}(\bar a^\top x, \|P^\top x\|^2)$ und
$$\Pr(a^\top x > b) = 1 - \Phi\!\left(\frac{b - \bar a^\top x}{\|P^\top x\|}\right).$$
Bei einer aktiven Zeile ($b - \bar a^\top x = \Omega \|P^\top x\|$) ist das genau $1 - \Phi(\Omega)$, **unabhängig von $d$**. Dagegen ist $\Pr(\|z\| \le \Omega) = \chi^2_d(\Omega^2)$ für $z \sim \mathcal{N}(0, I_d)$ (für $d = 2$: $1 - e^{-\Omega^2/2}$).
Gemeinsam über $m$ Zeilen gilt $\Pr(\text{irgendeine}) \le \sum_i \Pr(\text{Zeile } i)$ (Bonferroni); für das Ziel $\varepsilon$ genügt $\Omega = \Phi^{-1}(1 - \varepsilon/m)$.

**Literatur.** A. Ben-Tal, A. Nemirovski: *Robust solutions of uncertain linear programs*, Operations Research Letters (1999; Aussagen zur zeilenweisen Unsicherheit, zum konisch-quadratischen Gegenstück und zu Soyster am Text der Autorenkopie geprüft, Zeitschriftenangaben nicht bestätigt).
A. L. Soyster, *Convex programming with set-inclusive constraints and applications to inexact linear programming*, Operations Research (1973), S. 1154–1157 (nur über das Literaturverzeichnis von Ben-Tal und Nemirovski belegt).
L. El Ghaoui, H. Lebret: *Robust solutions to least-squares problems with uncertain data*, SIAM Journal on Matrix Analysis and Applications 18 (4), 1997 (nur die Listung geprüft).
D. Bertsimas, D. B. Brown, C. Caramanis: *Theory and applications of robust optimization*, SIAM Review 53 (3), 2011, S. 464–501 (Bibliografie geprüft; nicht am Text).
P. Mohajerin Esfahani, D. Kuhn: *Data-driven distributionally robust optimization using the Wasserstein metric*, Mathematical Programming 171, 2018, S. 115–166 (Bibliografie und Abstract geprüft; nicht am Text).
Die Formel $1 - \Phi(\Omega)$ und der Anteil im Ellipsoid sind hier hergeleitet und gemessen, nicht aus einer Quelle übernommen. Beweise der Arbeiten nicht nachgelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

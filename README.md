# Robust als Kegel: ellipsoidale Unsicherheit als SOCP – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-robust-demo.streamlit.app/)** -->

Stück 14 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst. Dieses Stück ist ein **Anwendungsstück** (in der Gliederung gelb): der Lorentz-Kegel aus Stück 2 bis 6 modelliert hier **Unsicherheit**. Es gibt keinen neuen Löser.

## Kernfrage

**Was kostet ellipsoidale Unsicherheit, und wo ist die Umformung exakt?** Liegen die Koeffizienten einer LP-Zeile `aᵀx ≤ b` in einem Ellipsoid `a = ā + Ω P u`, `‖u‖ ≤ 1`, so gilt die Zeile für **alle** möglichen `a` genau dann, wenn `āᵀx + Ω‖Pᵀx‖ ≤ b`: **eine Lorentz-Kegel-Zeile** (`(b − āᵀx ; −Ω Pᵀx) ∈ Q`), keine Näherung. Gemessen am Polygon mit sechs unsicheren Zeilen (Seed 0): der Zielwert fällt von 1,3369 (Ω = 0) auf **1,1483 bei Ω = 1** (−14,1 %), −25,2 % bei Ω = 2, −41,4 % bei Ω = 4; der nominale Plan verletzt von zufällig im Ellipsoid gezogenen Zeilen in der schlechtesten Zeile 50 %, der robuste Plan keine. **Ω hat zeilenweise eine Wahrscheinlichkeit, aber nicht als Menge:** bei unabhängig gaußschen Zeilen wird die aktive Zeile **exakt mit `1 − Φ(Ω)`** verletzt (Ω = 2: 2,3 %), unabhängig von der Dimension, während das Ellipsoid bei Ω = 2 nur 86 % (`d = 2`), 45 % (`d = 5`) und 5 % (`d = 10`) der Ziehungen enthält.

Das Beispiel ist das **geometrische** der Reihe: ein Polygon in der Ebene, dessen Ränder bei wachsendem Ω zu Kurven werden; für die Dimension ein Zufalls-LP.

## Modell

- **Kern** (`kegel_kern.py`, wortgleich mit Stück 2 bis 13, unverändert): `ConeProblem` (Orthant, Lorentz).
- **Löser** (`krb_solver.py`, `krb_cones.py`, aus Stück 6 übernommen): primal-dualer innerer Punkt (Mehrotra, Nesterov–Todd) mit Dualvariablen.
- **Robustes Programm** (`krb_robust.py`): die Instanz (Polygon, Zufalls-LP), die Kegelzeilen (`solve`), die Box als LP (`solve_box`), die Formel `worst_case` und der Maximierer `worst_case_point`, Ziehungen im und auf dem Ellipsoid, die **Schnittebenen**, die Zulässigkeitsgrenze `omega_max` (Bisektion mit sicherer Mindestzeile) und die Gauß-Rechnung (`1 − Φ(Ω)`, `χ²_d(Ω²)`, gemeinsame Verletzung, Bonferroni; `phi`, `phi_inv`, `chi2_cdf` ohne scipy).
- **Kein Eingriff in den Kern:** alles Neue steht in `krb_*.py`; BLAS-Threads laufen mit einem Thread (`threadpoolctl`). Laufzeit: numpy, plotly, pandas, fpdf2.

## Methodik

Fünf Experimente mit live berechnetem Verdikt: (1) **der Plan gegen gezogene Zeilen**: je Zeile Abstand zur Grenze, aktiv ja/nein und der Anteil der im Ellipsoid gezogenen Zeilen, die der nominale beziehungsweise robuste Plan verletzt; (2) **warum die Umformung exakt ist**: im a-Raum das Ellipsoid, gezogene Zeilen, der Maximierer `a*` und die Gerade `aᵀx = b`, daneben die **Schnittebenen**, die Runde für Runde auf den SOCP-Wert fallen; (3) **der Preis der Robustheit** (Zielwert gegen Ω, Ω_max mit sicherer Mindestzeile) und die **Form der Menge** (Kugel innen, Box als LP, Kugel außen); (4) **was Ω bedeutet** (gaußsche Zeilen: Verletzung gegen `1 − Φ(Ω)`, Anteil im Ellipsoid gegen `χ²`, gemeinsame Verletzung und die Bonferroni-Wahl); (5) **die Dimension** (Zufalls-LP `d` = 2, 3, 5, 8, 10). Regler: Seed, Radius Ω (0 bis 6), sichere Mindestzeile an/aus, Zeile (Experiment 2), Abstand δ (Experiment 3); ein Schnellstart lädt sechs Szenarien, die Adresszeile spiegelt die Konfiguration.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen Rundung, Rechner und Ziehungen, steht die gemessene Zahl trotzdem hier). Zeiten, Iterationszahlen, der Status des Lösers nahe Ω_max und alle Anteile aus Ziehungen werden nur in Bändern getestet. Standard: Polygon, Seed 0, `m = 6`, `d = 2`.

| Frage | Befund |
|---|---|
| Der Standardplan, Ω = 1 | nominal **1,3369**, robust **1,1483** (−14,1 %), Zeilen 1 und 2 aktiv; der nominale Plan verletzt von je 50 000 im Ellipsoid gezogenen Zeilen in der schlechtesten Zeile 50,3 %, in irgendeiner 75 % (bei unabhängig gezogenen Zeilen); der robuste Plan **keine**. Löser: 8 Iterationen, Lücke Primal/Dual 2·10⁻¹¹ |
| Ist die Formel exakt? | Zeile 2 am robusten Plan: Formel `āᵀx + Ω‖Pᵀx‖` = 1,045899; der größte von 400 000 Werten in und auf dem Ellipsoid liegt um 5·10⁻¹¹ darunter, der Maximierer `a* = ā + Ω P u*` trifft sie. Die Zeile ist aktiv (Abstand 2·10⁻¹³ zur Grenze 1,0459). Über 8 Zeilen und je 2 Mio. Punkten (Phase 0) höchstens 5·10⁻¹² |
| Ein anderer Rechenweg: Schnittebenen | Das LP mit Halbräumen `(ā + Ω P u_k)ᵀx ≤ b`, Runde für Runde um `u = Pᵀx/‖Pᵀx‖` der verletzten Zeilen ergänzt: Standardfall **3 Runden mit 4 zusätzlichen Halbräumen** (zu den 6 nominalen), Unterschied zum SOCP 2,5·10⁻¹⁰; Seeds 0 bis 2 mit Ω = 0,5 / 1 / 2: immer höchstens 12 Runden. Phase 0 mit dem Löser HiGHS: 7 von 9 Fällen 3 bis 4 Runden, **2 von 9 in 199 Runden** bei 10⁻⁸; Ursache ist kein Schleppen, sondern eine Prüftoleranz unterhalb der Genauigkeit von HiGHS (primale Zulässigkeit 10⁻⁷): die größte Verletzung bleibt ab Runde 3 bei 2,5·10⁻⁸, mit Verletzungstoleranz 10⁻⁶ enden alle 9 Fälle in 3 bis 4 Runden. Bei höherer Dimension brauchte der eigene Löser 15 bis 30 Runden (`d` = 3 bis 8, Ω = 2); der Kegel braucht keine Runden |
| Preis der Robustheit (Seed 0) | Ω = 0,25 / 0,5 / 1 / 2 / 3 / 4: −3,9 / −7,5 / **−14,1** / **−25,2** / −34,1 / **−41,4 %**; die Zuwächse schrumpfen (−14,1, −11,1, −8,9, −7,2 Prozentpunkte je Einheit). Fläche des robusten Gebiets (Gitter 161 × 161): bei Ω = 1 / 2 / 4 noch 76 / 59 / 39 % des nominalen |
| Zulässigkeitsgrenze mit sicherer Zeile `cᵀx ≥ 0,6` | `Ω_max` = **6,7265 / 7,4771 / 7,8460 / 9,1057** (Seeds 0 bis 3; Seed 7: 5,53); bei Ω = 6 bleibt Seed 0 bei 0,6379 (−52,3 %) mit einer aktiven Zeile. Ohne diese Zeile ist `x = 0` immer zulässig |
| Mengenform, gleiche Abstände `δ` je Koordinate (Seed 0) | Preis Kugel innen (Radius `δ`) / Box (LP, `‖x‖₁`) / Kugel außen (Radius `δ√2`): δ = 0,1: 10,0 / 13,2 / 13,5 %; δ = 0,2: 18,1 / 23,3 / 23,9 %; δ = 0,3: 24,9 / 31,3 / 32,0 %; Seeds 1 und 2 gleich geordnet: die Box liegt in 2D fast an der umschreibenden Kugel |
| Gauß: die aktive Zeile | `aᵢ ~ N(āᵢ, Pᵢ Pᵢᵀ)` unabhängig, 200 000 Ziehungen je Zeile: Ω = 0,5 / 1 / 1,645 / 2 / 2,5 / 3 gemessen 30,6 / 15,9 / 5,0 / 2,3 / 0,63 / 0,14 %, Formel `1 − Φ(Ω)` 30,9 / 15,9 / 5,0 / 2,3 / 0,62 / 0,13 %. Die Formel gilt für die aktive Zeile (kleinster Abstand in Standardabweichungen = Ω), inaktive Zeilen werden seltener verletzt |
| Gauß: Anteil im Ellipsoid | `d = 2`: 11,8 / 39,3 / 74,2 / 86,4 / 95,6 / 98,9 % für dieselben Ω (Formel `1 − e^(−Ω²/2)`). Bei Ω = 1: 15,9 % Verletzung bei nur 39,3 % der Ziehungen im Ellipsoid |
| Gauß: gemeinsam | Summe über die sechs Zeilen als Schranke (Bonferroni): Ω = 1,645 gibt je Zeile 5 %, **gemeinsam rund 10 %** (gemessen 9,9 %, Summe 10,0 %), Zielwert −21,6 %; die Bonferroni-Wahl `Ω = Φ⁻¹(1 − 0,05/6)` = 2,394 hält **1,7 %** gemeinsam (vorsichtig, Ziel 5 %) für **−29,0 %** |
| Die Dimension (Zufalls-LP, `m = 2d`, `b = 1`; je `d` eine andere Instanz) | Preis bei Ω = 1 / 2 / 3: `d = 2`: −7,5 / −13,7 / −18,2 %; `d = 3`: −18,1 / −24,2 / −28,9; `d = 5`: −28,4 / −36,3 / −42,2; `d = 8`: −37,6 / −56,8 / −61,6; `d = 10`: −24,5 / −49,1 / −70,5 %. **Im Trend wachsend, nicht monoton** (`d = 8` teurer als `d = 10` bei Ω ≤ 2); der Preis hängt hier an `d` und an `m = 2d` Zeilen zugleich. Über 40 Instanzen je `d` (Nachmessung der Prüfung, nicht in den Tests) kostet die Absicherung bei Ω = 2 im Mittel 25 % bei `d = 2` und 51 % bei `d = 8` und `d = 10`; je Instanz ist der Verlauf nicht monoton. Verletzung der aktiven Zeile bei Ω = 2 in jedem `d` rund 2,3 %; Anteil im Ellipsoid 86,5 / 73,8 / 44,7 / 14,5 / 5,3 % (`χ²_d(4)`: 86,5 / 73,9 / 45,1 / 14,3 / 5,3 %). Iterationen 8 bis 18; Zeit je Lösung 4 ms (`d = 2`) bis 110 ms (`d = 10`), Vormessung `d = 20`: 1,9 s (`m = 40` Kegel der Größe 21) |
| Gegen cvxpy | Ein **eigener** Aufbau in cvxpy (`a @ x + Ω * cp.norm(P.T @ x) <= b`, Clarabel) gibt denselben Zielwert (10⁻⁶) und Plan (10⁻⁵) auf dem Polygon (3 Seeds × Ω = 0 / 0,5 / 1 / 2), auf dem Zufalls-LP `d = 5` und für die Box (`cp.norm(x, 1)`); Ω_max stimmt mit der Bisektion in Clarabel überein; ein Halbraum-LP mit 200 Randpunkten je Ellipsoid (scipy, HiGHS) liegt von oben höchstens 10⁻³ über dem SOCP-Wert |

## Befunde und Korrekturen gegenüber dem Plan

- **Der Preis wächst mit `d` nur im Trend** (Plan: „wächst mit der Dimension“): die Zufalls-LPs sind je `d` andere Instanzen mit `m = 2d` Zeilen (Dimension und Zeilenzahl wachsen zugleich); `d = 8` ist bei Ω ≤ 2 teurer als `d = 10`. Verdikt und Text sagen „nicht monoton“.
- **Die Halbräume zählen als „zusätzlich“:** der Plan nannte „4 bis 6 Halbräume“ für das, was die Runden nachholen; gezählt werden hier die zusätzlichen zu den `m` Zeilen des nominalen LP (Standardfall 4 zusätzliche, 10 insgesamt).
- **Schnittebenen:** mit dem eigenen Löser (Toleranz 10⁻¹⁰, Verletzungstoleranz 10⁻⁸) enden alle neun Fälle der Phase 0 in höchstens 12 Runden; die 199 Runden der Phase 0 stammen vom LP-Löser HiGHS mit einer Verletzungstoleranz unterhalb seiner Genauigkeit (kein Schleppen des Verfahrens). Der Text sagt, dass solche Toleranzen nicht enden (die App deckelt auf 30 Runden).
- **Der Löser bleibt der eigene** auch für das LP der Schnittebenen und die Box (Orthant-Zeilen); scipy ist nur Orakel im dev-Satz.
- **Preset „Mindestzeile, Ω = 6“** statt einer Ω-Wahl über Ω_max: der Regler reicht bis 6, Seed 0 hat `Ω_max` = 6,73; für eine leere Region braucht es einen anderen Seed (Seed 7, `Ω_max` = 5,53; als Test).
- **Ein HTML-Kommentar im Markdown (`lint-ok`) wird in `st.markdown` sichtbar gedruckt** (Lehre aus Stück 13); hier steht keiner im App-Text.
- **Quellen:** geprüft am 2026-11-03. **Am Text (Autorenkopie, pdftotext):** Ben-Tal und Nemirovski, *Robust solutions of uncertain linear programs*, Operations Research Letters (1999): zeilenweise Unsicherheit, Ellipsoide, das robuste Gegenstück ist ein konisch-quadratisches Programm, spaltenweise Unsicherheit nach Soyster ist extrem konservativ; **die Zeitschriftenangaben (Band, Seiten) sind nicht bestätigt** (die Autorenkopie trägt sie nicht). **Nur Bibliografie oder Zusammenfassung:** El Ghaoui, Lebret, *Robust solutions to least-squares problems with uncertain data*, SIAM Journal on Matrix Analysis and Applications 18 (4), 1997 (Listung, Datum 1. Oktober 1997, Seitenzahlen nicht bestätigt); Bertsimas, Brown, Caramanis, *Theory and applications of robust optimization*, SIAM Review 53 (3), 2011, S. 464–501, DOI 10.1137/080734510; Mohajerin Esfahani, Kuhn, *Data-driven distributionally robust optimization using the Wasserstein metric*, Mathematical Programming 171, 2018, S. 115–166 (Abstract); Soyster, *Operations Research* (1973), S. 1154–1157 (nur über das Literaturverzeichnis von Ben-Tal und Nemirovski). **Nicht geprüft:** Boyd und Vandenberghe, *Convex Optimization*, Abschnitt 4.4.2 (Abruf scheiterte am Zertifikat); die Formel `1 − Φ(Ω)` ist hier **hergeleitet und gemessen**.
- **Abgrenzung:** `robust-opt-demo` (Optimierung-unter-Unsicherheit-Linie, Stück 4) behandelt **polyedrische** Unsicherheit mit Budget Γ als LP an einer Kapazitätsvorhaltung mit Zukauf, geprüft an lognormalen Szenarien; hier ist die Menge ein **Ellipsoid**, die Umformung ein **Kegel** und die Wahrscheinlichkeitsdeutung gaußsch. Keine Zahlen werden übernommen; die Box erscheint nur als Vergleich.

## Ehrliche Grenzen

- **Zeilenweise Ellipsoide, lineare Nebenbedingungen:** keine Schnitte von Ellipsoiden, kein Budget Γ, keine unsicheren quadratischen oder Kegel-Nebenbedingungen (dort braucht es andere Umformungen), keine Zwei-Stufen-Entscheidungen (dafür steht `robust-opt-demo`). Eine unsichere Zielfunktion ist über die Epigraphform exakt (`min t` mit `cᵀx ≤ t`, Fußnote bei Ben-Tal und Nemirovski), hier nicht gebaut.
- **Die Gauß-Aussagen gelten für unabhängig gaußsche Zeilen,** die aktive Zeile; andere Verteilungen sind nicht gemessen, eine verteilungsfreie Schranke wird nicht gezeigt. Der Anteil im Ellipsoid und die Verletzung der Zeile sind verschiedene Größen.
- **Die Zufalls-LPs sind Setzungen** (positive normierte Mittel, `b = 1`, Ellipsen mit dem Verhältnis 1 : 0,4), kein Maß für Praxisprobleme.
- **`Ω_max` kommt aus einer Bisektion;** nahe der Grenze meldet der Löser nicht „optimal“ (Clarabel wirft `SolverError`): die Genauigkeit liegt im Band 10⁻³, der Status wird nicht getestet.
- **Der Löser skaliert dicht:** `d = 20` braucht rund 2 s (Vormessung, nicht live), die Reihe endet bei `d = 10`. Zeiten hängen vom Rechner ab und werden nicht getestet.
- **Distributionelle Robustheit (Wasserstein):** nur als Zeile im Text; für lineare Verluste ist der Aufschlag ein Normterm (Mohajerin Esfahani, Kuhn), die Aussage ist am Text nicht geprüft und nicht gerechnet.

## Verifikation

- **Robustes Programm** (`tests/test_robust.py`): Handwerte (`d = 1`: `b/(ā + Ωρ)`; Quadrat mit Kugeln: Wert `2/(1 + √2 ω)` bei `x = (t, t)`; eine Zeile mit `P = ρI`: `b/(1 + Ωρ)`; `u*` und die Formel von Hand), die Formel als Obergrenze, die angenommen wird (Stichprobe, `d` = 2 bis 5), der robuste Plan verletzt keine gezogene Zeile und der nominale rund die Hälfte, **eine gemeinsame Störung über alle Zeilen ändert bei harten Nebenbedingungen nichts, solange jede Zeile für sich das volle Ellipsoid durchläuft**, Zielwert fällt in Ω und Ω = 0 ist das LP, Skalieren von `P` gegen Ω, Drehung, Dualitätslücke, Kugel innen ≤ Box ≤ Kugel außen, Box von Hand (`1/(1 + δ)`), Schnittebenen erreichen den SOCP-Wert (Rundenzahl höchstens 12) und enden an der Rundengrenze, `Ω_max`, `Φ`, `Φ⁻¹`, `χ²` gegen bekannte Werte (`d = 1, 2` geschlossen, `d = 5, 10` Zahlen), die Gauß-Verletzung der aktiven Zeile gegen `1 − Φ(Ω)` im Band von vier Standardfehlern, Bonferroni.
- **Orakel** (`tests/test_oracle_kegel_robust.py`): ein **eigener** Aufbau in cvxpy (Clarabel) gegen Wert und Plan (Polygon, Zufalls-LP, Box, mit sicherer Zeile), `Ω_max` gegen die Bisektion in Clarabel, ein Halbraum-LP mit Randpunkten in scipy.
- **Randfälle** (`tests/test_edge_cases.py`): Ω = 0 und `P = 0` (nominal), `P` mit Rang 1 (Strecke), `x = 0` und Nullzielfunktion, nominal unzulässiges Gebiet (Status, `is_feasible`, `Ω_max = 0`), eine Zeile und eine Dimension, Ω = 1000 (winziges Gebiet), alle Reglerecken, Reproduzierbarkeit, **Serialisierbarkeit der zwischengespeicherten Ergebnisse (pickle)**, jede Zeilenwahl.
- **Verdikte** (`tests/test_verdicts.py`, Werte von Hand gesetzt), **Claims** (`tests/test_claims.py`: jede Zahl dieser README und der Hilfetexte), **Oberfläche** (AppTest: jeder Button, jeder Regler an erstem und letztem Wert, jeder Radius, jede Zeile, leeres Gebiet bei Seed 7, PDF, Tabellen), **Abbildungen** (aus den gezeichneten Daten zurückgerechnet: die Gerade ist `aᵀx = b`, der Rand des Ellipsoids liegt auf einer Seite, das Gitter stimmt mit den Nebenbedingungen überein), Presets, Permalink, Layout, Footer, Laufzeit-Importe, Kern (Prüfsumme gegen das Schwester-Repo).
- **Umfang:** 153 Tests (15 davon, die cvxpy-Orakel gegen Clarabel, überspringen sich ohne cvxpy; die Orakel-Datei hat 16 Tests, der sechzehnte braucht nur scipy; mit cvxpy laufen alle 16 grün), rund 1,5 Minuten (2026-11-03, Windows); kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy und scipy sind Gegenproben im dev-Satz. Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2 bis 13 |
| `krb_robust.py` | Instanzen, Kegelzeilen, Box, Schnittebenen, `Ω_max`, Formel und Maximierer, Gauß-Rechnung |
| `krb_solver.py`, `krb_cones.py` | Primal-dualer innerer Punkt (Mehrotra, Nesterov–Todd) mit Dualvariablen, aus Stück 6 |
| `krb_constants.py` | Regler-Reihen, feste Reihen, Farben, Presets |
| `krb_evaluation.py` | Experimente, Verdikte, Formatierung |
| `krb_layout.py` | Fester Kopf (Stylesheet) |
| `krb_pdf_export.py` | PDF-Export (fpdf2) |
| `krb_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `krb_visualization.py` | Plotly-Abbildungen (robustes Gebiet, a-Raum mit Ellipsoid und Gerade, Schnittebenen, Preis, Gauß, Dimension; Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Robustes Programm, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe, Kern, Löser |

## Bewusst nicht umgesetzt

- **Distributionell robuste Optimierung (Wasserstein)** als Rechnung: nur die Zeile im Text.
- **Budget Γ (Bertsimas, Sim), Schnitte von Ellipsoiden, Zwei-Stufen-Entscheidungen:** die Schwester-Demo `robust-opt-demo` und die Unsicherheit-Linie.
- **Unsichere quadratische oder Kegel-Nebenbedingungen** (andere Umformungen) und eine **verteilungsfreie Wahrscheinlichkeitsschranke:** nicht gebaut, nicht am Text geprüft.
- **`d = 20` live:** die Zeit steigt dicht auf 2 s.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).

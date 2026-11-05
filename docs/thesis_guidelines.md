# Bachelorarbeit — writing rules (extracted from the guideline PDF)

**Source:** Prof. Dr.-Ing. Achim Ibenthal, *Hinweise zu Abschlussarbeiten*, HAWK,
23 August 2024, 20 pages — `docs/private/Anleitung_Abschlussarbeiten.pdf` (not
redistributed). Read completely on 2026-11-05; section numbers (§) refer to the PDF.
Where this file and the PDF differ, the PDF wins. The PPA template
(`docs/report/main.tex`, graded 1.0) is the technical starting point; the deviations
it has from the PDF are listed in part H and must be fixed in the thesis.

## A. Structure of an engineering thesis (§1.1, §3, §8)

Length: no fixed page count; **50 pages is rather few, more than 100 is too many**.
Recommended structure (chapters may be merged where sensible, e.g. system overview
with concept development; design methodology may get its own chapter):

0. **Abstract** — initial situation, problem, the solutions and results of the work,
   **4–5 sentences**, written **last**, "to the point".
1. **Introduction** — background, project environment, motivation (*what is the
   problem?*), objectives, task description, delimitation (*what is not part of the
   project*), short overview of the chapters.
2. **System overview** — mathematical/technical description of the system environment,
   block diagram / system model, interface description.
3. **State of research/technology** — literature research; structuring, classification
   and analysis of existing solutions **and their deficits**; identification of
   performance criteria / test metrics; summary of the open points to be resolved.
4. **Requirements analysis** — functional requirements (system behaviour),
   non-functional requirements (e.g. real-time behaviour, fault tolerance), scenarios and
   use cases, methodology, **requirements validation (correctness, completeness,
   consistency, comprehensibility)**.
5. **Concept development** — detailed problem analysis, solution approach incl. detailed
   block diagram, design/research methodology, tools and design flow.
6. **Implementation** — algorithms, software, hardware.
7. **Test** — test concept, test cases, test vectors, functional verification, formal
   verification, (compatibility test), module tests, system integration and system test,
   (field test).
8. **Benchmarking** (if applicable) — comparison with existing solutions, using the
   literature results, the defined performance criteria and test metrics;
   **visualisation of the results**.
9. **Summary** — summary of the starting point and of the solution approaches; **list of
   the essential own contributions**; essential evaluation criteria; summarising
   evaluation of the results; difficulties encountered and the strategies used; open
   points; outlook (for a Bachelor thesis: implementation/product development);
   recommendations.
10. **References** — own chapter after "Summary and Outlook" ("Bibliography"/"References").

**Roter Faden (§1.1, §8):** every chapter **begins** with its motivation, an overview and
what has been achieved so far; every chapter **ends** with a summary of the questions
solved, the remaining open points and the motivation for the next chapter. Chapter N
should motivate chapter N+1; the chapter overview closes the introduction.

**Methodology questions every thesis must answer (§3):** objective (why, what to show,
hypothesis); prerequisites (basis, system environment, tools and prior work received,
**why these tools**, what had to be developed, development flow and tool interfaces);
development concept (assumptions and how they are checked later; test concept: formal
correctness, metrics and comparison measures, effect of disturbances — test series over
the parameter ranges of input and disturbance variables, test vectors and references,
test equipment, conformity with standards; how a benchmark is performed);
implementation (what, in which steps, difficulties, efficiency gains); test
(observations at commissioning, results of the test series, additional tests due to new
insights, results **in tables or graphs over the parameter ranges**, **error analysis —
are errors minor or significant, judged by defined criteria**, error sources found and
removed); evaluation (target state, starting basis, why the result was achieved,
explanations for measurements; for experimental work: what the observations prove, new
insight, hypotheses confirmed, new hypotheses, what would clarify them; objective
reached — if not, why; open points; recommendations).

## B. Front matter (§1.2, §5.2)

- Title page; **Selbständigkeitserklärung** (required, also for English theses).
- Table of contents (**must not be missing**).
- **Glossary directly after the table of contents**: table of abbreviations with meaning;
  the guideline's own glossary underlines the letters that form each acronym.
- **Symbol list** recommended when there are many formulas.
- Lists of figures/tables: optional ("Auf Tabellen- und Bildverzeichnisse lege ich keinen Wert").
- Abbreviations are explained at their **first use** in the text.

## C. Style (§1.3, §5)

- Factual and neutral. Not allowed: marketing phrases, novel-like praise of tools or
  "software gurus", pictures unrelated to the topic, humour, comments about fatigue.
- Correct spelling and grammar (spellchecker is mandatory practice); English is welcome.
- Non-discriminatory language; no abbreviated gender forms (German texts).
- Do not explain lecture basics (e.g. advantages of OOP); describe tools briefly and
  **justify why** they were chosen ("it was free" is not a justification) (§8).
- **The central purpose of the document is to document the personal contribution** (§8).

## D. Deductions listed explicitly (§1.4)

- Diagram axes without units and labels.
- Images too small or blurred (use detail enlargements where it matters).
- Missing figure captions / table headings.
- Missing numbering of figures, tables, equations.
- Insufficient textual explanation of methods, equations, figures, diagrams.
- Spelling errors.

## E. Literature and citations (§2)

- Literature research is mandatory; prefer **peer-reviewed** journals and conferences
  (IEEE, SPIE, …), identify and obtain the key papers.
- Existing methods are compared **systematically**, by system-critical parameters and
  quality criteria **defined before the evaluation**, **under identical conditions**
  (same test data / standard test sets); if literature does not allow a comparison,
  implement/simulate the most promising methods. Document the systematics and the
  conclusions at the end of the chapter.
- **Every** piece of information not developed personally needs a source (books, articles,
  reports, theses, **images**).
- Internet sources only as an exception; Wikipedia only for free images.
- Verbatim quotes in quotation marks; paraphrases without quotes; source at the end.
- **Numbered in order of first appearance**, square brackets, e.g. [1], [2,3,4]; the
  reference stands **before the full stop** at the end of the sentence [5]. In LaTeX:
  `cite` package. BibTeX recommended (guideline names `english.bst`, which is not part of
  TeX Live; `IEEEtran.bst` gives the same order-of-appearance numbering).
- Minimum fields: author, title, journal; book: publisher and place; conference:
  location; journal: volume and number; page(s); month/year.

## F. Equations, figures, tables, listings (§6)

- LaTeX recommended. Mathematical description of the technical behaviour is welcome.
- Equations numbered **relative to the chapter** (e.g. (6.1)); multi-line equations
  aligned at "="; in a multi-line derivation only the **last line** is numbered.
- Typesetting: scalar variables italic (except Greek capitals); vectors and matrices
  **bold**; numbers, functions (sin, log) and symbols upright; **units upright with a
  space** (t = 3 s, 512 px).
- Figures **≥ 600 dpi**; **vector graphics (PDF) preferred**.
- Foreign images: only with permission or a free licence (e.g. Creative Commons), and
  **always with the source**.
- Figures, tables and equations numbered relative to chapters; **every one explained in
  the text**; captions on every figure, headings on every table.
- Avoid long source listings (block diagrams / UML instead); short excerpts with the
  `listings` package **with line numbers**.

## G. Planning, Kolloquium, formalities (§4, §8–§10)

- Phase planning with deliverables, acceptance criteria, risk assessment, reviews per
  phase; outline sent to the supervisor; a draft can be reviewed if sent ≥ 2 weeks before
  the deadline, as PDF.
- Coordinate early and regularly with the examiners (§8).
- **Kolloquium (§9):** max. 45 min in total, talk about 20, **max. 25 min**; about
  **13 slides per half hour**; focus on the personal contribution and the "added value";
  suggested slides: contents (1), introduction/motivation (1), system overview (1),
  problem definition (1), methodology/solution (1–2), results (1–3), summary — evaluation,
  problems, outlook (1). Rehearse; clear speech, varied pace, eye contact, readable fonts,
  no overloaded slides, **correct spelling**. Public unless a Sperrvermerk exists
  (announcement: topic, author, place/time, 2–3 sentence summary).
- Prüfungsamt forms attached to the final version; optional archiving as PDF/A (§10).

## H. Deviations of the PPA template that the thesis must fix

| PPA template | Rule | Fix in the thesis |
|---|---|---|
| `natbib` + `plainnat` (alphabetical) | order of first appearance (§2.2) | `cite` + `IEEEtran.bst` |
| glossary after lists of figures/tables | glossary directly after the TOC (§1.2) | move it; add symbol list |
| title page names one supervisor | both examiners grade | name Erstprüferin + Zweitprüferin |
| abstract with PPA figures | thesis recomputes everything | new abstract, written last |
| `\counterwithin` by chapter | relative numbering ✔ | keep |

## I. Thesis-specific agreements (examiners, 2026-10-28)

- Contribution = the self-retraining system and an honestly measured data ceiling; no
  clinical-accuracy claim.
- The first technical part: U-Net vs YOLOv8-seg on the same mask data, the better one is
  used for the rest of the work (answers PPA comments: boxes vs masks, why no U-Net).
- All numbers from this work's result files; none copied from the PPA.

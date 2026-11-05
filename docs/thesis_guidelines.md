# Bachelorarbeit — writing requirements checklist

Sources, in order of authority:
1. **Prof. Dr.-Ing. Achim Ibenthal, "Hinweise zu Abschlussarbeiten" (HAWK, Aug 2024)** —
   applies to every HAWK Arbeit (Haus-, Projekt-, Bachelor-, Masterarbeit). The PDF is
   at `docs/private/Anleitung_Abschlussarbeiten.pdf` (20 pages, gitignored — not ours to
   redistribute). **Read it in full and extract its rules before writing any thesis text;
   where it differs from this checklist, the PDF wins.** Until then, the rules below are
   the summary kept in `docs/project_context.md` plus what `docs/report/main.tex` already
   implements. That template earned the PPA a **1.0**, so it is the validated baseline.
2. HAWK Prüfungsverwaltung (confirmed 2026-10): registration only after all exams pass;
   **9 weeks** writing time; submission **not before week 7**; two examiners; Kolloquium.
3. Agreements with the examiners (meeting 2026-10-28).

## A. Structure (the "roter Faden")

The PPA followed Ibenthal's engineering-thesis chapter sequence; keep it:

| # | Chapter | Purpose |
|---|---|---|
| 1 | Introduction | context, motivation, objectives, scope/delimitation, chapter overview |
| 2 | System Overview | block diagram, interfaces |
| 3 | State of the Art | clinical background, related work, metrics, identified gaps |
| 4 | Requirements Analysis | functional / non-functional requirements, use cases |
| 5 | Concept Development | problem analysis, solution approach, design |
| 6 | Implementation | what was built and how |
| 7 | Testing and Validation | test concept, test cases, results |
| 8 | Benchmarking / Comparison | **U-Net vs YOLOv8-seg** (agreed first work step) |
| 9 | Conclusion and Outlook | contributions, objectives met?, challenges, outlook |
| — | Appendix | catalogs, parameters, mappings, statistics |

- [ ] **Every chapter opens with its motivation** and **ends with a summary plus a
      transition** to the next chapter.
- [ ] Objectives stated in ch. 1 are explicitly assessed one-by-one in the conclusion.

## B. Front matter (order as in the template)

- [ ] Title page — must name **both examiners**: Erstprüferin Prof. Dr. Claire Chalopin,
      Zweitprüferin Annika Willnat (the PPA title page named only Chalopin).
- [ ] **Selbständigkeitserklärung** — German text, Ort/Datum/Unterschrift (template has it).
- [ ] **Abstract: 4–5 sentences**.
- [ ] Table of contents, list of figures, list of tables, glossary/abbreviations.

## C. Formatting

- [ ] A4, 12 pt, two-sided; margins left 3 cm / right 2.5 cm / top & bottom 2.5 cm,
      binding offset 0.5 cm (as in template).
- [ ] **Chapter-relative numbering** of equations, figures and tables (`\counterwithin`).
- [ ] Every figure/table has `\caption{}` **and** `\label{}` **and is referenced in the text**.
- [ ] Figures **≥ 600 dpi**, prefer **vector** (PDF/SVG) for plots and diagrams.
- [ ] Source code via the **listings** package **with line numbers**.
- [ ] Citations numeric **[1], [2]**. Guideline says BibTeX + `english.bst` + `cite`;
      the PPA template uses `natbib` + `plainnat` (also numeric, accepted with a 1.0).
      Decide once and keep it consistent.

## D. Language and content

- [ ] English, **factual and neutral (sachlich und neutral)**; no marketing phrases,
      no informal language.
- [ ] Every claim backed by a measurement in this project or a citation.
- [ ] Thesis framing (agreed): the contribution is the **self-updating system** and an
      **honestly measured data ceiling**, not a clinical-accuracy claim.
- [ ] All numbers come from the rebuilt pipeline of this thesis (counting independent
      photographs). Never copy figures from the PPA abstract/text.
- [ ] Leakage effect, if cited: **≈ 12 points mAP@50 (0.61 → 0.48)**, same tool both sides.

## E. Presentation / Kolloquium

- [ ] Talk **max. 25 min**, **10–13 slides**, total incl. Q&A **max. 45 min**.
- [ ] Both examiners present; prepare short German answers to likely questions.

## F. Process deadlines

- [ ] Everything practical (data, experiments, figures, most text) done **before the
      January 2027 exam**, so only finishing remains after registration.
- [ ] 9-week writing window; earliest submission in week 7; agree grading time with
      both examiners in writing.

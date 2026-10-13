# Error analysis: the 15 hardest SciFact test queries for the best arm (hybrid)

Source: `results/errors.json` / `results/errors_dump.md`, produced by `python -m ragbench run` (`ragbench/errors.py`).
The 15 queries are the ones with the lowest nDCG@10 for the hybrid arm (all 15 scored 0.0); ties are broken by how deep
the first gold document sits. Each one was then read by hand (query, gold title + abstract opening, top-3 retrieved) and
given one primary category. The categories are judgement calls, not model output.

| # | qid | Query (abridged) | Gold doc (rank in hybrid top-100) | Category | Why |
|---|---|---|---|---|---|
| 1 | 1110 | Suboptimal nutrition is not predictive of chronic disease | GBD 2015 comparative risk assessment of 79 risks (>100) | multi-hop | Gold is a broad risk-factor study; you need to know that dietary risks are among the 79 risks it quantifies. |
| 2 | 1175 | The PPR MDA5 has two N-terminal CARD domains | Immune signaling by RIG-I-like receptors (>100) | numeric/entity | Gold names MDA5 as an RLR but the query hinges on "PPR"/"CARD", which the opening does not use. |
| 3 | 1199 | Benefits of colchicine achieved with high-dose statins... | 4-year cardiovascular event rates in atherothrombosis outpatients (>100) | label noise | Gold never mentions colchicine; it is relevant only through the citing paper's context. |
| 4 | 1213 | Prolonged activation of monocytes is deleterious in inflammatory disease | Myeloid KLF2 inactivation promotes atherosclerosis (>100) | lexical mismatch | "monocytes / inflammatory diseases" vs "myeloid / macrophage / atherosclerosis". |
| 5 | 1332 | TNF-a and IL-1 inhibit IL-6 and IL-10 | Inter-individual variability of cytokine responses (>100) | multi-hop | Top hits are about TNF/IL-6 directly; the gold supports the claim only via its results on cytokine co-regulation. |
| 6 | 437 | No animal model for MDS genomic alterations | Human iPSC models of glioma-initiating cells (>100) | label noise | Gold is about glioma, not myelodysplastic syndrome; only "genomic" and "model" overlap. |
| 7 | 502 | Efficiency in crowded delivery centres impaired by improvements... | HIV treatment gap: financial resources for ART (>100) | label noise | Topic mismatch; overlap is the generic words "delivery" and "efficiency". |
| 8 | 560 | Immune responses produce Th17 cells and iTregs | Compromised epithelial barrier induces adaptive immune compensation (>100) | multi-hop | Gold reaches Th17/Treg only through a colitis mechanism several steps away from the claim wording. |
| 9 | 913 | PPAR-RXRs are inhibited by PPAR ligands | Allosteric interactions among nuclear hormone receptors (>100) | numeric/entity | Abbreviation-heavy entity claim; gold talks about RXR heterodimers generically (0 shared content words >4 chars). |
| 10 | 975 | Primary pro-inflammatory cytokines induce secondary mediators | Inter-individual variability of cytokine responses (>100) | lexical mismatch | Same gold as #5; claim vocabulary ("mediators", "primary/secondary") is absent from the gold. |
| 11 | 834 | NOX2-independent pathways generate peroxynitrite | Altered antigen recognition, CD8+ T-cell tolerance in cancer (87) | multi-hop | Gold is about MDSCs nitrating the TCR; peroxynitrite is the bridge, not the topic. |
| 12 | 716 | Low expression of miR7a has a function in testis | Assessing the ceRNA hypothesis (82) | numeric/entity | Specific miRNA name ("miR7a") does not match the gold's generic miRNA wording. |
| 13 | 421 | Flexible molecules face more steric hindrance in the tumour microenvironment | Quantum dots distinguish species in the tumour milieu (67) | lexical mismatch | "microenviroment" (misspelt in the query) vs "milieu"; dense retrieval helps but not enough. |
| 14 | 1191 | Public DNA data doubles every 10 years | The EMBL nucleotide sequence database (60) | numeric/entity | Hinges on a number (doubling period) and "DNA data" vs "nucleotide sequence database". |
| 15 | 13 | 5% of perinatal mortality is due to low birth weight | Global prevalence of childhood underweight (60) | numeric/entity | A percentage claim; "low birth weight" vs "underweight in children under 5". |

## Summary

| Category | Count |
|---|---|
| numeric/entity | 5 |
| multi-hop | 4 |
| lexical mismatch | 3 |
| label noise | 3 |

- 10 of the 15 gold documents are not in the hybrid top-100 at all, so no re-ranker over that candidate list could fix
  them. The bottleneck for these queries is first-stage recall, not ordering.
- Numeric and entity claims (percentages, gene/miRNA names, abbreviations) are the largest group. Both BM25 (exact
  tokens) and a small embedding model handle them poorly when the gold abstract uses a broader term.
- About one in five of the hardest queries looks like a labelling artefact of how SciFact was built (claims written from
  citation sentences, so the cited abstract may only support the claim indirectly). These cap what any retriever can
  score and should be kept in mind when reading small nDCG differences.

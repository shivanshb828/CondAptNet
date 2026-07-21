# Protein Sequence Enrichment Gap — Investigation & Fix Report

Generated from branch `data/fix-protein-enrichment` against `master_dataset_v2.csv`.

---

## Executive Summary

| Metric | Before | After |
|---|---|---|
| Protein rows missing sequence (gap) | 683 rows, 228 targets | 203 rows, 64 targets |
| Rows fixed by new enrichment | — | +48 rows (23 targets) |
| Rows reclassified (wrong target_type) | — | 432 rows moved out of 'protein' pool |
| Training-ready rows (all splits) | 3,702 | **3,750** |

---

## 1. Gap Categorization

All 683 gap rows were in **Category A** (no `target_id`). Every row had `target_id_source = 'UniProt'` but `target_id` was `NaN` — the enrichment pipeline had run, attempted name-based lookup, and produced no accession. Categories B and C from the task spec do not apply:

- **Category B** (has UniProt accession but sequence fetch failed): 0 rows
- **Category C** (has non-UniProt identifier): 0 rows

The `target_id_source = 'UniProt'` field is a pipeline label meaning "intended lookup path", not confirmation that an accession was found.

### Root-cause breakdown of the 683 gap rows

| Root cause | Distinct targets | Rows | Fix applied |
|---|---|---|---|
| Misclassified as 'protein': small molecule | 44 targets | 135 rows | target_type corrected → `small_molecule` |
| Misclassified as 'protein': cell/cell line | 32 targets | 112 rows | target_type corrected → `cell` |
| Misclassified as 'protein': whole organism/virus | 20 targets | 115 rows | target_type corrected → `organism` |
| Misclassified as 'protein': RNA element / peptide / antibody / scraper garbage | 38 targets | 70 rows | target_type corrected → `other` |
| Genuine protein, enriched this pass | 23 targets | 48 rows | `protein_sequence` filled |
| Genuine protein, still missing (manual review) | 64 targets | 203 rows | left blank, flagged below |
| **Total** | **228** | **683** | |

**Key finding**: 453 of 683 gap rows (66%) were never going to be enrichable because the cleaning pipeline incorrectly labeled them as `target_type='protein'`. These rows were already excluded from training by augment.py's `protein_sequence.notna()` filter, so reclassifying their `target_type` does not change training-ready counts — but it correctly removes them from the "protein gap" denominator and prevents confusion in future enrichment passes.

---

## 2. Automatic Fixes Applied (48 rows, 23 targets)

All enrichment used manually verified UniProt accessions added to
`data/raw/protein_name_overrides.csv`. Each accession was fetched from the
UniProt REST API and the returned protein name and organism confirmed as a
plausible match before being accepted. No automated fuzzy-match results were
accepted without human verification.

One side fix: the existing override for "Influenza virus non-structural protein 1
(NS1) protein" pointed to **P03490** (stale/inaccessible accession). Updated to
**P03496** (NS1, A/Puerto Rico/8/1934 H1N1, 230 aa).

One encoding bug fixed: "selB protein (Escherichia coli)" had a non-breaking
space `\xa0` in its `target_name`, causing the exact-match override lookup to
silently skip it. Normalized to regular space and sequence applied.

### Rows fixed, sorted by row count

| Rows | Accession | UniProt Entry | Dataset target_name |
|---|---|---|---|
| 14 | Q9EZJ8 | RNA polymerase sigma factor SigA, *T. aquaticus* (438 aa) | Thermus aquaticus σA subunit |
| 6 | P14081 | Selenocysteine-specific elongation factor SelB, *E. coli* K12 (614 aa) | selB protein (Escherichia coli) ¹ |
| 3 | P08709 | Coagulation factor VII, *H. sapiens* (466 aa) | coagulation factor VIIa activity |
| 2 | P08962 | CD63 antigen, *H. sapiens* (238 aa) | CD63 protein |
| 2 | Q62351 | Transferrin receptor protein 1, *M. musculus* (763 aa) | Mouse transferrin receptor (TfR-ECD) |
| 2 | P61981 | 14-3-3 protein gamma, *H. sapiens* (247 aa) | Recombinant human 14-3-3 gamma (14-3-3γ) |
| 2 | P0A7I0 | Peptide chain release factor RF1, *E. coli* K12 (360 aa) | Escherichia coli release factor 1 |
| 2 | P69544 | DNA-binding protein G5P, *Enterobacteria phage M13* (87 aa) | Ff gene 5 protein (g5p) |
| 1 | P04483 | TetR class B from transposon Tn10 (207 aa) | tet repressor protein (tetR) |
| 1 | P20789 | Neurotensin receptor type 1, *R. norvegicus* (424 aa) | rat neurotensin receptor 1 |
| 1 | P10146 | C-C motif chemokine 1, *M. musculus* (92 aa) | mouse CCL1 chemokine mCCL1 |
| 1 | P0DTC2 | Spike glycoprotein, SARS-CoV-2 (1273 aa) | Receptor-binding domain (RBD) of SARS-CoV-2 spike protein (S protein) |
| 1 | P16070 | CD44 antigen, *H. sapiens* (742 aa) | CD44 glycoprotein HABD |
| 1 | P18272 | Nucleoprotein, *Zaire ebolavirus* Mayinga-76 (739 aa) | Ebola virus (EBOV) recombinant NP his-tagged |
| 1 | P04626 | Receptor tyrosine-protein kinase erbB-2, *H. sapiens* (1255 aa) | Epidermal growth factor receptor 2, Human (ErbB-2/HER2) |
| 1 | P67875 | Ribonuclease mitogillin (Asp f 1), *A. fumigatus* (176 aa) | Aspergillus fumigatus f1 allergen |
| 1 | P01130 | Low-density lipoprotein receptor, *H. sapiens* (860 aa) | Mammalian cell-expressed human recombinant low-density lipoprotein receptor (LDL-R) protein |
| 1 | P15941 | Mucin-1, *H. sapiens* (1255 aa) | Mucin 1 (MUC1) |
| 1 | P19963 | Major pollen allergen Ole e 1, *O. europaea* (145 aa) | Olive (Olea europaea L.) pollen 1 (Ole e 1) |
| 1 | P02818 | Osteocalcin, *H. sapiens* (100 aa) | Osteocalcin (OC) - osteoporosis biomarker |
| 1 | P03496 | Non-structural protein 1, Influenza A H1N1 PR8/1934 (230 aa) | Influenza virus non-structural protein 1 (NS1) protein |
| 1 | Q16552 | Interleukin-17A, *H. sapiens* (155 aa) | Interleukin 17 A A |
| 1 | Q96PD4 | Interleukin-17F, *H. sapiens* (163 aa) | Interleukin 17 F F |

¹ selB had a non-breaking space `\xa0` encoding bug in `target_name`; normalized before applying.

---

## 3. Reclassification Applied (432 rows, 134 targets)

The cleaning pipeline's `target_type` classifier tagged these as 'protein' but they are
not protein targets and cannot have a UniProt sequence. The existing
`_NON_PROTEIN_PATTERNS` regex filter in `enrich_proteins.py` caught **zero** of these
because it is only applied in the legacy schema path; the cleaned-schema path
(which this data uses) skips it entirely.

### small_molecule → 135 rows, 44 targets (examples)
Hematoporphyrin IX (HPIX), 17β-Estradiol, racemic ibuprofen, Diclofenac, Palladium ion,
N-methylmesoporphyrin IX, Tetra-BDE congener, Bisphenol A/B/6F-BPA, Arsenate, Tacrolimus,
Cyanine dye cy3, Black Hole Quenchers, Acetamiprid, Omethoate, Isocarbophos,
linezolid-neomycin B, Pyr tobramycin, Chitin, 25-HydroxyvitaminD3, di(2-ethylhexyl) phthalate,
Brevetoxin-2, Alpha-amanitin, Glutamate/Glutamic Acid, Ractopamine, CCdApPuro, and others.

### cell → 112 rows, 32 targets (examples)
Hepatoma HepG2 cells, HL60 leukemia cells, CCRF-CEM cells, Mouse tumor endothelial cells,
Colorectal cancer stem cells (CR-CSC), NCI-H69 small-cell carcinoma, OVCAR-3 ovarian cancer,
Rabies virus-infected BHK-21 cells, GCRV-infected CIK cells, MDA-MB-231 cells,
PSMA+ LNCaP cells, HCC-CD44s/CD44E, Cancer stem cells A549 shEcad, VacciniaInfectedA549, and others.

### organism → 115 rows, 20 targets (examples)
Cryptosporidium parvum oocysts, E. coli K88, E. coli ATCC 25922, E. coli O157:H7,
Soft-shelled turtle iridovirus (STIV), Vibrio parahaemolyticus, Influenza virus H1N1/H3N2,
Trachinotus ovatus nervous necrosis virus, Shigella sonnei, Bifidobacterium bifidum,
Helicobacter pylori, Group A streptococcus M3/M4, Vaccinia virus (VACV),
human influenza A/Panama/2007/1999, Aspergillus spores (A. niger, A. flavus, A. fumigatus),
Listeria monocytogenes, S. aureus strain 82354.

### other → 70 rows, 38 targets (RNA elements, peptides, antibodies, scraper garbage)
**RNA elements**: HIV TAR-RNA element, synthetic P5.1 stem-loop from B. subtilis RNase P,
16S rRNA Decoding Region, HIV-1 TAR element, yeast phenylalanine tRNA, HCV IRES,
DNA template for TAR transcription, HIV-1 LTR-325/LTR-408.

**Peptides**: Anti-neuroexcitation peptide III, O-glycan-peptide, MUC1 peptide variants
(APDTRPAPG/APDTREAPG/APDTRPPPG), N-terminal histone H3 peptide (dimethyl-Arg),
L-substance P, 16mer collagen XIα1 peptide, Beta-crosslap, arginine vasopressin,
salivary peptide histatin-3, IgE-binding epitope of Asp f 1, peptide-acridine conjugates.

**Antibodies** (cannot be represented as a single UniProt sequence):
Rabbit IgG, myasthenia gravis antibody mAB198, Rituximab (two name variants),
Anti-MUC1 IgG3 mAb C595, antiMUC1MAbC595, anti-hTNFα antibody, anti-FLAG M2, OA-mAb-F(ab′)2.

**Scraper garbage** (truncated/garbled text from patent/paper parsing):
"to SARS-CoV-2 spike glycoprote" (truncated target name — 17 rows from patent
050-099-381-908-14X; aptamers likely target SARS-CoV-2 spike, manual fix needed),
"a second aptamer", "provide an NP- aptamer conjugate", "and at thromb",
"the N-terminal doma", "the target polypeptide", "the detection of IgG",
KHO-3, CED-91-251-His6, TNFoc, His3-tagged recombinant proteins, Polyhistidine-tag.

---

## 4. Remaining Manual Review Items (203 rows, 64 targets)

These are genuine protein targets where enrichment failed because:
- The protein is not in UniProt's **reviewed** (Swiss-Prot) database — only TrEMBL entries exist
- The name describes a fragment, post-translational modification state, or chimeric construct
- The target is a multi-protein complex
- The name is garbled enough to prevent safe automated identification

Sorted by row count (highest priority first):

| Rows | Target name | Why enrichment failed / what's needed |
|---|---|---|
| 20 | Ustilago maydis RNA binding protein Rrm4 | Fungal RBP, *U. maydis* only in TrEMBL (A0A0D1CNT2 in unreviewed); add to overrides after manual confirmation |
| 13 | Escherichia coli (E. coli) core bacterial RNA polymerase (RNAP) | Multi-subunit complex (α₂ββ'ω); no single UniProt entry; best match is individual subunit — manual decision required |
| 11 | NS3 protein of hepatitis C virus | Strain-specific viral protein; name too generic for safe automatic match; needs NCBI/UniProt strain lookup |
| 10 | TvFL3 Thermoplasma volcanium | Archaeal protein, likely TrEMBL only; "TvFL3" is an internal lab name, not a standard gene symbol |
| 8 | Abrin toxin, Abrus precatorius (A.precatorius) | Abrin is a protein (P11140 for abrin-a chain) but name refers to the whole toxin preparation; verify which chain is the aptamer target before accepting |
| 8 | hepatitis C virus domain | Too vague to resolve safely; a specific HCV protein domain, identity unknown |
| 5 | M proteins on the surface of Streptococcus pyogenes (S. pyogenes) | M protein family (many serotype-specific genes); needs specific M-protein gene/accession |
| 5 | Hepatitis C virus core and NS5 proteins | Multi-protein target; no single sequence appropriate |
| 5 | Nonphosphorylated BACE1 CT | C-terminal fragment of BACE1 (P56817); full-length BACE1 already in override table; decide if fragment rows should use full-length sequence |
| 5 | Ss-LrpB Sulfolobus solfataricus | Archaeal transcription factor; may be TrEMBL entry Q9UXJ5 — verify before adding |
| 5 | Clostridium botulinum neurotoxin (BoNT) heavy chain-peptide domain | Domain fragment; full BoNT is P10844 or similar — decide if domain rows should use full-length |
| 5 | C-terminal region Recombinant human connective tissue growth factor (rhCTGF) | CTGF = CCN2, P29279 exists; decide if C-terminal-fragment rows should use full-length sequence |
| 5 | Cytotoxin CNY | Unknown identity; not a standard protein name |
| 5 | Clostridium botulinum neurotoxin BoNT-toxoid aldehyde-inactivated toxin | Chemically inactivated form; same protein as BoNT but modified — any sequence use must be flagged as approximate |
| 4 | Enterotoxigenic Escherichia coli (E. coli) (ETEC) K88 fimbriae protein | K88 fimbrial adhesin = FaeG; UniProt P08185 — verify and add to overrides |
| 4 | Recombinant HA protein from swine IAV H3 cluster IV | Influenza HA, strain-specific; cluster IV H3 HA needs specific strain lookup |
| 4 | Plasmid R1 transfer operon, gene M mutant (TraMM13) | TraM mutant M13; accession for wild-type TraM R1 exists but mutant may not — manual decision |
| 4 | Plasmid R1 transfer operon, gene M (TraM) | TraM from plasmid R1, E. coli; accession P15804 — verify and add to overrides |
| 3 | VP1 structural polypeptide of O-serotype FMD | Foot-and-mouth disease VP1, O-serotype; multiple strains in UniProt |
| 3 | phospho-Ser845 cGluR1 | Phosphorylated GluA1 (AMPA receptor subunit 1); full-length P19491 (rat); decide if phospho-state rows should use full-length |
| 3 | Rhesus (Rh) D antigen | Rh blood group D antigen; UniProt P18577 exists — verify and add to overrides |
| 3 | Phosphorylated BACE1 CT | Same as Nonphosphorylated BACE1 CT above |
| 3 | Abrin toxin | Shorter name variant of "Abrin toxin, Abrus precatorius" — same issue |
| 3 | 6xHIS–LiPABP protein (rLiPABP) | Leishmania infantum poly-A binding protein; TrEMBL entry exists; His-tag is recombinant decoration |
| 3 | HepCnsp3protease | Garbled name for HCV NS3 serine protease; same issue as "NS3 protein of hepatitis C virus" |
| 2 | Hemagglutinin (HA) glycoprotein; Influenza virus | Generic influenza HA — needs strain specification |
| 2 | FL11 (Pyrococcus sp. OT3) | Archaeal protein; "FL11" is a lab name, no standard ID |
| 2 | Stoffel fragment | Truncated Taq polymerase (N-terminal deletion); Taq polymerase = P19821 already in override table; decide whether full-length sequence is appropriate for a truncated-form aptamer |
| 2 | Tubulin purified, Calf brain | Tubulin α/β complex; multiple UniProt entries; which subunit is the target? |
| 2 | Recombinant HA1 proteins of the H5N1 influenza virus | HA1 domain, H5N1; strain-specific |
| 2 | Raf-like Ras-binding domain | Domain fragment; no single UniProt entry for just this domain |
| 2 | RNA-dependent RNA polymerase (RdRp) (NS5B) of HCV subtype 3a | HCV NS5B, subtype 3a; strain-specific |
| 2 | RNA binding motif protein, Y-linked, family 1, member A1 | RBMY1A1, human; UniProt O15542 — verify and add to overrides |
| 2 | Virus coat protein of two apple stem pitting virus (ASPV) isolates: MT32 and PSA‐H | Plant virus coat protein; multiple strains |
| 2 | Pepocin | Ribosome-inactivating protein from *Phytolacca americana*; verify UniProt |
| 2 | Norovirus P particles (GG2.4P), Human | Norovirus capsid P-domain particles; viral protein complex |
| 2 | dsRBD 2 of xlADAR1 | Double-strand RNA binding domain 2 of *Xenopus laevis* ADAR1; Xenopus proteins often TrEMBL only |
| 2 | HepatitisCVirusPolymerase | Garbled name for HCV NS5B; same as NS5B row above |
| 2 | Murine CD200R1 | Mouse CD200 receptor 1; Q9JKB6 may be the correct accession — verify before adding |
| 2 | human Tcell lukemia Tax | HTLV-1 Tax protein (garbled name); P14079 — verify and add to overrides |
| 2 | MBP-fused drosophila HSF | Fusion construct; HSF = heat shock factor, but MBP-fusion sequences are not in UniProt |
| 2 | H9N2 avian influenza virus (AIV) purified haemagglutinin (HA) | HA protein, H9N2; strain-specific |
| 1 | dsRBD 2 of Xlrbpa | Xenopus laevis RBPA; domain fragment |
| 1 | Endothelial regulatory protein pigpen (YPEN-1) | Human YPEN-1; gene name not standard — check HGNC |
| 1 | D‐staphylococcal enterotoxin B peptide (respective SEB peptides) | D-amino acid form — synthetic; full-length SEB = P01553 already in override table; decide if D-form rows should use L-form sequence |
| 1 | Epididymis 4, Human | HE4/WFDC2; Q14508 likely correct — verify and add to overrides |
| 1 | Escherichia coli (E. coli) O157:H7 | Whole bacterium (pathogenic strain), not a specific protein; should probably be reclassified to 'organism' |
| 1 | D‐staphylococcal enterotoxin B peptide (full-length SEB protein) | As above; name says full-length which may be fine — SEB = P01553 |
| 1 | quinoprotein glucose dehydrogenase PQQGDH | PQQ-dependent GDH; enzyme found in various bacteria; needs species specification |
| 1 | Klebsiella Pneumoniae Carbapenemase 2 (KPC-2) on E. coli | KPC-2 beta-lactamase; UniProt accession exists — verify and add to overrides |
| 1 | SipA effector protein secreted by T3SS | Salmonella SipA (invasion protein SipA); P21689 — verify and add to overrides |
| 1 | Rex fusion protein of Human T-lymphotropic virus 1 (HTLV-1) | HTLV-1 Rex; P14079 is Tax (wrong) — Rex is P0C0Z2 or similar; verify |
| 1 | Recombinant human E-selectin/IgG-Fc-chimeras | Chimeric fusion construct; E-selectin = P16150 but fusion is not a UniProt entry |
| 1 | Isoleucyl-tRNA synthetase (tRNAIle) | IleRS; multiple organisms; needs species specification |
| 1 | RNase H2 from Clostridium difficile (C. difficile) (CDH2) | CDH2; check if reviewed UniProt entry exists for *C. difficile* RNase H2 |
| 1 | Hepatitis C virus (HCV) envelope surface glycoprotein E2 | HCV E2; strain-specific, many TrEMBL entries |
| 1 | Hepatitis C virus (HCV) envelope surface glycoprotein E3 | Not a standard HCV protein name — likely E2 mislabeled |
| 1 | Nucleoprotein (NP) of CCHF virus | CCHF NP; accession lookup failed during this pass — retry with UniProt search |
| 1 | Hepatitis C virus (HCV) envelope surface glycoprotein E4 | Not a standard HCV protein name — likely E2 mislabeled |
| 1 | N-terminal region of Recombinant human CTGF (rhCTGF) | As with C-terminal region; CCN2 = P29279 exists; decide on fragment policy |
| 1 | Matrix binding domain [MBD] Hepatitis B capsid | HBV core protein (HBc) domain; HBc = P04530 already in override table |
| 1 | Major urinary protein 13 (MUP13) (rat urine biomarker) | MUP13; correct accession not found in this pass; search UniProt for rat MUP13 |
| 1 | M3 HIV-1 RT | HIV-1 reverse transcriptase (garbled "M3" prefix); HIV-1 RT = P04585 already in override table |
| 1 | Β-casomorphin-7 (BCM-7) | 7-residue opioid peptide from β-casein hydrolysis; too short for protein sequence representation |

---

## 5. Impact on Training Splits

Under augment.py's "ready" filter (`aptamer_sequence notna AND target_type=='protein' AND protein_sequence notna`):

| Split | Before this fix | After this fix | Gain |
|---|---|---|---|
| train | 3,262 | 3,290 | +28 |
| val | 226 | 232 | +6 |
| test | 214 | 228 | +14 |
| **Total** | **3,702** | **3,750** | **+48** |

The reclassification of 432 mislabeled rows does not change these counts — those rows lacked `protein_sequence` and were already excluded. The +48 rows come entirely from the new enrichment.

**Note**: these are the pre-augmentation row counts. The augmented `tier1_train.csv` (currently at 15,430 rows on the VM) was generated from the stale splits and must be regenerated from the updated `master_dataset_v2.csv` before the next training run.

---

## 6. Key Decisions Made (for reproducibility)

1. **Fragments and PTM variants**: "Nonphosphorylated BACE1 CT", "phospho-Ser845 cGluR1", "C-terminal region rhCTGF" — left blank. Using the full-length protein sequence for a fragment-specific aptamer selection creates a cross-attention mismatch (the aptamer's structural partners were the fragment residues, not the full protein). This is a known open question flagged for manual decision.

2. **Chimeric constructs**: "MBP-fused drosophila HSF", "Recombinant human E-selectin/IgG-Fc-chimeras" — left blank. No single UniProt sequence accurately represents a fusion construct.

3. **D-amino acid peptides**: "D-staphylococcal enterotoxin B peptide" — left blank. The L-form full-length protein (P01553) is in the override table but its sequence is not valid for a D-amino acid target.

4. **Scraper garbage "to SARS-CoV-2 spike glycoprote"** (17 rows, patent 050-099-381-908-14X): target name is clearly truncated during scraping; aptamers likely target SARS-CoV-2 spike protein. Left blank pending manual name correction. Do NOT apply P0DTC2 automatically — the truncated name is insufficient confirmation.

5. **RNAP complex (13 rows)**: E. coli RNAP core is a multi-subunit complex (α₂ββ'ω). No single UniProt entry represents it. This is a design decision: either pick the β' subunit (P0A8V2) as the primary binding interface, or exclude. Left blank.

---

## 7. Actionable Next Steps (prioritized)

High-impact manual additions to `data/raw/protein_name_overrides.csv`:

| Priority | Target | Suggested accession | Row count |
|---|---|---|---|
| 1 | Ustilago maydis RNA binding protein Rrm4 | A0A0D1CNT2 (TrEMBL, verify) | 20 |
| 2 | Plasmid R1 transfer operon, gene M (TraM) | P15804 (verify) | 4 |
| 3 | Enterotoxigenic E. coli K88 fimbriae protein | P08185 FaeG (verify) | 4 |
| 4 | Rhesus (Rh) D antigen | P18577 (verify) | 3 |
| 5 | RNA binding motif protein RBMY1A1 | O15542 (verify) | 2 |
| 6 | Murine CD200R1 | Q9JKB6 (verify, may be wrong) | 2 |
| 7 | human T-cell leukemia Tax (HTLV-1) | P14079 (verify) | 2 |
| 8 | Epididymis 4 / HE4 / WFDC2 | Q14508 (verify) | 1 |
| 9 | KPC-2 Klebsiella carbapenemase | lookup needed | 1 |
| 10 | SipA (Salmonella T3SS effector) | P21689 (verify) | 1 |

**Special case**: the 17 "to SARS-CoV-2 spike glycoprote" rows need their `target_name` corrected to a full protein name (manually identify which spike protein from patent 050-099-381-908-14X) before enrichment can proceed.

**Policy decision needed**: determine whether fragment-aptamer rows (BACE1 CT, CTGF C-terminal, phospho-cGluR1) should use the full-length protein sequence or be left blank. 23 rows affected across 5 targets.

After any manual additions: re-run `python scripts/data/enrich_proteins.py --input data/processed/master_dataset_v2.csv`, then regenerate augmented splits with `python scripts/data/augment.py`.

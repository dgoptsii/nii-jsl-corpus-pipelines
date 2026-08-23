# JSL Corpus Statistics

Counts, vocabulary tables and mouth-action analysis for the Japanese Sign
Language Dialogue Corpus.

Two inputs, both folders:

* **parsed annotations** — the CSVs produced by the annotation parser, one per
  document, named after the document (`FO_01-02_AniN_parsed.csv`).
* **the corpus** — searched recursively for `.eaf` files. Used for recording
  durations and for the MouthAction tiers.

```bash
pip install -r requirements.txt

python3 run_pipeline.py \
    --elan-list   ../parsing_pipeline/output/selected_elan_files.csv \
    --annotations /path/to/parsed \
    --out         corpus_statistics_output
```

**Use `--elan-list`, not `--corpus`.** The corpus tree holds the same recording
in several places — the region folder, a gesture-annotation pass, an old copy, a
file-sync conflicted copy — and those copies are different *versions*, not
byte-identical duplicates. `--corpus` walks the tree and indexes every one of
them, so a duplicated recording contributes its duration and its signers more
than once. `--elan-list` reads `selected_elan_files.csv`, the manifest the parser
writes naming the one copy of each recording it actually parsed, so the
statistics describe the same corpus the annotations came from.

`--corpus` still works, for a tree with no duplicates or a quick look; step 1
warns when it indexes two files with the same recording name.

Everything is written under `--out`. Nothing is written anywhere else, and no
input file is modified.

---

## Stages

Each stage reads only what an earlier stage wrote, so any of them can be rerun
alone with `--steps`:

| Step | What it does | Cost |
|------|--------------|------|
| 1 | Index the `.eaf` files: duration, tiers, participants, MouthAction presence | slow — the only pass over the corpus |
| 2 | Concatenate the parsed CSVs into one table and add the classification flags | fast |
| 3 | Every count, globally and per prefecture, plus the LaTeX tables | fast |
| 4 | MouthAction overlap (re-reads only the `.eaf` files that have those tiers) | medium |
| 5 | Figures | fast |

```bash
# changed a threshold in config.py — recompute without touching the corpus
python3 run_pipeline.py --steps 3 5 --out corpus_statistics_output

# restyle the figures only
python3 run_pipeline.py --steps 5 --out corpus_statistics_output
```

Prefecture is read from the filename prefix (`FO_…` → Fukuoka). The mapping is
`REGION_NAMES` in `config.py`: GM Gunma, NR Nara, NS Nagasaki, FO Fukuoka,
IS Ishikawa, TY Toyama, IK Ibaraki.

---

## Definitions used throughout

**Successfully parsed** — the row carries neither `compound` nor `ambiguous`.
Both are legitimate parser outcomes rather than failures: a compound sign is not
the sum of its parts, so it is deliberately left unparsed, and an ambiguous row
is one the parser refused to guess at. They are counted, but they are not part
of the analysable set, and every percentage that has "of parsed" in its name
uses the analysable set as its denominator.

**Unique** — reported twice, for two different things. Unique *annotation
strings* measures how much notational variety the annotators produced. Unique
*lexical items* measures vocabulary size. Only the second is what a
machine-learning user means by "vocabulary".

**Blocking markers** (`D`, `FAL`, `UN`) — these say the annotation could not be
read, which is a fact about the recording rather than about the language. They
are counted separately from the linguistic markers everywhere.

---

## What comes out

### `tables/` — CSV, one file per table

| File | Contents |
|------|----------|
| `summary.csv` | One row for the corpus and one per prefecture: files, duration, signers, annotations, parsed / ambiguous / compound, vocabulary, hapax |
| `keys.csv` | Each marker: count, share of parsed annotations, signers, distinct values |
| `per_file.csv` | The same counts per document — where a regional total comes from |
| `gloss_statistics.csv` | Each lexical item: occurrences, signers, files, regions |
| `coverage.csv` | **The headline vocabulary table** (see below) |
| `coverage_curve.csv` | Token share covered at each vocabulary cutoff, with the OOV rate |
| `coverage_curve_full.csv` | Rank against cumulative share, one row per gloss |
| `class_sizes.csv` | Vocabulary and tokens (raw and capped) surviving each (examples, signers) floor |
| `glosses_top_100.csv` … | The gloss lists behind each row of the coverage table |
| `duration_distribution.csv` | Segment duration percentiles, overall and per marker |
| `signing_rate.csv` | Annotations per minute and median segment length, per signer |
| `signer_balance.csv` | Token share per signer, cumulated |
| `split_feasibility.csv` | What a signer-disjoint held-out split would cost |
| `examples_per_signer.csv` | For the frequent glosses, how examples spread across signers |
| `marker_cooccurrence.csv` | Which markers appear on the same annotation |
| `mouth_overlap.csv` | Mouth action co-occurring with each marker and with bare lexical items |
| `mouth_categories.csv` | Every MouthAction label by category, split into agreed and disagreed |
| `mouth_key_categories.csv` | Each marker x each category: labels, agreement, and the marker's own rows reached |
| `mouth_coverage.csv` | How much of the corpus the mouth analysis speaks for |

### `tables/*.tex` — ready to `\input`

`tab_corpus_summary.tex`, `tab_vocabulary_coverage.tex`,
`tab_marker_frequency.tex`, `tab_mouth_overlap.tex`.

Each is a **complete** `tabular` environment, not a fragment — a fragment ending
in `\\` breaks the `\bottomrule` of whatever includes it. They need `booktabs`,
and, because Japanese passes through unchanged, XeLaTeX or LuaLaTeX with a
CJK-capable main font.

### `figures/` — PNG at 200 dpi

| Figure | Shows |
|--------|-------|
| `fig_corpus_overview` | Files, recording time, signers |
| `fig_annotation_breakdown` | The parser's outcome for the whole corpus |
| `fig_marker_frequency` | Each marker as a count and as a share of parsed annotations |
| `fig_coverage_curve` | Token share covered at the top 100 / 200 / 500 / 900 cutoffs |
| `fig_vocabulary_coverage` | The coverage table drawn: tokens each vocabulary covers, raw and capped |
| `fig_class_sizes` | Glosses meeting each (examples, signers) floor, as a labelled grid |
| `fig_top_glosses` | The 25 most frequent lexical items |
| `fig_region_outcome` | Parsed / ambiguous / compound by prefecture |
| `fig_region_vocabulary` | Lexical variety per 1,000 tokens, by prefecture |
| `fig_mouth_categories` | MouthAction labels per category, agreed vs disagreed |
| `fig_mouth_key_categories` | Each marker x each category, agreed vs disagreed |
| `fig_duration_distribution` | Segment duration percentiles per marker |
| `fig_signer_balance` | How evenly the corpus is sampled across signers |

Every mark carries its exact value, so the figures can be read as tables. Where a
quantity splits into a confirmed and a contested part, the contested part is a
lighter tint of the same hue with a hatch over it -- so the split survives
greyscale printing and colour-vision deficiency, not only colour.

The palette is validated for colour-vision deficiency (worst adjacent pair dE 9.1
under protanopia, 22.9 in normal vision).

Figures with Japanese on an axis need a CJK font. Without one they are skipped
and step 5 prints how to install it (`apt-get install fonts-noto-cjk`).

### `diagnostics/`

`mouth_disagreements.csv` — the annotations whose MouthAction tiers classified
the same moment differently. A count is not actionable; this is the list to open
in ELAN.

---

## The vocabulary coverage table

`tables/coverage.csv` and `tab_vocabulary_coverage.tex`. One row per candidate
vocabulary, each with its own signer floor:

| Vocabulary | Min. signers |
|------------|--------------|
| Top 100 | 5 |
| Top 200 | 5 |
| Top 500 | 5 |
| Top 900 | 3 |
| All glosses | 3 |

Occurrences are capped at 500 per gloss. The cap is not cosmetic: a handful of
very frequent glosses would otherwise dominate every total, and a training set
built from them would be just as imbalanced, so `total_occurrences_capped`
answers the more useful question — how many examples survive if you keep at most
500 per gloss.

Two things about how the rows are built are worth knowing:

* **The signer filter is applied before the cutoff.** "Top 100 with at least 5
  signers" means the 100 most frequent glosses *among those that qualify*, not
  the qualifying members of the overall top 100 — which would silently return
  fewer than 100 rows.
* **Regional rows are recomputed, not filtered.** Restricting to a prefecture
  recounts signers and occurrences inside it, so a regional row never describes
  the whole corpus while claiming to describe a subset.

Change the rows by editing `TOP_N_SPECS` in `config.py`, then rerun step 3.

---

## Calling the top-N functions directly

```python
from io_utils import read_csv_safely
from topn import top_glosses, write_top_glosses, gloss_statistics_for_regions, coverage_table

annotations = read_csv_safely("corpus_statistics_output/annotations.csv")

# top 200 glosses in Fukuoka and Nagasaki, at least 5 signers each, to a file
write_top_glosses(annotations, "top200_FO_NS.csv",
                  top_n=200, min_signers=5, regions=["FO", "NS"])

# the whole corpus, as a DataFrame
stats = gloss_statistics_for_regions(annotations)
top_glosses(stats, top_n=100, min_signers=5)

# the coverage table for one prefecture
coverage_table(gloss_statistics_for_regions(annotations, ["GM"]), scope="GM")
```

---

## MouthAction overlap

For every parsed annotation in a recording that has MouthAction tiers, the mouth
labels overlapping it in time are found and classified as **Mouthing**,
**MouthGesture** or **Others**. Results are aggregated for each marker and for
annotations carrying a lexical item and no marker — the plain lexical signs,
which are the natural comparison class.

Percentages use the annotations that had *some* mouth label as the denominator,
never the whole corpus. A marker that never appears in a mouth-annotated
recording has no evidence either way and should not be reported as 0%.

The categories can co-occur on one annotation, so they do not sum to 100%.

### Agreement between annotators

Agreement is measured per **label**, not per annotation, so the numbers can be
read against the total inventory of MouthAction labels. A label counts as
*disagreed* when a label from another tier overlaps it and puts it in a
different category, and as *agreed* otherwise.

Labels that no other tier overlaps therefore land in "agreed" — nothing
contradicts them. That is a weaker claim than genuine confirmation, so they are
also counted on their own as `n_uncontested`; if that column is close to
`n_labels`, the agreement rate is mostly measuring single-annotator files rather
than annotator consensus.

A disagreement is usually not an error — it normally means the annotation spans
a boundary between two mouth actions — which is why the individual cases go to
`diagnostics/mouth_disagreements.csv` rather than being only counted.

---

## For machine-learning users

The tables above describe the corpus as a linguistic object. These describe it
as a dataset:

* **`split_feasibility.csv` is the one to read first.** A random train/test split
  leaks the same signer into both halves, and the accuracy it reports then
  measures memorisation of a person rather than recognition of a sign. This table
  holds signers out instead — smallest first, so the test set spans as many
  different people as the budget allows — and reports `oov_token_percent`: the
  share of test tokens whose gloss never appears in training. That number is the
  honest ceiling on a closed-vocabulary recogniser, and it is invisible in any
  randomly-split evaluation.
* **`class_sizes.csv`** — how the trainable vocabulary shrinks as the minimum
  examples and minimum signers rise. Raising the signer floor usually costs far
  more vocabulary than raising the example floor. Tokens are reported raw and
  capped at 500 per gloss, since a balanced training set would keep the capped
  number. The grid is `CLASS_SIZE_THRESHOLDS` x `CLASS_SIZE_SIGNER_FLOORS` in
  `config.py` — by default 1/20/50/100/200/500/900 examples against 1/3/5/8
  signers.
* **`coverage_curve.csv`** — how much of the corpus a fixed vocabulary covers,
  and what falls outside it.
* **`duration_distribution.csv`** — sets the input window and the frame budget.
  Reported as percentiles, because durations are strongly right-skewed and a mean
  would describe no actual sign.
* **`examples_per_signer.csv`** — `max_signer_share` near 1 means one person
  produced nearly every example of that gloss.
* **`signing_rate.csv`, `signer_balance.csv`** — how far the corpus is from
  evenly sampled, and how much pace varies between signers.
* **`marker_cooccurrence.csv`** — which markers travel together, before deciding
  which deserve their own head in a multi-task model.

---

## Configuration

`config.py` holds everything a maintainer is likely to change: the parser's
column names, the region map, `TOP_N_SPECS`, `OCCURRENCE_CAP`,
`COVERAGE_CUTOFFS`, `COVERAGE_CURVE_MARKERS`, `CLASS_SIZE_THRESHOLDS`,
`CLASS_SIZE_SIGNER_FLOORS`, `MIN_OVERLAP_MS`, and the tier-name
hints used to recognise MouthAction and Word tiers. If the parser's schema
changes, `KEY_COLUMNS` is the only place that has to follow.

## Tests

```bash
python3 -m pytest -q
```

101 tests. They build a miniature corpus rather than depending on the real one:
small enough that every expected number can be checked by hand, and able to
contain the awkward cases on purpose — a `REF_ANNOTATION` chain, two MouthAction
tiers that disagree, a gloss produced by a single signer.

## Requirements

Python 3.9+, `pandas`, `numpy`, `matplotlib`. No ELAN or Java dependency: the
`.eaf` files are read as XML.

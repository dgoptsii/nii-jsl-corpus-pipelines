# ELAN Annotation Parsing Pipeline

Parse JSL Word-tier annotations in ELAN (`.eaf`) files and write **new** ELAN
files that keep every original tier and add the parsed columns as child tiers.

```
corpus/*.eaf  ──▶  extract Word-jp  ──▶  rule parser  ──▶  *_parsed.eaf
```

Python 3.9 or newer. No other dependency.

---

## Quick start

```bash
python3 run_pipeline.py ./corpus --list-only          # what would be processed
python3 run_pipeline.py ./corpus -output_folder ./output \
    --save-debug --exceptions-file input_lists/exceptions.txt
```

`./corpus` holds the `.eaf` files; `./output` is where results go. The script
can be run from any directory.

### One file per recording, or nothing

Every stage names its output after the input file stem, so two files with the
same name would overwrite each other. Rather than pick one, the pipeline
**rejects the whole folder**: it prints every duplicated name with all its
paths, parses nothing, and exits `1`.

Which copy of a recording is correct is an annotation question. Whichever one a
heuristic chose would decide what every downstream number describes,
invisibly. Curate the folder, then run again. `--list-only` still prints the
file list when duplicates are present.

---

## Options

| Option | What it does |
| --- | --- |
| `-output_folder FOLDER` | Where results go. Default `pipeline_output`. Use an absolute path. |
| `--flat` | Only the `.eaf` files directly in the folder; default searches the tree |
| `-f FILE`, `--file-list FILE` | Process only the files listed in a text file |
| `--regions FO NS` | Only files whose name starts with these prefixes |
| `--save-debug` | Also write the intermediate CSVs and tier reports |
| `--exceptions-file FILE` | Apply manual "this is not ambiguous" decisions |
| `--no-overwrite` | Fail instead of overwriting an existing output `.eaf` |
| `--list-only` | Print which files would be processed, then stop |

A `--file-list` takes one entry per line; blank lines and `#` comments are
ignored. Only the file stem is used, and matching ignores case and punctuation,
so `FO_01-02_AniN`, `TY_05-06_AniN.eaf` and `/mnt/corpus/GM_09-10_AniN.eaf` all
work. Listed files that do not exist are reported, so a typo cannot silently
shrink a run.

Exit codes: `0` all files fine, `2` finished with failures (listed in the
summary), `1` bad arguments, nothing found, or duplicate names. A corrupt
`.eaf` is caught and logged; the rest still process.

---

## What you get

```
output/
├── parsed_elan_files/               always written; open these in ELAN
│   └── FO_01-02_AniN_parsed.eaf
└── debug/                           only with --save-debug
    ├── extracted_word_annotations/  raw annotations + timings
    ├── parsed_annotations/          the full parsed table
    ├── ambiguous_annotations/       the file you review by hand
    └── tier_reports/                what was created, what did not match
```

Without `--save-debug` nothing intermediate is written; the stages hand data
over in memory.

Every original tier survives untouched. For each speaker and each non-empty
parsed column one new tier is added, as a `Symbolic_Association` child of the
Word-jp tier, so every parsed value sits under the annotation it came from and
inherits its timing:

```
FO_07_FK_40F-Word-jp                     original, untouched
  └── FO_07_FK_40F-PARSED-lexical_item   new
  └── FO_07_FK_40F-PARSED-pt             new
FO_07_FK_40F-Word-roman                  original, untouched
```

---

## Running the steps separately

Only needed to hand-correct something between stages; otherwise
`run_pipeline.py` does all three. Each script prints the next command.

```bash
python3 step1_extract.py ./corpus -output_folder ./extracted
python3 step2_parse.py ./extracted -output_folder ./parsed \
    --exceptions-file input_lists/exceptions.txt
#   ... hand-correct ./parsed/parsed/*.csv here ...
python3 step3_build_elan.py ./corpus -parsed_csv_folder ./parsed/parsed \
    -output_folder ./final_eaf
```

---

## Manual review

Run with `--save-debug` and open
`output/debug/ambiguous_annotations/<file>_ambiguous_rows.csv`. Each row keeps
its speaker and timings, so it can be found in ELAN immediately. For each row:

* the annotation is wrong, fix it in ELAN and re-run;
* the annotation may keep Latin letters after parsing, add the string to
  `input_lists/exceptions.txt`;
* the rule is too strict in general, discuss before changing it, since a rule
  change affects the whole corpus.

Matching uses the normalised, lower-cased annotation. With no exceptions file,
no exceptions apply.

---

## The parsed columns

| Column | Meaning |
| --- | --- |
| `speaker_id`, `tier_id`, `time_start`, `time_end` | metadata, used to join back onto ELAN |
| `annotation` | the original string, verbatim |
| `lexical_item` | lexical material with all markers removed |
| `pt` | pointing number, `0`, or `dw` |
| `dw` | depicting word |
| `fs` | fingerspelling |
| `aw` | air writing |
| `lh`, `rh` | left / right hand |
| `d` | disfluency (**blocking**) |
| `cl` | classifier |
| `m` | mouth action |
| `ges` | gesture |
| `nmm` | non-manual marker |
| `rep` | repetition, as `word(count;hand)` |
| `stop`, `hold`, `index`, `keep` | boolean articulation flags |
| `fal` | false start (**blocking**) |
| `un` | unclear (**blocking**) |
| `qm`, `past`, `neg` | boolean flags |
| `compound` | verbatim annotation, for `<...>` compound groups |
| `ambiguous` | verbatim annotation, for rows needing manual review |

Empty means the rule did not fire; `TRUE` means the marker was present with no
value; multiple values in one column are joined with `;`. When a **blocking**
marker (`d`, `fal`, `un`) is detected every other column is cleared: those
markers mean the annotation's content cannot be trusted.

Full rules, surface forms and worked examples are in a short reference document
generated from the parser itself, so it cannot drift from the implementation:

```bash
python3 docs/generate_simple_rules.py
cd docs && latexmk -xelatex parsing_rules_simple.tex
```

---

## What each file does

| File | Purpose |
| --- | --- |
| `run_pipeline.py` | the main script: all three stages plus the orchestration |
| `step1_extract.py`, `step2_parse.py`, `step3_build_elan.py` | one stage each |
| `config.py` | column schema, tier naming, tolerances: **tweak here first** |
| `locate_elan_files.py` | finding `.eaf` files (recursive/flat, file list, regions) |
| `extract.py` | reading Word-jp annotations out of ELAN XML |
| `parsing.py` | **the rule engine**: the only file with linguistic knowledge |
| `elan_builder.py` | writing the new `.eaf`, keeping every original tier |
| `io_utils.py` | CSV reading/writing with Japanese encoding fallbacks |
| `input_lists/` | `exceptions.txt`, `files_of_interest.txt` |

---

## Using it from Python

```python
from pathlib import Path
from config import PipelineConfig
from run_pipeline import run_pipeline
from parsing import parse_annotation

print(parse_annotation("pt3(dw:5種類)"))          # check one rule

result = run_pipeline(PipelineConfig(
    elan_folder=Path("/data/corpus"),
    output_folder=Path("/data/output"),
    save_debug=True,
))
print(f"{result.parsed_percentage:.1f}% resolved across {len(result.files)} files")
for item in result.failed:
    print("FAILED:", item.file_stem, item.error)
```

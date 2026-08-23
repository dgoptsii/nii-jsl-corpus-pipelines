# ELAN Annotation Parsing Pipeline

Parse JSL Word-tier annotations in ELAN (`.eaf`) files and write **new** ELAN
files that keep every original tier and add the parsed columns as child tiers.

```
corpus/*.eaf  ──▶  extract Word-jp  ──▶  rule parser  ──▶  *_parsed.eaf
```


---

## Requirements

Python 3.9 or newer.

---

## Quick start

```bash
git clone <your-repo-url>
cd elan-parsing-pipeline

python3 run_pipeline.py ./corpus -output_folder ./output
```

`./corpus` is the folder holding your `.eaf` files. `./output` is where results
go. That's the whole thing.

Before a big run, check what would be processed:

```bash
python3 run_pipeline.py ./corpus --list-only
```

A realistic run:

```bash
python3 run_pipeline.py ./corpus -output_folder ./output --save-debug --exceptions-file input_lists/exceptions.txt
```

### One file per recording, or nothing

Every stage names its output after the input file stem, so two files with the
same name would overwrite each other's results. Rather than pick one, the
pipeline **rejects the whole folder**: it prints every duplicated name with all
of its paths, parses nothing, writes no manifest, and exits `1`.

```
ERROR: 3 name(s) exist in more than one place (4 extra copy(ies)).
  FO_05-06_Cur
    /corpus/04_Fukuoka/Curry/FO_05-06_Cur.eaf
    /corpus/old/FO_05-06_Cur.eaf
...
Nothing was parsed. Remove the extra copies, or point --elan-folder at a
folder holding one file per recording.
```

Which version of a recording is the correct one is an annotation question, not
something a heuristic should settle silently — whichever copy it chose would
decide what every downstream number describes, invisibly. Curate the folder,
then run again. `--list-only` still prints the file list when duplicates are
present, since diagnosing them is what it is for.

You can run the script from anywhere — it does not care about your current
directory:

```bash
python3 /path/to/elan-parsing-pipeline/run_pipeline.py /data/corpus -output_folder /data/output
```

---

## Options

```
python3 run_pipeline.py ELAN_FOLDER [options]
```

| Option | What it does |
| --- | --- |
| `-output_folder FOLDER`, `--output-folder FOLDER` | Where results go. Default: `pipeline_output` |
| `--flat` | Read only the `.eaf` files directly in the folder. Default is to search the whole tree. |
| `-f FILE`, `--file-list FILE` | Process only the files listed in a text file |
| `--regions FO NS` | Process only files whose name starts with these prefixes |
| `--save-debug` | Also write the intermediate CSVs and tier reports |
| `--exceptions-file FILE` | Apply your manual "this is not ambiguous" decisions |
| `--no-overwrite` | Fail instead of overwriting an existing output `.eaf` |
| `--list-only` | Print which files would be processed, then stop |
| `-h`, `--help` | Show all options |

### Choosing which files to process

**Everything, recursively** (the default):

```bash
python3 run_pipeline.py ./corpus -output_folder ./output
```

**One folder only, no subfolders:**

```bash
python3 run_pipeline.py ./corpus -output_folder ./output --flat
```

**Only the files you list:**

```bash
python3 run_pipeline.py ./corpus -output_folder ./output --file-list input_lists/files_of_interest.txt
```

One entry per line. Blank lines and `#` comments are ignored. Any of these
forms works — only the file stem is used, and matching ignores case and
punctuation:

```
# input_lists/files_of_interest.txt
FO_01-02_AniN
TY_05-06_AniN.eaf
NS_07-08_AniN_word_annotations.csv
/mnt/corpus/GM_09-10_AniN.eaf
```

Files you list that don't exist are reported explicitly, so a typo doesn't
silently shrink your run.

**By region prefix:**

```bash
python3 run_pipeline.py ./corpus -output_folder ./output --regions FO NS IS
```

---

## What you get

```
output/
├── parsed_elan_files/               ← always written; open these in ELAN
│   └── FO_01-02_AniN_parsed.eaf
└── debug/                           ← only with --save-debug
    ├── extracted_word_annotations/  raw annotations + timings
    ├── parsed_annotations/          the full parsed table
    ├── ambiguous_annotations/       ← the file you review by hand
    └── tier_reports/                what was created, what didn't match
```

Without `--save-debug`, nothing intermediate is written to disk — extraction
and parsing hand data over in memory.

### The new ELAN files

Every original tier survives untouched. For each speaker and each non-empty
parsed column, one new tier is added:

```
FO_07_FK_40F-Word-jp                  ← original, untouched
  └── FO_07_FK_40F-PARSED-lexical_item  ← new
  └── FO_07_FK_40F-PARSED-pt          ← new
  └── FO_07_FK_40F-PARSED-cl          ← new
  └── ...
FO_07_FK_40F-Word-roman               ← original, untouched
FO_07_FK_40F-LOCALIZATION             ← original, untouched
```

New tiers are `Symbolic_Association` children of the Word-jp tier, so every
parsed value sits under the annotation it came from and inherits its timing.


Notes:

- **Use absolute paths** for `-output`. The default output folder is relative and
  lands wherever the job started.
- **Exit codes**: `0` all files fine, `2` finished with some failures (listed in
  the summary), `1` bad arguments, nothing found, or duplicate file names.
- **One bad file doesn't kill the run.** A corrupt `.eaf` is caught, logged, and
  the rest still process.

---

## Running the steps separately

Use these only if you want to hand-correct something between stages. Otherwise
`run_pipeline.py` does all three.

```bash
# 1. extract Word-jp annotations to CSV
python3 step1_extract.py ./corpus -output_folder ./extracted

# 2. parse them
python3 step2_parse.py ./extracted -output_folder ./parsed --exceptions-file input_lists/exceptions.txt

#    ... hand-correct ./parsed/parsed/*.csv here if you want ...

# 3. write the new ELAN files
python3 step3_build_elan.py ./corpus -parsed_csv_folder ./parsed/parsed -output_folder ./final_eaf
```

Each script prints the next command to run when it finishes.

---

## Manual review

Run with `--save-debug` and open
`output/debug/ambiguous_annotations/<file>_ambiguous_rows.csv`. Each row keeps
its speaker and timings, so you can find it in ELAN immediately.

For each row, decide:

- **the annotation is wrong** → fix it in ELAN, re-run;
- **the annotation is allowed have Latin letters after parsing** → add the string to
  `input_lists/exceptions.txt`;
- **the rule is too strict in general** → discuss before changing it; a rule
  change affects the whole corpus.

`input_lists/exceptions.txt` looks like this:

```
# reviewed 2026-08-06
pt:L(LH)
```

Matching uses the normalised, lower-cased annotation. 
If no exceptions file passed, no exceptions apply.

---

## The parsed columns

| Column | Meaning |
| --- | --- |
| `speaker_id`, `time_start`, `time_end` | metadata, used to join back onto ELAN |
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

Conventions: empty means *the rule did not fire*; `TRUE` means *the marker was
present with no value*; multiple values in one column are joined with `;`.

**Blocking markers**: when `d`, `fal` or `un` is detected, every other column is
cleared — those markers mean "do not trust this annotation's content".

Full rules, surface forms and worked examples: `reports/parsing_rules_simple.pdf`,
regenerated from the parser itself with `python3 docs/generate_simple_rules.py`
so the documentation cannot drift away from the implementation.

---

## What each file does

**Scripts you run:**

| File | Purpose |
| --- | --- |
| `run_pipeline.py` | **the main script** — all three stages, plus the orchestration code |
| `step1_extract.py` | stage 1 only, for the separate-steps workflow |
| `step2_parse.py` | stage 2 only |
| `step3_build_elan.py` | stage 3 only |

**Modules the scripts import** (you don't run these directly):

| File | Purpose |
| --- | --- |
| `config.py` | column schema, tier naming, time-matching tolerances — **tweak here first** |
| `locate_elan_files.py` | finding `.eaf` files (recursive/flat, file list, regions) |
| `extract.py` | reading Word-jp annotations out of ELAN XML |
| `parsing.py` | **the rule engine** — the only file with linguistic knowledge in it |
| `elan_builder.py` | writing the new `.eaf`, keeping every original tier |
| `io_utils.py` | CSV reading/writing with Japanese encoding fallbacks |

**Everything else:**

| Path | Purpose |
| --- | --- |
| `input_lists/exceptions.txt` | your manual exceptions decisions |
| `input_lists/files_of_interest.txt` | template for `--file-list` |

---

## Using it from Python

```python
import sys
sys.path.insert(0, "/path/to/elan-parsing-pipeline")   # not needed if you're in the repo folder

from pathlib import Path
from config import PipelineConfig
from run_pipeline import run_pipeline
from parsing import parse_annotation

# check one rule interactively
print(parse_annotation("pt3(dw:5種類)"))

# run over a corpus and work with the numbers
result = run_pipeline(PipelineConfig(
    elan_folder=Path("/data/corpus"),
    output_folder=Path("/data/output"),
    save_debug=True,
))
print(f"{result.parsed_percentage:.1f}% resolved across {len(result.files)} files")
for item in result.failed:
    print("FAILED:", item.file_stem, item.error)
```
# nii-jsl-corpus-pipelines

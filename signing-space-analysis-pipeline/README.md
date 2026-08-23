# Signing Space Analysis

Takes parsed JSL annotations, cuts a video clip per annotation, extracts a
reduced landmark set, classifies hand positions into signing-space regions, and
produces tables and figures with signer-level uncertainty.

```
parsed annotations ─▶ clips ─▶ landmarks ─▶ region counts ─▶ tables + figures
```

Companion to the annotation-parsing pipeline: its `*_parsed.csv` files are this
repo's input.

---

## Quick start

```bash
git clone <your-repo-url>
cd signing-space-analysis
pip install -r requirements.txt

python3 run_pipeline.py /path/to/parsed_annotations/parsed \
    --video-root /path/to/ELAN_W_OpenPose \
    -output_folder ./out \
    --keywords cl fs lexical_item \
    --regions FO GM \
    --signers-file input_lists/signers.csv
```

Trial run on a handful of clips first — landmark extraction is the slow part:

```bash
python3 run_pipeline.py /path/to/parsed --video-root /path/to/videos \
    -output_folder ./trial --limit 5
```

---

## The five stages

| Script | Does | Needs |
| --- | --- | --- |
| `step1_select_rows.py` | picks parsed rows for each keyword | pandas |
| `step2_extract_clips.py` | cuts one cropped clip per annotation | ffmpeg |
| `step3_extract_landmarks.py` | detects and stores the reduced landmark set | mediapipe, opencv, ultralytics |
| `step4_region_counts.py` | classifies landmarks into signing-space regions | numpy, pandas |
| `step5_analyze.py` | builds tables and figures | + matplotlib |

`run_pipeline.py` runs all five. Each also runs standalone, which is what you
want when only the analysis changed:

```bash
python3 step5_analyze.py ./out --signers-file input_lists/signers.csv
```

That last one is the important shortcut. **Age bands and gender are resolved in
step 5, not baked into the clip index**, so regrouping the ages, fixing a
signer's gender or correcting an age costs seconds — no clips are re-cut and no
landmarks re-extracted. Only two things force earlier stages:

| Change | Re-run from |
| --- | --- |
| age bands (`AGE_GROUPS`), gender, any metadata label | step 5 |
| excluding a bad clip | step 5 |
| region geometry, yaw correction | step 4 |
| **handedness** — it decides whether a signer is mirrored | step 4 |
| a different keyword or region selection | step 1 |

Landmark extraction is by far the most expensive stage. It skips clips that
already have output unless you pass `--overwrite`, and it takes `--workers N` to
use more than one core — see [Speed](#speed-cpu-gpu-and---workers).

---

## What gets stored, and why it is fast

MediaPipe reports 33 pose + 21 per hand + up to 478 face landmarks per frame.
This pipeline stores only what the analysis uses:

| Group | Points | Why |
| --- | --- | --- |
| pose | shoulders, elbows, wrists | normalisation anchors, yaw estimation |
| hands | wrist + the knuckle of every finger | the points that get classified |
| face | chin, top of head | the two anatomical reference lines |

About **50 floats per frame instead of ~1200** — roughly 24× smaller files and
proportionally faster loading.

Detection cost is reduced separately, because MediaPipe computes everything
regardless of what you keep. The defaults are `--model-complexity 1` and the
unrefined face mesh, since the refined 478-point mesh only improves eyes and
lips. To reproduce the older, slower settings exactly:

```bash
python3 step3_extract_landmarks.py ./out --model-complexity 2 --refine-face
```

If chin or head-top tracking looks worse on your footage, that flag is the first
thing to switch back.

---

## The signers file

One optional CSV carries everything about the people in the corpus:

```csv
signer_id,handedness,age,gender
FO_07_FK_40F,left,34,F
GM_02_XX_60M,,67,M
NS_01_AB_25M,right,25,F
```

`input_lists/signers.csv` is the real one, built from the corpus
annotation-progress spreadsheet by `tools/signers_from_xlsx.py`:

```bash
pip install openpyxl
python3 tools/signers_from_xlsx.py 進捗状況Annotation\ information3.xlsx \
    -output_folder input_lists
```

That writes the CSV and a companion `input_lists/signers.txt` recording every
judgement the sheet forced — which ages were copied from a pair partner, which
handedness cells were ambiguous. Re-run it when the spreadsheet changes rather
than editing the CSV by hand.

Three decisions live in that script, all reversible in one place:

- **Only a bare `L` is mirrored.** `R(L)`, `R (L)` and `R L` mark a signer who
  uses the left hand sometimes but leads with the right; mirroring them would
  move their dominant hand to the wrong side of the space. `?` falls back to
  right-handed, the same default an unlisted signer gets.
- **A blank age means "same as the pair partner"**, since the second signer of a
  pair usually has no age written. Every value filled this way is listed in
  `signers.txt`.
- **`REGION_CODES`** maps each prefecture to the two-letter filename prefix
  (`Gunma → GM`, `Ibaraki → IK`, …). Wrong codes are the one failure here that
  is otherwise silent, which is why step 2 prints how many signers matched.

Only the ID is required. **Blank handedness means right-handed**, blank age
means the signer shows as "unknown" in the age table rather than being dropped,
and anyone absent from the file gets both defaults — so the file only has to
list what differs.

Column names are matched loosely, so a spreadsheet someone else maintains will
usually just work: the ID may be `signer_id`, `speaker_id` or `participant_id`;
handedness may instead be a `left_handed` column holding TRUE/1/yes/x; age may
be `age`, `age_years` or `years`; gender may be `gender`, `sex`, `性` or `性別`,
written `M`/`F`/`male`/`男`/`女`. Any other gender value is carried through
verbatim rather than folded away.

IDs match on a prefix, so `FO_07` covers `FO_07_FK_40F`. Zero-padding is
ignored — `GM_5` and `GM_05` are the same signer — and a prefix may not end
inside a number, so `GM_1` cannot claim `GM_11`.

```bash
python3 run_pipeline.py ./parsed --video-root ~/corpus \
    --signers-file input_lists/signers.csv
```

### What marking someone left-handed does

For a signer marked `left`, the pipeline:

1. **negates the horizontal axis** after shoulder normalisation, so their
   signing space lands where a right-hander's does; and
2. **files their left hand as `dominant`**, the right as `non_dominant`.

Without step 2, a left-hander's dominant-hand data would sit in the `left`
column while every right-hander's sat in `right`, so averaging either column
would silently halve your sample. Columns are therefore named by **role**, never
by side:

```
dominant_upper_torso, dominant_p_right_upper_torso, ...
non_dominant_upper_torso, ...
```

`handedness` is carried on every row, so the original left/right identity stays
recoverable.

---

## Dropping bad crops

Some clips are unusable and no automatic check catches them: the signer walks
out of frame, the wrong panel was cropped, the hands are never visible. Look
through `out/clips/`, list the bad ones, and pass the list:

```bash
python3 run_pipeline.py ... --exclude-file input_lists/excluded_clips.txt
python3 step5_analyze.py ./out --exclude-file input_lists/excluded_clips.txt
```

```
# input_lists/excluded_clips.txt
FS_000123          # signer leaves frame at 0:02
FS_000481          # hands below the crop
FS/NS/NS_07-08_AniN    # whole recording: camera too far right
```

Name a clip however is easiest to copy out of your file browser — the bare name
`FS_000123`, with `.mp4`, the full `clip_id`, or an absolute path. Case and
slash direction are ignored. A leading folder drops everything under it, so
`FS/NS/NS_07-08_AniN` removes one recording and `CL/GM` removes a keyword from
one prefecture. Substrings do *not* match: `FS_0001` will not take `FS_000123`.

**An entry that matches nothing is reported at the end of the run.** A typo here
would otherwise leave the bad clip silently in the averages, which is the whole
failure this file exists to prevent.

The clips are never deleted — they are skipped from landmark extraction onward.
Delete a line and re-run to bring one back. Because the filtering also happens
in step 5, you can exclude a clip whose landmarks and counts already exist
without recomputing anything:

```bash
python3 step5_analyze.py ./out --exclude-file input_lists/excluded_clips.txt
```

---

## Which side is "left"?

**Region names are in the signer's frame.** `p_right_upper_torso` is the space beside the
signer's *right* shoulder — which appears on the **left** of the image, because
you are facing them.

Normalisation puts the signer's left shoulder at +x, so the negative-x half of
the signing space is the signer's right side. Two tests pin this down
(`test_region_names_are_in_the_signers_frame`), because getting it backwards
would silently swap every lateral statistic without any error.

The body-map figures are drawn mirrored, as if facing the signer, so the
signer's right appears on the viewer's left — the usual convention for an
anatomical diagram. The figure mirrors the *drawing*, never the names, so the
CSVs and the figures cannot drift apart.

### One caveat for left-handed signers

Mirroring flips a left-hander's space so it pools with the right-handers. After
that flip, `p_right_upper_torso` for a mirrored signer means *the side a right-hander's
dominant hand would use* — the dominant side — not literally their anatomical
right. For the unmirrored majority the names are literal.

If you need the literal anatomy for a left-hander, `handedness` is on every row:
swap left and right back for rows where it is `left`.

---

## Uncertainty

Every interval is a **95% confidence interval bootstrapped over signers**, not
over clips.

One signer contributes many clips, so clips are not independent observations. A
clip-level interval would come out far too narrow and would make a result
resting on two or three people look precise:

```
avg regions per clip = 3.4
   CI over clips    →  [3.3, 3.5]     looks solid
   CI over signers  →  [2.6, 4.1]     honest for 5 signers
```

The procedure: draw *n* signers with replacement (*n* = distinct signers), pool
all their clips, recompute, repeat 2000 times, take the 2.5th and 97.5th
percentiles.

Every table carries `n_signers` beside its interval, and cells with fewer than
five signers are marked `ci_reliable=False` and hatched in the figures — the
reviewer's point about small subsets, made visible rather than hidden.

---

## Outputs

```
out/
├── key_rows/<KEYWORD>/ALL_<KEYWORD>_rows.csv     selected annotations
├── clips/<KEYWORD>/<REGION>/.../<CLIP>.mp4       cropped video
├── clips/clip_index.csv                          ← signer, handedness, age group
├── landmarks/<CLIP_ID>/landmarks.npz             the reduced set
├── region_counts/<CLIP_ID>/region_counts.csv     per frame, per hand role
├── tables/
│   ├── by_region_and_keyword.csv        ← Table 1, one row per hand role
│   ├── by_age_group_and_keyword.csv     ← Table 2, one row per hand role
│   ├── by_gender_and_keyword.csv        ← Table 3, one row per hand role
│   ├── by_region_age_gender.csv         all three crossed
│   ├── clip_level.csv                     one row per clip and hand role
│   ├── region_distribution.csv            share per region, with CIs
│   ├── region_groups.csv                  coarse anatomical groups
│   └── central_periphery_summary.csv      central / periphery / extreme
├── figures/
│   ├── body_map_<KEYWORD>_<REGION>.png     signing space, shaded by share
│   ├── avg_regions_by_region.png
│   ├── avg_regions_by_age_group.png
│   ├── avg_regions_by_gender.png
│   └── region_groups.png
└── debug/                                  only with --save-debug
    └── <CLIP_ID>/
        ├── yolo_person_masks.jpg           which signer was kept
        ├── mediapipe_input.jpg             the frame MediaPipe received
        └── signing_space.mp4               clip beside its classified space
```

`clip_index.csv` is the join table: it carries signer ID, handedness, age and
gender into every later stage, so nothing downstream has to re-read annotations.
Step 5 re-resolves the age band and gender from `--signers-file` when you pass
one, so those labels are never stale.

### Age bands

The bands are the corpus decades — `20`, `30`, `40`, `50`, `60`, `70`, `80` —
because the spreadsheet records a decade band per signer rather than an exact
age. Grouping them any coarser would imply a precision the source does not have.
`<20` exists only as a guard; no signer in the corpus is under 20.

They live in `AGE_GROUPS` in `config.py`. Change them there and re-run step 5.

### The summary tables

All four have the same shape; only the grouping differs.

| Table | Grouped by |
| --- | --- |
| `by_region_and_keyword.csv` | prefecture × keyword |
| `by_age_group_and_keyword.csv` | age band × keyword |
| `by_gender_and_keyword.csv` | gender × keyword |
| `by_region_age_gender.csv` | prefecture × age × gender × keyword |

The last one is where an interaction would show — whether an age effect holds
in every prefecture, whether it differs by gender. It is also the one that
fragments fastest: 7 prefectures × 7 age bands × 2 genders splits 122 signers
very thin, so most cells will be `ci_reliable=False`. Read those as "not enough
people", not as a finding. The one-variable tables are what belongs on a poster;
the cross-table is for deciding which comparison is worth making.

| Column | Meaning |
| --- | --- |
| `region_code` / `age_group` / `gender` | the grouping |
| `keyword` | annotation keyword |
| `hand_role` | `dominant` or `non_dominant` — **one row each** |
| `n_annotations` | clips in this cell **in which this hand was detected** |
| `n_clips` | clips in this cell, hand detected or not |
| `hand_present_percent` | `n_annotations` / `n_clips` — how two-handed the keyword is |
| `n_signers` | unique signers |
| `n_left_handed_signers` | how many of those are left-handed |
| `avg_regions` | average signing-space regions used per clip |
| `ci_low`, `ci_high` | 95% CI, bootstrapped over signers |
| `ci_reliable` | False when fewer than 5 signers |
| `sd_regions`, `median_regions` | spread |

### One row per hand

The two hands are averaged **separately**, never pooled:

```
region_code keyword    hand_role  n_annotations  n_clips  hand_present_percent  n_signers  avg_regions  ci_low  ci_high
         FO      FS     dominant            153      153                 100.0         17       1.9608  1.6846   2.2500
         FO      FS non_dominant             68      153                  44.4         14       1.2794  1.0100   1.5900
```

A single pooled average would sit between two numbers that mean different
things: the dominant hand carries the sign, while the non-dominant one is
frequently idle or acting as a static base. Pooling would also mix a hand
detected in every clip with one detected in less than half of them, so the mean
would move whenever tracking quality changed rather than whenever signing did.

`n_annotations` therefore differs between the roles — it counts only the clips
in which *that* hand was detected, because a hand that never appeared has no
signing space to average. `n_clips` keeps the full denominator, and
`hand_present_percent` is the ratio: read it as a **two-handedness rate** per
keyword and region, which is a result in its own right and one the reviewers
are likely to ask about.

The body-map figures already drew both hands side by side. The bar charts show
one role at a time — two roles × two keywords × several regions is unreadable
on one axis — and `--hand-role` picks which, with the role named in the title.
The CSVs are unaffected by that flag.

---

## Options worth knowing

```
--keywords cl fs pt lexical_item   which annotation columns to analyse
--regions FO GM NS                 which geographical regions
--lexical-only                     for lexical_item, exclude rows with any marker
--limit 5                          cap clips per keyword (trial runs)
--signers-file FILE                signer_id,handedness,age,gender (one CSV)
--exclude-file FILE                clip names to drop after inspection
--model-complexity {0,1,2}         detection cost/accuracy
--refine-face                      478-point face mesh (slower, unused here)
--no-person-mask                   skip YOLO; faster, but the other signer may leak in
--no-yaw                           plain shoulder normalisation, no yaw correction
--hand-role dominant|non_dominant  which hand the bar figures draw (tables hold both)
--workers 8 | auto                 parallel landmark extraction (see below)
--figure-font-scale 1.6            bigger type and figures, for a poster
--from-stage / --to-stage          run part of the pipeline
--overwrite                        redo work that already has output
--save-debug                       diagnostic renders (see below)
--debug-limit 5 | all              how many clips get them
--debug-max-frames 60 | all        how much of each clip to render
```

---

## Diagnostic renders

Off by default, because they are slow and produce more video than the corpus.
Turn them on when a number looks wrong and you need to see why:

```bash
python3 run_pipeline.py ./parsed --video-root ~/corpus \
    -output_folder ./out --save-debug --debug-limit 5
```

### How many to render

`--debug-limit` takes a **number of clips** or the word **`all`**:

```bash
--save-debug --debug-limit 5      # five clips: the usual spot-check
--save-debug --debug-limit 50     # a sample big enough to spot a pattern
--save-debug --debug-limit all    # every clip; expect it to dominate the runtime
```

The default is 5. `all` is spelled out rather than written as `0`, because
`--debug-limit 0` reads like "none" and would mean the opposite.

`--debug-max-frames` caps how much of each clip is drawn, which is the other way
to keep the output small — thirty clips of one second each usually tells you more
than five whole ones:

```bash
--save-debug --debug-limit 30 --debug-max-frames 20
```

Two things get written, both under `out/debug/<CLIP_ID>/`.

**`signing_space.mp4`** — the cropped clip with its landmarks drawn, beside the
normalised signing space, frame by frame. Every counted hand point is coloured
by the region it was assigned to, the anatomical reference lines are drawn, and
the current yaw and handedness are printed on the frame. This is the render that
shows you a chin anchor jumping, a yaw estimate shearing the space sideways, or
a hand that is being tracked but landing in the wrong cell.

The two halves face the same way: the canvas is *not* mirrored, because
normalisation already puts the signer's left at +x, which is the image right. If
a hand appears on opposite sides of the two panels, something is wrong — and one
of the tests pins exactly that.

**`yolo_person_masks.jpg` / `mediapipe_input.jpg`** — the first frame with
YOLO's kept (green) and removed (red) silhouettes, and the masked frame
MediaPipe actually received. When a hand belongs to the other signer, or the
wrong person was chosen as the target, it is visible here and nowhere else.

Either can be requested on its own from the stage that produces it, which is
what you want when the landmarks are already extracted:

```bash
python3 step3_extract_landmarks.py ./out --save-debug-images --debug-limit 3
python3 step4_region_counts.py    ./out --save-debug-video  --debug-limit 3
python3 step4_region_counts.py    ./out --save-debug-video  --debug-limit all
```

Step 4 renders a video even for a clip whose region counts already exist —
asking for the video is usually the *second* run, not the first. A render that
fails (an unreadable or moved clip) is reported and skipped; it never fails the
run, since a broken diagnostic ending a long job is worse than no diagnostic.

### Figures for a poster

The defaults are sized for reading on screen. For an A0 poster read from two
metres, scale the type and the canvas together:

```bash
python3 step5_analyze.py ./out --figure-font-scale 1.6
```

Region shading is capped at a light tint (`MAX_SHADE` in `figures.py`) so the
percentages stay black-on-light and remain legible at a distance, rather than
turning white-on-dark in the busiest region.

**All six stored hand points are counted**, the thumb knuckle included. Thumb
position distinguishes real handshapes, and an abducted thumb frequently falls
in a different region from the other knuckles, so counting only the four finger
knuckles and the wrist would discard a genuine part of the signing space.

---

## Speed: CPU, GPU, and `--workers`

**Everything here runs on the CPU**, and step 3 is essentially the whole
runtime. A GPU barely helps:

| Part | Device | Helped by a GPU? |
| --- | --- | --- |
| MediaPipe Holistic — pose, face, hands | CPU only | No. The `mediapipe` Python package has no GPU path for the Holistic solution. |
| YOLO person segmentation | GPU if PyTorch sees CUDA | Yes — but it is `yolov8n-seg` on 1 frame in 5, a minority of the cost |
| ffmpeg clip cutting (step 2) | CPU | Marginal |
| Steps 1, 4, 5 | CPU, numpy/pandas | No — these take seconds |

What *does* help is cores. Clips are independent, so `--workers` extracts
several at once, one process each:

```bash
python3 run_pipeline.py ./parsed --video-root ~/corpus --workers auto
python3 step3_extract_landmarks.py ./out --workers 16
```

`auto` uses one process per core, less one. The default is 1, so nothing
changes unless you ask.

Two things to size it against:

- **RAM.** Each worker holds its own MediaPipe graph and YOLO model — budget
  1–2 GB each. Sixteen workers on a 16 GB machine will swap, and swapping is
  slower than running serially.
- **Threads.** Each worker is pinned to one thread (`OMP_NUM_THREADS=1`,
  `cv2.setNumThreads(1)`). Without that, N workers each grab every core and
  contend; this is the usual reason naive parallelism comes out *slower*.

Workers are spawned, not forked, because MediaPipe and torch hold state that
does not survive a fork. Startup therefore costs a second or two per worker —
irrelevant for a real run, wasteful for five clips.

The output is identical either way: both paths call the same `process_clip`, and
a test asserts a serial and a parallel run reach the same verdicts. Only the
ordering of the progress lines changes, since they print on completion.

### Other ways to cut the time

- `--no-person-mask` drops YOLO entirely — the biggest single saving, if the
  other signer is not actually leaking into your crops. Check a few
  `--save-debug` mask images before deciding.
- `--model-complexity 0` for a pass that only needs coarse positions.
- Step 3 skips clips that already have landmarks, so extract once and then
  iterate on steps 4 and 5, which take seconds.

---

## Running on a server

```bash
nohup python3 /path/to/run_pipeline.py /data/parsed \
      --video-root /data/videos -output_folder /data/out \
      --workers auto > /data/out/run.log 2>&1 &
tail -f /data/out/run.log
```

- Use absolute paths for `-output_folder`.
- One bad clip is logged and skipped; the run continues.
- Exit codes: `0` clean, `2` finished with some failures, `1` bad arguments.
- YOLO downloads its weights on first use, so the first run needs network access.
  Afterwards `--no-person-mask` is not required.
- Time a small batch before committing to the full corpus:

  ```bash
  python3 run_pipeline.py /data/parsed --video-root /data/videos \
      -output_folder /data/trial --limit 20 --workers auto
  ```

  Step 3 prints `Throughput: N clips/s`, which is what to multiply by.

---

## Tests

```bash
pip install pytest
pytest
```

189 tests covering normalisation, yaw estimation, mirroring, region
classification, the bootstrap, the summary tables, the diagnostic renders,
parallel extraction, the exclusion list, the spreadsheet reader, and an
end-to-end run of steps 1, 4 and 5 on synthetic landmarks. Steps 2 and 3 need ffmpeg, real video
and MediaPipe, so the tests build their outputs directly and exercise everything
downstream.

Worth knowing what the tests pin down:

- normalisation is scale-invariant and removes camera roll;
- a mirrored left-hander lands in the same region as the equivalent right-hander;
- a signer-level interval is genuinely wider than a clip-level one when one
  signer dominates;
- both summary tables report the dominant and non-dominant hand on separate
  rows, each averaged over its own clips;
- a parallel extraction run reaches the same verdicts as a serial one;
- an exclusion entry that matches nothing is reported rather than ignored;
- step 5 follows the signers file, so regrouping ages needs no re-extraction;
- region shares always sum to 100%.

---

## Repository layout

```
run_pipeline.py       all five stages
step1..step5_*.py     one stage each, runnable alone

config.py             landmark sets, geometry, tolerances — tweak here first
geometry.py           normalisation, yaw, region classification
landmarks.py          detection, person masking, the reduced landmark store
region_counts.py      landmarks → per-frame counts, handedness resolved here
signers.py            signer ID, handedness and age lookups
stats.py              aggregation and the signer-level bootstrap
figures.py            matplotlib figures with error bars
debug_render.py       optional example videos and YOLO mask images
io_utils.py           CSV and argument helpers

exclusions.py         the manual drop-list for bad crops

input_lists/          signers.csv, signers.txt, excluded_clips.txt
tools/                signers_from_xlsx.py — rebuilds them from the spreadsheet
tests/
```

## License

MIT

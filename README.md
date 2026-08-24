# JSL Dialogue Corpus: annotation normalisation and signing-space analysis

Three pipelines that turn the free-form gloss annotations of the Japanese Sign
Language Dialogue Corpus into structured, analysable data, and then measure how
much of the signing space signers actually use.

```
ELAN .eaf  ─▶  1. parsing  ─▶  parsed annotations  ─┬─▶  2a. corpus statistics
   + video                     + rebuilt .eaf       │
                                                    └─▶  2b. signing space
                                                            (clips ▸ landmarks
                                                             ▸ regions ▸ tables)
```

| Folder | What it does |
| --- | --- |
| [`parsing_pipeline/`](parsing_pipeline/) | Splits each gloss string into its layers (pointing, classifiers, fingerspelling, mouthing, gestures, non-manual markers) and writes them back as time-aligned ELAN tiers. |
| [`corpus-statistics-pipeline/`](corpus-statistics-pipeline/) | Counts what is in the corpus: recordings, signers, duration, vocabulary, marker frequency, parse outcome. |
| [`signing-space-analysis-pipeline/`](signing-space-analysis-pipeline/) | Cuts one clip per annotation, extracts pose and hand landmarks, normalises them on the signer's body, and classifies every hand point into one of 26 body-relative regions. |
| [`reports/`](reports/) | The technical report. |

Each folder has its own README with the full command reference. Start there.

## Quick start

```bash
# 1. parse the annotations
cd parsing_pipeline
python3 run_pipeline.py /path/to/corpus -output_folder output \
    --save-debug --exceptions-file input_lists/exceptions.txt

# 2a. what is in the corpus
cd ../corpus-statistics-pipeline
python3 run_pipeline.py \
    --elan-list ../parsing_pipeline/output/selected_elan_files.csv \
    --annotations ../parsing_pipeline/output/debug/parsed_annotations \
    --out corpus_statistics_output

# 2b. where the hands go  (needs the video; hours, not seconds)
cd ../signing-space-analysis-pipeline
python3 run_pipeline.py ../parsing_pipeline/output/debug/parsed_annotations \
    --video-root /path/to/video -output_folder output \
    --keywords cl fs lexical_item --signers-file input_lists/signers.csv
```


## What is not in this repository

Corpus media and annotations, and everything derived from them: video clips,
extracted landmarks, per-frame region counts, generated tables and figures. All
of it regenerates from the corpus and the code, and together it runs to several
gigabytes. The compiled reports are committed as PDFs.

## Requirements

Python 3.9+. Each pipeline lists its own dependencies in `requirements.txt`;
the signing-space pipeline additionally needs `ffmpeg`, and its landmark stage
uses MediaPipe and a YOLOv8n-seg checkpoint.

## Citing the corpus

M. Bono et al., "Utterance unit annotation for the Japanese Sign Language
Dialogue Corpus," John Benjamins, 2023, pp. 353-382.
<https://research.nii.ac.jp/jsl-corpus/public/en/>

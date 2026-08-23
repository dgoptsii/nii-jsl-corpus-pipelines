# input_lists

The two plain-text files you edit by hand. Both are optional, and both are
meant to be committed to git so that decisions accumulate across interns.

| File | Passed with | Purpose |
| --- | --- | --- |
| `files_of_interest.txt` | `-file_list` | Restrict a run to specific ELAN files |
| `exceptions.txt` | `--exceptions_file` | Annotations that are allowed to have Latin characters after parsing and must not be marked ambiguous |

```bash
python3 run_pipeline.py ./corpus -output_folder ./output \
    -file_list input_lists/files_of_interest.txt \
    --exceptions_file input_lists/exceptions.txt
```

Blank lines and lines starting with `#` are ignored in both files.

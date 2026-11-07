The `oldString-mask-v1` experiment replaces the previous schema experiment.
Messages, tools and templates are not rewritten. `003-oldstring-mask.patch`
integrates `matcher.h` into native common sampling, before the chain and on
its grammar retry. The sampler accepts generated tokens only, clones mask state
on init/clone/copy, and resets it with the sampler. Backend sampling is rejected
and speculative decoding is disabled for both experiment conditions.

`native.h` uses the native PEG tool-call AST. It supports raw JSON and Qwen's
XML arguments; other constructed formats fail explicitly at configuration.
Incomplete UTF-8, JSON escapes and partial XML closing markers are preserved.
The native grammar's own lazy triggers and AST-proven open strings provide
shortcuts for candidate extensions that cannot cross argument boundaries.

Each file has a separate suffix array, built lazily from a bounded snapshot.
The manifest comes from Git's ignore implementation against an empty private
index, including in directories without a repository. File contents are limited
to 16 MiB each and the directory fallback to 128 MiB. Fallbacks are logged and
reported as such; a late valid path can reactivate masking after a union-budget
fallback if the already emitted prefix matches it. Snapshot caches belong to a
single request. No file contents enter mask telemetry.

Build through `llama/server` as usual, then run the focused native targets:

```sh
cmake --build build/llama-server-cpu --target ollama-edit-mask-test ollama-edit-mask-native-test -j8
build/llama-server-cpu/ollama-edit-mask-test /tmp/edit-fixtures path/to/sled/src/lib.rs
build/llama-server-cpu/ollama-edit-mask-native-test path/to/qwen-template.jinja /tmp/native-edit-fixtures path/to/model.gguf path/to/qwen-json-template.jinja
go test ./llm ./server
```

The matcher test has a randomized substring oracle. Native tests exercise the
actual Qwen AST, token-boundary splits, prose, another tool, JSON and XML,
Unicode, late paths, previews, and common sampler clone/copy/reset. The model
argument loads vocabulary only for the sampler state tests. Set
`OLLAMA_EDIT_MASK_SAMPLING_TEST=1` to load weights and test both grammar-first
sampling and a forced grammar retry; the retry test checks that the mask is
reapplied before resampling.

Run a paired OpenCode experiment against a clean sled checkout with the built
Ollama payload. The harness sets the experiment switch for both conditions,
reuses the same absolute working-directory path, and archives each worktree:

```sh
rustc --edition=2021 -O llama/edit-mask/experiment.rs -o /tmp/opencode-oldstring-mask-experiment
/tmp/opencode-oldstring-mask-experiment --repo path/to/sled --ollama path/to/ollama --models path/to/models --model MODEL --task-file llama/edit-mask/sled-doc-comment-task.txt --n 1 --context-length 131072 --timeout-secs 1800 --keep-worktrees --out path/to/results
python3 llama/edit-mask/score_edit.py path/to/results --repo path/to/sled
```

`opencode` must be on PATH, or set `--opencode` to its executable. The CSV records
self-report and process completion; the separate scorer checks the exact file
change and successful confirming reads. `prompt_equality.py` saves native render
and token equality evidence before and after a read against a running native
runner; use `--help` for its URL, working-directory and output arguments.

The OpenAI-compatible provider may ignore the model option `think=false`.
Record the effective native thinking setting instead of assuming it was disabled.
The recorded functionality check is summarized in `VALIDATION.md`.

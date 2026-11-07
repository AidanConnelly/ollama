The native mask was built and checked on Linux with an AVX2 CPU payload,
llama.cpp b11963 (39febd2adb947b70da35e95f9351b3bf58ad170a), and Qwen3.5-9B Q6_K.

The native payload and Go server built successfully. `go test ./llm ./server`
passed. The focused matcher and native parser/sampler tests passed, including
model-backed grammar-first sampling and a forced grammar retry. The retry check
verified that the mask was reapplied before resampling.

The fresh N=1 pair used seed 20261108, context 131072, and a 1800-second timeout
per condition. Speculative and backend sampling were disabled in both runs.
Payload hashes were unchanged throughout the pair. The effective thinking
setting was the native template default; the requested provider option
`think=false` was not forwarded through the OpenAI-compatible route.

| Condition | Exact edit | Confirmed read/edit/read | Edit attempts | Runtime | Masked candidate logits | Fallbacks |
| --- | --- | --- | --- | --- | --- | --- |
| Control | Pass | Pass | 1 | 487.935 s | 0 | 0 |
| Constrained | Pass | Pass | 1 | 1462.516 s | 3,964,587 | 0 |

Independent scoring verified that only `src/lib.rs` changed, its contents exactly
matched the requested one-comment insertion, and the confirming read included
the comment. Both processes exited cleanly and reported success separately.

All four actual request stages had identical tool schemas, rendered prompts and
prompt token IDs: 7394, 8906, 9048 and 10551. Later raw messages differed only in
opaque tool-call IDs that the native template does not render. Counterfactual
checks with identical messages also passed before and after a read.

Local logs and exported telemetry agreed on the rejection total. The selected
snapshot contained 13,117 bytes, with a 52,468-byte suffix index. Mask telemetry
contained only status, reasons and sizes/counts. The telemetry SDK dropped some
per-token/progress events at its event limit; the activation event and final
rejection total were retained.

The constrained run took about three times as long on this CPU. This pair checks
functionality and does not establish a success-rate improvement. Directory-wide
fallback and exceptional-file cases were exercised by focused tests rather than
counted as evidence of successful masking in this pair.

Full reports, prompt/token equality evidence, independent scores, transcripts,
patches, telemetry, build logs and payload hashes remain in the shared workspace
under `qwen35-oldstring-mask-v1-final-20261107-n1/`. Generated run artifacts and
archived worktrees are intentionally outside this Git repository. Earlier schema
experiments and aborted setup attempts are preserved separately.

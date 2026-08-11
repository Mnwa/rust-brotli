# Changelog

## 9.0.0 — first release as `simd-brotli`

Forked from [`brotli`](https://crates.io/crates/brotli) 8.0.4
(`dropbox/rust-brotli` at `9651aa3`). The encoder's hot paths are now vectorized with
[`fearless_simd`](https://crates.io/crates/fearless_simd), which runtime-dispatches to the
widest instruction set the running CPU supports, on stable Rust and without `unsafe`.

**Compressed output is unchanged.** Every optimization below is bit-identical to the scalar
code it replaces: streams produced by this crate match upstream `brotli` byte for byte at every
quality level, and remain bitwise identical to the C brotli engine at levels 0–9.

### Verification and measurements

Encoder output was diffed against the 8.0.4 fork base over 60 (corpus, quality) pairs — four
corpora (`alice29.txt`, `asyoulik.txt`, `random_org_10k.bin`, `monkey`) × qualities 0–11
including 9.5, 9.5x and 9.5y, all at `-w22` — and every pair is byte-identical. The 100-test
suite passes unchanged.

Wall clock on a 3.1 MB varied corpus (English text, a word list and three shared libraries),
Apple M5 Pro, rustc 1.97.1, release profile with LTO, best of three runs:

| quality | 8.0.4 base | this fork | change |
| ------- | ---------- | --------- | ------ |
| q9      | 0.18 s     | 0.16 s    | −11%   |
| q10     | 0.82 s     | 0.70 s    | −15%   |
| q11     | 1.93 s     | 1.77 s    | −8%    |

These are NEON numbers from one machine and one corpus; the win depends on the instruction set
the CPU offers and on how much of the input reaches the slow paths. Treat them as an indication
of scale, not a guarantee.

### Packaging

- Renamed the package to `simd-brotli`; the library target is `simd_brotli`, so this crate and
  upstream `brotli` can sit in the same dependency graph. Migration is a rename of the import
  (`use brotli::…` → `use simd_brotli::…`); the API surface is otherwise untouched.
- The `brotli` and `catbrotli` binaries keep their names.
- Added Mikhail Panfilov to the author list, alongside the upstream authors. Licensing is
  unchanged: BSD-3-Clause AND MIT, with both upstream license files shipped.

### SIMD backend: `packed_simd`/`portable_simd` → `fearless_simd`

- **Removed the `simd` cargo feature and the nightly requirement it carried.** Vectorization is
  now unconditional and builds on stable: the `#![feature(portable_simd)]` gate is gone, as is
  the `core::simd` dependency behind it.
- **Removed `src/enc/compat.rs`** (392 lines of hand-written scalar fallbacks for
  `Compat16x16` / `CompatF8` / `Compat32x8`). There is no longer a scalar shim to keep in sync
  with a vector path — one implementation serves both.
- `s16`, `v8` and `s8` are now `Mem16x16`, `Mem256f` and `Mem256i` from
  `enc::vectorization`. These are plain `Copy + Default` arrays so they can live in the
  encoder's allocator-backed slices; arithmetic happens on `fearless_simd` registers loaded via
  `to_simd` / stored via `from_simd`.
- **Runtime dispatch.** `enc::vectorization::detect_level()` picks the instruction set (AVX2,
  NEON, …) on `std` and wasm builds, caching the answer in a `OnceLock` — the probe is a dozen
  feature tests, and clustering calls into it once per histogram. `no_std` builds fall back to
  the level the crate was compiled for, so the `no-stdlib` configuration still works.
- Because detection is not free, the `dispatch!` regions are hoisted to the outermost loop that
  can own them: a whole Zopfli block, a whole hasher store range, a whole match-finder walk.
  Inner functions take an already-detected `S: Simd` instead of probing per call.
- Bumped MSRV to **1.89.0** and the edition to **2024**, both required by `fearless_simd`.
- `no_std` support is preserved through `fearless_simd`'s `libm` feature; the crate's `std`
  feature forwards to `fearless_simd/std`.

### Newly vectorized encoder paths

Upstream vectorized two cost loops. This fork extends wide-lane work to the stages that
profiling showed dominate compression time:

- **H10 binary-tree match finder** (`backward_references/hash_to_binary_tree.rs`) —
  `StoreAndFindMatchesH10`, `Store`, `StoreRange` and `BulkStoreRange` gained SIMD variants.
  The tree walk measures a match length per node, up to 64 per position, so detection is
  hoisted to the range walk rather than paid per comparison.
- **Zopfli shortest path** (`backward_references/hq.rs`) — `UpdateNodes`,
  `StitchToPreviousBlockH10`, `FindAllMatchesH10`, the per-position sweep of
  `BrotliZopfliComputeShortestPath` and `ZopfliIterate` all run on a per-block detected
  instruction set, so one probe covers the match finder, the node update and the hasher store
  for the whole block.
- **Static-dictionary match length** (`static_dict.rs`) — `FindMatchLengthWithLimit` now
  resolves short matches (the overwhelming majority) with a narrow scalar probe and never looks
  at the CPU feature set; only once a match passes the wide-compare threshold does it detect an
  instruction set and switch to 32-byte compares. `FindMatchLengthWithLimitSimd` is available
  for callers that already hold one.
- **Block splitter** (`block_splitter.rs`) — `FindBlocks`' per-histogram cost scan keeps a
  running per-lane winner and reduces across lanes once per row instead of branching per
  histogram. Ties resolve to the lowest histogram id, exactly as the scalar scan did.
- **Population cost** (`bit_cost.rs`) — the symbol-cost walk tests eight buckets per compare
  and only falls back to per-bucket work for populated ones. Histograms are mostly empty (a
  distance alphabet has 544 buckets and a metablock rarely touches a tenth of them), so the
  empty runs are what the wide compare is for. Empty runs still land in `depth_histo` exactly
  as a bucket-at-a-time scan would leave them.
- **Prior evaluation** (`prior_eval.rs`) — the per-literal cost update takes a level detected
  once when the evaluator is built.

### Hot-path optimizations

- **Histogram clustering no longer materializes histogram sums.** `BrotliPopulationCostOfSum`
  costs the histogram that adding `b` into `a` *would* produce, walking the two bucket arrays
  lane-wise instead of cloning one and running a separate add pass over it. The float
  accumulation sees the same values in the same order, so the result is bit-identical. Applied
  in `BrotliHistogramBitCostDistance` and in the clustering pair queue
  (`BrotliCompareAndPushToQueue`), which is where nearly all population-cost calls come from —
  and on large varied input, histogram clustering is roughly half of a quality-10 compression.
- The population-cost walk was refactored behind a `Buckets` trait (`Own` / `Sum`) so the
  single-histogram and summed-histogram forms share one implementation and cannot drift apart.

### Profiling support

- Added an optional [`hotpath`](https://crates.io/crates/hotpath) instrumentation layer over
  the encoder pipeline, behind three features: `hotpath` (wall clock), `hotpath-cpu` (CPU
  time) and `hotpath-alloc` (allocation counts and bytes). Every call site is a `cfg_attr`, so
  a default build neither links `hotpath` nor pays any runtime cost.
- Instrumented stages: `encode_data`, `copy_input_to_ring_buffer`, `WriteMetaBlockInternal`,
  `ChooseContextMap`, `DecideOverLiteralContextModeling`, `compress_stream_fast`, the three
  `store_meta_block*` writers, `LogMetaBlock`, `BrotliCreateBackwardReferences` and the Zopfli
  entry points, `BrotliBuildMetaBlock` / `Greedy` / `BrotliOptimizeHistograms`,
  `BrotliSplitBlock` and its internals, the `cluster.rs` histogram-clustering functions,
  `BrotliEstimateBitCostsForLiterals`, and the two `compress_fragment` fast paths.
- Instrumentation sits at metablock granularity, not per byte, so overhead stays under
  measurement noise (q9/q10/q11 on a 3.9 MB corpus timed within 1% of an uninstrumented build).
- See [Profiling the encoder](README.md#profiling-the-encoder) for usage.

### Modernization

- Migrated the crate to **edition 2024**, including the FFI layer (`src/ffi/`), the threading
  and worker-pool modules, and the binaries. `unsafe` blocks inside `unsafe fn` are now
  explicit, as edition 2024 requires.
- Removed the now-dead `extern crate` / SIMD `use` statements the old feature gate needed.

### Upstream

For changes inherited from `brotli` 8.0.4 and earlier, see the version history in
[README.md](README.md#whats-new-in-804).

# simd-brotli 10.0.0 compared with Google Brotli 1.2.0

This document compares the current Rust implementation with the official
[Google Brotli 1.2.0 release](https://github.com/google/brotli/releases/tag/v1.2.0). Both produce
standard [RFC 7932](https://www.rfc-editor.org/rfc/rfc7932) Brotli streams, but the encoders are no
longer mechanically identical implementations.

## Executive summary

- **Format compatibility is complete in this test.** Google Brotli decoded every Rust stream from
  q0 through q11, and the Rust decoder decoded every Google stream. All 24 decoded files matched
  the original input.
- **Byte identity is quality-dependent.** The two encoders emitted identical bytes at q5, q6, q7,
  and q8 on the benchmark corpus. Other qualities produced valid but different streams.
- **Google C is faster from q0 through q9 on this small corpus, except for a statistically weak q4
  result.** The largest measured Rust deficits are q8 (40% more wall time) and q9 (22% more).
- **Rust is substantially faster at q10 and q11.** It used 42% less wall time at q10 and 45% less
  at q11, corresponding to approximately 1.74x and 1.81x the C encoder's throughput.
- **Compression density is close except at q1.** Rust q11 was 36 bytes smaller than C, Rust q10 was
  30 bytes larger, and Rust q1 was 9,861 bytes (6.32%) larger on this input.
- **The Rust implementation's main engineering differences are safe Rust encoder internals,
  `no_std` support, allocator-generic APIs, runtime-dispatched SIMD, built-in multithreaded
  compression, optional hot-path profiling, and concatenation extensions.** Google C retains the
  smaller and more established native ABI, build ecosystem, and better q5-q9 latency here.

The short answer is: use the Rust implementation when Rust integration, memory safety, `no_std`,
custom allocation, or q10/q11 throughput is the priority. Use Google C when a mature system C ABI,
small dynamically linked deployment, or the best q5-q9 latency on this machine matters most.

## Benchmark scope

| Item                | Value                                          |
|---------------------|------------------------------------------------|
| Date                | 2026-08-18                                     |
| Rust implementation | `simd-brotli` 10.0.0, release profile with LTO |
| C implementation    | Homebrew Google Brotli 1.2.0                   |
| Hardware            | Apple M5 Pro, arm64, 18 cores, 24 GiB RAM      |
| Operating system    | macOS 26.6.1                                   |
| Rust toolchain      | rustc/cargo 1.97.1                             |
| Input               | `testdata/random_then_unicode`, 272,666 bytes  |
| Encoder settings    | quality 0-11, `lgwin=22`, single-threaded CLI  |
| Timing tool         | Hyperfine, 3 warmups and 20 measured runs      |
| Reported statistic  | Mean end-to-end CLI wall time                  |

The benchmark includes process startup, input/output file handling, allocation, compression, and
shutdown. It is representative of small-file CLI use, not a pure in-process encoder-kernel
benchmark. Results below 5 ms carry significant startup and scheduler noise; q0-q4 should therefore
be treated as directional only.

No size hint, custom dictionary, multithreading, `float64`, `hotpath`, or nonstandard stream option
was enabled in the main table.

## Encoder results across all quality levels

In the `Rust time vs C` column, a positive value means Rust took longer; a negative value means
Rust was faster. `Size delta` is Rust bytes minus C bytes.

| Quality | Rust time |   C time | Rust time vs C | Rust bytes | C bytes | Size delta | Byte-identical |
|--------:|----------:|---------:|---------------:|-----------:|--------:|-----------:|:--------------:|
|       0 |    1.7 ms |   1.6 ms |          +5.4% |    169,645 | 169,963 |       -318 |       No       |
|       1 |    2.3 ms |   2.2 ms |          +5.9% |    165,999 | 156,138 |     +9,861 |       No       |
|       2 |    2.9 ms |   2.8 ms |          +3.5% |    148,707 | 148,643 |        +64 |       No       |
|       3 |    3.2 ms |   2.8 ms |         +13.0% |    147,754 | 147,664 |        +90 |       No       |
|       4 |    4.0 ms |   4.1 ms |          -3.0% |    144,144 | 144,125 |        +19 |       No       |
|       5 |    5.9 ms |   5.4 ms |          +7.7% |    138,327 | 138,327 |          0 |    **Yes**     |
|       6 |    6.4 ms |   5.7 ms |         +12.7% |    137,238 | 137,238 |          0 |    **Yes**     |
|       7 |    8.8 ms |   7.9 ms |         +12.1% |    132,633 | 132,633 |          0 |    **Yes**     |
|       8 |   11.7 ms |   8.4 ms |         +40.2% |    132,305 | 132,305 |          0 |    **Yes**     |
|       9 |   12.7 ms |  10.5 ms |         +21.7% |    132,102 | 132,107 |         -5 |       No       |
|      10 |   57.3 ms |  99.5 ms |     **-42.4%** |    129,867 | 129,837 |        +30 |       No       |
|      11 |  127.4 ms | 231.1 ms |     **-44.9%** |    124,674 | 124,710 |        -36 |       No       |

The q10/q11 timings are not algorithm-identical comparisons because their output streams differ.
They compare the same quality setting, input, window, and compatible format, but each encoder made
slightly different parsing and coding decisions.

### What the numbers mean

- q5-q8 byte identity proves that both encoders reached exactly the same command, block, and entropy
  coding decisions on this corpus. Their timing differences are implementation overhead only.
- At q6, Google C has approximately 11% more throughput. The Rust tagged matcher is already SIMD
  accelerated; remaining time is primarily candidate validation and surrounding match-finder work.
- q8 and q9 are the clearest remaining default-mode performance gaps. They deserve profiling before
  more q6 work because their absolute and relative deficits are larger.
- q10 and q11 show the value of the Rust fork's SIMD H10 filtering, wide common-prefix scans,
  compact Zopfli nodes, and vectorized node updates. The gain is large enough to survive CLI and I/O
  overhead.
- The q1 size difference is too large to dismiss as floating-point tie-breaking. It indicates real
  fast-path or streaming/size-hint decision drift and should be investigated separately if q1
  density matters.

## Output compatibility versus output identity

Brotli defines the decoded representation, not one canonical compressed byte sequence. Two
encoders may choose different matches, block boundaries, context maps, or Huffman trees and still
produce equally valid streams.

For this corpus:

- Rust and C streams were byte-identical at q5-q8.
- They differed at q0-q4 and q9-q11.
- Google Brotli decoded all 12 Rust streams to the original input.
- simd-brotli decoded all 12 Google streams to the original input.

Consequently, **interoperability can be relied upon; byte identity cannot be treated as a general
API guarantee**. Any regression test requiring byte identity should pin the implementation,
version, feature set, target, and encoder parameters.

The q11 Rust stream is 36 bytes smaller than the C stream on this input, but that 0.029% difference
is too small to establish a general compression-ratio advantage. A multi-corpus benchmark is
required for such a claim.

## `float64` mode

The optional Rust `float64` feature now keeps eight cost lanes vectorized with `f64x8`, uses
`u64x8` winner indices, and stores prior/block-splitting costs as real `f64` values. It does not
narrow those costs through `f32` and does not fall back to a scalar eight-element loop.

The table below uses medians because the q11 `float64` sample had unusually high run-to-run
variance.

| Quality |  Rust f32 |  Rust f64 |  Google C |                   f64 effect | Output observation                                                         |
|--------:|----------:|----------:|----------:|-----------------------------:|----------------------------------------------------------------------------|
|       6 |   6.74 ms |   6.55 ms |   6.33 ms |                 Within noise | All three streams are the same 137,238 bytes                               |
|      11 | 125.83 ms | 144.53 ms | 232.94 ms | f64 is 14.9% slower than f32 | f32 and f64 are both 124,674 bytes but differ bytewise; C is 124,710 bytes |

At q11, `f64x8` represents 512 bits of logical lane data. On NEON this requires more registers and
traffic than the default 256 bits of `f32x8` data. The higher precision still beats Google C by
about 38% wall time in this run, but it is not free. `float64` should be selected for numerical
behavior, not expected speed.

The default `ZopfliNode` is 16 bytes, matching the compact C union-style layout. Enabling `float64`
uses a 64-bit phase payload and makes the Rust node larger, increasing q11 node-array traffic.

## Decoder smoke comparison

Both CLIs decoded the same Rust-generated streams. With 5 warmups and 50 runs:

| Stream             | Rust decoder | Google C decoder |
|--------------------|-------------:|-----------------:|
| q6, 137,238 bytes  |       1.8 ms |           1.8 ms |
| q11, 124,674 bytes |       1.9 ms |           1.9 ms |

These sub-2-ms results are dominated by process startup and file I/O. They establish that neither
CLI has an obvious end-to-end regression for this input, but they do not establish equal decoder
core throughput. A persistent in-process harness over a much larger corpus is needed for that.

## Observed peak memory

Hyperfine reported the following approximate peak resident memory. This is a process-level
measurement, not allocator-only working memory.

| Quality |      Rust |  Google C |
|--------:|----------:|----------:|
|       0 |  2.22 MiB |  2.52 MiB |
|       1 |  2.52 MiB |  3.03 MiB |
|     2-3 |  3.09 MiB |  3.09 MiB |
|       4 |  5.31 MiB |  5.31 MiB |
|       5 |  6.14 MiB |  6.14 MiB |
|       6 |  7.45 MiB |  7.45 MiB |
|       7 | 13.05 MiB | 13.05 MiB |
|       8 | 21.05 MiB | 21.05 MiB |
|    9-11 | 38.69 MiB | 38.69 MiB |

The near-identical high-quality figures are consistent with the shared algorithm lineage and
equivalent hash/tree capacities. More detailed allocator profiling is available in Rust through
the `hotpath-alloc` feature.

## Implementation differences

| Area             | Google Brotli 1.2.0 C                                                               | simd-brotli 10.0.0 Rust                                                                                                                 | Practical consequence                                                                        |
|------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Format           | RFC 7932 encoder/decoder                                                            | RFC 7932 encoder/decoder                                                                                                                | Streams interoperate in both directions                                                      |
| Lineage          | Reference implementation                                                            | Safe-Rust port that has accumulated independent optimizations                                                                           | Same core design, but no permanent byte-identity guarantee                                   |
| Memory safety    | Pointer arithmetic, macros, manual allocation                                       | Safe encoder core with bounds-checked slices; `unsafe` is concentrated in FFI, platform CLI code, and tests                             | Stronger internal safety boundary in normal Rust use                                         |
| Standard library | Native C runtime/platform layer                                                     | Crate is `#![no_std]`; `std` is an optional default feature                                                                             | Rust can run in kernels, embedded systems, and custom runtimes                               |
| Allocation       | C allocator callbacks                                                               | Generic typed allocators plus standard heap adapters                                                                                    | Rust can statically enforce more allocator/storage relationships                             |
| SIMD selection   | Compiler/platform macros and C implementation choices                               | Stable runtime dispatch through `fearless_simd`; compile-time selection for `no_std`                                                    | One Rust binary can select NEON/AVX2-capable paths without unsafe intrinsics in encoder code |
| SIMD scope       | Primarily reference scalar/word-at-a-time encoder paths plus platform optimizations | H10 and tagged matchers, common-prefix scans, static dictionary, block splitting, histogram costs, prior evaluation, and Zopfli updates | Main reason Rust wins at q10/q11                                                             |
| q5/q6 matching   | Reference quick hash matcher with cache-prefetch-oriented layout                    | H58/H68 tagged tables reject 16/32 candidates together and overlap tag-table access with recent-distance checks                         | Same q5/q6 streams here; C still has lower latency                                           |
| q7/q8 matching   | Reference forgetful-chain hashers                                                   | Rust implementations of H40/H41/H42 with allocator-backed storage                                                                       | Same streams here; q8 Rust overhead remains significant                                      |
| q10/q11 matching | H10 hash-to-binary-tree matcher and scalar recent-candidate screening               | H10 plus SIMD screening of up to 32 recent candidates and wide match-length checks                                                      | Fewer expensive candidate visits on varied data                                              |
| Zopfli work      | Reference node updates and C union payload                                          | SIMD node updates, compact phase payload, reduced repeated indexing, reused SIMD dispatch                                               | Lower q10/q11 node-processing cost                                                           |
| Floating costs   | Reference encoder precision/layout                                                  | Default f32 plus optional fully vectorized `float64` mode                                                                               | Rust offers an explicit precision/performance tradeoff                                       |
| Multithreading   | Core reference API is principally single-stream/single-thread                       | `CompressMulti`, reusable worker pools, and scoped borrowed-input compression                                                           | Rust exposes built-in parallel chunk compression                                             |
| Concatenation    | v1.2.0 CLI can decode concatenated streams                                          | Appendable/catable/bare modes plus `catbrotli` boundary processing                                                                      | Rust provides additional stream-construction workflows                                       |
| Profiling        | External profilers                                                                  | `hotpath`, `hotpath-cpu`, and `hotpath-alloc` features                                                                                  | Stage-level time and allocation attribution is built in                                      |
| API surface      | Stable official C ABI and bindings across ecosystems                                | Native Rust readers/writers, custom-I/O APIs, allocator APIs, plus optional compatible C FFI                                            | Choice depends on integration language and deployment model                                  |
| Extra qualities  | Integer q0-q11                                                                      | q0-q11 plus q9.5, q9.5x, and q9.5y experimental modes                                                                                   | Rust exposes extra encoder trade-off points not present in the C CLI                         |

Relevant source mappings:

- Google C encoder sources: [`c/enc` at v1.2.0](https://github.com/google/brotli/tree/v1.2.0/c/enc)
- Rust tagged q5/q6 matcher: [`src/enc/backward_references/tagged.rs`](src/enc/backward_references/tagged.rs)
- Rust H10 structure: [`src/enc/backward_references/hash_to_binary_tree.rs`](src/enc/backward_references/hash_to_binary_tree.rs)
- Rust q10/q11 Zopfli path: [`src/enc/backward_references/hq.rs`](src/enc/backward_references/hq.rs)
- Rust block splitting: [`src/enc/block_splitter.rs`](src/enc/block_splitter.rs)
- Rust SIMD storage and dispatch: [`src/enc/vectorization.rs`](src/enc/vectorization.rs)
- Rust built-in profiling points: [`README.md`](README.md#profiling-the-encoder)

## Binary footprint

The release Rust CLI was 1.8 MiB before stripping and 1.6 MiB after stripping. The Homebrew C CLI
was 32 KiB, but it dynamically loads approximately 887 KiB of Brotli dylibs (`common`, `decoder`,
and `encoder`). These figures are not directly comparable:

- the Rust executable packages substantially more implementation and Rust support code into one
  binary;
- the C executable delegates almost all work to shared libraries;
- filesystem size does not measure loaded pages, shared-page amortization, or an application that
  statically links either library.

For a deployment decision, compare stripped application artifacts built with the same link mode,
target, panic strategy, and feature set.

## Where each implementation currently wins

### Rust advantages

- q10/q11 throughput on this Apple Silicon corpus;
- safe native Rust API and a safe encoder core;
- `no_std` and typed custom-allocation support;
- runtime-dispatched stable SIMD without encoder-side unsafe intrinsics;
- built-in multithreaded compression and scoped thread-pool integration;
- integrated time, CPU, and allocation profiling;
- appendable/catable construction and q9.5 variants.

### Google C advantages

- lower q5-q9 latency in this test, especially q8 and q9;
- much better q1 compression density on this corpus;
- official reference status, mature ABI, and broader existing language/toolchain integration;
- smaller dynamically linked CLI deployment;
- less dependence on Rust monomorphization and compile times.

## Highest-value remaining Rust work

1. **Profile and optimize q8.** It is 40% slower with byte-identical output, making it the cleanest
   apples-to-apples optimization target.
2. **Profile q9 separately.** It is 22% slower and differs by only five output bytes. Determine
   whether the cost lies in the advanced hasher, block splitting, or histogram clustering.
3. **Investigate q1 density.** A 6.32% output penalty is more important than its sub-millisecond
   timing difference. Compare fast-fragment commands and size-hint behavior with C before tuning
   instruction-level performance.
4. **Close the q6/q7 candidate-validation gap.** Tag generation is already cheap; reduce dependent
   position loads and match-length work after a tag hit.
5. **Reduce `float64` q11 node traffic.** Eight f64 lanes and 24-byte nodes increase bandwidth and
   register pressure. Separate cold phase data or use phase-specific arrays if profiling confirms
   memory pressure.
6. **Add a persistent cross-language benchmark harness.** Call both encoder libraries in-process,
   alternate their run order, record compression size and peak allocation, and use several text,
   binary, random, and repetitive corpora. This removes CLI startup noise and prevents conclusions
   from being tied to one 272 KiB sample.
7. **Keep cross-decoding mandatory.** Every performance change should be checked with both decoders;
   byte identity should only be required for paths intentionally preserving C decisions.

## Reproduction

Build the Rust CLI:

```bash
cargo build --release --bin brotli
```

Confirm the reference C version:

```bash
/opt/homebrew/bin/brotli --version
```

Run the all-quality timing comparison:

```bash
hyperfine --warmup 3 --runs 20 \
  --parameter-list q 0,1,2,3,4,5,6,7,8,9,10,11 \
  'target/release/brotli -c -q{q} -w22 testdata/random_then_unicode /tmp/rust.br' \
  '/opt/homebrew/bin/brotli -f -q {q} -w 22 -o /tmp/google-c.br testdata/random_then_unicode'
```

Generate comparable q6 streams:

```bash
target/release/brotli -c -q6 -w22 testdata/random_then_unicode /tmp/rust-q6.br
/opt/homebrew/bin/brotli -f -q 6 -w 22 -o /tmp/c-q6.br testdata/random_then_unicode
cmp /tmp/rust-q6.br /tmp/c-q6.br
```

Build and test the high-precision Rust mode:

```bash
cargo test --features float64
cargo build --release --features float64 --bin brotli
```

## Conclusion

simd-brotli is format-compatible with Google Brotli but should now be viewed as an independently
optimized Rust encoder, not a byte-for-byte transcription. The current implementation is strongest
at q10/q11, where its SIMD and compact Zopfli work produce a major throughput advantage. Google C
remains faster in the q5-q9 range and exposes a serious q1 density gap that Rust should address.

The most valuable next step is q8 profiling: it has identical output, the same observed peak
memory, and a 40% timing deficit, so improvements there can be measured without compression-ratio
or algorithm-selection ambiguity.

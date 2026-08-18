# simd-brotli

[![crates.io](https://img.shields.io/crates/v/simd-brotli.svg)](https://crates.io/crates/simd-brotli)
[![docs.rs](https://img.shields.io/docsrs/simd-brotli)](https://docs.rs/simd-brotli/)

A Brotli compressor and decompressor whose encoder hot paths use runtime-dispatched SIMD. It is a
fork of [`brotli`](https://crates.io/crates/brotli), produces byte-identical compressed output, and
keeps the upstream API while using the crate name `simd_brotli` so both packages can coexist in one
dependency graph.

- Stable Rust with no `unsafe` in the Rust implementation
- Runtime dispatch to AVX2, NEON, and other supported instruction sets
- `no_std` support with pluggable allocators
- Rust stream, low-level, command-line, and C-compatible interfaces

## Installation

Add the crate from the command line:

```bash
cargo add simd-brotli
```

Or add it to `Cargo.toml`:

```toml
[dependencies]
simd-brotli = "10"
```

The default feature set enables the standard-library stream APIs. For a `no_std` project, disable
default features:

```toml
[dependencies]
simd-brotli = { version = "10", default-features = false }
```

The minimum supported Rust version is 1.89.0.

## Quick start

This example compresses into a `Vec<u8>` and then decompresses it again:

```rust
use simd_brotli::{CompressorWriter, Decompressor};
use std::io::{Read, Write};

fn main() -> std::io::Result<()> {
    let input = b"Brotli works especially well on repeated text. \
                  Brotli works especially well on repeated text.";

    let mut compressor = CompressorWriter::new(Vec::new(), 4096, 9, 22);
    compressor.write_all(input)?;
    let compressed = compressor.into_inner();

    let mut decompressor = Decompressor::new(compressed.as_slice(), 4096);
    let mut decoded = Vec::new();
    decompressor.read_to_end(&mut decoded)?;

    assert_eq!(decoded, input);
    Ok(())
}
```

The last two arguments to `CompressorWriter::new` are the quality and window size. Quality may be
0–11; an `lgwin` value between 20 and 22 is a good general-purpose choice.

## API examples

Choose an adapter based on which side of your pipeline should implement `Read` or `Write`:

| Task | Read adapter | Write adapter | Copy helper |
| --- | --- | --- | --- |
| Compress | `CompressorReader` | `CompressorWriter` | `BrotliCompress` |
| Decompress | `Decompressor` | `DecompressorWriter` | `BrotliDecompress` |

### Compress from a reader

`CompressorReader` turns any `Read` input into a stream of compressed bytes:

```rust
use simd_brotli::CompressorReader;
use std::io::{self, Cursor};

fn main() -> io::Result<()> {
    let source = Cursor::new(b"data to compress");
    let mut compressed = CompressorReader::new(source, 4096, 9, 22);
    io::copy(&mut compressed, &mut io::stdout())?;
    Ok(())
}
```

### Decompress into a writer

`DecompressorWriter` accepts compressed bytes and forwards decoded bytes to its inner writer:

```rust
use simd_brotli::DecompressorWriter;
use std::io::{self, Write};

fn decode(compressed: &[u8]) -> io::Result<Vec<u8>> {
    let mut decoder = DecompressorWriter::new(Vec::new(), 4096);
    decoder.write_all(compressed)?;
    decoder
        .into_inner()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "incomplete Brotli stream"))
}
```

### Copy a complete stream

The copy helpers are convenient when the input and output already implement the standard I/O
traits:

```rust
use simd_brotli::enc::BrotliEncoderParams;
use simd_brotli::{BrotliCompress, BrotliDecompress};
use std::io::{self, Cursor};

fn main() -> io::Result<()> {
    let original = b"compress a complete stream";
    let mut params = BrotliEncoderParams::default();
    params.quality = 9;
    params.lgwin = 22;

    let mut compressed = Vec::new();
    BrotliCompress(&mut Cursor::new(original), &mut compressed, &params)?;

    let mut decoded = Vec::new();
    BrotliDecompress(&mut compressed.as_slice(), &mut decoded)?;
    assert_eq!(decoded, original);
    Ok(())
}
```

Use `CompressorReader::with_params` or `CompressorWriter::with_params` when you need the same
parameter control with a stream adapter.

### Compress with multiple threads

`CompressMultiScoped` splits borrowed input into independently compressed chunks and concatenates
them into one Brotli stream. `StdThreadScope` uses scoped operating-system threads, so no input copy
or thread pool is required:

```rust
use simd_brotli::enc::threading::{CompressMultiScoped, StdThreadScope};
use simd_brotli::enc::{
    BrotliEncoderMaxCompressedSizeMulti, BrotliEncoderParams, StandardAlloc,
};
use simd_brotli::BrotliDecompress;

let input = vec![b'a'; 1 << 20];
let thread_count = 4;
let mut params = BrotliEncoderParams::default();
params.quality = 9;

let mut output = vec![
    0;
    BrotliEncoderMaxCompressedSizeMulti(input.len(), thread_count)
];
let mut allocators = vec![Some(StandardAlloc::default()); thread_count];

let compressed_size = CompressMultiScoped(
    &params,
    &input,
    &mut output,
    &mut allocators,
    &StdThreadScope,
)
.expect("threaded compression failed");
output.truncate(compressed_size);

let mut decoded = Vec::new();
BrotliDecompress(&mut output.as_slice(), &mut decoded).unwrap();
assert_eq!(decoded, input);
```

Use one allocator per chunk. For applications that already use Rayon, implement the lightweight
`ThreadScope` adapter shown in its API documentation to run chunks on the existing pool.

The repository also includes stdin/stdout examples:

```bash
cargo run --release --example compress < input.txt > input.txt.br
cargo run --release --example decompress < input.txt.br > restored.txt
```

## What this fork changes

- **Stable, runtime-dispatched SIMD.** [`fearless_simd`](https://crates.io/crates/fearless_simd)
  selects an available implementation at runtime, so one standard-library build can use AVX2 on
  compatible x86-64 machines and NEON on Apple Silicon. `no_std` builds use the SIMD level selected
  at compile time.
- **Broader encoder vectorization.** SIMD paths cover the H10 and tagged match finders, Zopfli node
  updates, static-dictionary matching, block splitting, and population-cost calculations.
- **Hot-path algorithm improvements.** Histogram clustering, for example, calculates the cost of
  combined histograms without materializing their sum.
- **Optional profiling.** Feature-gated instrumentation attributes encoder time or allocations by
  pipeline stage without affecting default builds.

Compressed streams remain fully compatible with the upstream implementation; some quality levels
and corpora are byte-identical, but byte identity is not a general guarantee. The test matrix
compares qualities 0–11, including the 9.5 variants, across multiple corpora. On one 3.1 MB varied
corpus (Apple M5 Pro, NEON, release with LTO), encoding was about 11% faster at q9, 15% at q10, and
8% at q11. Results depend on the CPU and input, so benchmark your own workload. See the
[complete comparison with Google Brotli 1.2.0](C_BROTLI_COMPARISON.md) for all-quality timings,
output sizes, compatibility checks, implementation differences, and reproduction commands.

See the [changelog](CHANGELOG.md) for release details and verification results.

## Migrating from `brotli`

Replace the package and change Rust imports from `brotli` to `simd_brotli`:

```rust
// Before: use brotli::CompressorWriter;
use simd_brotli::CompressorWriter;
```

The public API is otherwise unchanged. The `brotli` and `catbrotli` binary names are also retained.

## `no_std` and custom allocation

Without the default `std` feature, use the custom-I/O and allocator-backed APIs. The low-level
decompression flow mirrors the C API:

1. Provide allocators for bytes, `u32` values, and Huffman codes.
2. Construct a `BrotliState`.
3. Call `BrotliDecompressStream` until it returns success or failure.

This allows all working memory to be allocated up front, which is useful in kernels, embedded
systems, and sandboxed processes. See the crate documentation and the `no_std` tests for complete
allocator examples.

## Command-line tools

Build and run the compatible `brotli` CLI through Cargo:

```bash
cargo run --release --bin brotli -- -c -q9 input.txt output.br
cargo run --release --bin brotli -- -d output.br restored.txt
```

## C interface

The optional C interface is a drop-in replacement for the official
[`google/brotli`](https://github.com/google/brotli) library. Build it from the `c` directory:

```bash
cd c
make
```

This produces `c/target/release/libbrotli.so` and the C command-line tool. The Rust implementation
is safe; the FFI bindings necessarily form an unsafe boundary.

## Stream concatenation

Brotli streams can be prepared as independently compressed chunks and joined for streaming use
cases. All chunks must use the same window size.

### Direct byte concatenation

Use a bare appendable first stream and bare catable subsequent streams, then append the Brotli
finalization byte:

```bash
brotli -c -bare -appendable -w22 input1.txt > base.br
brotli -c -bare -catable -w22 input2.txt > part2.br
brotli -c -bare -catable -w22 input3.txt > part3.br
(cat base.br part2.br part3.br; printf '\x03') > combined.br
brotli -d combined.br -o output.txt
```

This method is fast because joining requires no Brotli processing. Catable streams automatically
disable dictionary references across the chunk boundary.

### Size-optimized concatenation

`catbrotli` spends CPU time processing the stream boundaries and can produce a smaller result:

```bash
brotli -c -appendable input1.txt > appendable.br
brotli -c -catable input2.txt > catable1.br
brotli -c -catable input3.txt > catable2.br
catbrotli appendable.br catable1.br catable2.br > combined.br
```

Parameter dependencies are normalized automatically by the library:

- `catable = true` also sets `appendable = true` and `use_dictionary = false`.
- `bare_stream = true` also sets `byte_align = true`.
- `appendable = false` sets `byte_align = false`.

These rules apply whether parameters are set through `set_parameter` or directly on
`BrotliEncoderParams`.

## Profiling the encoder

The optional [`hotpath`](https://docs.rs/hotpath/) instrumentation reports time or allocations for
individual encoder stages:

```bash
# Wall-clock time
cargo run --release --features hotpath --bin brotli -- -c -q11 input.bin /dev/null

# CPU time
cargo run --release --features hotpath-cpu --bin brotli -- -c -q11 input.bin /dev/null

# Allocation counts and bytes
cargo run --release --features hotpath-alloc --bin brotli -- -c -q11 input.bin /dev/null
```

The report is printed when the process exits. Set `HOTPATH_OUTPUT_FORMAT=json-pretty` for the full
report as JSON. Most measurements are taken once per metablock, but match-finder and Zopfli details
are measured per position and add profiler overhead. Use an uninstrumented release build for
end-to-end benchmarks and a sampling profiler such as `sample` or `perf` for instruction-level
attribution.

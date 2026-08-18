// These exports mirror a C ABI: pointer validity contracts live in the public C API/header.
#![allow(clippy::missing_safety_doc)]

pub mod alloc_util;
pub mod broccoli;
pub mod compressor;
pub mod decompressor;
pub mod multicompress;

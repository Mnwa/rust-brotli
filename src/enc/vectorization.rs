//! Fixed-width vector storage for the encoder.
//!
//! The types here are plain arrays: `Default` + `Copy`, so they can live in the
//! encoder's allocator-backed slices and be handed to `Allocator<T>`. They carry no
//! arithmetic of their own. Math is done on [`fearless_simd`] vectors instead: inside a
//! `dispatch!` region, [`Mem256f::to_simd`] (and friends) loads a register and
//! [`Mem256f::from_simd`] stores it back.

use core::ops::{Index, IndexMut};
use core::slice::SliceIndex;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{vminq_f32, vminvq_f32, vminvq_u32};
#[cfg(all(target_arch = "aarch64", feature = "float64"))]
use core::arch::aarch64::{vminq_f64, vminvq_f64};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use fearless_simd::aarch64::Neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::x86::{Avx2, Avx512, Sse2, Sse4_2};
use fearless_simd::{Level, Simd, SimdBase, SimdInto, SimdSplit, f32x8, i16x16, i32x8, u32x8};
#[cfg(feature = "float64")]
use fearless_simd::{f64x8, u64x8};

#[cfg(target_arch = "aarch64")]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_f32x8_neon(_simd: Neon, v: f32x8<Neon>) -> f32 {
        let (left, right) = v.split();
        vminvq_f32(vminq_f32(left.into(), right.into()))
    }
);

#[cfg(target_arch = "aarch64")]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_u32x8_neon(_simd: Neon, v: u32x8<Neon>) -> u32 {
        let (left, right) = v.split();
        vminvq_u32(left.min(right).into())
    }
);

#[cfg(all(target_arch = "aarch64", feature = "float64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_f64x8_neon(_simd: Neon, v: f64x8<Neon>) -> f64 {
        let (left, right) = v.split();
        let (left, right) = left.min(right).split();
        vminvq_f64(vminq_f64(left.into(), right.into()))
    }
);

#[cfg(all(target_arch = "aarch64", feature = "float64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_u64x8_neon(_simd: Neon, v: u64x8<Neon>) -> u64 {
        let lanes: [u64; 8] = v.into();
        let a = lanes[0].min(lanes[1]);
        let b = lanes[2].min(lanes[3]);
        let c = lanes[4].min(lanes[5]);
        let d = lanes[6].min(lanes[7]);
        a.min(b).min(c.min(d))
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_f32x8_avx2(_simd: Avx2, v: f32x8<Avx2>) -> f32 {
        let v: __m256 = v.into();
        let v = _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let v = _mm_min_ps(v, _mm_movehl_ps(v, v));
        _mm_cvtss_f32(_mm_min_ss(v, _mm_shuffle_ps::<0x55>(v, v)))
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_f32x8_sse2(_simd: Sse2, v: f32x8<Sse2>) -> f32 {
        let (left, right) = v.split();
        let v = _mm_min_ps(left.into(), right.into());
        let v = _mm_min_ps(v, _mm_movehl_ps(v, v));
        _mm_cvtss_f32(_mm_min_ss(v, _mm_shuffle_ps::<0x55>(v, v)))
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_u32x8_avx2(_simd: Avx2, v: u32x8<Avx2>) -> u32 {
        let v: __m256i = v.into();
        let v = _mm_min_epu32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let v = _mm_min_epu32(v, _mm_shuffle_epi32::<0x4e>(v));
        _mm_cvtsi128_si32(_mm_min_epu32(v, _mm_shuffle_epi32::<0xb1>(v))) as u32
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_u32x8_sse4(_simd: Sse4_2, v: u32x8<Sse4_2>) -> u32 {
        let (left, right) = v.split();
        let v = _mm_min_epu32(left.into(), right.into());
        let v = _mm_min_epu32(v, _mm_shuffle_epi32::<0x4e>(v));
        _mm_cvtsi128_si32(_mm_min_epu32(v, _mm_shuffle_epi32::<0xb1>(v))) as u32
    }
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_u32x8_sse2(_simd: Sse2, v: u32x8<Sse2>) -> u32 {
        let min = |a: __m128i, b: __m128i| {
            let sign = _mm_set1_epi32(i32::MIN);
            let gt = _mm_cmpgt_epi32(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign));
            _mm_or_si128(_mm_and_si128(gt, b), _mm_andnot_si128(gt, a))
        };
        let (left, right) = v.split();
        let v = min(left.into(), right.into());
        let v = min(v, _mm_shuffle_epi32::<0x4e>(v));
        _mm_cvtsi128_si32(min(v, _mm_shuffle_epi32::<0xb1>(v))) as u32
    }
);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "float64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_f64x8_avx512(_simd: Avx512, v: f64x8<Avx512>) -> f64 {
        _mm512_reduce_min_pd(v.into())
    }
);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "float64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_f64x8_avx2(_simd: Avx2, v: f64x8<Avx2>) -> f64 {
        let (left, right) = v.split();
        let v = _mm256_min_pd(left.into(), right.into());
        let v = _mm_min_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd::<1>(v));
        _mm_cvtsd_f64(_mm_min_sd(v, _mm_unpackhi_pd(v, v)))
    }
);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "float64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_f64x8_sse2(_simd: Sse2, v: f64x8<Sse2>) -> f64 {
        let (left, right) = v.split();
        let (left0, left1) = left.split();
        let (right0, right1) = right.split();
        let v = _mm_min_pd(
            _mm_min_pd(left0.into(), right0.into()),
            _mm_min_pd(left1.into(), right1.into()),
        );
        _mm_cvtsd_f64(_mm_min_sd(v, _mm_unpackhi_pd(v, v)))
    }
);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "float64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn min_lane_u64x8_avx512(_simd: Avx512, v: u64x8<Avx512>) -> u64 {
        _mm512_reduce_min_epu64(v.into())
    }
);

/// The instruction set the vectorized encoder paths run on.
///
/// Detected at runtime where the platform allows it (`std` builds, wasm), otherwise the
/// best level this crate was compiled for. The `std` answer is cached: probing costs a
/// dozen feature tests, and callers such as [`crate::enc::bit_cost::BrotliPopulationCost`]
/// dispatch once per histogram, deep inside the clustering loops.
#[cfg(feature = "std")]
#[inline]
pub fn detect_level() -> Level {
    static LEVEL: std::sync::LazyLock<Level> =
        std::sync::LazyLock::new(|| Level::try_detect().unwrap_or_else(Level::baseline));
    *LEVEL
}

/// See the `std` variant above; without `std` there is nothing to cache, as
/// `try_detect` cannot probe the CPU and always resolves to the compiled-for level.
#[cfg(not(feature = "std"))]
#[inline]
pub fn detect_level() -> Level {
    Level::try_detect().unwrap_or_else(Level::baseline)
}

/// The smallest lane of `v`, folded in `log2(8)` steps.
#[inline(always)]
pub fn min_lane_f32x8<S: Simd>(v: f32x8<S>) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = v.witness().level().as_neon() {
        return min_lane_f32x8_neon(neon, <[f32; 8]>::from(v).simd_into(neon));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let level = v.witness().level();
        if let Some(avx2) = level.as_avx2() {
            return min_lane_f32x8_avx2(avx2, <[f32; 8]>::from(v).simd_into(avx2));
        }
        if let Some(sse2) = level.as_sse2() {
            return min_lane_f32x8_sse2(sse2, <[f32; 8]>::from(v).simd_into(sse2));
        }
    }

    let (left, right) = v.split();
    let arr = left.min(right).as_array();
    arr[0].min(arr[1]).min(arr[2].min(arr[3]))
}

/// The smallest lane of `v`, folded in `log2(8)` steps.
#[inline(always)]
pub fn min_lane_u32x8<S: Simd>(v: u32x8<S>) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = v.witness().level().as_neon() {
        return min_lane_u32x8_neon(neon, <[u32; 8]>::from(v).simd_into(neon));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let level = v.witness().level();
        if let Some(avx2) = level.as_avx2() {
            return min_lane_u32x8_avx2(avx2, <[u32; 8]>::from(v).simd_into(avx2));
        }
        if let Some(sse4) = level.as_sse4_2() {
            return min_lane_u32x8_sse4(sse4, <[u32; 8]>::from(v).simd_into(sse4));
        }
        if let Some(sse2) = level.as_sse2() {
            return min_lane_u32x8_sse2(sse2, <[u32; 8]>::from(v).simd_into(sse2));
        }
    }

    let (left, right) = v.split();
    let arr = left.min(right).as_array();
    arr[0].min(arr[1]).min(arr[2].min(arr[3]))
}

/// The smallest lane of `v`, folded in `log2(8)` steps.
#[cfg(feature = "float64")]
#[inline(always)]
pub fn min_lane_f64x8<S: Simd>(v: f64x8<S>) -> f64 {
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = v.witness().level().as_neon() {
        return min_lane_f64x8_neon(neon, <[f64; 8]>::from(v).simd_into(neon));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let level = v.witness().level();
        if let Some(avx512) = level.as_avx512() {
            return min_lane_f64x8_avx512(avx512, <[f64; 8]>::from(v).simd_into(avx512));
        }
        if let Some(avx2) = level.as_avx2() {
            return min_lane_f64x8_avx2(avx2, <[f64; 8]>::from(v).simd_into(avx2));
        }
        if let Some(sse2) = level.as_sse2() {
            return min_lane_f64x8_sse2(sse2, <[f64; 8]>::from(v).simd_into(sse2));
        }
    }

    let (left, right) = v.split();
    let arr = left.min(right).as_array();
    arr[0].min(arr[1]).min(arr[2].min(arr[3]))
}

/// The smallest lane of `v`, folded in `log2(8)` steps.
#[cfg(feature = "float64")]
#[inline(always)]
pub fn min_lane_u64x8<S: Simd>(v: u64x8<S>) -> u64 {
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = v.witness().level().as_neon() {
        return min_lane_u64x8_neon(neon, <[u64; 8]>::from(v).simd_into(neon));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let level = v.witness().level();
        // AVX-512 has native unsigned 64-bit minimum instructions. The AVX2/SSE
        // compare-and-blend emulation is slower than the portable scalarized fold.
        if let Some(avx512) = level.as_avx512() {
            return min_lane_u64x8_avx512(avx512, <[u64; 8]>::from(v).simd_into(avx512));
        }
    }

    let v = v.min(v.slide::<4>(v));
    let v = v.min(v.slide::<2>(v));
    let v = v.min(v.slide::<1>(v));
    v[0]
}

macro_rules! define_vector {
    ($(#[$attr:meta])* $name:ident, $elem:ty, $lanes:literal, $simd:ident) => {
        $(#[$attr])*
        #[derive(Default, Copy, Clone, Debug)]
        pub struct $name([$elem; $lanes]);

        impl $name {
            /// Load the lanes into a SIMD register.
            #[inline(always)]
            pub fn to_simd<S: Simd>(self, simd: S) -> $simd<S> {
                self.0.simd_into(simd)
            }

            /// Store a SIMD register back into plain memory.
            #[inline(always)]
            pub fn from_simd<S: Simd>(value: $simd<S>) -> Self {
                Self(value.into())
            }
        }

        impl From<[$elem; $lanes]> for $name {
            #[inline(always)]
            fn from(value: [$elem; $lanes]) -> Self {
                Self(value)
            }
        }

        impl<I: SliceIndex<[$elem]>> Index<I> for $name {
            type Output = I::Output;

            #[inline(always)]
            fn index(&self, index: I) -> &Self::Output {
                &self.0[index]
            }
        }

        impl<I: SliceIndex<[$elem]>> IndexMut<I> for $name {
            #[inline(always)]
            fn index_mut(&mut self, index: I) -> &mut Self::Output {
                &mut self.0[index]
            }
        }
    };
}

#[cfg(not(feature = "float64"))]
define_vector!(Mem256f, f32, 8, f32x8);
#[cfg(feature = "float64")]
define_vector!(Mem256f, f64, 8, f64x8);
define_vector!(Mem256i, i32, 8, i32x8);
define_vector!(Mem16x16, i16, 16, i16x16);
define_vector!(
    /// A 16-bucket probability distribution.
    ///
    /// Same shape as [`Mem16x16`], but deliberately a separate type: `BrotliAlloc`
    /// requires `Allocator<PDF>` and `Allocator<s16>` as distinct bounds, so the two
    /// cannot be aliases of each other. Re-exported as [`crate::enc::pdf::PDF`].
    PDF,
    i16,
    16,
    i16x16
);

pub type v256 = Mem256f;
pub type v256i = Mem256i;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_lane_32_finds_every_lane() {
        fearless_simd::dispatch!(detect_level(), simd => {
            for lane in 0..8 {
                let mut floats = [10.0f32; 8];
                floats[lane] = -1.0;
                assert_eq!(min_lane_f32x8(floats.simd_into(simd)), -1.0);

                let mut integers = [u32::MAX; 8];
                integers[lane] = 1;
                assert_eq!(min_lane_u32x8(integers.simd_into(simd)), 1);
            }
        });
    }

    #[cfg(feature = "float64")]
    #[test]
    fn min_lane_64_finds_every_lane() {
        fearless_simd::dispatch!(detect_level(), simd => {
            for lane in 0..8 {
                let mut floats = [10.0f64; 8];
                floats[lane] = -1.0;
                assert_eq!(min_lane_f64x8(floats.simd_into(simd)), -1.0);

                let mut integers = [u64::MAX; 8];
                integers[lane] = 1;
                assert_eq!(min_lane_u64x8(integers.simd_into(simd)), 1);
            }
        });
    }
}

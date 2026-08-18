use core;
use core::cmp::{max, min};

use fearless_simd::{Select, Simd, SimdBase, SimdMask};
#[cfg(not(feature = "float64"))]
use fearless_simd::{f32x8 as FloatX8, u32x8 as IndexX8};
#[cfg(feature = "float64")]
use fearless_simd::{f64x8 as FloatX8, u64x8 as IndexX8};

use super::super::alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use super::backward_references::BrotliEncoderParams;
use super::bit_cost::BrotliPopulationCost;
use super::block_split::BlockSplit;
use super::cluster::{BrotliHistogramBitCostDistance, BrotliHistogramCombine, HistogramPair};
use super::command::Command;
use super::histogram::{
    ClearHistograms, CostAccessors, HistogramAddHistogram, HistogramAddItem, HistogramAddVector,
    HistogramClear, HistogramCommand, HistogramDistance, HistogramLiteral,
};
use super::util::FastLog2;
use super::vectorization::{Mem256f, detect_level};
#[cfg(not(feature = "float64"))]
use super::vectorization::{
    min_lane_f32x8 as min_lane_floatx8, min_lane_u32x8 as min_lane_indexx8,
};
#[cfg(feature = "float64")]
use super::vectorization::{
    min_lane_f64x8 as min_lane_floatx8, min_lane_u64x8 as min_lane_indexx8,
};
use crate::enc::combined_alloc::allocate;
use crate::enc::floatX;

/// Lane offsets, added to a vector's base index to recover a histogram id.
#[cfg(not(feature = "float64"))]
type LaneIndex = u32;
#[cfg(feature = "float64")]
type LaneIndex = u64;
static LANE_INDICES: [LaneIndex; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

static kMaxLiteralHistograms: usize = 100usize;

static kMaxCommandHistograms: usize = 50usize;

static kLiteralBlockSwitchCost: floatX = 28.1;

static kCommandBlockSwitchCost: floatX = 13.5;

static kDistanceBlockSwitchCost: floatX = 14.6;

static kLiteralStrideLength: usize = 70usize;

static kCommandStrideLength: usize = 40usize;

static kSymbolsPerLiteralHistogram: usize = 544usize;

static kSymbolsPerCommandHistogram: usize = 530usize;

static kSymbolsPerDistanceHistogram: usize = 544usize;

static kMinLengthForBlockSplitting: usize = 128usize;

static kIterMulForRefining: usize = 2usize;

static kMinItersForRefining: usize = 100usize;

#[inline(always)]
fn update_cost_and_signal<S: Simd>(
    simd: S,
    num_histograms32: u32,
    ix: usize,
    min_cost: floatX,
    block_switch_cost: floatX,
    cost: &mut [Mem256f],
    switch_signal: &mut [u8],
) {
    let ymm_min_cost = FloatX8::splat(simd, min_cost);
    let ymm_block_switch_cost = FloatX8::splat(simd, block_switch_cost);

    for (index, cost_it) in cost[..((num_histograms32 as usize + 7) >> 3)]
        .iter_mut()
        .enumerate()
    {
        let costk_minus_min_cost = cost_it.to_simd(simd) - ymm_min_cost;
        // One bit per lane that is at least a block switch away from the cheapest histogram.
        switch_signal[ix + index] |= costk_minus_min_cost
            .simd_ge(ymm_block_switch_cost)
            .to_bitmask() as u8;
        *cost_it = Mem256f::from_simd(costk_minus_min_cost.min(ymm_block_switch_cost));
        //println_stderr!("{:} ss {:} c {:?}", (index << 3) + 7, switch_signal[ix + index],*cost_it);
    }
}

#[inline(always)]
fn update_histogram_costs<S: Simd>(
    simd: S,
    insert_costs: &[floatX],
    num_histograms: usize,
    cost: &mut [Mem256f],
) -> (floatX, u8) {
    let num_vectors = num_histograms >> 3;
    let vectorized_offset = num_vectors << 3;
    let mut min_cost = 1e38;
    let mut block_id = 0;
    let mut min_lanes = FloatX8::splat(simd, min_cost);
    let mut id_lanes = IndexX8::splat(simd, LaneIndex::MAX);
    let (insert_cost_chunks, insert_cost_tail) = insert_costs[..vectorized_offset].as_chunks::<8>();
    debug_assert!(insert_cost_tail.is_empty());
    for (v_index, (cost_iter, insert_costs)) in cost[..num_vectors]
        .iter_mut()
        .zip(insert_cost_chunks)
        .enumerate()
    {
        let base_index = v_index << 3;
        let updated = cost_iter.to_simd(simd) + FloatX8::load_array_ref(simd, insert_costs);
        *cost_iter = Mem256f::from_simd(updated);
        let improved = updated.simd_lt(min_lanes);
        min_lanes = improved.select(updated, min_lanes);
        id_lanes = improved.select(
            IndexX8::load_array(simd, LANE_INDICES) + base_index as LaneIndex,
            id_lanes,
        );
    }
    if num_vectors != 0 {
        let best = min_lane_floatx8(min_lanes);
        if best < min_cost {
            min_cost = best;
            block_id = min_lane_indexx8(
                min_lanes
                    .simd_eq(FloatX8::splat(simd, best))
                    .select(id_lanes, IndexX8::splat(simd, LaneIndex::MAX)),
            ) as u8;
        }
    }
    for (lane, insert_cost) in insert_costs[vectorized_offset..num_histograms]
        .iter()
        .enumerate()
    {
        let histogram_id = vectorized_offset + lane;
        let scalar_cost = &mut cost[histogram_id >> 3][histogram_id & 7];
        *scalar_cost += *insert_cost;
        if *scalar_cost < min_cost {
            min_cost = *scalar_cost;
            block_id = histogram_id as u8;
        }
    }
    (min_cost, block_id)
}
fn CountLiterals(cmds: &[Command], num_commands: usize) -> usize {
    let mut total_length: usize = 0usize;
    for cmd in cmds.iter().take(num_commands) {
        total_length = total_length.wrapping_add(cmd.insert_len_ as usize);
    }
    total_length
}

fn CopyLiteralsToByteArray(
    cmds: &[Command],
    num_commands: usize,
    data: &[u8],
    offset: usize,
    mask: usize,
    literals: &mut [u8],
) {
    let mut pos: usize = 0usize;
    let mut from_pos: usize = offset & mask;
    for cmd in cmds.iter().take(num_commands) {
        let mut insert_len: usize = cmd.insert_len_ as usize;
        if from_pos.wrapping_add(insert_len) > mask {
            let head_size: usize = mask.wrapping_add(1).wrapping_sub(from_pos);
            literals[pos..(pos + head_size)]
                .copy_from_slice(&data[from_pos..(from_pos + head_size)]);
            from_pos = 0usize;
            pos = pos.wrapping_add(head_size);
            insert_len = insert_len.wrapping_sub(head_size);
        }
        if insert_len > 0usize {
            literals[pos..(pos + insert_len)]
                .copy_from_slice(&data[from_pos..(from_pos + insert_len)]);
            pos = pos.wrapping_add(insert_len);
        }
        from_pos = from_pos
            .wrapping_add(insert_len)
            .wrapping_add(cmd.copy_len() as usize)
            & mask;
    }
}

fn MyRand(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(16807);
    if *seed == 0u32 {
        *seed = 1u32;
    }
    *seed
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn InitialEntropyCodes<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors,
    IntegerType: Sized + Clone,
>(
    data: &[IntegerType],
    length: usize,
    stride: usize,
    num_histograms: usize,
    histograms: &mut [HistogramType],
) where
    u64: core::convert::From<IntegerType>,
{
    let mut seed: u32 = 7u32;
    let block_length: usize = length.wrapping_div(num_histograms);
    ClearHistograms(histograms, num_histograms);
    for (i, histogram) in histograms.iter_mut().enumerate().take(num_histograms) {
        let mut pos: usize = length.wrapping_mul(i).wrapping_div(num_histograms);
        if i != 0usize {
            pos = pos.wrapping_add((MyRand(&mut seed) as usize).wrapping_rem(block_length));
        }
        if pos.wrapping_add(stride) >= length {
            pos = length.wrapping_sub(stride).wrapping_sub(1);
        }
        HistogramAddVector(histogram, &data[pos..], stride);
    }
}

fn RandomSample<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors,
    IntegerType: Sized + Clone,
>(
    seed: &mut u32,
    data: &[IntegerType],
    length: usize,
    mut stride: usize,
    sample: &mut HistogramType,
) where
    u64: core::convert::From<IntegerType>,
{
    let pos: usize;
    if stride >= length {
        pos = 0usize;
        stride = length;
    } else {
        pos = (MyRand(seed) as usize).wrapping_rem(length.wrapping_sub(stride).wrapping_add(1));
    }
    HistogramAddVector(sample, &data[pos..], stride);
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn RefineEntropyCodes<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors + core::default::Default,
    IntegerType: Sized + Clone,
>(
    data: &[IntegerType],
    length: usize,
    stride: usize,
    num_histograms: usize,
    histograms: &mut [HistogramType],
) where
    u64: core::convert::From<IntegerType>,
{
    let mut iters: usize = kIterMulForRefining
        .wrapping_mul(length)
        .wrapping_div(stride)
        .wrapping_add(kMinItersForRefining);
    let mut seed: u32 = 7u32;
    iters = iters
        .wrapping_add(num_histograms)
        .wrapping_sub(1)
        .wrapping_div(num_histograms)
        .wrapping_mul(num_histograms);
    for iter in 0usize..iters {
        let mut sample = HistogramType::default();
        HistogramClear(&mut sample);
        RandomSample(&mut seed, data, length, stride, &mut sample);
        HistogramAddHistogram(&mut histograms[iter.wrapping_rem(num_histograms)], &sample);
    }
}

fn BitCost(count: usize) -> floatX {
    if count == 0usize {
        -2.0
    } else {
        FastLog2(count as u64)
    }
}

/// Entry point into the vectorized cost loop: picks the best instruction set available
/// and runs [`FindBlocksSimd`] with it.
fn FindBlocks<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors,
    IntegerType: Sized + Clone,
>(
    data: &[IntegerType],
    length: usize,
    block_switch_bitcost: floatX,
    num_histograms: usize,
    histograms: &[HistogramType],
    insert_cost: &mut [floatX],
    cost: &mut [Mem256f],
    switch_signal: &mut [u8],
    block_id: &mut [u8],
) -> usize
where
    u64: core::convert::From<IntegerType>,
{
    dispatch!(detect_level(), simd => FindBlocksSimd(
        simd,
        data,
        length,
        block_switch_bitcost,
        num_histograms,
        histograms,
        insert_cost,
        cost,
        switch_signal,
        block_id,
    ))
}

#[inline(always)]
fn FindBlocksSimd<
    S: Simd,
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors,
    IntegerType: Sized + Clone,
>(
    simd: S,
    data: &[IntegerType],
    length: usize,
    block_switch_bitcost: floatX,
    num_histograms: usize,
    histograms: &[HistogramType],
    insert_cost: &mut [floatX],
    cost: &mut [Mem256f],
    switch_signal: &mut [u8],
    block_id: &mut [u8],
) -> usize
where
    u64: core::convert::From<IntegerType>,
{
    if num_histograms == 0 {
        return 0;
    }
    let data_size: usize = histograms[0].slice().len();
    let bitmaplen = num_histograms.div_ceil(8);
    let mut num_blocks: usize = 1;
    if num_histograms <= 1 {
        block_id[..length].fill(0);
        return 1;
    }
    let insert_cost = &mut insert_cost[..(data_size * num_histograms)];
    insert_cost.fill(0.0);
    let (initial_costs, remaining_costs) = insert_cost.split_at_mut(num_histograms);
    let histograms = &histograms[..num_histograms];
    for (initial_cost, histogram) in initial_costs.iter_mut().zip(histograms) {
        *initial_cost = FastLog2(histogram.total_count() as u32 as u64);
    }
    for (symbol, symbol_costs) in remaining_costs
        .chunks_exact_mut(num_histograms)
        .enumerate()
        .rev()
    {
        let symbol = symbol + 1;
        for ((symbol_cost, initial_cost), histogram) in symbol_costs
            .iter_mut()
            .zip(initial_costs.iter())
            .zip(histograms)
        {
            *symbol_cost = *initial_cost - BitCost(histogram.slice()[symbol] as usize);
        }
    }
    for (initial_cost, histogram) in initial_costs.iter_mut().zip(histograms) {
        *initial_cost -= BitCost(histogram.slice()[0] as usize);
    }
    cost.fill(Mem256f::default());
    switch_signal[..(length * bitmaplen)].fill(0);
    for (byte_ix, (data_byte_ix, block_id_ptr)) in data[..length]
        .iter()
        .zip(block_id[..length].iter_mut())
        .enumerate()
    {
        let ix = byte_ix * bitmaplen;
        let insert_cost_ix: usize =
            u64::from(data_byte_ix.clone()).wrapping_mul(num_histograms as u64) as usize;
        let mut block_switch_cost: floatX = block_switch_bitcost;
        let insert_cost_slice = &insert_cost[insert_cost_ix..];
        let (min_cost, best_id) =
            update_histogram_costs(simd, insert_cost_slice, num_histograms, cost);
        *block_id_ptr = best_id;
        if byte_ix < 2000usize {
            block_switch_cost *= 0.77 + 0.07 * (byte_ix as floatX) / 2000.0;
        }
        update_cost_and_signal(
            simd,
            num_histograms as u32,
            ix,
            min_cost,
            block_switch_cost,
            cost,
            switch_signal,
        );
    }
    let mut cur_id = block_id[length - 1];
    for (signal, previous_id) in switch_signal[..(length * bitmaplen)]
        .chunks_exact(bitmaplen)
        .zip(block_id[..length].iter_mut())
        .rev()
        .skip(1)
    {
        let mask = 1u8 << (cur_id & 7);
        if signal[(cur_id >> 3) as usize] & mask != 0 && cur_id != *previous_id {
            cur_id = *previous_id;
            num_blocks += 1;
        }
        *previous_id = cur_id;
    }
    num_blocks
}

fn RemapBlockIds(
    block_ids: &mut [u8],
    length: usize,
    new_id: &mut [u16],
    num_histograms: usize,
) -> usize {
    static kInvalidId: u16 = 256u16;
    let mut next_id: u16 = 0u16;
    new_id[..num_histograms].fill(kInvalidId);
    for i in 0usize..length {
        if new_id[(block_ids[i] as usize)] as i32 == kInvalidId as i32 {
            new_id[(block_ids[i] as usize)] = {
                let _old = next_id;
                next_id = (next_id as i32 + 1) as u16;
                _old
            };
        }
    }
    for i in 0usize..length {
        block_ids[i] = new_id[(block_ids[i] as usize)] as u8;
    }
    next_id as usize
}

fn BuildBlockHistograms<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors,
    IntegerType: Sized + Clone,
>(
    data: &[IntegerType],
    length: usize,
    block_ids: &[u8],
    num_histograms: usize,
    histograms: &mut [HistogramType],
) where
    u64: core::convert::From<IntegerType>,
{
    ClearHistograms(histograms, num_histograms);
    for i in 0usize..length {
        HistogramAddItem(
            &mut histograms[(block_ids[i] as usize)],
            u64::from(data[i].clone()) as usize,
        );
    }
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn ClusterBlocks<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors + core::default::Default + Clone,
    Alloc: alloc::Allocator<u8>
        + alloc::Allocator<u32>
        + alloc::Allocator<HistogramType>
        + alloc::Allocator<HistogramPair>,
    IntegerType: Sized + Clone,
>(
    alloc: &mut Alloc,
    data: &[IntegerType],
    length: usize,
    num_blocks: usize,
    scratch_space: &mut HistogramType::i32vec,
    block_ids: &mut [u8],
    split: &mut BlockSplit<Alloc>,
) where
    u64: core::convert::From<IntegerType>,
{
    // All dynamic u32 scratch is bounded by num_blocks and has one shared
    // lifetime. Keep it in one standard allocation instead of four cells.
    let mut block_data = allocate::<u32, _>(alloc, 4 * num_blocks);
    let (histogram_symbols, scratch) = block_data.slice_mut().split_at_mut(num_blocks);
    let (block_lengths, scratch) = scratch.split_at_mut(num_blocks);
    let (cluster_size, clusters) = scratch.split_at_mut(num_blocks);
    let expected_num_clusters: usize = (16usize)
        .wrapping_mul(num_blocks.wrapping_add(64).wrapping_sub(1))
        .wrapping_div(64);
    let mut all_histograms_size: usize = 0usize;
    let mut all_histograms_capacity: usize = expected_num_clusters;
    let mut all_histograms = allocate::<HistogramType, _>(alloc, all_histograms_capacity);
    let mut cluster_size_size: usize = 0usize;
    let mut num_clusters: usize = 0usize;
    let mut histograms = allocate::<HistogramType, _>(alloc, min(num_blocks, 64));
    let mut max_num_pairs: usize = (64i32 * 64i32 / 2i32) as usize;
    let pairs_capacity: usize = max_num_pairs.wrapping_add(1);
    let mut pairs = allocate::<HistogramPair, _>(alloc, pairs_capacity);
    let mut pos: usize = 0usize;

    static kInvalidIndex: u32 = u32::MAX;
    let mut i: usize;
    let mut sizes: [u32; 64] = [0; 64];
    let mut new_clusters: [u32; 64] = [0; 64];
    let mut symbols: [u32; 64] = [0; 64];
    let mut remap: [u32; 64] = [0; 64];
    {
        let mut block_idx: usize = 0usize;
        i = 0usize;
        while i < length {
            {
                {
                    let _rhs = 1;
                    let _lhs = &mut block_lengths[block_idx];
                    *_lhs = (*_lhs).wrapping_add(_rhs as u32);
                }
                if i.wrapping_add(1) == length
                    || block_ids[i] as i32 != block_ids[i.wrapping_add(1)] as i32
                {
                    block_idx = block_idx.wrapping_add(1);
                }
            }
            i = i.wrapping_add(1);
        }
    }
    i = 0usize;
    while i < num_blocks {
        {
            let num_to_combine: usize = min(num_blocks.wrapping_sub(i), 64);

            for j in 0usize..num_to_combine {
                HistogramClear(&mut histograms.slice_mut()[j]);
                for _k in 0usize..block_lengths[i.wrapping_add(j)] as usize {
                    HistogramAddItem(
                        &mut histograms.slice_mut()[j],
                        u64::from(data[pos].clone()) as usize,
                    );
                    pos = pos.wrapping_add(1);
                }
                let new_cost = BrotliPopulationCost(&histograms.slice()[j], scratch_space);
                (histograms.slice_mut()[j]).set_bit_cost(new_cost);

                new_clusters[j] = j as u32;
                symbols[j] = j as u32;
                sizes[j] = 1u32;
            }
            let num_new_clusters: usize = BrotliHistogramCombine(
                histograms.slice_mut(),
                &mut sizes[..],
                &mut symbols[..],
                &mut new_clusters[..],
                pairs.slice_mut(),
                num_to_combine,
                num_to_combine,
                64usize,
                max_num_pairs,
                scratch_space,
            );
            {
                if all_histograms_capacity < all_histograms_size.wrapping_add(num_new_clusters) {
                    let mut _new_size: usize = if all_histograms_capacity == 0usize {
                        all_histograms_size.wrapping_add(num_new_clusters)
                    } else {
                        all_histograms_capacity
                    };
                    while _new_size < all_histograms_size.wrapping_add(num_new_clusters) {
                        _new_size = _new_size.wrapping_mul(2);
                    }
                    let mut new_array = allocate::<HistogramType, _>(alloc, _new_size);
                    new_array.slice_mut()[..all_histograms_capacity]
                        .clone_from_slice(&all_histograms.slice()[..all_histograms_capacity]);
                    <Alloc as Allocator<HistogramType>>::free_cell(
                        alloc,
                        core::mem::replace(&mut all_histograms, new_array),
                    );
                    all_histograms_capacity = _new_size;
                }
            }
            debug_assert!(cluster_size_size + num_new_clusters <= num_blocks);
            for j in 0usize..num_new_clusters {
                all_histograms.slice_mut()[all_histograms_size] =
                    histograms.slice()[new_clusters[j] as usize].clone();
                all_histograms_size = all_histograms_size.wrapping_add(1);
                cluster_size[cluster_size_size] = sizes[new_clusters[j] as usize];
                cluster_size_size = cluster_size_size.wrapping_add(1);
                remap[new_clusters[j] as usize] = j as u32;
            }
            for j in 0usize..num_to_combine {
                histogram_symbols[i.wrapping_add(j)] =
                    (num_clusters as u32).wrapping_add(remap[symbols[j] as usize]);
            }
            num_clusters = num_clusters.wrapping_add(num_new_clusters);
        }
        i = i.wrapping_add(64);
    }
    <Alloc as Allocator<HistogramType>>::free_cell(alloc, core::mem::take(&mut histograms));
    max_num_pairs = min(
        (64usize).wrapping_mul(num_clusters),
        num_clusters.wrapping_div(2).wrapping_mul(num_clusters),
    );
    if pairs_capacity < max_num_pairs.wrapping_add(1) {
        let new_cell = allocate::<HistogramPair, _>(alloc, max_num_pairs.wrapping_add(1));
        <Alloc as Allocator<HistogramPair>>::free_cell(
            alloc,
            core::mem::replace(&mut pairs, new_cell),
        );
    }
    i = 0usize;
    for item in clusters[..num_clusters].iter_mut() {
        *item = i as u32;
        i = i.wrapping_add(1);
    }
    let num_final_clusters: usize = BrotliHistogramCombine(
        all_histograms.slice_mut(),
        cluster_size,
        histogram_symbols,
        clusters,
        pairs.slice_mut(),
        num_clusters,
        num_blocks,
        256usize,
        max_num_pairs,
        scratch_space,
    );
    <Alloc as Allocator<HistogramPair>>::free_cell(alloc, core::mem::take(&mut pairs));

    // Final clustering no longer needs cluster sizes; reuse those lanes for
    // the old-to-new histogram index.
    let new_index = &mut cluster_size[..num_clusters];
    new_index.fill(kInvalidIndex);
    pos = 0usize;
    {
        let mut next_index: u32 = 0u32;
        for i in 0usize..num_blocks {
            let mut histo: HistogramType = HistogramType::default();
            let mut best_out: u32;
            let mut best_bits: floatX;
            HistogramClear(&mut histo);
            for _j in 0usize..block_lengths[i] as usize {
                HistogramAddItem(&mut histo, u64::from(data[pos].clone()) as usize);
                pos = pos.wrapping_add(1);
            }
            best_out = if i == 0usize {
                histogram_symbols[0]
            } else {
                histogram_symbols[i.wrapping_sub(1)]
            };
            best_bits = BrotliHistogramBitCostDistance(
                &histo,
                &all_histograms.slice_mut()[(best_out as usize)],
                scratch_space,
            );
            for &cluster in clusters.iter().take(num_final_clusters) {
                let cur_bits: floatX = BrotliHistogramBitCostDistance(
                    &histo,
                    &all_histograms.slice_mut()[cluster as usize],
                    scratch_space,
                );
                if cur_bits < best_bits {
                    best_bits = cur_bits;
                    best_out = cluster;
                }
            }
            histogram_symbols[i] = best_out;
            if new_index[best_out as usize] == kInvalidIndex {
                new_index[best_out as usize] = next_index;
                next_index = next_index.wrapping_add(1);
            }
        }
    }
    <Alloc as Allocator<HistogramType>>::free_cell(alloc, core::mem::take(&mut all_histograms));
    {
        if split.types_alloc_size() < num_blocks {
            let mut _new_size: usize = if split.types_alloc_size() == 0usize {
                num_blocks
            } else {
                split.types_alloc_size()
            };
            while _new_size < num_blocks {
                _new_size = _new_size.wrapping_mul(2);
            }
            let mut new_array = allocate::<u8, _>(alloc, _new_size);
            new_array.slice_mut()[..split.types_alloc_size()]
                .copy_from_slice(&split.types.slice()[..split.types_alloc_size()]);
            <Alloc as Allocator<u8>>::free_cell(
                alloc,
                core::mem::replace(&mut split.types, new_array),
            );
        }
    }
    {
        if split.lengths_alloc_size() < num_blocks {
            let mut _new_size: usize = if split.lengths_alloc_size() == 0usize {
                num_blocks
            } else {
                split.lengths_alloc_size()
            };
            while _new_size < num_blocks {
                _new_size = _new_size.wrapping_mul(2);
            }
            let mut new_array = allocate::<u32, _>(alloc, _new_size);
            new_array.slice_mut()[..split.lengths_alloc_size()]
                .copy_from_slice(split.lengths.slice());
            <Alloc as Allocator<u32>>::free_cell(
                alloc,
                core::mem::replace(&mut split.lengths, new_array),
            );
        }
    }
    {
        let mut cur_length: u32 = 0u32;
        let mut block_idx: usize = 0usize;
        let mut max_type: u8 = 0u8;
        for i in 0usize..num_blocks {
            cur_length = cur_length.wrapping_add(block_lengths[i]);
            if i.wrapping_add(1) == num_blocks
                || histogram_symbols[i] != histogram_symbols[i.wrapping_add(1)]
            {
                let id: u8 = new_index[histogram_symbols[i] as usize] as u8;
                split.types.slice_mut()[block_idx] = id;
                split.lengths.slice_mut()[block_idx] = cur_length;
                max_type = max(max_type, id);
                cur_length = 0u32;
                block_idx = block_idx.wrapping_add(1);
            }
        }
        split.num_blocks = block_idx;
        split.num_types = (max_type as usize).wrapping_add(1);
    }
    <Alloc as Allocator<u32>>::free_cell(alloc, block_data);
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn SplitByteVector<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors + core::default::Default + Clone,
    Alloc: alloc::Allocator<u8>
        + alloc::Allocator<u16>
        + alloc::Allocator<u32>
        + alloc::Allocator<floatX>
        + alloc::Allocator<Mem256f>
        + alloc::Allocator<HistogramType>
        + alloc::Allocator<HistogramPair>,
    IntegerType: Sized + Clone,
>(
    alloc: &mut Alloc,
    data: &[IntegerType],
    length: usize,
    literals_per_histogram: usize,
    max_histograms: usize,
    sampling_stride_length: usize,
    block_switch_cost: floatX,
    params: &BrotliEncoderParams,
    scratch_space: &mut HistogramType::i32vec,
    split: &mut BlockSplit<Alloc>,
) where
    u64: core::convert::From<IntegerType>,
{
    let data_size: usize = HistogramType::default().slice().len();
    let mut num_histograms: usize = length.wrapping_div(literals_per_histogram).wrapping_add(1);
    if num_histograms > max_histograms {
        num_histograms = max_histograms;
    }
    if length == 0usize {
        split.num_types = 1;
        return;
    } else if length < kMinLengthForBlockSplitting {
        {
            if split.types_alloc_size() < split.num_blocks.wrapping_add(1) {
                let mut _new_size: usize = if split.types_alloc_size() == 0usize {
                    split.num_blocks.wrapping_add(1)
                } else {
                    split.types_alloc_size()
                };

                while _new_size < split.num_blocks.wrapping_add(1) {
                    _new_size = _new_size.wrapping_mul(2);
                }
                let mut new_array = allocate::<u8, _>(alloc, _new_size);
                new_array.slice_mut()[..split.types_alloc_size()]
                    .copy_from_slice(&split.types.slice()[..split.types_alloc_size()]);
                <Alloc as Allocator<u8>>::free_cell(
                    alloc,
                    core::mem::replace(&mut split.types, new_array),
                );
            }
        }
        {
            if split.lengths_alloc_size() < split.num_blocks.wrapping_add(1) {
                let mut _new_size: usize = if split.lengths_alloc_size() == 0usize {
                    split.num_blocks.wrapping_add(1)
                } else {
                    split.lengths_alloc_size()
                };
                while _new_size < split.num_blocks.wrapping_add(1) {
                    _new_size = _new_size.wrapping_mul(2);
                }
                let mut new_array = allocate::<u32, _>(alloc, _new_size);
                new_array.slice_mut()[..split.lengths_alloc_size()]
                    .copy_from_slice(&split.lengths.slice()[..split.lengths_alloc_size()]);
                <Alloc as Allocator<u32>>::free_cell(
                    alloc,
                    core::mem::replace(&mut split.lengths, new_array),
                );
            }
        }
        split.num_types = 1;
        split.types.slice_mut()[split.num_blocks] = 0u8;
        split.lengths.slice_mut()[split.num_blocks] = length as u32;
        split.num_blocks = split.num_blocks.wrapping_add(1);
        return;
    }
    let mut histograms = allocate::<HistogramType, _>(alloc, num_histograms);

    InitialEntropyCodes(
        data,
        length,
        sampling_stride_length,
        num_histograms,
        histograms.slice_mut(),
    );
    RefineEntropyCodes(
        data,
        length,
        sampling_stride_length,
        num_histograms,
        histograms.slice_mut(),
    );
    {
        let mut num_blocks: usize = 0usize;
        let bitmaplen: usize = num_histograms.wrapping_add(7) >> 3;
        let mut block_workspace = allocate::<u8, _>(alloc, length * (bitmaplen + 1));
        let (block_ids, switch_signal) = block_workspace.slice_mut().split_at_mut(length);
        let mut insert_cost = allocate::<floatX, _>(alloc, data_size.wrapping_mul(num_histograms));
        let mut cost = allocate::<Mem256f, _>(alloc, ((num_histograms + 7) >> 3));
        let mut new_id = allocate::<u16, _>(alloc, num_histograms);
        let iters: usize = (if params.quality <= 11 { 3i32 } else { 10i32 }) as usize;
        for _i in 0usize..iters {
            num_blocks = FindBlocks(
                data,
                length,
                block_switch_cost,
                num_histograms,
                histograms.slice_mut(),
                insert_cost.slice_mut(),
                cost.slice_mut(),
                switch_signal,
                block_ids,
            );
            num_histograms = RemapBlockIds(block_ids, length, new_id.slice_mut(), num_histograms);
            BuildBlockHistograms(
                data,
                length,
                block_ids,
                num_histograms,
                histograms.slice_mut(),
            );
        }
        <Alloc as Allocator<floatX>>::free_cell(alloc, insert_cost);
        <Alloc as Allocator<Mem256f>>::free_cell(alloc, cost);
        <Alloc as Allocator<u16>>::free_cell(alloc, new_id);
        <Alloc as Allocator<HistogramType>>::free_cell(alloc, histograms);
        ClusterBlocks::<HistogramType, Alloc, IntegerType>(
            alloc,
            data,
            length,
            num_blocks,
            scratch_space,
            block_ids,
            split,
        );
        <Alloc as Allocator<u8>>::free_cell(alloc, block_workspace);
    }
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn BrotliSplitBlock<
    Alloc: alloc::Allocator<u8>
        + alloc::Allocator<u16>
        + alloc::Allocator<u32>
        + alloc::Allocator<floatX>
        + alloc::Allocator<Mem256f>
        + alloc::Allocator<HistogramLiteral>
        + alloc::Allocator<HistogramCommand>
        + alloc::Allocator<HistogramDistance>
        + alloc::Allocator<HistogramPair>,
>(
    alloc: &mut Alloc,
    cmds: &[Command],
    num_commands: usize,
    data: &[u8],
    pos: usize,
    mask: usize,
    params: &BrotliEncoderParams,
    lit_scratch_space: &mut <HistogramLiteral as CostAccessors>::i32vec,
    cmd_scratch_space: &mut <HistogramCommand as CostAccessors>::i32vec,
    dst_scratch_space: &mut <HistogramDistance as CostAccessors>::i32vec,
    literal_split: &mut BlockSplit<Alloc>,
    insert_and_copy_split: &mut BlockSplit<Alloc>,
    dist_split: &mut BlockSplit<Alloc>,
) {
    {
        /*for (i, cmd) in cmds[..num_commands].iter().enumerate() {
            println_stderr!("C {:} {:} {:} {:} {:} {:}",
                            i, cmd.insert_len_, cmd.copy_len_, cmd.dist_extra_, cmd.cmd_prefix_, cmd.dist_prefix_);
        }*/
        let literals_count: usize = CountLiterals(cmds, num_commands);
        let mut literals = allocate::<u8, _>(alloc, literals_count);
        CopyLiteralsToByteArray(cmds, num_commands, data, pos, mask, literals.slice_mut());
        SplitByteVector::<HistogramLiteral, Alloc, u8>(
            alloc,
            literals.slice(),
            literals_count,
            kSymbolsPerLiteralHistogram,
            kMaxLiteralHistograms,
            kLiteralStrideLength,
            kLiteralBlockSwitchCost,
            params,
            lit_scratch_space,
            literal_split,
        );
        <Alloc as Allocator<u8>>::free_cell(alloc, literals);
    }
    let mut prefix_codes = allocate::<u16, _>(alloc, num_commands);
    {
        for (code, cmd) in prefix_codes.slice_mut().iter_mut().zip(cmds.iter()) {
            *code = cmd.cmd_prefix_;
        }
        SplitByteVector::<HistogramCommand, Alloc, u16>(
            alloc,
            prefix_codes.slice(),
            num_commands,
            kSymbolsPerCommandHistogram,
            kMaxCommandHistograms,
            kCommandStrideLength,
            kCommandBlockSwitchCost,
            params,
            cmd_scratch_space,
            insert_and_copy_split,
        );
    }
    {
        let mut j: usize = 0usize;
        for cmd in cmds.iter().take(num_commands) {
            if cmd.copy_len() != 0 && cmd.cmd_prefix_ >= 128 {
                prefix_codes.slice_mut()[j] = cmd.dist_prefix_ & 0x03ff;
                j = j.wrapping_add(1);
            }
        }
        SplitByteVector::<HistogramDistance, Alloc, u16>(
            alloc,
            prefix_codes.slice(),
            j,
            kSymbolsPerDistanceHistogram,
            kMaxCommandHistograms,
            kCommandStrideLength,
            kDistanceBlockSwitchCost,
            params,
            dst_scratch_space,
            dist_split,
        );
    }
    <Alloc as Allocator<u16>>::free_cell(alloc, prefix_codes);
}

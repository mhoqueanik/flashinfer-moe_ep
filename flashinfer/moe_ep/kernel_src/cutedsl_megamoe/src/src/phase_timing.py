# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""clock64 phase-timing slot map for the fused MegaMoE kernel.

A DSL-4.5.2-compatible fallback for IKET: each warp role accumulates
``%clock64`` deltas for its phases in registers and stores them once, at the
end of its role body, into a per-CTA slot row of the ``phase_timing`` local
workspace region (store-only; each launch overwrites, so the values read
back after a run are the LAST launch's).  The slot layout mirrors the
iket.range_push/range_pop marker names so a future run-iket trace and this
breakdown are cross-comparable.

Layout: ``(PT_MAX_CTAS, PT_NUM_SLOTS)`` Int64 cycles, row = cta_linear_id.

Enabled by ``MegaMoENvfp4Config.enable_phase_timing`` (compile-key member;
default off -> the region exists but no timing code is generated, keeping
default-path codegen byte-identical).
"""

PT_MAX_CTAS = 256
PT_NUM_SLOTS = 32

# Slot indices.  Grouped by the warp role that OWNS (writes) the slot.
# "accum" slots sum deltas across the role's work loop; the rest are one-shot.
PT = {
    # sched warp (w7)
    "sched_pre_init_wait": 0,  # cross-rank dispatch arrival (Sched_PreInit_Wait)
    "sched_loop_total": 1,  # gen/publish loop, first work to sentinel
    "sched_publish": 2,  # accum: producer_acquire backpressure
    # TMA-A warp (w5)
    "tma_a_consume_work": 3,  # accum: idle waiting for scheduled work
    "tma_a_loop_total": 4,
    # TMA-B warp (w6)
    "tma_b_consume_work": 5,  # accum
    "tma_b_fc1_wait": 6,  # accum: dispatch->fc1 token arrival spin
    "tma_b_fc2_wait": 7,  # accum: fc1->fc2 handoff spin
    "tma_b_loop_total": 8,
    # MMA warp (w4)
    "mma_consume_work": 9,  # accum
    "mma_fc1": 10,  # accum: fc1 mainloop
    "mma_fc2": 11,  # accum: fc2 mainloop
    "mma_loop_total": 12,
    # epilogue warps (w0-3; warp 0 lane 0 stores)
    "epi_consume_work": 13,  # accum
    "epi_fc1_wait": 14,  # accum: acc TMEM consumer_wait inside the fc1 call
    "epi_fc1": 15,  # accum: WHOLE fc1 epilogue call (wait included; work = 15-14)
    "epi_fc2_wait": 16,  # accum: consumer_wait inside the fc2 call
    "epi_fc2": 17,  # accum: WHOLE fc2 call incl. combine STG (work = 17-16)
    "epi_drain_barrier": 18,  # accum: TMA drain + epi named barrier
    "epi_flag": 19,  # accum: fc1/fc2 done-flag publish
    "epi_loop_total": 20,
    # dispatch warps (w8-11; local warp 0 lane 0 stores)
    "dispatch_prep": 21,
    "dispatch_barrier": 22,  # cross-rank send-count exchange
    "dispatch_pull": 23,  # per-token pull loop
    "dispatch_total": 24,
    # kernel tail (dispatch local warp 0 lane 0)
    "tail_rendezvous": 25,  # all-warp NamedBarrier (thread 0's view)
    "tail_nvlink_drain": 26,
    "tail_shared_reset": 27,
    "tail_nvlink_publish": 28,
    "tail_local_reset": 29,
    # whole kernel (thread 0)
    "kernel_total": 30,
}

PT_ACCUM_SLOTS = {
    "sched_publish",
    "tma_a_consume_work",
    "tma_b_consume_work",
    "tma_b_fc1_wait",
    "tma_b_fc2_wait",
    "mma_consume_work",
    "mma_fc1",
    "mma_fc2",
    "epi_consume_work",
    "epi_fc1_wait",
    "epi_fc1",
    "epi_fc2_wait",
    "epi_fc2",
    "epi_drain_barrier",
    "epi_flag",
}

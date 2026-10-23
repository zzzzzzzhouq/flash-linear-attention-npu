/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 L1<->UB chunk_fwd_o path.
 *
 * Stage1/2: validated (precision passed); do not regress.
 * Stage2 (AIC): L0/L1 streamSlot ping-pong + ordered CrossCore ready chains.
 * AIV stages run in separate head rounds: S1(all heads), S3(all heads), then S5(all heads).
 * Stage3 uses mode=0x2 pair handshakes and independent A-prime/V-to-MTE3 ping-pong,
 *   without a global barrier.
 * Stage4 (AIC): A-prime/V MTE2 prefetch + independent L0 ping-pong + FP32 Fixpipe to owner AIV.
 * Stage5 (AIV): owner fuses O_s-prime/O_l in FP32 and writes bf16 O through MTE3.
 */

#ifndef CHUNK_FWD_O_ARCH35_A5_H
#define CHUNK_FWD_O_ARCH35_A5_H

#include "kernel_operator.h"
#include "kernel_utils/gdn_timer/AscendTimerV2_scope.hpp"
#include "../chunk_fwd_o_struct.h"
#include "chunk_fwd_o_common.h"
#include "chunk_fwd_o_cube.h"
#include "chunk_fwd_o_vector.h"

namespace GDN {

template <typename GT, bool UseExp2>
class ChunkFwdOA5 {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g, GM_ADDR cuSeqlens,
                                GM_ADDR chunkOffsets, GM_ADDR o, GM_ADDR workspace,
                                const ChunkFwdOTilingData *tilingData, GM_ADDR timerAddr = nullptr)
    {
        q_ = q;
        k_ = k;
        v_ = v;
        h_ = h;
        g_ = g;
        cuSeqlens_ = cuSeqlens;
        chunkOffsets_ = chunkOffsets;
        o_ = o;
        workspace_ = workspace;
        tiling_ = *tilingData;
        // 设备侧计时器：本 launch 独占 timer 张量�?SLICE_FWD_O 切片�?        // timerAddr 为空时所�?Tik/Tok 直接返回，数值路径不变�?        if (timerAddr != nullptr) {
            timer_.Init(timerAddr + AscendTimer::SliceOffsetWords(AscendTimer::SLICE_FWD_O) *
                                        static_cast<int64_t>(sizeof(int64_t)));
            timer_.TikNoBarrier();
            timerKernelStartCycle_ = timer_.GetStartCycle();
        }
        // Init -> InitDone 记为 MAIN_INIT�?        timer_.TokNoBarrier<Overwrite>(AscendTimer::MAIN_INIT);
    }

    __aicore__ inline void Process()
    {
        timer_.TikNoBarrier(); // MAIN_BODY 起点（InitDone�?        ProcessCore();
        // 收尾：MAIN_BODY -> �?stage 实际迭代次数 -> �?barrier �?KERNEL 总窗口�?        timer_.TokNoBarrier<Overwrite>(AscendTimer::MAIN_BODY);
        timer_.PublishCounts();
        timer_.Tik(timerKernelStartCycle_);
        timer_.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
    }

private:
    __aicore__ inline void ProcessCore()
    {
        const uint32_t aicCoreIdx = AscendC::GetBlockIdx();
        const uint32_t aicCoreNum = AscendC::GetBlockNum();
        const uint32_t aivCoreIdx = AscendC::GetBlockIdx() / 2U;
        const uint32_t aivCoreNum = AscendC::GetBlockNum();

        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            ChunkFwdOA5VectorProcess<GT, UseExp2> vector(q_, k_, v_, h_, g_, cuSeqlens_, chunkOffsets_, o_,
                                                       workspace_);
            vector.Init(tiling_, &pipe, &timer_);
            vector.Process(aivCoreIdx, aivCoreNum);
            return;
        }

        if ASCEND_IS_AIC {
            ChunkFwdOA5CubeProcess cube(q_, k_, v_, h_, g_, cuSeqlens_, chunkOffsets_, o_, workspace_);
            cube.Init(tiling_, &timer_);
            cube.Process(aicCoreIdx, aicCoreNum);
        }
    }

    GM_ADDR q_ = nullptr;
    GM_ADDR k_ = nullptr;
    GM_ADDR v_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR g_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkOffsets_ = nullptr;
    GM_ADDR o_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    ChunkFwdOTilingData tiling_{};
    // 计时（SLICE_FWD_O 切片）；timer 为空张量时整类退化为无操作�?    AscendTimerDevice timer_;
    int64_t timerKernelStartCycle_ = 0;
};

template <typename GT, bool UseExp2>
__aicore__ inline void ChunkFwdOA5Dispatch(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                           GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o, GM_ADDR workspace,
                                           const ChunkFwdOTilingData *tilingData, GM_ADDR timer)
{
    ChunkFwdOA5<GT, UseExp2> op;
    op.Init(q, k, v, h, g, cuSeqlens, chunkOffsets, o, workspace, tilingData, timer);
    op.Process();
}

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_A5_H

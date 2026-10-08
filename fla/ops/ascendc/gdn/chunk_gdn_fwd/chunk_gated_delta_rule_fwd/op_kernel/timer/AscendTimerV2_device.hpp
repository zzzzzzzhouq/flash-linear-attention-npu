/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * chunk_gated_delta_rule_fwd NPU 端计时器实现。
 * 移植自 vllm-ascend fused_sparse_attention_overlap 的 AscendTimerV2_device，
 * 去除 MergeKV 专用的延迟累加（defer）机制；本算子动态项粒度为调度任务级，
 * 直接使用 TikNoBarrier/TokNoBarrier 逐迭代写槽即可。
 */
#ifndef CHUNK_GATED_DELTA_RULE_FWD_TIMER_DEVICE_H
#define CHUNK_GATED_DELTA_RULE_FWD_TIMER_DEVICE_H

#include "AscendTimerV2.hpp"
#include "kernel_operator.h"
#include <type_traits>

#define GDN_TIMER_TOTAL_BUFFER_SIZE \
    (GdnTimer::N_TIMING_COUNTER + GdnTimer::DYNAMIC_ITER_TOTAL_SIZE)

// ========== 1. NPU 端专属类型定义 ==========
typedef struct AccumulateTag_ Accumulate;
struct AccumulateTag_
{
};
typedef struct OverwriteTag_ Overwrite;
struct OverwriteTag_
{
};

class AscendTimerDevice
{
public:
    GM_ADDR t_addr;
    int64_t start_cycle;
    int64_t dynamic_start_cycles[GdnTimer::DYNAMIC_TYPE_COUNT];
    int dynamic_start_iters[GdnTimer::DYNAMIC_TYPE_COUNT];
    bool dynamic_valid[GdnTimer::DYNAMIC_TYPE_COUNT];
    int block_id;
    int r;
    int group_id;
    int subblockid;
    int core_id; // 组内展开后的核编号：AIC=group*3，AIV=group*3+1+subblock

    // ========== 2. 构造函数 ==========
    __aicore__ inline AscendTimerDevice()
    {
        t_addr = NULL;
        start_cycle = 0;
        for (int i = 0; i < GdnTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = 0;
            dynamic_start_iters[i] = -1;
            dynamic_valid[i] = false;
        }
        block_id = 0;
        r = 0;
        group_id = 0;
        subblockid = 0;
        core_id = 0;
    }

    __aicore__ inline AscendTimerDevice(GM_ADDR timeAddr)
    {
        t_addr = timeAddr;
        start_cycle = 0;
        for (int i = 0; i < GdnTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = 0;
            dynamic_start_iters[i] = -1;
            dynamic_valid[i] = false;
        }
        block_id = AscendC::GetBlockIdx();
        r = AscendC::GetTaskRation();
        group_id = block_id;
        core_id = block_id;

        if ASCEND_IS_AIV
        {
            group_id = group_id / r;
            subblockid = AscendC::GetSubBlockIdx();
            core_id = group_id * 3 + subblockid + 1; // {1,2}, {4,5}
        }
        else
        {
            subblockid = 0;
            core_id = group_id * 3; // 0,3,6
        }
    }

    // ========== 3. 拷贝构造 ==========
    __aicore__ inline AscendTimerDevice(const AscendTimerDevice &other)
    {
        t_addr = other.t_addr;
        start_cycle = other.start_cycle;
        for (int i = 0; i < GdnTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = other.dynamic_start_cycles[i];
            dynamic_start_iters[i] = other.dynamic_start_iters[i];
            dynamic_valid[i] = other.dynamic_valid[i];
        }
        block_id = other.block_id;
        r = other.r;
        group_id = other.group_id;
        subblockid = other.subblockid;
        core_id = other.core_id;
    }

    // ========== 4. 初始化（构造后手动注入地址时使用） ==========
    __aicore__ inline void Init(const GM_ADDR timeAddr)
    {
        t_addr = (GM_ADDR)timeAddr;
        start_cycle = 0;
        for (int i = 0; i < GdnTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = 0;
            dynamic_start_iters[i] = -1;
            dynamic_valid[i] = false;
        }
        block_id = AscendC::GetBlockIdx();
        r = AscendC::GetTaskRation();
        group_id = block_id;
        core_id = block_id;

        if ASCEND_IS_AIV
        {
            group_id = group_id / r;
            subblockid = AscendC::GetSubBlockIdx();
            core_id = group_id * 3 + subblockid + 1;
        }
        else
        {
            subblockid = 0;
            core_id = group_id * 3;
        }
    }

    // ========== 5. 基础接口 ==========
    __aicore__ __inline__ GM_ADDR GetTAddr() { return t_addr; }
    __aicore__ __inline__ int64_t GetStartCycle() { return start_cycle; }
    __aicore__ __inline__ int GetCoreId() { return core_id; }

    // ========== 6. 设置当前核心指定动态项的实际迭代次数（供解析端截断列） ==========
    __aicore__ __inline__ void setDynamicActualIter(GdnTimer::DynamicTimingType type, int64_t iter)
    {
        if (t_addr == nullptr) { return; }
        if (iter > GdnTimer::MAX_DYNAMIC_ITER) {
            AscendC::printf("[WARNING] NPU端动态项 %d 实际迭代次数 %d 超出上限 %d，已截断为 %d\n",
                            (int)type, (int)iter, GdnTimer::MAX_DYNAMIC_ITER, GdnTimer::MAX_DYNAMIC_ITER);
            iter = GdnTimer::MAX_DYNAMIC_ITER;
        }
        if (core_id < 0 || core_id >= GdnTimer::N_CORE_COUNT) {
            return;
        }
        if ((int)type < 0 || (int)type >= GdnTimer::DYNAMIC_TYPE_COUNT) {
            return;
        }

        int32_t idx = GdnTimer::N_TIMING_COUNTER + core_id * GdnTimer::DYNAMIC_ITER_PER_CORE_ALIGN + (int)type;

        if (idx >= GDN_TIMER_TOTAL_BUFFER_SIZE) {
            AscendC::printf("[WARNING] NPU端动态迭代次数索引 %d 超出总缓冲区大小 %d\n",
                            idx, GDN_TIMER_TOTAL_BUFFER_SIZE);
            return;
        }

        AscendC::GlobalTensor<int64_t> iterTensor;
        iterTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        iterTensor(idx) = iter;
    }

    // ========== 7. 开始计时（固定项：记录到 start_cycle） ==========
    __aicore__ __inline__ void Tik()
    {
        if (t_addr == nullptr) { return; }
        AscendC::PipeBarrier<PIPE_ALL>();
        start_cycle = AscendC::GetSystemCycle();
    }

    // 无屏障 Tik：仅记录下发时刻，不排空流水（阶段级计时统一使用）
    __aicore__ __inline__ void TikNoBarrier() {
        if (t_addr == nullptr) { return; }
        start_cycle = AscendC::GetSystemCycle();
    }

    // 无屏障 Tik 动态项版：iter 为本核调度任务号
    __aicore__ __inline__ void TikNoBarrier(GdnTimer::DynamicTimingType type, int iter)
    {
        if (t_addr == nullptr) { return; }
        if ((int)type < 0 || (int)type >= GdnTimer::DYNAMIC_TYPE_COUNT || iter < 0 || iter >= GdnTimer::MAX_DYNAMIC_ITER) {
            if (iter == GdnTimer::MAX_DYNAMIC_ITER) {   // 只在第一次溢出时打，天然去重
                AscendC::printf("[WARNING] NPU端动态项 %d 迭代 %d 超出上限 %d，该迭代起不再记录计时\n",
                                (int)type, iter, GdnTimer::MAX_DYNAMIC_ITER);
            }
            return;
        }

        int64_t start = AscendC::GetSystemCycle();          // 无 PipeBarrier
        dynamic_start_cycles[(int)type] = start;
        dynamic_start_iters[(int)type] = iter;
        dynamic_valid[(int)type] = true;
    }

    // ========== 8. 结束计时 ==========
    template <typename ModeTag = Accumulate>
    __aicore__ __inline__ void Tok(int32_t timing_idx)
    {
        if (t_addr == nullptr) { return; }
        if (timing_idx >= GdnTimer::N_TIMING_ITEM_PER_CORE)
        {
            AscendC::printf("[WARNING] NPU端Tok索引 %d 超出每核最大计时项数 %d\n",
                            timing_idx, GdnTimer::N_TIMING_ITEM_PER_CORE);
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();
        int64_t elapsed_cycle = end_cycle - start_cycle;

        if constexpr (std::is_same_v<ModeTag, Accumulate>)
        {
            accumulate_time(timing_idx, start_cycle, elapsed_cycle);
        }
        else if constexpr (std::is_same_v<ModeTag, Overwrite>)
        {
            write_time(timing_idx, start_cycle, end_cycle);
        }
    }

    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void Tok()
    {
        Tok<ModeTag>(GdnTimer::KERNEL_TIMING_IDX);
    }

    // 无屏障 Tok 固定项版：以最近一次 TikNoBarrier() 的 start_cycle 为起点
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void TokNoBarrier(int32_t timing_idx)
    {
        if (t_addr == nullptr) { return; }
        if (timing_idx >= GdnTimer::N_TIMING_ITEM_PER_CORE) { return; }
        int64_t end_cycle = AscendC::GetSystemCycle();
        int64_t elapsed_cycle = end_cycle - start_cycle;
        if constexpr (std::is_same_v<ModeTag, Accumulate>) { accumulate_time(timing_idx, start_cycle, elapsed_cycle); }
        else { write_time(timing_idx, start_cycle, end_cycle); }
    }

    // 无屏障 Tok 动态项版：与 TikNoBarrier(type, iter) 配对
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void TokNoBarrier(GdnTimer::DynamicTimingType type, int iter)
    {
        if (t_addr == nullptr) { return; }
        int timing_idx = GdnTimer::FIXED_TIMING_COUNT + (int)type * GdnTimer::MAX_DYNAMIC_ITER + iter;
        if (timing_idx >= GdnTimer::N_TIMING_ITEM_PER_CORE) { return; }
        if (!dynamic_valid[(int)type] || dynamic_start_iters[(int)type] != iter) { return; }  // 防错配
        int64_t end_cycle = AscendC::GetSystemCycle();      // 无 PipeBarrier
        int64_t start = dynamic_start_cycles[(int)type];
        int64_t elapsed_cycle = end_cycle - start;
        if constexpr (std::is_same_v<ModeTag, Accumulate>) {
            accumulate_time(timing_idx, start, elapsed_cycle);
        } else {
            write_time(timing_idx, start, end_cycle);
        }
        dynamic_valid[(int)type] = false;
        dynamic_start_iters[(int)type] = -1;
    }

private:
    // ========== 9. 核内索引计算 ==========
    __aicore__ __inline__ int32_t calculate_index(int32_t idx)
    {
        if (idx >= GdnTimer::N_TIMING_ITEM_PER_CORE)
        {
            AscendC::printf("[WARNING] NPU端计算索引时，输入idx %d 超出每核最大计时项数 %d\n",
                            idx, GdnTimer::N_TIMING_ITEM_PER_CORE);
            return 0;
        }

        int32_t adjusted_idx = 0;
        if ASCEND_IS_AIV
        {
            adjusted_idx = group_id * GdnTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3 +
                GdnTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * (1 + subblockid) + idx * 2;
        }
        else
        {
            adjusted_idx = group_id * GdnTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3 + idx * 2;
        }

        if (adjusted_idx + 1 >= GdnTimer::N_TIMING_COUNTER)
        {
            AscendC::printf("[WARNING] NPU端计算后索引 %d 超出总对齐最大值 %d\n",
                            adjusted_idx, GdnTimer::N_TIMING_COUNTER);
            return 0;
        }

        return adjusted_idx;
    }

    // ========== 10. 累加/覆盖计时值 ==========
    __aicore__ __inline__ void accumulate_time(int32_t idx, int64_t start, int64_t time)
    {
        int32_t adjusted_idx = calculate_index(idx);
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        int64_t current_start = timeTensor(adjusted_idx);
        int64_t current_end = timeTensor(adjusted_idx + 1);
        if (current_start == 0 && current_end == 0) {
            timeTensor(adjusted_idx) = start;
            timeTensor(adjusted_idx + 1) = start + time;
        } else {
            timeTensor(adjusted_idx + 1) = current_end + time;
        }
    }

    __aicore__ __inline__ void write_time(int32_t idx, int64_t start, int64_t end)
    {
        int32_t adjusted_idx = calculate_index(idx);
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        timeTensor(adjusted_idx) = start;
        timeTensor(adjusted_idx + 1) = end;
    }
};

#endif // CHUNK_GATED_DELTA_RULE_FWD_TIMER_DEVICE_H

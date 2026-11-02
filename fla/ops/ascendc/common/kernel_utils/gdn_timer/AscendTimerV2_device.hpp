#ifndef ASCEND_TIMER_DEVICE_H
#define ASCEND_TIMER_DEVICE_H

#include "AscendTimerV2.hpp"
#include <type_traits>

#define ASCEND_TIMER_TOTAL_BUFFER_SIZE (AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE)

// ========== 1. NPU端专属类型定义 ==========
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
    // ========== 2. 成员变量 ==========
    GM_ADDR t_addr;
    int64_t start_cycle;
    int64_t dynamic_start_cycles[AscendTimer::DYNAMIC_TYPE_COUNT > 0 ? AscendTimer::DYNAMIC_TYPE_COUNT : 1];
    int dynamic_start_iters[AscendTimer::DYNAMIC_TYPE_COUNT > 0 ? AscendTimer::DYNAMIC_TYPE_COUNT : 1];
    bool dynamic_valid[AscendTimer::DYNAMIC_TYPE_COUNT > 0 ? AscendTimer::DYNAMIC_TYPE_COUNT : 1];
    int block_id;
    int r;
    int group_id;
    int subblockid;
    int core_id; // 当前核心ID

    // ========== 3. 构造函数 ==========
    __aicore__ inline AscendTimerDevice()
    {
        t_addr = NULL;
        start_cycle = 0;
        for (int i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = 0;
            dynamic_start_iters[i] = -1;
            dynamic_valid[i] = false;
            next_iter_[i] = 0;
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
        for (int i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = 0;
            dynamic_start_iters[i] = -1;
            dynamic_valid[i] = false;
            next_iter_[i] = 0;
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

    // ========== 4. 拷贝构造 ==========
    __aicore__ inline AscendTimerDevice(const AscendTimerDevice &other)
    {
        t_addr = other.t_addr;
        start_cycle = other.start_cycle;
        for (int i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
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

    // ========== 5. 初始化函数 ==========
    __aicore__ inline void Init(const GM_ADDR timeAddr)
    {
        t_addr = (GM_ADDR)timeAddr;
        for (int i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = 0;
            dynamic_start_iters[i] = -1;
            dynamic_valid[i] = false;
            next_iter_[i] = 0;
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

    __aicore__ inline void Init(const GM_ADDR timeAddr, int64_t startCycle)
    {
        Init(timeAddr);
        start_cycle = startCycle;
    }

    // ========== 6. 基础接口 ==========
    __aicore__ __inline__ GM_ADDR GetTAddr() { return t_addr; }
    __aicore__ __inline__ int64_t GetStartCycle() { return start_cycle; }
    __aicore__ __inline__ int GetCoreId() { return core_id; } // 获取当前核心ID

    // ========== 7. 初始化计时值为0 ==========
    __aicore__ __inline__ void zero(int32_t timing_idx)
    {
        if (t_addr == nullptr) { return; }
        if (timing_idx < 0 || timing_idx >= AscendTimer::N_TIMING_ITEM_PER_CORE)
        {
            AscendC::printf("[WARNING] NPU端计时索引 %d 超出每核最大计时项数 %d\n",
                            timing_idx, AscendTimer::N_TIMING_ITEM_PER_CORE);
            return;
        }
        int32_t adjusted_idx = calculate_index(timing_idx);
        if (adjusted_idx < 0) { return; }
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        timeTensor(adjusted_idx) = 0;
        timeTensor(adjusted_idx + 1) = 0;
    }

    __aicore__ __inline__ void zero()
    {
        zero(AscendTimer::KERNEL_TIMING_IDX);
    }

    // ========== 8. 设置当前核心指定动态项的实际迭代次数 ==========
    __aicore__ __inline__ void setDynamicActualIter(AscendTimer::DynamicTimingType type, int64_t iter)
    {
        if (t_addr == nullptr || iter < 0) { return; }
        // 校验iter不可以超出上限
        if (iter > AscendTimer::MAX_DYNAMIC_ITER) {
            AscendC::printf("[WARNING] NPU端动态项 %d 实际迭代次数 %d 超出上限 %d，已截断为 %d\n", (int)type, (int)iter, AscendTimer::MAX_DYNAMIC_ITER, AscendTimer::MAX_DYNAMIC_ITER);
            iter = AscendTimer::MAX_DYNAMIC_ITER;
        }
        // 校验核心ID和动态项类型
        if (core_id < 0 || core_id >= AscendTimer::N_CORE_COUNT)
        {
            AscendC::printf("[WARNING] NPU端核心ID %d 超出范围（0-%d）\n", core_id, AscendTimer::N_CORE_COUNT - 1);
            return;
        }
        if ((int)type < 0 || (int)type >= AscendTimer::DYNAMIC_TYPE_COUNT)
        {
            AscendC::printf("[WARNING] NPU端动态项类型 %d 超出范围（0-%d）\n", (int)type, AscendTimer::DYNAMIC_TYPE_COUNT - 1);
            return;
        }

        int32_t idx = AscendTimer::N_TIMING_COUNTER + core_id * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN + (int)type;

        // 校验总偏移
        if (idx >= ASCEND_TIMER_TOTAL_BUFFER_SIZE)
        {
            AscendC::printf("[WARNING] NPU端动态迭代次数索引 %d 超出总缓冲区大小 %d\n",
                            idx, ASCEND_TIMER_TOTAL_BUFFER_SIZE);
            return;
        }

        // 设置迭代次数
        AscendC::GlobalTensor<int64_t> iterTensor;
        iterTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        iterTensor(idx) = iter;
    }

    // ========== 9. 初始化当前核心所有动态项的迭代次数为0 ==========
    __aicore__ __inline__ void zeroDynamicActualIterAll()
    {
        if (t_addr == nullptr) { return; }
        if (core_id < 0 || core_id >= AscendTimer::N_CORE_COUNT)
        {
            AscendC::printf("[WARNING] NPU端核心ID %d 超出范围（0-%d）\n", core_id, AscendTimer::N_CORE_COUNT - 1);
            return;
        }

        int32_t coreOffset = AscendTimer::N_TIMING_COUNTER + core_id * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN;
        // 初始化当前核心的动态迭代次数区（仅初始化有效动态项，对齐部分留空）
        for (int type = 0; type < AscendTimer::DYNAMIC_TYPE_COUNT; ++type)
        {
            int32_t idx = coreOffset + type;
            if (idx >= ASCEND_TIMER_TOTAL_BUFFER_SIZE)
                break;
            AscendC::GlobalTensor<int64_t> iterTensor;
            iterTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
            iterTensor(idx) = 0;
        }
    }
    // ========== 9.1 清零本核全部动态计时槽位（跨launch防脏累加） ==========
    __aicore__ __inline__ void zeroDynamicTimingAll()
    {
        if (t_addr == nullptr) { return; }
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        for (int type = 0; type < AscendTimer::DYNAMIC_TYPE_COUNT; ++type) {
            for (int iter = 0; iter < AscendTimer::MAX_DYNAMIC_ITER; ++iter) {
                int32_t timing_idx =
                    AscendTimer::FIXED_TIMING_COUNT + type * AscendTimer::MAX_DYNAMIC_ITER + iter;
                int32_t adjusted_idx = calculate_index(timing_idx);
                if (adjusted_idx < 0) { return; }
                timeTensor(adjusted_idx) = 0;
                timeTensor(adjusted_idx + 1) = 0;
            }
        }
    }
    // ========== 10. 开始计时 ==========
    __aicore__ __inline__ void Tik()
    {
        if (t_addr == nullptr) { return; }
        AscendC::PipeBarrier<PIPE_ALL>();
        start_cycle = AscendC::GetSystemCycle();
    }

    __aicore__ __inline__ void Tik(int64_t start)
    {
        if (t_addr == nullptr) { return; }
        start_cycle = start;
    }

    __aicore__ __inline__ void Tik(AscendTimer::DynamicTimingType type, int iter)
    {
        if (t_addr == nullptr) { return; }
        if ((int)type < 0 || (int)type >= AscendTimer::DYNAMIC_TYPE_COUNT || iter < 0 || iter >= AscendTimer::MAX_DYNAMIC_ITER) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t start = AscendC::GetSystemCycle();
        dynamic_start_cycles[(int)type] = start;
        dynamic_start_iters[(int)type] = iter;
        dynamic_valid[(int)type] = true;

        int timing_idx = AscendTimer::FIXED_TIMING_COUNT + (int)type * AscendTimer::MAX_DYNAMIC_ITER + iter;
        int32_t adjusted_idx = calculate_index(timing_idx);
        if (adjusted_idx < 0) { return; }
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        //timeTensor(adjusted_idx) = start;
    }

    // ========== 无屏障 Tik：仅记录下发时刻，不排空流水 ==========
    __aicore__ __inline__ void TikNoBarrier() {
        if (t_addr == nullptr) { return; }
        start_cycle = AscendC::GetSystemCycle();
    }
    // ========== 无屏障 Tik DynamicTimingType版：仅记录下发时刻，不排空流水 ==========
    __aicore__ __inline__ void TikNoBarrier(AscendTimer::DynamicTimingType type, int iter)
    {
        if (t_addr == nullptr) { return; }
        if ((int)type < 0 || (int)type >= AscendTimer::DYNAMIC_TYPE_COUNT || iter < 0 || iter >= AscendTimer::MAX_DYNAMIC_ITER) {
            if (iter == AscendTimer::MAX_DYNAMIC_ITER) {   // 只在第一次溢出时打，天然去重
                AscendC::printf("[WARNING] NPU端动态项 %d 迭代 %d 超出上限 %d，该迭代起不再记录计时\n", (int)type, iter, AscendTimer::MAX_DYNAMIC_ITER);
            }
            return;
        }

        int64_t start = AscendC::GetSystemCycle();          // 无 PipeBarrier
        dynamic_start_cycles[(int)type] = start;
        dynamic_start_iters[(int)type] = iter;
        dynamic_valid[(int)type] = true;
        int timing_idx = AscendTimer::FIXED_TIMING_COUNT + (int)type * AscendTimer::MAX_DYNAMIC_ITER + iter;
        int32_t adjusted_idx = calculate_index(timing_idx);
        if (adjusted_idx < 0) { return; }
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        //timeTensor(adjusted_idx) = start;
    }

    // ========== 11. 结束计时 ==========
    template <typename ModeTag = Accumulate>
    __aicore__ __inline__ void Tok(int32_t timing_idx)
    {
        if (t_addr == nullptr) { return; }
        if (timing_idx < 0 || timing_idx >= AscendTimer::N_TIMING_ITEM_PER_CORE)
        {
            AscendC::printf("[WARNING] NPU端Tok索引 %d 超出每核最大计时项数 %d\n",
                            timing_idx, AscendTimer::N_TIMING_ITEM_PER_CORE);
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
        Tok<ModeTag>(AscendTimer::KERNEL_TIMING_IDX);
    }

    // 动态计时专用Tok（移除迭代次数更新）
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void Tok(AscendTimer::DynamicTimingType type, int iter)
    {
        if (t_addr == nullptr) { return; }
        if ((int)type < 0 || (int)type >= AscendTimer::DYNAMIC_TYPE_COUNT || iter < 0 || iter >= AscendTimer::MAX_DYNAMIC_ITER) { return; }
        // 索引计算
        int timing_idx = AscendTimer::FIXED_TIMING_COUNT + (int)type * AscendTimer::MAX_DYNAMIC_ITER + iter;

        // 校验索引
        if (timing_idx < 0 || timing_idx >= AscendTimer::N_TIMING_ITEM_PER_CORE)
        {
            AscendC::printf("[WARNING] NPU端动态项 %d 迭代 %d 索引 %d 超出每核最大计时项数 %d\n",
                            (int)type, iter, timing_idx, AscendTimer::N_TIMING_ITEM_PER_CORE);
            return;
        }

        if (!dynamic_valid[(int)type] || dynamic_start_iters[(int)type] != iter) { return; }

        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();
        int64_t start = dynamic_start_cycles[(int)type];
        int64_t elapsed_cycle = end_cycle - start;

        if constexpr (std::is_same_v<ModeTag, Accumulate>)
        {
            accumulate_time(timing_idx, start, elapsed_cycle);
        }
        else if constexpr (std::is_same_v<ModeTag, Overwrite>)
        {
            write_time(timing_idx, start, end_cycle);
        }
        dynamic_valid[(int)type] = false;
        dynamic_start_iters[(int)type] = -1;
    }

    // ========== 无屏障 Tok：不排空流水，由调用者在 WaitFlag 后调用表示异步操作已完成 ==========
    template <typename ModeTag = Accumulate>
    __aicore__ __inline__ void TokNoBarrier(int32_t timing_idx)
    {
        if (t_addr == nullptr) { return; }
        if (timing_idx < 0 || timing_idx >= AscendTimer::N_TIMING_ITEM_PER_CORE) { return; }
        int64_t end_cycle = AscendC::GetSystemCycle();
        int64_t elapsed_cycle = end_cycle - start_cycle;
        if constexpr (std::is_same_v<ModeTag, Accumulate>) { accumulate_time(timing_idx, start_cycle, elapsed_cycle); }
        else write_time(timing_idx, start_cycle, end_cycle);
    }
    // ========== 无屏障 Tok DynamicTimingType版：不排空流水，由调用者在 WaitFlag 后调用表示异步操作已完成 ==========
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void TokNoBarrier(AscendTimer::DynamicTimingType type, int iter)
    {
        if (t_addr == nullptr) { return; }
        if ((int)type < 0 || (int)type >= AscendTimer::DYNAMIC_TYPE_COUNT || iter < 0 || iter >= AscendTimer::MAX_DYNAMIC_ITER) { return; }
        int timing_idx = AscendTimer::FIXED_TIMING_COUNT + (int)type * AscendTimer::MAX_DYNAMIC_ITER + iter;
        if (timing_idx < 0 || timing_idx >= AscendTimer::N_TIMING_ITEM_PER_CORE) { return; }
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

    // ========== 12. 打印调试信息 ==========
    __aicore__ __inline__ void print()
    {
        AscendC::printf("AscendTimer: core_id=%d, block_id=%d, group_id=%d, subblockid=%d, t_addr=%p\n",
                        core_id, block_id, group_id, subblockid, t_addr);
    }

    // Optional per-core GM publication: call once after all final timestamps.
    __aicore__ __inline__ void Publish()
    {
#if ENABLE_TIMER && ASCEND_TIMER_EXPLICIT_GM_PUBLISH
        constexpr int cacheLineWords = 64 / sizeof(int64_t);
        static_assert(AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN % cacheLineWords == 0);
        static_assert(AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN % cacheLineWords == 0);
        if (t_addr == nullptr || core_id < 0 || core_id >= AscendTimer::N_CORE_COUNT ||
            reinterpret_cast<uint64_t>(t_addr) % 64 != 0) { return; }
        int row = calculate_index(0);
        if (row < 0) { return; }
        const int offsets[2] = {row, AscendTimer::N_TIMING_COUNTER + core_id * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN};
        const int sizes[2] = {AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN, AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN};
        for (int part = 0; part < 2; ++part) {
            AscendC::GlobalTensor<int64_t> data;
            data.SetGlobalBuffer((__gm__ int64_t *)t_addr + offsets[part]);
            for (int word = 0; word < sizes[part]; word += cacheLineWords) {
                AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                AscendC::DcciDst::CACHELINE_OUT>(data[word]);
            }
        }
#endif
    }

    // ========== 12.1 动态迭代槽位分配（每个 stage、每个核单调递增） ==========
    __aicore__ __inline__ int32_t NextIter(AscendTimer::DynamicTimingType type)
    {
        const int32_t t = static_cast<int32_t>(type);
        if (t < 0 || t >= AscendTimer::DYNAMIC_TYPE_COUNT) { return 0; }
        int32_t iter = next_iter_[t];
        if (iter >= AscendTimer::MAX_DYNAMIC_ITER) { iter = AscendTimer::MAX_DYNAMIC_ITER - 1; }
        else { next_iter_[t] = iter + 1; }
        return iter;
    }

    // kernel 收尾调用一次：把每个 stage 的实际调用次数写入计数区。
    __aicore__ __inline__ void PublishCounts()
    {
        for (int32_t i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
            if (next_iter_[i] > 0) {
                setDynamicActualIter(static_cast<AscendTimer::DynamicTimingType>(i), next_iter_[i]);
            }
        }
    }

private:
    int32_t next_iter_[AscendTimer::DYNAMIC_TYPE_COUNT];
    // ========== 13. 核内索引计算 ==========
    __aicore__ __inline__ int32_t calculate_index(int32_t idx)
    {
        if (core_id < 0 || core_id >= AscendTimer::N_CORE_COUNT || idx < 0 || idx >= AscendTimer::N_TIMING_ITEM_PER_CORE)
        {
            AscendC::printf("[WARNING] NPU端计算索引时，输入idx %d 超出每核最大计时项数 %d\n",
                            idx, AscendTimer::N_TIMING_ITEM_PER_CORE);
            return -1;
        }

        int32_t adjusted_idx = 0;
        if ASCEND_IS_AIV
        {
            adjusted_idx = group_id * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3 +
                AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * (1 + subblockid) + idx * 2;
        }
        else
        {
            adjusted_idx = group_id * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3 + idx * 2;
        }

        if (adjusted_idx + 1 >= AscendTimer::N_TIMING_COUNTER)
        {
            AscendC::printf("[WARNING] NPU端计算后索引 %d 超出总对齐最大值 %d\n",
                            adjusted_idx, AscendTimer::N_TIMING_COUNTER);
            return -1;
        }

        return adjusted_idx;
    }

    // ========== 14. 累加/覆盖计时值 ==========
    __aicore__ __inline__ void accumulate_time(int32_t idx, int64_t start, int64_t time)
    {
        int32_t adjusted_idx = calculate_index(idx);
        if (adjusted_idx < 0) { return; }
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
        if (adjusted_idx < 0) { return; }
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        timeTensor(adjusted_idx) = start;
        timeTensor(adjusted_idx + 1) = end;
    }
};

#endif // ASCEND_TIMER_DEVICE_H

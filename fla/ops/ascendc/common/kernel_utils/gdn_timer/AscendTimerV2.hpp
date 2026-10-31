#ifndef ASCEND_TIMER_COMMON_H
#define ASCEND_TIMER_COMMON_H

#include <stdint.h>

namespace AscendTimer
{
// CPU/NPU 共用布局。三个 kernel 各自使用一个完整 slice，不能共享核内槽位。
enum FixedTiming : int32_t
{
    KERNEL_TIMING_IDX = 0,
    FIXED_TIMING_COUNT
};

enum DynamicTimingType : int32_t
{
    S1_BATE = 0,
    S2_KKT,
    S3_MASK,
    S4_Y0,
    S4_Y1,
    S5_A,
    S6_VB_KBG,
    S7_WU,
    H_V0_CAST,
    H_C1_WH,
    H_V1_VALUE,
    H_C2_KG_RIGHT,
    H_V2_STATE,
    O_V1_GATE,
    O_C1_QK,
    O_C2_QH,
    O_V2_MASK,
    O_C3_ATTNV,
    O_V3_OUTPUT,
    DYNAMIC_TYPE_COUNT_ENUM
};

static constexpr int MAX_DYNAMIC_ITER = 1024;
static constexpr int GDN_TIMER_KERNEL_COUNT = 3;
static constexpr int GDN_PREPARE_TIMER_SLICE = 0;
static constexpr int GDN_FWD_H_TIMER_SLICE = 1;
static constexpr int GDN_FWD_O_TIMER_SLICE = 2;

// CPU端：对齐相关常量（128字节(16元素)对齐，避免多核写一个cache line）
// 每个计时项占 2 个 int64 槽位: [start_cycle, end_cycle]
static constexpr int N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER;
static constexpr int N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2;                                     // 每核原始计数器数
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;                // 16倍对齐后每核计数器数 128B对齐 每个核写入N_TIMING_COUNTER_PER_CORE_ALIGN个数
static constexpr int N_CORE_COUNT = 96; // 计时缓冲预留的核数上限（AIC+AIV），按目标硬件型号与分核策略调整，这里保留为96
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;                           // 总基础计数器数

// 动态迭代次数存储区对齐配置（每核动态项存储区16元素对齐）
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;                         // 动态项类型总数
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;    // 每核动态迭代次数区16对齐 128B对齐
static constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT; // 动态迭代次数总存储区大小，每个核cacheline不重叠
}

#define ASCEND_TIMER_TOTAL_BUFFER_SIZE (AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE)
#define GDN_TIMER_TOTAL_BUFFER_NUMEL (AscendTimer::GDN_TIMER_KERNEL_COUNT * ASCEND_TIMER_TOTAL_BUFFER_SIZE)
// ========== 4. 计时开关宏 ==========
#ifndef ENABLE_TIMER
#define ENABLE_TIMER 1 // 构建可覆盖为0；所有计时调用均用TIMER_BLOCK。
#endif
#ifndef ASCEND_TIMER_EXPLICIT_GM_PUBLISH
#define ASCEND_TIMER_EXPLICIT_GM_PUBLISH 0 // 仅确认目标发布需求/API/对齐后启用。
#endif
#if ENABLE_TIMER
#define TIMER_BLOCK(code) \
    do                    \
    {                     \
        code ;            \
    } while (0);
#else
#define TIMER_BLOCK(code) \
    do                    \
    {                     \
    } while (0);
#endif

#endif // ASCEND_TIMER_COMMON_H

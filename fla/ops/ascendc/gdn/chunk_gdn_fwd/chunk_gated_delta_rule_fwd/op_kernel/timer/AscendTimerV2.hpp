/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * chunk_gated_delta_rule_fwd 设备侧计时器（CPU/NPU 共用定义）。
 * 移植自 vllm-ascend fused_sparse_attention_overlap 的 AscendTimerV2，
 * 计时项按本算子 Phase6 融合 kernel 的阶段重新划分。
 *
 * 布局（int64 计数，每个计时项 2 槽 [start_cycle, end_cycle]）：
 *   [0, N_TIMING_COUNTER)          : 每核计时 start/end 对
 *   [N_TIMING_COUNTER, TOTAL_SIZE) : 每核动态项实际迭代次数
 */
#ifndef CHUNK_GATED_DELTA_RULE_FWD_TIMER_COMMON_H
#define CHUNK_GATED_DELTA_RULE_FWD_TIMER_COMMON_H

#include <stdint.h>

namespace GdnTimer
{

// ========== 1. 固定计时项（每核一份，Overwrite 覆盖写） ==========
enum FixedTiming
{
    KERNEL_TIMING_IDX = 0, // 整个 kernel 的墙钟时间
    AIC_KKT_CUBE,          // AIC: KKT cube（k·k^T 得分矩阵）
    AIV_CUMSUM,            // AIV: chunk 内门控 cumsum
    SYNC_SCORE_CUMSUM,     // SyncAll：发布 score 与 cumsum workspace
    AIV_KKT_EPILOGUE,      // AIV: KKT fused-cumsum epilogue（产出 solve 输入）
    SOLVE_TRI,             // SolveTri（AIC+AIV 均记录）
    SOLVE_SYNC_WAIT,       // solve→recompute 交接的跨核等待
    RECOMPUTE_WU,          // Recompute W/U（AIC+AIV 均记录）
    WRITE_GCUMSUM,         // 公共 g_cumsum BHT→BTH 转置写出
    FWD_H,                 // ChunkFwdH 全程（含内部细分动态项）
    HO_SYNC,               // H→O 全核 SyncAll
    FWD_O,                 // ChunkFwdO 全程（含内部细分动态项）
    FINAL_SYNC,            // HO 空闲流水开启时的最终 SyncAll
    FIXED_TIMING_COUNT
};

// ========== 2. 动态计时项（每迭代一个槽位，iter 为本核调度任务号） ==========
enum DynamicTimingType
{
    FWH_AIC_C1,     // FwdH AIC: C1 v_work = w @ h（整段，含流水等待）
    FWH_AIC_C1_WAIT, // FwdH AIC C1 内 CrossCoreWaitFlag(vec2Done) 纯等待
    FWH_AIC_C2,     // FwdH AIC: C2 h' = k^T @ v_work（整段，含流水等待）
    FWH_AIC_C2_WAIT, // FwdH AIC C2 内 CrossCoreWaitFlag(vec1Done) 纯等待
    FWH_AIV_VEC1,   // FwdH AIV: V1 epilogue（vnew / v_update 计算）
    FWH_AIV_VEC2,   // FwdH AIV: V2 epilogue（h 状态更新 / final_state）
    FWO_AIC_QK,     // FwdO AIC: Cube1 attn = q @ k^T
    FWO_AIC_QH,     // FwdO AIC: Cube2 段（含 H 预取、vec 等待与 h_inter = q @ h）
    FWO_AIC_ATTENV, // FwdO AIC: Cube3 段（o_inter = attn_mask @ vnew）
    FWO_AIV_QKMASK, // FwdO AIV: Vec1 qkmask epilogue
    FWO_AIV_OUTPUT, // FwdO AIV: Vec2 output epilogue
    DYNAMIC_TYPE_COUNT_ENUM
};

// ========== 3. 公共常量（CPU/NPU 共用） ==========
// 动态项每核最大迭代槽位数。FwdH/FwdO 每核任务数为
// ceil(totalTasks / coreNum)，长序列下按 128 预留；超出部分截断并告警。
static constexpr int MAX_DYNAMIC_ITER = 128;

// 每个计时项占 2 个 int64 槽位: [start_cycle, end_cycle]
static constexpr int N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER;
static constexpr int N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2;
// 128B（16 个 int64）对齐，避免多核写同一 cache line
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;
// 计时缓冲预留的物理核槽位上限（AIC+AIV 合计）。MIX_AIC_1_2 每组
// 1 AIC + 2 AIV 共 3 核，core_id = group*3 + slot，核内区间起点为
// core_id * N_TIMING_COUNTER_PER_CORE_ALIGN。A5/950 为 32 组（96 核），
// 预留 96 槽位；组数超过 N_CORE_COUNT/3 的设备侧写入会被拒绝并告警。
static constexpr int N_CORE_COUNT = 96;
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;

// 动态迭代次数存储区对齐配置（每核动态项存储区 16 元素对齐），
// 与计时区相同按物理核槽位预留。
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;
static constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT;

static constexpr int TOTAL_BUFFER_SIZE = N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE;

} // namespace GdnTimer

// ========== 4. 计时开关宏 ==========
// 计时本身会轻微扰动流水；关闭开关可编译出无打点的对照版本。
#define ENABLE_TIMER 1
#ifdef ENABLE_TIMER
#define TIMER_BLOCK(code) \
    do                    \
    {                     \
        code;             \
    } while (0)
#else
#define TIMER_BLOCK(code) \
    do                    \
    {                     \
    } while (0)
#endif

// 判空调用包装：timer 指针未注入（如独立调用子 kernel）时整句消失，
// 与 TIMER_BLOCK 叠加使用。
#define GDN_TIMER_CALL(ptr, code) \
    do                            \
    {                             \
        if ((ptr) != nullptr)     \
        {                         \
            code;                 \
        }                         \
    } while (0)

#endif // CHUNK_GATED_DELTA_RULE_FWD_TIMER_COMMON_H

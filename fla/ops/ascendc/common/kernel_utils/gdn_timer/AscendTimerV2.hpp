#ifndef ASCEND_TIMER_COMMON_H
#define ASCEND_TIMER_COMMON_H

#include <stdint.h>

namespace AscendTimer
{
// ========== 1. 枚举定义（CPU/NPU 共用） ==========
// 按 chunk_gated_delta_rule_fwd 三算子路径（Prepare / FwdH / FwdO）的实际阶段命名。
// 规则：只给"计算/下发"窗口打点，不给 WaitFlag / CrossCoreWaitFlag / SyncAll 打点。
// 固定计时项（整个 kernel 只跑一次）
enum FixedTiming : int32_t
{
    KERNEL_TIMING_IDX = 0, // 整体 kernel 墙钟（带 barrier 收尾）
    MAIN_INIT,             // Init -> InitDone
    MAIN_BODY,             // InitDone -> kernel 结束
    FIXED_TIMING_COUNT
};

// 动态计时项类型（循环内按 iter 重复）
enum DynamicTimingType : int32_t
{
    // ---- Prepare（S0-S7）----
    PREP_RESIDENT_AUX,   // S0：常驻辅助（I_vcs/索引/清零）
    PREP_PAD_TAIL,       // S1-pre：尾块补齐（仅 tail 命中）
    PREP_K_UPLOAD,       // S1a：k̂ 归一化 + L1 上传（NotifyAicStage1Done 在窗口内）
    PREP_GATE_BETA,      // S1b：gate cumsum / β sigmoid / 导出
    PREP_KKT_MM,         // S2：KKT = k̂·k̂ᵀ 的 MMAD（每 (chunk, HK) 一次）
    PREP_KKT_GATE,       // S3a：G=exp*(g_i−g_j) 与 I_vcs 拷贝（可与 KKT 并行）
    PREP_KKT_EPI,        // S3b：L=KKT⊙G 下三角 + 打包/上传（Tik 在 WaitCubeKktDone 之后）
    PREP_SOLVE_PREFILL,  // S4-pre：I·I 预填充
    PREP_SOLVE_ISSUE,    // S4-issue：Y=I+LeafLeft·(−L)
    PREP_SOLVE_DUMP,     // S4-dump / S5-dump：L0C→L1/UB
    PREP_SOLVE,          // S5：(I+L)⁻¹
    PREP_WU_V,           // S6a：vb = v·β（GM→UB→L1）
    PREP_WU_K,           // S6b：kbg = k̂·β·e^g（GM→UB→L1）
    PREP_W_ISSUE,        // S7a：W = A·kbg 的 MMAD 下发
    PREP_U_ISSUE,        // S7b：U = A·vb 的 MMAD 下发（V=256 时为两个半块）
    PREP_WU_DRAIN,       // 下一包窗口内的 w/u 回写（UB→GM）

    // ---- FwdH ----
    FWH_WH_MM,           // C1：P = w·S 的 MMAD（Tik 在 L0/W/H 等待之后）
    FWH_VNEW,            // V1：v_new epilogue
    FWH_KV_MM,           // C2：h_work = kᵀ·v_new 的 MMAD
    FWH_STATE,           // V2：状态更新 epilogue
    FWH_PREFETCH,        // W/K lookahead 的 MTE2 搬入

    // ---- FwdO ----
    FWO_GATE,            // V1：exp*(g) 因子（gateA/gateO）
    FWO_QK,              // C1a：attn = q·kᵀ
    FWO_QH,              // C1b：O_s = q·h
    FWO_MASK,            // V2：tril + 衰减 → A′；O_s′
    FWO_ATTN_V,          // C3：O_l = A′·v_new
    FWO_OUTPUT,          // V3：o = scale·(O_s′+O_l)

    DYNAMIC_TYPE_COUNT_ENUM
};

// ========== 2. 公共常量定义（CPU/NPU 共用） ==========
static constexpr int MAX_DYNAMIC_ITER = 64;

static constexpr int N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER;
static constexpr int N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2;
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;
static constexpr int N_CORE_COUNT = 96; // AIC+AIV 预留核数（A5 每组 1 AIC + 2 AIV）
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;

static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;
static constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT;

// ========== 3. 多 launch 切片 ==========
// Prepare / FwdH / FwdO 是三个独立 launch，共用一个 INT64 timer 张量。
// 每个 launch 使用自己的切片，避免互相覆盖；切片内布局与单 kernel 相同。
enum KernelSlice : int32_t
{
    SLICE_PHASE6 = 0, // 融合 Phase6 路径（本轮不打点，保留占位）
    SLICE_PREPARE,
    SLICE_FWD_H,
    SLICE_FWD_O,
    KERNEL_SLICE_COUNT
};

static constexpr int64_t SLICE_WORDS = static_cast<int64_t>(N_TIMING_COUNTER) + DYNAMIC_ITER_TOTAL_SIZE;
static constexpr int64_t TOTAL_BUFFER_WORDS = SLICE_WORDS * KERNEL_SLICE_COUNT;

// 切片在 timer 张量中的 int64 偏移（host 与 kernel 共用）。
// 用宏而非 constexpr 函数：部分 CCE 编译选项下不会内联，链接期会报未定义符号。
#define ASCEND_TIMER_SLICE_OFFSET_WORDS(slice) \
    (static_cast<int64_t>(slice) * AscendTimer::SLICE_WORDS)
} // namespace AscendTimer

// ========== 4. 计时开关宏 ==========
#ifndef ENABLE_TIMER
#define ENABLE_TIMER 1
#endif
#ifndef ASCEND_TIMER_EXPLICIT_GM_PUBLISH
#define ASCEND_TIMER_EXPLICIT_GM_PUBLISH 0
#endif
#if ENABLE_TIMER
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

#endif // ASCEND_TIMER_COMMON_H

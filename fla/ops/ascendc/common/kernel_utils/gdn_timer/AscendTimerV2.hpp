#ifndef ASCEND_TIMER_COMMON_H
#define ASCEND_TIMER_COMMON_H

#include <stdint.h>

namespace AscendTimer
{
// ========== 1. 枚举定义（CPU/NPU 共用） ==========
// 按 chunk_gated_delta_rule_fwd 三算子路径（Prepare / FwdH / FwdO）的实际阶段命名。
// 规则：默认只给"计算/下发"窗口打点；诊断用的 Init 拆分与等待窗口在末尾单独列出，
//       它们只包已有同步，不新增任何业务同步。
// 固定计时项（整个 kernel 只跑一次）
enum FixedTiming : int32_t
{
    KERNEL_TIMING_IDX = 0, // 整体 kernel 墙钟（带 barrier 收尾）
    MAIN_INIT,             // Init -> InitDone
    MAIN_BODY,             // InitDone -> kernel 结束
    ANCHOR,                // 每核在 body 起点的标记；各核减自己的 ANCHOR 得到“本核 body 时间”
    FIXED_TIMING_COUNT
};

// 动态计时项类型（循环内按 iter 重复）
// 命名规则：名字前缀 = 所处阶段，看导出列名就知道是哪一步。
//   Prepare: S0 / S1A(S1a) / S1B(S1b) / S1_PRE / S2 / S3A / S3B / S4 / S5 / S6 / S7
//   FwdH:    H_C0 / H_C1 / H_V1 / H_C2 / H_V2
//   FwdO:    O_V1 / O_C1A / O_C1B / O_V2 / O_C3 / O_V3
//   等待：    <消费阶段>_WAIT_<生产阶段>，如 S2_WAIT_S1
enum DynamicTimingType : int32_t
{
    // ---- Prepare（S0-S7）----
    S0_AUX,              // S0：常驻辅助（I_vcs/索引/清零）
    S1_PRE_TAIL,         // S1-pre：尾块补齐（仅 tail 命中）
    S1A_KUPLOAD,         // S1a：k̂ 归一化 + L1 上传（NotifyAicStage1Done 在窗口内）
    S1A_QUPLOAD,         // S1a'：q̂ 归一化 + 导出（仅 useQkL2norm=1 时执行）
    S1B_GATE_BETA,       // S1b：gate cumsum / β sigmoid / 导出
    S2_KKT,              // S2：KKT = k̂·k̂ᵀ 的 MMAD（每 (chunk, HK) 一次）
    S3A_GATE,            // S3a：G=exp*(g_i−g_j) 与 I_vcs 拷贝（可与 KKT 并行）
    S3B_EPI,             // S3b：L=KKT⊙G 下三角 + 打包/上传（Tik 在 WaitCubeKktDone 之后）
    S4_PREFILL,          // S4-pre：I·I 预填充
    S4_ISSUE,            // S4-issue：Y=I+LeafLeft·(−L)
    S4_DUMP,             // S4-dump：Y 从 L0C→GM→L1
    S5_SOLVE,            // S5：(I+L)⁻¹
    S5_DUMP,             // S5-dump：A 从 L0C→GM/L1
    S6_VB,               // S6a：vb = v·β（GM→UB→L1）
    S6_KBG,              // S6b：kbg = k̂·β·e^g（GM→UB→L1）
    S7_W,                // S7a：W = A·kbg 的 MMAD 下发
    S7_U,                // S7b：U = A·vb 的 MMAD 下发（V=256 时为两个半块）
    S7_DRAIN,            // 下一包窗口内的 w/u 回写（UB→GM）

    // ---- FwdH ----
    H_C0_PREFETCH,       // C0：W/K/H 的 GM→L1 预取
    H_C1_WH,             // C1：P = w·S 的 MMAD（Tik 在 L0/W/H 等待之后）
    H_V1_VNEW,           // V1：v_new epilogue
    H_C2_KV,             // C2：h_work = kᵀ·v_new 的 MMAD
    H_V2_STATE,          // V2：状态更新 epilogue

    // ---- FwdO ----
    O_V1_GATE,           // V1：exp*(g) 因子（gateA/gateO）
    O_C1A_QK,            // C1a：attn = q·kᵀ
    O_C1B_QH,            // C1b：O_s = q·h
    O_V2_MASK,           // V2：tril + 衰减 → A′；O_s′
    O_C3_ATTNV,          // C3：O_l = A′·v_new
    O_V3_OUTPUT,         // V3：o = scale·(O_s′+O_l)

    // ---- 诊断：Init 拆分 + 等待窗口（不改变上面任何既有窗口的定义）----
    // Init 拆分用于回答"AIV Init 比 AIC 多 1.5us 花在哪"。
    INIT_MAP,            // Init：InitOnChip 整体
    INIT_WS,             // Init：workspace 绑定
    INIT_GM,             // Init：GM SetGlobalBuffer 批次
    // 等待窗口：只包已有 CrossCoreWaitFlag，不新增任何同步。
    // 导出时用于断言 wait_end ≈ producer notify。
    S2_WAIT_S1,          // AIC：WaitAivStage1Done（producer = S1_NOTIFY）
    S4_WAIT_S3,          // AIC：WaitAivStage3Done（producer = S3B_EPI）
    S7_WAIT_S6,          // AIC：WaitAivStage6Done（producer = S6_KBG）
    S1_WAIT_S4,          // AIV：WaitAicStage4Done（producer = S4_DUMP）
    S3B_WAIT_KKT,        // AIV：WaitCubeKktDone（producer = S2_KKT）
    S1_NOTIFY,           // AIV：NotifyAicStage1Done 的瞬时点，跨核时间基准的校准锚

    // ---- 搬运等待（诊断）----
    // MTE2：阻塞在"输入数据从 GM 到达"上（消费侧 WaitFlag<MTE2_V>）
    // MTE3：阻塞在"上一轮写回完成 / 缓冲槽可复用"上（WaitFlag<MTE3_MTE2> 或 <MTE3_V>）
    MTE2,
    MTE3,

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
#define ASCEND_TIMER_EXPLICIT_GM_PUBLISH 1
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

#ifndef ASCEND_TIMER_SCOPE_H
#define ASCEND_TIMER_SCOPE_H

#include "AscendTimerV2_device.hpp"

// RAII 计时窗口：构造时分配�?stage 的下一�?iter �?TikNoBarrier�?// 析构�?TokNoBarrier<Overwrite>。iter �?timer 内部单调分配，跨 pack 不会覆盖�?// 约定�?//   * 只包"计算 / 下发"代码，不�?WaitFlag / CrossCoreWaitFlag / SyncAll�?//     若某段计算前面存在功能性等待，把本对象放在等待之后�?//   * 同一作用域内多个窗口请各自放在独立的 {} 块里，避免变量名冲突�?//   * timer 指针为空（普通调用）时整体退化为无操作�?class AscendTimerScope
{
public:
    __aicore__ inline AscendTimerScope(AscendTimerDevice *timer,
                                       AscendTimer::DynamicTimingType type)
        : timer_(timer), type_(type), iter_(0)
    {
        if (timer_ != nullptr) {
            iter_ = timer_->NextIter(type_);
            timer_->TikNoBarrier(type_, iter_);
        }
    }

    __aicore__ inline ~AscendTimerScope()
    {
        if (timer_ != nullptr) {
            timer_->template TokNoBarrier<Overwrite>(type_, iter_);
        }
    }

    AscendTimerScope(const AscendTimerScope &) = delete;
    AscendTimerScope &operator=(const AscendTimerScope &) = delete;

private:
    AscendTimerDevice *timer_;
    AscendTimer::DynamicTimingType type_;
    int32_t iter_;
};

#define GDN_TIMER_SCOPE(timer_ptr, stage_type) \
    AscendTimerScope gdn_timer_scope_guard(timer_ptr, stage_type)

#endif // ASCEND_TIMER_SCOPE_H

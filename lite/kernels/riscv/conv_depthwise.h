#pragma once

#include <cmath>
#include <string>
#include <vector>
// #include "lite/backends/arm/math/conv_impl.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace riscv {

template <PrecisionType Ptype, PrecisionType Otype>
class DepthwiseConv : public KernelLite<TARGET(kARM), Ptype> {
 public:
  typedef void (*conv_dw_impl)(const void* din,
                               void* dout,
                               int num,
                               int ch_out,
                               int h_out,
                               int w_out,
                               int ch_in,
                               int h_in,
                               int w_in,
                               const void* weights,
                               const float* bias,
                               const operators::ConvParam& param,
                               ARMContext* ctx,
                               const float* scale);
  DepthwiseConv() = default;
  ~DepthwiseConv() {}
  virtual void PrepareForRun();
  virtual void ReInitWhenNeeded();
  virtual void Run();

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }

  std::string kernel_func_name_{"NotImplForConvDW"};
#define PROFILE_INFO(dtype1, dtype2)                                        \
  template <>                                                               \
  void DepthwiseConv<PRECISION(dtype1), PRECISION(dtype2)>::                \
      SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) { \
    ch->kernel_func_name = kernel_func_name_;                               \
  }

#define KERNEL_FUNC_NAME(kernel_func_name) kernel_func_name_ = kernel_func_name;

#else
#define PROFILE_INFO(dtype1, dtype2)
#define KERNEL_FUNC_NAME(kernel_func_name)
#endif

 private:
  using param_t = operators::ConvParam;
  Tensor weights_;
  Tensor bias_;
  DDim last_shape_;
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  conv_dw_impl impl_{nullptr};
  std::vector<float> w_scale_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

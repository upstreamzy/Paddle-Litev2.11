// #include "lite/kernels/arm/conv_compute.h"
#include "lite/kernels/riscv/conv_compute.h"
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/arm/conv_depthwise.h"
// #include "lite/kernels/arm/conv_direct.h"
// #include "lite/kernels/arm/conv_gemmlike.h"
// #include "lite/kernels/arm/conv_winograd.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace riscv {
#define PARAM_INIT                                                           \
  auto& param = this->Param<param_t>();                                      \
  auto w_dims = param.filter->dims();                                        \
  auto& ctx = this->ctx_->template As<ARMContext>();                         \
  auto paddings = *param.paddings;                                           \
  auto dilations = *param.dilations;                                         \
  int ic = w_dims[1] * param.groups;                                         \
  int oc = w_dims[0];                                                        \
  int kh = w_dims[2];                                                        \
  int kw = w_dims[3];                                                        \
  int pad_h = paddings[0];                                                   \
  int pad_w = paddings[2];                                                   \
  int stride = param.strides[0];                                             \
  int sh = param.strides[1];                                                 \
  int sw = param.strides[0];                                                 \
  int threads = ctx.threads();                                               \
  int chin = param.x->dims()[1];                                             \
  int hin = param.x->dims()[2];                                              \
  int win = param.x->dims()[3];                                              \
  int chout = param.output->dims()[1];                                       \
  int hout = param.output->dims()[2];                                        \
  int wout = param.output->dims()[3];                                        \
  bool pads_equal =                                                          \
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));        \
  bool pads_all_equal = (pads_equal && pad_h == pad_w);                      \
  bool ks_equal = (sw == sh) && (kw == kh);                                  \
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);             \
  bool kps_equal = (pad_h == pad_w) && ks_equal;                             \
  bool flag_dw_3x3 = (kw == 3) && (kh == 3) && (stride == 1 || stride == 2); \
  bool flag_dw_5x5 = (kw == 5) && (kh == 5) && (stride == 1 || stride == 2); \
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

template <>
void ConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  PARAM_INIT
  /// select conv impl
  if (param.groups == ic && ic == oc && ks_equal && no_dilation && flag_dw) {
    impl_ = new DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>;
    VLOG(3) << "invoking dw conv";
  } else if (param.groups == 1 && kw == 3 && stride == 1 && ks_equal &&
             no_dilation) {
    impl_ = new WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>;
    VLOG(3) << "invoking winograd conv";
  } else if (param.groups == 1 && kw == 3 && stride == 2 &&
             chin * chout < 4 * hin * win && ks_equal && no_dilation) {
    impl_ = new DirectConv<PRECISION(kFloat), PRECISION(kFloat)>;
    VLOG(3) << "invoking direct conv";
  } else {
    impl_ = new GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>;
    VLOG(3) << "invoking gemm like conv";
  }
  impl_->SetContext(std::move(this->ctx_));
  impl_->SetParam(param);
  impl_->PrepareForRun();
  is_first_epoch_ = false;
}



}
}
}
}

typedef paddle::lite::kernels::riscv::ConvCompute<PRECISION(kFloat),
                                                PRECISION(kFloat)>
    ConvFp32;

REGISTER_LITE_KERNEL(conv2d, kRISCV, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kRISCV))})
    .BindInput("SecondInput", {LiteType::GetTensorTy(TARGET(kRISCV))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kRISCV))})
    .BindInput("Prelu_alpha", {LiteType::GetTensorTy(TARGET(kRISCV))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kRISCV))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kRISCV))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();
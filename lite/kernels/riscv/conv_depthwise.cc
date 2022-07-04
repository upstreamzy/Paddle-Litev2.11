// #include "lite/kernels/arm/conv_depthwise.h"
// #include "lite/backends/arm/math/conv_block_utils.h"
// #include "lite/backends/arm/math/conv_impl.h"
// #ifdef ENABLE_ARM_FP16
// #include "lite/backends/arm/math/fp16/conv_impl_fp16.h"
// #endif
#include "lite/kernels/riscv/conv_depthwise.h"
#include "lite/backends/riscv/math/conv_impl.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace riscv {

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::ReInitWhenNeeded() {
  auto& param = this->template Param<param_t>();
  auto x_dims = param.x->dims();
  if (last_shape_ == x_dims) {
    return;
  }
  auto w_dims = param.filter->dims();
  auto kw = w_dims[3];
  auto win = param.x->dims()[3];
  auto paddings = *param.paddings;
  // select dw conv kernel
  if (kw == 3) {
    bool pads_less = ((paddings[1] < 2) && (paddings[3] < 2));
    if (pads_less && paddings[0] == paddings[2] &&
        (paddings[0] == 0 || paddings[0] == 1)) {
      flag_trans_weights_ = false;
    } else {
      // trans weights
      constexpr int cblock = 4;
      auto oc = w_dims[0];
      auto kh = w_dims[2];
      auto cround = ROUNDUP(oc, cblock);
      weights_.Resize({cround, 1, kh, kw});
      auto w_data = weights_.mutable_data<float>();
      auto w_data_in = param.filter->data<float>();
      lite::arm::math::conv_trans_weights_numc(
          w_data_in, w_data, oc, 1, cblock, kh * kw);
      flag_trans_weights_ = true;
    }
    impl_ = lite::arm::math::conv_depthwise_3x3_fp32;
    KERNEL_FUNC_NAME("conv_depthwise_3x3_fp32")
  } else if (kw == 5) {
    auto strides = param.strides;
    bool pads_equal = (paddings[0] == paddings[2]) && (paddings[0] == 2);
    // todo s1 profile is not great than c4
    bool s1_equal =
        0 &&
        (strides[0] == 1 && strides[1] == 1 && pads_equal &&
         static_cast<int>(param.activation_param.active_type) < 4 && win > 8);
    bool s2_equal =
        (strides[0] == 2 && strides[1] == 2 && pads_equal &&
         static_cast<int>(param.activation_param.active_type) < 4 && win > 16);
    if (s1_equal || s2_equal) {
      flag_trans_weights_ = false;
    } else {
      // trans weights
      constexpr int cblock = 4;
      auto oc = w_dims[0];
      auto kh = w_dims[2];
      auto cround = ROUNDUP(oc, cblock);
      weights_.Resize({cround, 1, kh, kw});
      auto w_data = weights_.mutable_data<float>();
      auto w_data_in = param.filter->data<float>();
      lite::arm::math::conv_trans_weights_numc(
          w_data_in, w_data, oc, 1, cblock, kh * kw);
      flag_trans_weights_ = true;
    }
    impl_ = lite::arm::math::conv_depthwise_5x5_fp32;
    KERNEL_FUNC_NAME("conv_depthwise_5x5_fp32")
  } else {
    LOG(FATAL) << "this type dw conv not impl: " << kw;
  }
  last_shape_ = x_dims;
}


template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<RISCVContext>();
  // select dw conv kernel
  ReInitWhenNeeded();
  last_shape_ = param.x->dims();
}



PROFILE_INFO(kFloat, kFloat)

#define CONV_DW_PARAM \
  i_data, o_data, bs, oc, oh, ow, ic, ih, iw, w_data, b_data, param, &ctx
template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<RISCVContext>();
  const auto* i_data = param.x->data<float>();
  const auto* w_data = flag_trans_weights_ ? weights_.data<float>()
                                           : param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  auto* o_data = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  impl_(CONV_DW_PARAM, w_scale_.data());
}


}
}
}
}
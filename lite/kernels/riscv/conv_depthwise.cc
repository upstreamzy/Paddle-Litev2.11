
#include "lite/kernels/riscv/conv_depthwise.h"
// #include "lite/backends/riscv/math/avx/conv_utils.h"
#include "lite/backends/riscv/math/conv_depthwise_impl.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace riscv {

#define CONV_DW_PARAM                                                         \
  i_data, o_data, bs, oc, oh, ow, ic, ih, iw, w_data, b_data, pad, flag_bias, \
      act_param

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {}

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);

  auto input_dims = param.x->dims();
  CHECK_EQ(input_dims.size(), 4UL);

  const auto* i_data = param.x->data<float>();
  const auto* w_data = param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto act_param = param.activation_param;
  const auto stride = param.strides[1];
  auto pad = (*param.paddings)[2];
  bool flag_bias = param.bias != nullptr;
  auto* o_data = param.output->mutable_data<float>();
  auto dilations = *param.dilations;
  bool pad_less = pad < 2;

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  int kh = w_dims[2];

  if (kh == 3) {
    if ((dilations[0] == 1) && (dilations[1] == 1) && pad_less) {
      if (stride == 1) {
        lite::x86::math::conv_depthwise_3x3s1_p01_direct(CONV_DW_PARAM);
      } else if (stride == 2) {
        lite::x86::math::conv_depthwise_3x3s2_p01_direct(CONV_DW_PARAM);
      }
    } else {
      lite::x86::math::conv_depthwise_3x3_pack(
          param, &input_padding_, &input_pack_, &filter_pack_, &output_pack_);
    }
  } else if (kh == 5) {
    if (stride == 1) {
      lite::x86::math::conv_depthwise_5x5s1(CONV_DW_PARAM);
    } else if (stride == 2) {
      lite::x86::math::conv_depthwise_5x5s2(CONV_DW_PARAM);
    }
  } else {
    LOG(FATAL) << "kw and kh only support 3 or 5";
  }
  KERNEL_FUNC_NAME("conv_depthwise_direct")
}

PROFILE_INFO(kFloat, kFloat)




}
}
}
}
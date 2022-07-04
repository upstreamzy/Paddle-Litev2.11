#pragma once

#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void conv_depthwise_3x3_fp32(const void* din,
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
                             const float* scale) {
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  const int pad_h = paddings[0];
  const int pad_w = paddings[2];
  int stride = param.strides[1];
  int pad = pad_w;
  bool flag_bias = param.bias != nullptr;
  bool pads_less = ((paddings[1] < 2) && (paddings[3] < 2));
  if (stride == 1) {
    if (pads_less && (pad_h == pad_w) && (pad < 2)) {  // support pad = [0, 1]
      conv_depthwise_3x3s1_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                pad,
                                flag_bias,
                                act_param,
                                ctx);
    } else {
      conv_3x3s1_depthwise_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                param,
                                act_param,
                                ctx);
    }
  } else if (stride == 2) {
    if (pads_less && pad_h == pad_w && (pad < 2)) {  // support pad = [0, 1]
      conv_depthwise_3x3s2_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                pad,
                                flag_bias,
                                act_param,
                                ctx);
    } else {
      conv_3x3s2_depthwise_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                param,
                                act_param,
                                ctx);
    }
  } else {
    LOG(FATAL) << "fp32 depthwise conv3x3 stride: " << stride << " unsupported";
  }
}

void conv_depthwise_5x5_fp32(const void* din,
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
                             const float* scale) {
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride = param.strides[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  ctx->ExtendWorkspace((w_in + w_out + 16) * sizeof(float));
  if (stride == 2) {
    if (pad_h == pad_w && pad_h == 2 &&
        static_cast<int>(act_param.active_type) < 4 && w_in > 16) {
      // only support conv + relu/relu6
      conv_depthwise_5x5s2p2_fp32(reinterpret_cast<float*>(dout),
                                  reinterpret_cast<const float*>(din),
                                  reinterpret_cast<const float*>(weights),
                                  bias,
                                  flag_bias,
                                  num,
                                  ch_out,
                                  h_out,
                                  w_out,
                                  ch_in,
                                  h_in,
                                  w_in,
                                  param,
                                  ctx);
    } else {
      conv_depthwise_5x5s2_fp32(reinterpret_cast<const float*>(din),
                                reinterpret_cast<float*>(dout),
                                num,
                                ch_out,
                                h_out,
                                w_out,
                                ch_in,
                                h_in,
                                w_in,
                                reinterpret_cast<const float*>(weights),
                                bias,
                                param,
                                act_param,
                                ctx);
    }
  } else if (stride == 1) {
    if (0 && pad_h == pad_w && pad_h == 2 &&
        static_cast<int>(act_param.active_type) < 4 && w_in > 8) {
      // only support conv + relu/relu6
      conv_depthwise_5x5s1p2_fp32(reinterpret_cast<float*>(dout),
                                  reinterpret_cast<const float*>(din),
                                  reinterpret_cast<const float*>(weights),
                                  bias,
                                  flag_bias,
                                  flag_relu,
                                  num,
                                  ch_in,
                                  h_in,
                                  w_in,
                                  h_out,
                                  w_out,
                                  param,
                                  ctx);
    } else {
      conv_depthwise_5x5s1_fp32(reinterpret_cast<float*>(dout),
                                reinterpret_cast<const float*>(din),
                                reinterpret_cast<const float*>(weights),
                                bias,
                                flag_bias,
                                flag_relu,
                                num,
                                ch_in,
                                h_in,
                                w_in,
                                h_out,
                                w_out,
                                pad_w,
                                pad_h,
                                param,
                                ctx);
    }
  } else {
    LOG(FATAL) << "unsupport this type 5x5 dw conv";
  }
}


}
}
}
}

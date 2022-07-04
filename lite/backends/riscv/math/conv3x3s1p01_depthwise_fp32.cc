#include "lite/backends/riscv/math/conv_depthwise.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace riscv {
namespace math {


void conv_depthwise_3x3s1_fp32(const float *din,
                               float *dout,
                               int num,
                               int ch_out,
                               int h_out,
                               int w_out,
                               int ch_in,
                               int h_in,
                               int w_in,
                               const float *weights,
                               const float *bias,
                               int pad,
                               bool flag_bias,
                               const operators::ActivationParam act_param,
                               ARMContext *ctx) {
  bool has_active = act_param.has_active;
  auto act_type = act_param.active_type;
  float tmp = act_param.Relu_clipped_coef;
  float ss = act_param.Leaky_relu_alpha;
  float vsix[4] = {tmp, tmp, tmp, tmp};
  float vscale[4] = {ss, ss, ss, ss};
  if (has_active) {
    switch (act_type) {
      case lite_api::ActivationType::kRelu:
        if (pad == 0) {
          if (w_in > 5) {
            conv_depthwise_3x3s1p0_bias_relu(dout,
                                             din,
                                             weights,
                                             bias,
                                             flag_bias,
                                             true,
                                             num,
                                             ch_in,
                                             h_in,
                                             w_in,
                                             h_out,
                                             w_out,
                                             ctx);
          } else {
            conv_depthwise_3x3s1p0_bias_s_relu(dout,
                                               din,
                                               weights,
                                               bias,
                                               flag_bias,
                                               true,
                                               num,
                                               ch_in,
                                               h_in,
                                               w_in,
                                               h_out,
                                               w_out,
                                               ctx);
          }
        }
        if (pad == 1) {
          if (w_in > 4) {
            conv_depthwise_3x3s1p1_bias_relu(dout,
                                             din,
                                             weights,
                                             bias,
                                             flag_bias,
                                             true,
                                             num,
                                             ch_in,
                                             h_in,
                                             w_in,
                                             h_out,
                                             w_out,
                                             ctx);
          } else {
            conv_depthwise_3x3s1p1_bias_s_relu(dout,
                                               din,
                                               weights,
                                               bias,
                                               flag_bias,
                                               true,
                                               num,
                                               ch_in,
                                               h_in,
                                               w_in,
                                               h_out,
                                               w_out,
                                               ctx);
          }
        }
        break;
      case lite_api::ActivationType::kRelu6:
        if (pad == 0) {
          if (w_in > 5) {
            conv_depthwise_3x3s1p0_bias_relu6(dout,
                                              din,
                                              weights,
                                              bias,
                                              vsix,
                                              flag_bias,
                                              num,
                                              ch_in,
                                              h_in,
                                              w_in,
                                              h_out,
                                              w_out,
                                              ctx);
          } else {
            conv_depthwise_3x3s1p0_bias_s_relu6(dout,
                                                din,
                                                weights,
                                                bias,
                                                vsix,
                                                flag_bias,
                                                num,
                                                ch_in,
                                                h_in,
                                                w_in,
                                                h_out,
                                                w_out,
                                                ctx);
          }
        }
        if (pad == 1) {
          if (w_in > 4) {
            conv_depthwise_3x3s1p1_bias_relu6(dout,
                                              din,
                                              weights,
                                              bias,
                                              vsix,
                                              flag_bias,
                                              num,
                                              ch_in,
                                              h_in,
                                              w_in,
                                              h_out,
                                              w_out,
                                              ctx);
          } else {
            conv_depthwise_3x3s1p1_bias_s_relu6(dout,
                                                din,
                                                weights,
                                                bias,
                                                vsix,
                                                flag_bias,
                                                num,
                                                ch_in,
                                                h_in,
                                                w_in,
                                                h_out,
                                                w_out,
                                                ctx);
          }
        }
        break;
      case lite_api::ActivationType::kLeakyRelu:
        if (pad == 0) {
          if (w_in > 5) {
            conv_depthwise_3x3s1p0_bias_leakyRelu(dout,
                                                  din,
                                                  weights,
                                                  bias,
                                                  vscale,
                                                  flag_bias,
                                                  num,
                                                  ch_in,
                                                  h_in,
                                                  w_in,
                                                  h_out,
                                                  w_out,
                                                  ctx);
          } else {
            conv_depthwise_3x3s1p0_bias_s_leakyRelu(dout,
                                                    din,
                                                    weights,
                                                    bias,
                                                    vscale,
                                                    flag_bias,
                                                    num,
                                                    ch_in,
                                                    h_in,
                                                    w_in,
                                                    h_out,
                                                    w_out,
                                                    ctx);
          }
        }
        if (pad == 1) {
          if (w_in > 4) {
            conv_depthwise_3x3s1p1_bias_leakyRelu(dout,
                                                  din,
                                                  weights,
                                                  bias,
                                                  vscale,
                                                  flag_bias,
                                                  num,
                                                  ch_in,
                                                  h_in,
                                                  w_in,
                                                  h_out,
                                                  w_out,
                                                  ctx);
          } else {
            conv_depthwise_3x3s1p1_bias_s_leakyRelu(dout,
                                                    din,
                                                    weights,
                                                    bias,
                                                    vscale,
                                                    flag_bias,
                                                    num,
                                                    ch_in,
                                                    h_in,
                                                    w_in,
                                                    h_out,
                                                    w_out,
                                                    ctx);
          }
        }
        break;
      default:
        LOG(FATAL) << "this act_type: " << static_cast<int>(act_type)
                   << " fuse not support";
    }
  } else {
    if (pad == 0) {
      if (w_in > 5) {
        conv_depthwise_3x3s1p0_bias_no_relu(dout,
                                            din,
                                            weights,
                                            bias,
                                            flag_bias,
                                            false,
                                            num,
                                            ch_in,
                                            h_in,
                                            w_in,
                                            h_out,
                                            w_out,
                                            ctx);
      } else {
        conv_depthwise_3x3s1p0_bias_s_no_relu(dout,
                                              din,
                                              weights,
                                              bias,
                                              flag_bias,
                                              false,
                                              num,
                                              ch_in,
                                              h_in,
                                              w_in,
                                              h_out,
                                              w_out,
                                              ctx);
      }
    }
    if (pad == 1) {
      if (w_in > 4) {
        conv_depthwise_3x3s1p1_bias_no_relu(dout,
                                            din,
                                            weights,
                                            bias,
                                            flag_bias,
                                            false,
                                            num,
                                            ch_in,
                                            h_in,
                                            w_in,
                                            h_out,
                                            w_out,
                                            ctx);
      } else {
        conv_depthwise_3x3s1p1_bias_s_no_relu(dout,
                                              din,
                                              weights,
                                              bias,
                                              flag_bias,
                                              false,
                                              num,
                                              ch_in,
                                              h_in,
                                              w_in,
                                              h_out,
                                              w_out,
                                              ctx);
      }
    }
  }
}

}
}
}
}
#pragma once

#include <cmath>
#include <vector>
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace riscv {
namespace math {
    void conv_depthwise_3x3s1p0_bias_s_relu(float* dout,
                                        const float* din,
                                        const float* weights,
                                        const float* bias,
                                        bool flag_bias,
                                        bool flag_relu,
                                        const int num,
                                        const int ch_in,
                                        const int h_in,
                                        const int w_in,
                                        const int h_out,
                                        const int w_out,
                                        RISCVContext* ctx);

    void conv_depthwise_3x3s1_fp32(const float* din,
                               float* dout,
                               int num,
                               int ch_out,
                               int h_out,
                               int w_out,
                               int ch_in,
                               int h_in,
                               int w_in,
                               const float* weights,
                               const float* bias,
                               int pad,
                               bool flag_bias,
                               const operators::ActivationParam act_param,
                               RISCVContext* ctx);
}
}
}
}
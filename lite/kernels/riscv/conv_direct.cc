#include "lite/kernels/riscv/conv_direct.h"
#include <cmath>
#include "lite/backends/riscv/math/conv_bias.h"
#include "lite/backends/riscv/math/conv_direct_fp32.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace riscv {

static inline float activation_ss(float v, int activation_type, const ncnn::Mat& activation_params)
{
  // if (activation_type == 1)
  // {
  //   v = fmax(v, 0.f);
  // }
  // else if (activation_type == 2)
  // {
  //   float slope = activation_params[0];
  //   v = v > 0.f ? v : v * slope;
  // }
  // else if (activation_type == 3)
  // {
  //   float min = activation_params[0];
  //   float max = activation_params[1];
  //   if (v < min)
  //       v = min;
  //   if (v > max)
  //       v = max;
  // }
  // else if (activation_type == 4)
  // {
    v = 1.f / (1.f + exp(-v));
  // }
  // else if (activation_type == 5)
  // {
  //   v = v * tanh(log(exp(v) + 1.f));
  // }

  return v;
}

// static int convolution(const float* i_data,
//                                float* o_data,
//                                int bs,
                              //  int ic,
                              //  int ih,
                              //  int iw,
                              //  int oc,
                              //  int oc_expand,
                              //  int oh,
                              //  int ow,
                              //  int ph,
                              //  int pw,
                              //  int wh,
                              //  int ww,
//                            vector<int> &stride) {
// void conv_direct::run(const float* i_data,
//                     const float* trans_weight,
//                     float* trans_out,
//                     int bs,
//                     int ic,
//                     int ih,
//                     int iw,
//                     int oc,
//                     int oc_expand,
//                     int oh,
//                     int ow,
//                     int ph,
//                     int pw,
//                     int wh,
//                     int ww,
//                     int strideh) {
static int convolution(const float* i_data, float* o_data, const float *weight_data, const float *bias_data, int ww, int wh, int stride_w, int stride_h, 
  int dilation_w, int dilation_h, int activation_type, const struct ActivationParam& activation_params)
{
  const int w = iw;
  const int inch = ic;

  const int outw = ow;
  const int outh = oh;
  const int outch = oc;

  const int bias_term = bias_data.empty() ? 0 : 1;

  kernel_w = ww;
  kernel_h = wh;
  const int maxk = kernel_w * kernel_h;

  stride_h = strides[0];
  stride_w = strides[1];


  float* outptr = o_data; 
  // kernel offsets
  std::vector<int> _space_ofs(maxk);
  int* space_ofs = &_space_ofs[0];
  {
    int p1 = 0;
    int p2 = 0;
    int gap = w * dilation_h - kernel_w * dilation_w;
    for (int i = 0; i < kernel_h; i++)
    {
        for (int j = 0; j < kernel_w; j++)
        {
            space_ofs[p1] = p2;
            p1++;
            p2 += dilation_w;
        }
        p2 += gap;
    }
  }

  // #pragma omp parallel for num_threads(opt.num_threads)
  for (int p = 0; p < outch; p++)
  {
    // float* outptr = top_blob.channel(p);

    for (int i = 0; i < outh; i++)
    {
      for (int j = 0; j < outw; j++)
      {
          float sum = 0.f;

          if (bias_term)
            sum = bias_data[p];

          const float* kptr = (const float*)weight_data + maxk * inch * p;

          for (int q = 0; q < inch; q++)
          {
            // const Mat m = bottom_blob.channel(q);

            const float* sptr = i_data + (p * iw * ih) + (i * stride_h) * stride_w + j * stride_w;

            for (int k = 0; k < maxk; k++) // 29.23
            {
                float val = sptr[space_ofs[k]]; // 20.72
                float wt = kptr[k];
                sum += val * wt; // 41.45
            }

            kptr += maxk;
          }

          outptr[j] = activation_ss(sum, activation_type, activation_params);
      }

      outptr += outw;
    }
  }

  return 0;
}

template <>
void DirectConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();

  vector<int> dilations = *(param.dilations.get());

  const auto* i_data = param.x->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int wh = param.filter->dims()[2];
  int ww = param.filter->dims()[3];

  const int ph = (*(param.paddings))[0];
  const int pw = (*(param.paddings))[2];

  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oc = o_dims[1];
  int oh = o_dims[2];
  int ow = o_dims[3];

  // float* trans_out = static_cast<float*>(
  //     TargetMalloc(TARGET(kRISCV), sizeof(float) * bs * oc_expand_ * oh * ow));
  // memset(trans_out, 0, sizeof(float) * oc * oh * ow * bs);

  

  auto act_param = param.activation_param;
  // code_->run(i_data,
  //            weights_.data<float>(),
  //            trans_out,
  //            bs,
  //            ic,
  //            ih,
  //            iw,
  //            oc,
  //            oc_expand_,
  //            oh,
  //            ow,
  //            ph,
  //            pw,
  //            wh,
  //            ww,
  //            param.strides[0]);
  // static int convolution(const float* i_data, float* o_data, const float *weight_data, const float *bias_data, int ww, int wh, int stride_w, int stride_h, 
  // int dilation_w, int dilation_h, int activation_type, const struct ActivationParam& activation_params)
  convolution(i_data, o_data, weights_.data<float>(), b_data, ww, wh, stride_w, stride_h, param.strides[0], param.strides[1], dilations[0], dilations[1], 
    act_param.active_type, act_param);

  // lite::x86::math::conv_direct_transpose_out(bs,
  //                                            oc,
  //                                            oh,
  //                                            ow,
  //                                            o_data,
  //                                            trans_out,
  //                                            b_data,
  //                                            act_param.active_type,
  //                                            act_param);
  // TargetFree(TARGET(kRISCV), trans_out);
  // KERNEL_FUNC_NAME("conv_3x3s2_direct_fp32")
}
}  // namespace riscv
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

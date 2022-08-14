// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <iostream>
#include "lite/kernels/x86/conv_compute.h"
#include <utility>
#include "lite/backends/x86/math/fill_bias_activate.h"
#include "lite/kernels/x86/conv_depthwise.h"
#include "lite/kernels/x86/conv_direct.h"
// using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {
#define INIT_PARAM                      \
  printf("conv_compute.cc init01 \n");   \
  auto& param = this->Param<param_t>(); \
  printf("conv_compute.cc init02 \n");   \
  auto x_dims = param.x->dims();        \
  auto w_dims = param.filter->dims();   \
  auto o_dims = param.output->dims();   \
  printf("conv_compute.cc init1 \n");   \
  int win = x_dims[3];                  \
  printf("the win is %d\n", win);       \
  int hin = x_dims[2];                  \
  printf("the hin is %d\n", hin);       \
  int chin = x_dims[1];                 \
  printf("the chin is %d\n", chin);     \
  int num = x_dims[0];                  \
  printf("the num is %d\n", num);       \
  int wout = o_dims[3];                 \
  printf("the wout is %d\n", wout);     \
  int hout = o_dims[2];                 \
  printf("the hout is %d\n", hout);     \
  int chout = o_dims[1];                \
  printf("the chout is %d\n", chout);     \
  int kw = w_dims[3];                   \
  printf("the kw is %d\n", kw);     \
  int kh = w_dims[2];                   \
  printf("the kh is %d\n", kh);     \
  int group = param.groups;             \
  printf("the group is %d\n", group);   \
  int m = chout / group;                \
  int n = hout * wout;                  \
  int k = chin * kw * kh / group;       
  

#define PREPARE_PARAM                                                         \
  auto& param = this->Param<param_t>();                                       \
  const int input_channel = param.x->dims()[1];                               \
  const int output_channel = param.filter->dims()[0];                         \
  const int groups = param.groups;                                            \
  const int kernel_h = param.filter->dims()[2];                               \
  const int kernel_w = param.filter->dims()[3];                               \
  const int stride_h = param.strides[0];                                      \
  const int stride_w = param.strides[1];                                      \
  auto paddings = *param.paddings;                                            \
  auto dilations = *param.dilations;                                          \
  bool dw_kernel = (input_channel == groups && output_channel == groups);     \
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);           \
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);              \
  bool kps_equal = (paddings[0] == paddings[2]) && ks_equal;                  \
  bool pads_equal =                                                           \
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));         \
  bool flag_dw_3x3 =                                                          \
      (kernel_h == 3) && (kernel_w == 3) && (stride_h == 1 || stride_h == 2); \
  bool flag_dw_5x5 =                                                          \
      (kernel_h == 5) && (kernel_w == 5) && (stride_h == 1 || stride_h == 2);

#define PREPARE_PARAM_INT8                                          \
  auto& param = this->Param<param_t>();                             \
  const int input_channel = param.x->dims()[1];                     \
  const int output_channel = param.filter->dims()[0];               \
  const int groups = param.groups;                                  \
  const int kernel_h = param.filter->dims()[2];                     \
  const int kernel_w = param.filter->dims()[3];                     \
  const int stride_h = param.strides[0];                            \
  const int stride_w = param.strides[1];                            \
  auto paddings = *param.paddings;                                  \
  auto dilations = *param.dilations;                                \
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w); \
  bool kps_equal = (paddings[0] == paddings[2]) && ks_equal;        \
  bool pads_equal =                                                 \
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));
// static inline float activation_ss(float v, int activation_type, const struct ActivationParam& activation_params)
static inline float activation_ss(float v)
{
  v = fmax(v, 0.f);
  return v;
}
static int convolution(const float* i_data, float* o_data, const float *weight_data, const float *bias_data, int iw, int ih, int ic,
  int ww, int wh, int ow, int oh, int oc, int stride_w, int stride_h, int dilation_w, int dilation_h)
  // int activation_type, const struct ActivationParam& activation_params)
{
  const int w = iw;
  const int inch = ic;

  const int outw = ow;
  const int outh = oh;
  const int outch = oc;

  // const int bias_term = bias_data.empty() ? 0 : 1;
  if (bias_data == NULL)
  {
    printf("convolution bias_data == NULL\n");
  }
  const int bias_term = NULL == 0 ? 0 : 1;

  int kernel_w = ww;
  int kernel_h = wh;
  const int maxk = kernel_w * kernel_h;

  // stride_h = strides[0];
  // stride_w = strides[1];

  const float* kptr = weight_data;
  
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
  
  int k = 0;
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

          // const float* kptr = (const float*)weight_data + maxk * inch * p;
          
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
          
          outptr[j] = activation_ss(sum);
          // outptr[j] = activation_ss(sum, activation_type, activation_params);
      }

      outptr += outw;
    }
  }
  printf("the outptr distance of outptr and odata is %ld \n", outptr - o_data);
  return 0;
}

template <>
void Conv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  PREPARE_PARAM
  //! todo add conv_5x5_depthwise implement
  printf("Conv2dCompute<float,float> PrepareForRun is running \n");
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;
  if (kernel_w == 1 && stride_w == 1 && paddings[0] == 0 && kps_equal &&
      pads_equal) {
    flag_1x1gemm_ = true;
  } else {
    flag_1x1gemm_ = false;
  }

  bool nodilations = true;
  for (auto ele : *(param.dilations))
    if (ele != 1) nodilations = false;

  bool pad_all_equal = (paddings[0] == paddings[1]) &&
                       (paddings[1] == paddings[2]) &&
                       (paddings[2] == paddings[3]);
  bool flag_p = paddings[0] <= stride_h;

  //! select conv impl
  if (dw_kernel && kps_equal && flag_dw && pads_equal &&
      ((flag_dw_5x5 && no_dilation) || (flag_dw_3x3 && (groups & 3) == 0))) {
    impl_ = new DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>;
    VLOG(3) << "invoking conv_depthwise_3x3p0p1 or conv_depthwise_5x5";
    std::cout << "invoking conv_depthwise_3x3p0p1 or conv_depthwise_5x5" << std::endl;
  }

  // support 3x3s1p01,5x5s1p01,7x7s1p01
  //  3x3s2p012,5x5s1p012,7x7s1p012
  if (output_channel % 8 == 0 && groups == 1 &&
      (kernel_h == 3 || kernel_h == 5 || kernel_h == 7) &&
      (stride_h == 2 || stride_h == 1) && nodilations && kps_equal &&
      pad_all_equal && flag_p) {
#if defined(_WIN64) || defined(__MINGW64__) || \
    (defined(__CYGWIN__) && defined(__x86_64__)) || defined(__x86_64__)
    impl_ = new DirectConv<PRECISION(kFloat), PRECISION(kFloat)>();
    VLOG(3) << "invoking directConv";
    std::cout << "invoking directConv" << std::endl;
#endif
  }

  if (impl_) {
    impl_->SetContext(std::move(this->ctx_));
    impl_->SetParam(param);
    impl_->PrepareForRun();
    is_first_epoch_ = false;
  }
}

template <>
void Conv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  printf("Conv2dCompute<float float> Run is running1\n");
  // if (impl_) {
  //   printf("Conv2dCompute<float float> impl_ will be run\n");
  //   return impl_->Run();
  // }
  // auto& ctx = ctx_->As<X86Context>();
  INIT_PARAM
  printf("Conv2dCompute<float float> Run is running2\n");
  bool flag_bias = (param.bias != nullptr);
  unsigned int group_size_out = m * n;
  unsigned int group_size_weights = m * k;
  unsigned int group_size_coldata = n * k;
  unsigned int channel_in_size = chin * hin * win;
  unsigned int channel_out_size = chout * hout * wout;
  auto paddings = *param.paddings;

  printf("the paddings is \n");
  for (int i = 0; i < 4; i++)
  {
    printf("%d ", paddings[i]);
  }
  printf("\n");

  auto dilations = *param.dilations;
  printf("Conv2dCompute<float float> Run is running3\n");
  auto din = param.x->data<float>();
  auto dout = param.output->mutable_data<float>();
  auto weights = param.filter->data<float>();
  const float* bias_ptr =
      flag_bias ? static_cast<const float*>(param.bias->data<float>())
                : nullptr;
  float* col_data = nullptr;
  auto act_param = param.activation_param;
  // convolution(const float* i_data, float* o_data, const float *weight_data, const float *bias_data, int iw, int ic,
  // int ww, int wh, int stride_w, int ow, int oh, int oc, int stride_h, int dilation_w, int dilation_h, int activation_type, 
  // const struct ActivationParam& activation_params)
  printf("the dilation_w is %d \n", dilations[0]);
  printf("the dilation_h is %d \n", dilations[1]);
  if ((kh > hin || kw > win) && win == 1)
  {
    memcpy(dout, din, channel_in_size);
    return ;
  }
    
  printf("Conv2dCompute<float float> Run is running4\n");
  printf("the input data is \n");
  for (int i = 0; i < win * hin * chin; i++)
  {
    if (i % 16 == 0)
    {
      printf("\n");
    }
    printf("%f ", din[i]);
  }
  printf("\n");
  printf("the weight data is \n");
  for (int i = 0; i < chin * chout * kw * kh; i++)
  {
    if (i % 16 == 0)
    {
      printf("\n");
    }
    printf("%f ", weights[i]);
  }
  printf("\n");
  // printf("Conv2dCompute<float float> Run convolution will be running \n");
  convolution(din, dout, weights, bias_ptr, win, hin, chin, kw, kh, wout, hout, chout, param.strides[1], param.strides[0], dilations[0], dilations[1]);
    // act_param.active_type, act_param);
  printf("the output data is \n");
  for (int i = 0; i < wout * hout * chout; i++)
  {
    if (i == 16)
    {
      printf("\n");
    }
    printf("%f ", dout[i]);
  }
  printf("\n");
  
  // if (!flag_1x1gemm_) {
  //   printf("flag_1x1gemm_ is false \n");
  //   size_t col_size = group_size_coldata * group;
  //   size_t col_data_size = static_cast<size_t>(col_size * sizeof(float));
  //   col_data = static_cast<float*>(TargetMalloc(TARGET(kX86), col_data_size));
  // }
  // auto act_param = param.activation_param;
  // paddle::lite::x86::math::Blas<lite::TargetType::kX86> matmul(ctx);
  // for (int i = 0; i < num; i++) {
  //   const float* din_batch = din + i * channel_in_size;
  //   float* dout_batch = dout + i * channel_out_size;
  //   const float* din_data = din_batch;
  //   if (!flag_1x1gemm_) {
  //     lite::x86::math::im2col<float>(din_batch,
  //                                    chin,
  //                                    hin,
  //                                    win,
  //                                    w_dims[2],
  //                                    w_dims[3],
  //                                    paddings[0],
  //                                    paddings[1],
  //                                    paddings[2],
  //                                    paddings[3],
  //                                    param.strides[0],
  //                                    param.strides[1],
  //                                    dilations[0],
  //                                    dilations[1],
  //                                    col_data);
  //     din_data = static_cast<const float*>(col_data);
  //   }

  //   for (int g = 0; g < group; g++) {
  //     const float* col_data_group = din_data + g * group_size_coldata;
  //     const float* weights_group = weights + g * group_size_weights;
  //     float* dout_group = dout_batch + g * group_size_out;
  //     if (n == 1) {
  //       matmul.GEMV<float>(
  //           false, m, k, 1.f, weights_group, col_data_group, 0.f, dout_group);
  //     } else {
  //       matmul.GEMM<float>(false,
  //                          false,
  //                          m,
  //                          n,
  //                          k,
  //                          1.f,
  //                          weights_group,
  //                          k,
  //                          col_data_group,
  //                          n,
  //                          0.f,
  //                          dout_group,
  //                          n);
  //     }
  //   }
  //   //! bias and activate
  //   lite::x86::math::fill_bias_act(
  //       dout_batch, bias_ptr, chout, wout * hout, flag_bias, &act_param);
  // }
  // if (!flag_1x1gemm_) TargetFree(TARGET(kX86), col_data);
}

template <>
void Conv2dCompute<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  PREPARE_PARAM_INT8
  if (kernel_w == 1 && stride_w == 1 && paddings[0] == 0 && kps_equal &&
      pads_equal) {
    flag_1x1gemm_ = true;
  } else {
    flag_1x1gemm_ = false;
  }

  auto o_dims = param.output->dims();
  int m = output_channel / groups;
  int n = o_dims[2] * o_dims[3];
  int k = input_channel * kernel_h * kernel_w / groups;
  int group_size_weights = m * k;
  auto weights = param.filter->data<int8_t>();
  bool flag_bias = (param.bias != nullptr);
  const float* bias_ptr =
      flag_bias ? static_cast<const float*>(param.bias->data<float>())
                : nullptr;
  auto w_scale_ = param.weight_scale;

  Tensor weight_s{};
  weight_s.Resize({param.filter->dims()[0]});
  weight_s.set_precision(PRECISION(kFloat));
  auto weight_tmp = weight_s.mutable_data<float>();

  if (w_scale_.size() != 1 && w_scale_.size() != param.filter->dims()[0]) {
    LOG(FATAL) << "weights scale size must equal to filter size";
  }
  if (w_scale_.size() == 1) {
    for (int i = 0; i < param.filter->dims()[0]; ++i) {
      weight_tmp[i] = (w_scale_[0]);
    }
  } else {
    for (int i = 0; i < param.filter->dims()[0]; ++i) {
      weight_tmp[i] = (w_scale_[i]);
    }
  }
  auto weight_scale = weight_s.data<float>();
  const float input_scale = param.input_scale;
  const float output_scale = param.output_scale;
  int relu_type = 0;
  float relu_alpha = 1.f;

  if (param.activation_param.active_type == lite_api::ActivationType::kRelu6) {
    relu_type = 2;
    relu_alpha = param.activation_param.Relu_clipped_coef;
  } else if (param.activation_param.active_type ==
             lite_api::ActivationType::kLeakyRelu) {
    relu_type = 3;
    relu_alpha = param.activation_param.Leaky_relu_alpha;
  } else if (param.activation_param.active_type ==
             lite_api::ActivationType::kRelu) {
    relu_type = 1;
  }
  for (int g = 0; g < groups; g++) {
    const int8_t* weights_group = weights + g * group_size_weights;
    auto gemm = new lite::x86::math::generate_gemm_s8u8_x86_kern<float>(
        false,
        false,
        m,
        n,
        k,
        weights_group,
        n,
        weight_scale + g * m,
        input_scale,
        output_scale,
        bias_ptr + g * m,
        relu_type,
        relu_alpha);
    gemm_s8_ptr_float_.push_back(gemm);
  }
}

template <>
void Conv2dCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  INIT_PARAM
  int group_size_coldata = n * k;
  int channel_size_in = hin * win;
  int channel_size_out = hout * wout;
  int chin_per_group = chin / group;
  int group_size_weights = m * k;
  int8_t* col_data = nullptr;
  auto din = param.x->data<int8_t>();
  auto dout = param.output->mutable_data<float>();
  auto weights = param.filter->data<int8_t>();
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  if (!flag_1x1gemm_) {
    int col_size = group * group_size_coldata;
    col_data = static_cast<int8_t*>(
        TargetMalloc(TARGET(kX86), col_size * sizeof(int8_t)));
  }
  for (int b = 0; b < num; ++b) {
    for (int g = 0; g < group; ++g) {
      float* dout_group = dout + (b * chout + g * m) * channel_size_out;
      const int8_t* din_group =
          din + (b * chin + g * chin_per_group) * channel_size_in;
      const int8_t* weights_group = weights + g * group_size_weights;

      if (!flag_1x1gemm_) {
        lite::x86::math::im2col<int8_t>(din_group,
                                        chin_per_group,
                                        hin,
                                        win,
                                        kh,
                                        kw,
                                        paddings[0],
                                        paddings[1],
                                        paddings[2],
                                        paddings[3],
                                        param.strides[0],
                                        param.strides[1],
                                        dilations[0],
                                        dilations[1],
                                        col_data);
        gemm_s8_ptr_float_[g]->compute(weights_group, col_data, dout_group);
      } else {
        gemm_s8_ptr_float_[g]->compute(weights_group, din_group, dout_group);
      }
    }
  }
  if (!flag_1x1gemm_) TargetFree(TARGET(kX86), col_data);
}

template <>
void Conv2dCompute<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
  PREPARE_PARAM_INT8
  if (kernel_w == 1 && stride_w == 1 && paddings[0] == 0 && kps_equal &&
      pads_equal) {
    flag_1x1gemm_ = true;
  } else {
    flag_1x1gemm_ = false;
  }

  auto o_dims = param.output->dims();
  int m = output_channel / groups;
  int n = o_dims[2] * o_dims[3];
  int k = input_channel * kernel_h * kernel_w / groups;
  int group_size_weights = m * k;
  auto weights = param.filter->data<int8_t>();
  bool flag_bias = (param.bias != nullptr);
  const float* bias_ptr =
      flag_bias ? static_cast<const float*>(param.bias->data<float>())
                : nullptr;
  auto w_scale_ = param.weight_scale;
  Tensor weight_s{};
  weight_s.Resize({param.filter->dims()[0]});
  weight_s.set_precision(PRECISION(kFloat));
  auto weight_tmp = weight_s.mutable_data<float>();

  if (w_scale_.size() != 1 && w_scale_.size() != param.filter->dims()[0]) {
    LOG(FATAL) << "weights scale size must equal to filter size";
  }
  if (w_scale_.size() == 1) {
    for (int i = 0; i < param.filter->dims()[0]; ++i) {
      weight_tmp[i] = (w_scale_[0]);
    }
  } else {
    for (int i = 0; i < param.filter->dims()[0]; ++i) {
      weight_tmp[i] = (w_scale_[i]);
    }
  }
  const float input_scale = param.input_scale;
  const float output_scale = param.output_scale;
  int relu_type = 0;
  float relu_alpha = 1.f;

  if (param.activation_param.has_active) {
    if (param.activation_param.active_type ==
        lite_api::ActivationType::kRelu6) {
      relu_type = 2;
      relu_alpha = param.activation_param.Relu_clipped_coef / output_scale;
    } else if (param.activation_param.active_type ==
               lite_api::ActivationType::kLeakyRelu) {
      relu_type = 3;
      relu_alpha = param.activation_param.Leaky_relu_alpha / output_scale;
    } else if (param.activation_param.active_type ==
               lite_api::ActivationType::kRelu) {
      relu_type = 1;
    }
  }

  auto weight_scale = weight_s.data<float>();
  for (int g = 0; g < groups; g++) {
    const int8_t* weights_group = weights + g * group_size_weights;
    auto gemm = new lite::x86::math::generate_gemm_s8u8_x86_kern<int8_t>(
        false,
        false,
        m,
        n,
        k,
        weights_group,
        n,
        weight_scale + g * m,
        input_scale,
        output_scale,
        bias_ptr + g * m,
        relu_type,
        relu_alpha);
    gemm_s8_ptr_int8_.push_back(gemm);
  }
}

template <>
void Conv2dCompute<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  INIT_PARAM
  int group_size_coldata = n * k;
  int channel_size_in = hin * win;
  int channel_size_out = hout * wout;
  int chin_per_group = chin / group;
  int group_size_weights = m * k;
  int8_t* col_data = nullptr;
  auto din = param.x->data<int8_t>();
  auto dout = param.output->mutable_data<int8_t>();
  auto weights = param.filter->data<int8_t>();
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  if (!flag_1x1gemm_) {
    int col_size = group * group_size_coldata;
    col_data = static_cast<int8_t*>(
        TargetMalloc(TARGET(kX86), col_size * sizeof(int8_t)));
  }
  for (int b = 0; b < num; ++b) {
    for (int g = 0; g < group; ++g) {
      int8_t* dout_group = dout + (b * chout + g * m) * channel_size_out;
      const int8_t* din_group =
          din + (b * chin + g * chin_per_group) * channel_size_in;
      const int8_t* weights_group = weights + g * group_size_weights;

      if (!flag_1x1gemm_) {
        lite::x86::math::im2col<int8_t>(din_group,
                                        chin_per_group,
                                        hin,
                                        win,
                                        kh,
                                        kw,
                                        paddings[0],
                                        paddings[1],
                                        paddings[2],
                                        paddings[3],
                                        param.strides[0],
                                        param.strides[1],
                                        dilations[0],
                                        dilations[1],
                                        col_data);
        gemm_s8_ptr_int8_[g]->compute(weights_group, col_data, dout_group);
      } else {
        gemm_s8_ptr_int8_[g]->compute(weights_group, din_group, dout_group);
      }
    }
  }
  if (!flag_1x1gemm_) TargetFree(TARGET(kX86), col_data);
}

#undef PREPARE_PARAM
#undef PREPARE_PARAM_INT8
#undef INIT_PARAM
}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::x86::Conv2dCompute<PRECISION(kFloat),
                                                  PRECISION(kFloat)>
    ConvFp32;
typedef paddle::lite::kernels::x86::Conv2dCompute<PRECISION(kInt8),
                                                  PRECISION(kFloat)>
    ConvInt8_Fp32;
typedef paddle::lite::kernels::x86::Conv2dCompute<PRECISION(kInt8),
                                                  PRECISION(kInt8)>
    ConvInt8_Int8;

REGISTER_LITE_KERNEL(conv2d, kX86, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("SecondInput", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kX86, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(conv2d, kX86, kInt8, kNCHW, ConvInt8_Int8, int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindInput("SecondInput",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(conv2d, kX86, kInt8, kNCHW, ConvInt8_Fp32, fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindInput("SecondInput",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(
    depthwise_conv2d, kX86, kInt8, kNCHW, ConvInt8_Int8, int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(
    depthwise_conv2d, kX86, kInt8, kNCHW, ConvInt8_Fp32, fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();

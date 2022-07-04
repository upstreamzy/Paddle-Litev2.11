#include "lite/backends/riscv/math/conv_depthwise.h"

#include "lite/core/parallel_defines.h"
namespace paddle {
namespace lite {
namespace riscv {
namespace math {
void conv_depthwise_3x3s1p0_bias_s_relu(float *dout,
                                        const float *din,
                                        const float *weights,
                                        const float *bias,
                                        bool flag_bias,
                                        bool flag_relu,
                                        const int num,
                                        const int ch_in,
                                        const int h_in,
                                        const int w_in,
                                        const int h_out,
                                        const int w_out,
                                        RISCVContext *ctx) {
  //! 3x3s1 convolution, implemented by direct algorithm
  //! pad is done implicit
  //! for 4x6 convolution window
  const int right_pad_idx[8] = {5, 4, 3, 2, 1, 0, 0, 0};
  const float zero_ptr[4] = {0.f, 0.f, 0.f, 0.f};

  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask_rp1 =
      vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(6 - w_in));
  uint32x4_t vmask_rp2 =
      vcgeq_s32(vld1q_s32(right_pad_idx + 4), vdupq_n_s32(6 - w_in));

  unsigned int vmask[8];
  vst1q_u32(vmask, vmask_rp1);
  vst1q_u32(vmask + 4, vmask_rp2);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    LITE_PARALLEL_BEGIN(i, tid, ch_in) {
      float *dout_channel = dout_batch + i * size_out_channel;
      const float *din_channel = din_batch + i * size_in_channel;
      const float *weight_ptr = weights + i * 9;
      float32x4_t wr0 = vld1q_f32(weight_ptr);
      float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
      float32x4_t wr2 = vld1q_f32(weight_ptr + 6);

#ifdef __aarch64__
      float32x4_t wbias;
      if (flag_bias) {
        wbias = vdupq_n_f32(bias[i]);
      } else {
        wbias = vdupq_n_f32(0.f);
      }
#endif  // __aarch64__

      float out_buf1[4];
      float out_buf2[4];
      float trash_buf[4];

      float *doutr0 = dout_channel;
      float *doutr1 = dout_channel + w_out;

      for (int j = 0; j < h_out; j += 2) {
        const float *dr0 = din_channel + j * w_in;
        const float *dr1 = dr0 + w_in;
        const float *dr2 = dr1 + w_in;
        const float *dr3 = dr2 + w_in;

        doutr0 = dout_channel + j * w_out;
        doutr1 = doutr0 + w_out;

        if (j + 3 >= h_in) {
          switch (j + 3 - h_in) {
            case 2:
              dr1 = zero_ptr;
            case 1:
              dr2 = zero_ptr;
              doutr1 = trash_buf;
            case 0:
              dr3 = zero_ptr;
              if (j + 2 > h_out) {
                doutr1 = trash_buf;
              }
            default:
              break;
          }
        }
#ifdef __aarch64__
        asm volatile(COMPUTE_S_S1_P0 RESULT_S_S1_RELU
                     : [din0] "+r"(dr0),
                       [din1] "+r"(dr1),
                       [din2] "+r"(dr2),
                       [din3] "+r"(dr3)
                     : [wr0] "w"(wr0),
                       [wr1] "w"(wr1),
                       [wr2] "w"(wr2),
                       [vbias] "w"(wbias),
                       [mask1] "w"(vmask_rp1),
                       [mask2] "w"(vmask_rp2),
                       [zero] "w"(vzero),
                       [out1] "r"(out_buf1),
                       [out2] "r"(out_buf2)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v11",
                       "v12",
                       "v13",
                       "v14",
                       "v15");
#else
        unsigned int *vmask_ptr = vmask;
        float bias_val = flag_bias ? bias[i] : 0.f;
        asm volatile(COMPUTE_S_S1_P0 RESULT_S_S1_RELU
                     : [din0] "+r"(dr0),
                       [din1] "+r"(dr1),
                       [din2] "+r"(dr2),
                       [din3] "+r"(dr3),
                       [vmask] "+r"(vmask_ptr)
                     : [wr0] "w"(wr0),
                       [wr1] "w"(wr1),
                       [wr2] "w"(wr2),
                       [vzero] "w"(vzero),
                       [bias_val] "r"(bias_val),
                       [out1] "r"(out_buf1),
                       [out2] "r"(out_buf2)
                     : "cc",
                       "memory",
                       "q4",
                       "q5",
                       "q6",
                       "q7",
                       "q8",
                       "q9",
                       "q10",
                       "q11",
                       "q12",
                       "q13",
                       "q14",
                       "q15");
#endif
        for (int w = 0; w < w_out; ++w) {
          *doutr0++ = out_buf1[w];
          *doutr1++ = out_buf2[w];
        }
      }  // end of processing heights
    }    // end of processing channels
    LITE_PARALLEL_END()
  }  // end of processing batchs
}

void conv_depthwise_3x3s1p0_bias_relu(float *dout,
                                      const float *din,
                                      const float *weights,
                                      const float *bias,
                                      bool flag_bias,
                                      bool flag_relu,
                                      const int num,
                                      const int ch_in,
                                      const int h_in,
                                      const int w_in,
                                      const int h_out,
                                      const int w_out,
                                      RISCVContext *ctx) {
  //! pad is done implicit
  const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  float *zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, (w_in + 6) * sizeof(float));
  float *write_ptr = zero_ptr + (w_in + 6);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  unsigned int vmask[4];
  auto &&res = right_mask_3x3s1_fp32(w_in, w_out, 0, vmask);
  uint32_t cnt_col = res.first;
  uint32_t remain = res.second;
  uint32_t right_pad_num = (4 - remain) * 4;

  float32x4_t vzero = vdupq_n_f32(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float *dout_ptr = dout_batch + c * size_out_channel;

      const float *din_ch_ptr = din_batch + c * size_in_channel;

      float bias_val = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};

      const float *wei_ptr = weights + c * w_stride;

      float32x4_t wr0 = vld1q_f32(wei_ptr);
      float32x4_t wr1 = vld1q_f32(wei_ptr + 3);
      float32x4_t wr2 = vld1q_f32(wei_ptr + 6);

      float *doutr0 = dout_ptr;
      float *doutr1 = doutr0 + w_out;
      float *doutr2 = doutr1 + w_out;
      float *doutr3 = doutr2 + w_out;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;
      const float *dr3 = dr2 + w_in;
      const float *dr4 = dr3 + w_in;
      const float *dr5 = dr4 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;
      const float *din_ptr3 = dr3;
      const float *din_ptr4 = dr4;
      const float *din_ptr5 = dr5;

      float *ptr_zero = const_cast<float *>(zero);
#ifdef __aarch64__
      for (int i = 0; i < h_out; i += 4) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;
        din_ptr5 = dr5;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        doutr2 = doutr1 + w_out;
        doutr3 = doutr2 + w_out;

        dr0 = dr4;
        dr1 = dr5;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;
        dr5 = dr4 + w_in;

        //! process bottom pad
        if (i + 5 >= h_in) {
          switch (i + 5 - h_in) {
            case 4:
              din_ptr1 = zero_ptr;
            case 3:
              din_ptr2 = zero_ptr;
            case 2:
              din_ptr3 = zero_ptr;
            case 1:
              din_ptr4 = zero_ptr;
            case 0:
              din_ptr5 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 4 > h_out) {
          switch (i + 4 - h_out) {
            case 3:
              doutr1 = write_ptr;
            case 2:
              doutr2 = write_ptr;
            case 1:
              doutr3 = write_ptr;
            default:
              break;
          }
        }

        int cnt = cnt_col;
        asm volatile(
            INIT_S1
            "ld1 {v8.4s}, [%[din_ptr4]], #16   \n"  /*vld1q_f32(din_ptr0)*/
            "ld1 {v10.4s}, [%[din_ptr5]], #16   \n" /*vld1q_f32(din_ptr0)*/
            "ext  v16.16b, v0.16b, v1.16b, #4 \n"   /* v16 = 1234 */
            "ext  v17.16b, v0.16b, v1.16b, #8 \n"   /* v17 = 2345 */
            "ld1 {v9.4s}, [%[din_ptr4]]   \n"       /*vld1q_f32(din_ptr0)*/
            "ld1 {v11.4s}, [%[din_ptr5]]   \n"      /*vld1q_f32(din_ptr0)*/
            MID_COMPUTE_S1 MID_RESULT_S1_RELU
            "cmp  %w[remain], #1             \n"
            "blt 0f                         \n" RIGHT_COMPUTE_S1
                RIGHT_RESULT_S1_RELU "0: \n"
            : PARAM1
            : PARAM2, [remain] "r"(remain)
            : ASM_PARAM);
        dout_ptr = dout_ptr + 4 * w_out;
      }
#else
      for (int i = 0; i < h_out; i += 2) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = dout_ptr + w_out;

        dr0 = dr2;
        dr1 = dr3;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        //! process bottom pad
        if (i + 3 >= h_in) {
          switch (i + 3 - h_in) {
            case 2:
              din_ptr1 = zero_ptr;
            case 1:
              din_ptr2 = zero_ptr;
            case 0:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;

        unsigned int *vmask_ptr = vmask;
        asm volatile(INIT_S1
                     "sub %[din0_ptr], #8 @ 0pad + 2 float data overlap\n"
                     "sub %[din1_ptr], #8 @ 0pad + 2 float data overlap\n"
                     "sub %[din2_ptr], #8 @ 0pad + 2 float data overlap\n"
                     "sub %[din3_ptr], #8 @ 0pad + 2 float data overlap\n"
                     "vext.32  q6, q8, q9, #1     @ 0012\n"
                     "vext.32  q7, q8, q9, #2     @ 1234\n" MID_COMPUTE_S1
                         MID_RESULT_S1_RELU
                     "cmp  %[remain], #1             \n"
                     "blt 0f                         \n" RIGHT_COMPUTE_S1
                         RIGHT_RESULT_S1_RELU "0:                         \n"
                     : PARAM1
                     : PARAM2, [remain] "r"(remain)
                     : ASM_PARAM);
        dout_ptr += 2 * w_out;
      }  //! end of processing mid rows
#endif
    }
    LITE_PARALLEL_END()
  }
}

}
}
}
}
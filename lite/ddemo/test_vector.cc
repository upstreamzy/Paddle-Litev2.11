// #define __riscv_vector
#include <riscv_vector.h>
int main(void)
{
    float in1[4] = {-1.f,0.f,+1.f,2.f};
    float out1=0;
    word_type vl = vsetvl_e32m8(4);
    vfloat32m8_t _add = vle32_v_f32m8(in1,vl);
    vfloat32m1_t _sum = vfmv_s_f_f32m1(vundefined_f32m1(),out1,vl);
    _sum = vfredosum_vs_f32m8_f32m1(_sum, _add, _sum, vl);
    out1 = vfmv_f_s_f32m1_f32(_sum);
    return 0;
}
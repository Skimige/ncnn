// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tanh_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#endif // __loongarch_sx

namespace ncnn {

TanH_loongarch::TanH_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif
}

int TanH_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __loongarch_sx
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = tanh_ps(_p);
            __lsx_vst(_p, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = tanhf(*ptr);
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn

// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void linear_coeffs(int w, int outw, int* xofs, float* alpha, int align_corner)
{
    double scale = (double)w / outw;
    if (align_corner)
    {
        scale = (double)(w - 1) / (outw - 1);
    }

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner)
        {
            fx = (float)(dx * scale);
        }

        int sx = floor(fx);
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
}

static void resize_bilinear_image(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src.row(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;

#if __riscv_vector
            const unsigned int* pxofs = (const unsigned int*)xofs;
            int n = w;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m4(n);

                vuint32m4_t _sx = __riscv_vmul_vx_u32m4(__riscv_vle32_v_u32m4(pxofs, vl), sizeof(float), vl);

                vfloat32m4x2_t _S1 = __riscv_vloxseg2ei32_v_f32m4x2(S1, _sx, vl);
                vfloat32m4_t _S1p0 = __riscv_vget_v_f32m4x2_f32m4(_S1, 0);
                vfloat32m4_t _S1p1 = __riscv_vget_v_f32m4x2_f32m4(_S1, 1);

                vfloat32m4x2_t _a = __riscv_vlseg2e32_v_f32m4x2(alphap, vl);
                vfloat32m4_t _a0 = __riscv_vget_v_f32m4x2_f32m4(_a, 0);
                vfloat32m4_t _a1 = __riscv_vget_v_f32m4x2_f32m4(_a, 1);

                vfloat32m4_t _rows1 = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(_S1p0, _a0, vl), _S1p1, _a1, vl);

                __riscv_vse32_v_f32m4(rows1p, _rows1, vl);

                pxofs += vl;
                alphap += vl * 2;
                rows1p += vl;
                n -= vl;
            }
#else  // __riscv_vector
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
#endif // __riscv_vector
        }
        else
        {
            // hresize two rows
            const float* S0 = src.row(sy);
            const float* S1 = src.row(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;

#if __riscv_vector
            const unsigned int* pxofs = (const unsigned int*)xofs;
            int n = w;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m4(n);

                vuint32m4_t _sx = __riscv_vmul_vx_u32m4(__riscv_vle32_v_u32m4(pxofs, vl), sizeof(float), vl);

                vfloat32m4x2_t _S0 = __riscv_vloxseg2ei32_v_f32m4x2(S0, _sx, vl);
                vfloat32m4x2_t _S1 = __riscv_vloxseg2ei32_v_f32m4x2(S1, _sx, vl);
                vfloat32m4_t _S0p0 = __riscv_vget_v_f32m4x2_f32m4(_S0, 0);
                vfloat32m4_t _S0p1 = __riscv_vget_v_f32m4x2_f32m4(_S0, 1);
                vfloat32m4_t _S1p0 = __riscv_vget_v_f32m4x2_f32m4(_S1, 0);
                vfloat32m4_t _S1p1 = __riscv_vget_v_f32m4x2_f32m4(_S1, 1);

                vfloat32m4x2_t _a = __riscv_vlseg2e32_v_f32m4x2(alphap, vl);
                vfloat32m4_t _a0 = __riscv_vget_v_f32m4x2_f32m4(_a, 0);
                vfloat32m4_t _a1 = __riscv_vget_v_f32m4x2_f32m4(_a, 1);

                vfloat32m4_t _rows0 = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(_S0p0, _a0, vl), _S0p1, _a1, vl);
                vfloat32m4_t _rows1 = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(_S1p0, _a0, vl), _S1p1, _a1, vl);

                __riscv_vse32_v_f32m4(rows0p, _rows0, vl);
                __riscv_vse32_v_f32m4(rows1p, _rows1, vl);

                pxofs += vl;
                alphap += vl * 2;
                rows0p += vl;
                rows1p += vl;
                n -= vl;
            }
#else  // __riscv_vector
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0] * a0 + S0p[1] * a1;
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
#endif // __riscv_vector
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);

#if __riscv_vector
        int n = w;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);

            vfloat32m8_t _rows0 = __riscv_vle32_v_f32m8(rows0p, vl);
            vfloat32m8_t _rows1 = __riscv_vle32_v_f32m8(rows1p, vl);

            vfloat32m8_t _Dp = __riscv_vfmacc_vf_f32m8(__riscv_vfmul_vf_f32m8(_rows0, b0, vl), b1, _rows1, vl);

            __riscv_vse32_v_f32m8(Dp, _Dp, vl);

            Dp += vl;
            rows0p += vl;
            rows1p += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < w; i++)
        {
            //             D[x] = rows0[x]*b0 + rows1[x]*b1;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }
#endif // __riscv_vector

        beta += 2;
    }
}

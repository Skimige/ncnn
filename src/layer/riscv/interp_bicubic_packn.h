// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void resize_bicubic_image_packn(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)packn * 4u, packn);
    Mat rowsbuf1(w, (size_t)packn * 4u, packn);
    Mat rowsbuf2(w, (size_t)packn * 4u, packn);
    Mat rowsbuf3(w, (size_t)packn * 4u, packn);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

    int prev_sy1 = -3;

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
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const float* S3p = S3 + sx;

                vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p - packn, vl);
                vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p, vl);
                vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + packn, vl);
                vfloat32m1_t _S33 = __riscv_vle32_v_f32m1(S3p + packn * 2, vl);
                vfloat32m1_t _rows3 = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                __riscv_vse32_v_f32m1(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                vfloat32m1_t _S20 = __riscv_vle32_v_f32m1(S2p - packn, vl);
                vfloat32m1_t _S21 = __riscv_vle32_v_f32m1(S2p, vl);
                vfloat32m1_t _S22 = __riscv_vle32_v_f32m1(S2p + packn, vl);
                vfloat32m1_t _S23 = __riscv_vle32_v_f32m1(S2p + packn * 2, vl);
                vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p - packn, vl);
                vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p, vl);
                vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + packn, vl);
                vfloat32m1_t _S33 = __riscv_vle32_v_f32m1(S3p + packn * 2, vl);
                vfloat32m1_t _rows2 = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat32m1_t _rows3 = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                __riscv_vse32_v_f32m1(rows2p + dx * packn, _rows2, vl);
                __riscv_vse32_v_f32m1(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            float* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            const float* S1 = src.row(sy);
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                vfloat32m1_t _S10 = __riscv_vle32_v_f32m1(S1p - packn, vl);
                vfloat32m1_t _S11 = __riscv_vle32_v_f32m1(S1p, vl);
                vfloat32m1_t _S12 = __riscv_vle32_v_f32m1(S1p + packn, vl);
                vfloat32m1_t _S13 = __riscv_vle32_v_f32m1(S1p + packn * 2, vl);
                vfloat32m1_t _S20 = __riscv_vle32_v_f32m1(S2p - packn, vl);
                vfloat32m1_t _S21 = __riscv_vle32_v_f32m1(S2p, vl);
                vfloat32m1_t _S22 = __riscv_vle32_v_f32m1(S2p + packn, vl);
                vfloat32m1_t _S23 = __riscv_vle32_v_f32m1(S2p + packn * 2, vl);
                vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p - packn, vl);
                vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p, vl);
                vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + packn, vl);
                vfloat32m1_t _S33 = __riscv_vle32_v_f32m1(S3p + packn * 2, vl);
                vfloat32m1_t _rows1 = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_S10, alphap[0], vl), alphap[1], _S11, vl), alphap[2], _S12, vl), alphap[3], _S13, vl);
                vfloat32m1_t _rows2 = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat32m1_t _rows3 = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                __riscv_vse32_v_f32m1(rows1p + dx * packn, _rows1, vl);
                __riscv_vse32_v_f32m1(rows2p + dx * packn, _rows2, vl);
                __riscv_vse32_v_f32m1(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const float* S0 = src.row(sy - 1);
            const float* S1 = src.row(sy);
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                vfloat32m1_t _S00 = __riscv_vle32_v_f32m1(S0p - packn, vl);
                vfloat32m1_t _S01 = __riscv_vle32_v_f32m1(S0p, vl);
                vfloat32m1_t _S02 = __riscv_vle32_v_f32m1(S0p + packn, vl);
                vfloat32m1_t _S03 = __riscv_vle32_v_f32m1(S0p + packn * 2, vl);
                vfloat32m1_t _S10 = __riscv_vle32_v_f32m1(S1p - packn, vl);
                vfloat32m1_t _S11 = __riscv_vle32_v_f32m1(S1p, vl);
                vfloat32m1_t _S12 = __riscv_vle32_v_f32m1(S1p + packn, vl);
                vfloat32m1_t _S13 = __riscv_vle32_v_f32m1(S1p + packn * 2, vl);
                vfloat32m1_t _S20 = __riscv_vle32_v_f32m1(S2p - packn, vl);
                vfloat32m1_t _S21 = __riscv_vle32_v_f32m1(S2p, vl);
                vfloat32m1_t _S22 = __riscv_vle32_v_f32m1(S2p + packn, vl);
                vfloat32m1_t _S23 = __riscv_vle32_v_f32m1(S2p + packn * 2, vl);
                vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p - packn, vl);
                vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p, vl);
                vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + packn, vl);
                vfloat32m1_t _S33 = __riscv_vle32_v_f32m1(S3p + packn * 2, vl);
                vfloat32m1_t _rows0 = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_S00, alphap[0], vl), alphap[1], _S01, vl), alphap[2], _S02, vl), alphap[3], _S03, vl);
                vfloat32m1_t _rows1 = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_S10, alphap[0], vl), alphap[1], _S11, vl), alphap[2], _S12, vl), alphap[3], _S13, vl);
                vfloat32m1_t _rows2 = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat32m1_t _rows3 = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                __riscv_vse32_v_f32m1(rows0p + dx * packn, _rows0, vl);
                __riscv_vse32_v_f32m1(rows1p + dx * packn, _rows1, vl);
                __riscv_vse32_v_f32m1(rows2p + dx * packn, _rows2, vl);
                __riscv_vse32_v_f32m1(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];
        float b2 = beta[2];
        float b3 = beta[3];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            vfloat32m1_t _rows0 = __riscv_vle32_v_f32m1(rows0p, vl);
            vfloat32m1_t _rows1 = __riscv_vle32_v_f32m1(rows1p, vl);
            vfloat32m1_t _rows2 = __riscv_vle32_v_f32m1(rows2p, vl);
            vfloat32m1_t _rows3 = __riscv_vle32_v_f32m1(rows3p, vl);

            vfloat32m1_t _Dp = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_rows0, b0, vl), b1, _rows1, vl), b2, _rows2, vl), b3, _rows3, vl);

            __riscv_vse32_v_f32m1(Dp, _Dp, vl);

            Dp += packn;
            rows0p += packn;
            rows1p += packn;
            rows2p += packn;
            rows3p += packn;
        }

        beta += 4;
    }
}

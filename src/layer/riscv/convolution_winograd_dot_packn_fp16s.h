// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_winograd_dot_packn_fp16sa_rvv(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);

    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 2u * packn, packn, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
    if (tiles >= 8)
        bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, batch, 2u * packn, packn, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, batch, 2u * packn, packn, opt.workspace_allocator);
    else if (tiles >= 2)
        bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, batch, 2u * packn, packn, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 2u * packn, packn, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
        for (; i + 7 < tiles; i += 8)
        {
            __fp16* tmpptr = tm2.row<__fp16>(i / 8);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * packn;

            for (int q = 0; q < inch; q++)
            {
#if C906
                for (int l = 0; l < packn; l++)
                {
                    tmpptr[0] = r0[l];
                    tmpptr[1] = r0[l + packn];
                    tmpptr[2] = r0[l + packn * 2];
                    tmpptr[3] = r0[l + packn * 3];
                    tmpptr[4] = r0[l + packn * 4];
                    tmpptr[5] = r0[l + packn * 5];
                    tmpptr[6] = r0[l + packn * 6];
                    tmpptr[7] = r0[l + packn * 7];
                    tmpptr += 8;
                }

                r0 += bottom_blob_tm.cstep * packn;
#else
                vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(r0, vl);
                vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(r0 + packn, vl);
                vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(r0 + packn * 2, vl);
                vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(r0 + packn * 3, vl);
                vfloat16m1_t _val4 = __riscv_vle16_v_f16m1(r0 + packn * 4, vl);
                vfloat16m1_t _val5 = __riscv_vle16_v_f16m1(r0 + packn * 5, vl);
                vfloat16m1_t _val6 = __riscv_vle16_v_f16m1(r0 + packn * 6, vl);
                vfloat16m1_t _val7 = __riscv_vle16_v_f16m1(r0 + packn * 7, vl);
                __riscv_vsseg8e16_v_f16m1x8(tmpptr, __riscv_vcreate_v_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                r0 += bottom_blob_tm.cstep * packn;
                tmpptr += packn * 8;
#endif
            }
        }
        for (; i + 3 < tiles; i += 4)
        {
            __fp16* tmpptr = tm2.row<__fp16>(i / 8 + (i % 8) / 4);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * packn;

            for (int q = 0; q < inch; q++)
            {
#if C906
                for (int l = 0; l < packn; l++)
                {
                    tmpptr[0] = r0[l];
                    tmpptr[1] = r0[l + packn];
                    tmpptr[2] = r0[l + packn * 2];
                    tmpptr[3] = r0[l + packn * 3];
                    tmpptr += 4;
                }

                r0 += bottom_blob_tm.cstep * packn;
#else
                vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(r0, vl);
                vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(r0 + packn, vl);
                vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(r0 + packn * 2, vl);
                vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(r0 + packn * 3, vl);
                __riscv_vsseg4e16_v_f16m1x4(tmpptr, __riscv_vcreate_v_f16m1x4(_val0, _val1, _val2, _val3), vl);

                r0 += bottom_blob_tm.cstep * packn;
                tmpptr += packn * 4;
#endif
            }
        }
        for (; i + 1 < tiles; i += 2)
        {
            __fp16* tmpptr = tm2.row<__fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * packn;

            for (int q = 0; q < inch; q++)
            {
#if C906
                for (int l = 0; l < packn; l++)
                {
                    tmpptr[0] = r0[l];
                    tmpptr[1] = r0[l + packn];
                    tmpptr += 2;
                }

                r0 += bottom_blob_tm.cstep * packn;
#else
                vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(r0, vl);
                vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(r0 + packn, vl);
                __riscv_vsseg2e16_v_f16m1x2(tmpptr, __riscv_vcreate_v_f16m1x2(_val0, _val1), vl);

                r0 += bottom_blob_tm.cstep * packn;
                tmpptr += packn * 2;
#endif
            }
        }
        for (; i < tiles; i++)
        {
            __fp16* tmpptr = tm2.row<__fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * packn;

            for (int q = 0; q < inch; q++)
            {
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(r0, vl);
                __riscv_vse16_v_f16m1(tmpptr, _val, vl);

                r0 += bottom_blob_tm.cstep * packn;
                tmpptr += packn;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 2u * packn, packn, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        __fp16* output0_tm = top_blob_tm.channel(p);

        const Mat kernel0_tm = kernel_tm.channel(p);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 7 < tiles; i += 8)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 8);
                const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                int nn = inch * packn; // inch always > 0

                vfloat16m1_t _sum0 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum1 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum2 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum3 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum4 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum5 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum6 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum7 = __riscv_vfmv_v_f_f16m1(0.f, vl);

                for (int j = 0; j < nn; j++)
                {
                    __fp16 val0 = *r0++;
                    __fp16 val1 = *r0++;
                    __fp16 val2 = *r0++;
                    __fp16 val3 = *r0++;
                    __fp16 val4 = *r0++;
                    __fp16 val5 = *r0++;
                    __fp16 val6 = *r0++;
                    __fp16 val7 = *r0++;
                    vfloat16m1_t _w0 = __riscv_vle16_v_f16m1(k0, vl);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, val1, _w0, vl);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, val2, _w0, vl);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, val3, _w0, vl);
                    _sum4 = __riscv_vfmacc_vf_f16m1(_sum4, val4, _w0, vl);
                    _sum5 = __riscv_vfmacc_vf_f16m1(_sum5, val5, _w0, vl);
                    _sum6 = __riscv_vfmacc_vf_f16m1(_sum6, val6, _w0, vl);
                    _sum7 = __riscv_vfmacc_vf_f16m1(_sum7, val7, _w0, vl);

                    k0 += packn;
                }

                __riscv_vse16_v_f16m1(output0_tm, _sum0, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn, _sum1, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn * 2, _sum2, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn * 3, _sum3, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn * 4, _sum4, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn * 5, _sum5, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn * 6, _sum6, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn * 7, _sum7, vl);

                output0_tm += packn * 8;
            }
            for (; i + 3 < tiles; i += 4)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4);
                const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                int nn = inch * packn; // inch always > 0

                vfloat16m1_t _sum0 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum1 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum2 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum3 = __riscv_vfmv_v_f_f16m1(0.f, vl);

                for (int j = 0; j < nn; j++)
                {
                    __fp16 val0 = *r0++;
                    __fp16 val1 = *r0++;
                    __fp16 val2 = *r0++;
                    __fp16 val3 = *r0++;
                    vfloat16m1_t _w0 = __riscv_vle16_v_f16m1(k0, vl);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, val1, _w0, vl);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, val2, _w0, vl);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, val3, _w0, vl);

                    k0 += packn;
                }

                __riscv_vse16_v_f16m1(output0_tm, _sum0, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn, _sum1, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn * 2, _sum2, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn * 3, _sum3, vl);

                output0_tm += packn * 4;
            }
            for (; i + 1 < tiles; i += 2)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2);
                const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                int nn = inch * packn; // inch always > 0

                vfloat16m1_t _sum0 = __riscv_vfmv_v_f_f16m1(0.f, vl);
                vfloat16m1_t _sum1 = __riscv_vfmv_v_f_f16m1(0.f, vl);

                for (int j = 0; j < nn; j++)
                {
                    __fp16 val0 = *r0++;
                    __fp16 val1 = *r0++;
                    vfloat16m1_t _w0 = __riscv_vle16_v_f16m1(k0, vl);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, val1, _w0, vl);

                    k0 += packn;
                }

                __riscv_vse16_v_f16m1(output0_tm, _sum0, vl);
                __riscv_vse16_v_f16m1(output0_tm + packn, _sum1, vl);

                output0_tm += packn * 2;
            }
            for (; i < tiles; i++)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
                const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                int nn = inch * packn; // inch always > 0

                vfloat16m1_t _sum = __riscv_vfmv_v_f_f16m1(0.f, vl);

                for (int j = 0; j < nn; j++)
                {
                    __fp16 val = *r0++;
                    vfloat16m1_t _w0 = __riscv_vle16_v_f16m1(k0, vl);
                    _sum = __riscv_vfmacc_vf_f16m1(_sum, val, _w0, vl);

                    k0 += packn;
                }

                __riscv_vse16_v_f16m1(output0_tm, _sum, vl);

                output0_tm += packn;
            }
        }
    }
}

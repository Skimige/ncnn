// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convdw3x3s1_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + g * 8) : _mm256_setzero_ps();

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);
        float* outptr1 = out.row(1);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);
        const float* r3 = img0.row(3);

        int i = 0;
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m256 _sum00 = _bias0;
                __m256 _sum01 = _bias0;
                __m256 _sum02 = _bias0;
                __m256 _sum03 = _bias0;
                __m256 _sum10 = _bias0;
                __m256 _sum11 = _bias0;
                __m256 _sum12 = _bias0;
                __m256 _sum13 = _bias0;

                __m256 _k00 = _mm256_load_ps(k0);
                __m256 _k01 = _mm256_load_ps(k0 + 8);
                __m256 _k02 = _mm256_load_ps(k0 + 16);

                __m256 _r00 = _mm256_load_ps(r0);
                __m256 _r01 = _mm256_load_ps(r0 + 8);
                __m256 _r02 = _mm256_load_ps(r0 + 16);
                __m256 _r03 = _mm256_load_ps(r0 + 24);
                __m256 _r04 = _mm256_load_ps(r0 + 32);
                __m256 _r05 = _mm256_load_ps(r0 + 40);

                _sum00 = _mm256_comp_fmadd_ps(_k00, _r00, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k00, _r01, _sum01);
                _sum02 = _mm256_comp_fmadd_ps(_k00, _r02, _sum02);
                _sum03 = _mm256_comp_fmadd_ps(_k00, _r03, _sum03);
                _sum00 = _mm256_comp_fmadd_ps(_k01, _r01, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k01, _r02, _sum01);
                _sum02 = _mm256_comp_fmadd_ps(_k01, _r03, _sum02);
                _sum03 = _mm256_comp_fmadd_ps(_k01, _r04, _sum03);
                _sum00 = _mm256_comp_fmadd_ps(_k02, _r02, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k02, _r03, _sum01);
                _sum02 = _mm256_comp_fmadd_ps(_k02, _r04, _sum02);
                _sum03 = _mm256_comp_fmadd_ps(_k02, _r05, _sum03);

                __m256 _k10 = _mm256_load_ps(k0 + 24);
                __m256 _k11 = _mm256_load_ps(k0 + 32);
                __m256 _k12 = _mm256_load_ps(k0 + 40);

                __m256 _r10 = _mm256_load_ps(r1);
                __m256 _r11 = _mm256_load_ps(r1 + 8);
                __m256 _r12 = _mm256_load_ps(r1 + 16);
                __m256 _r13 = _mm256_load_ps(r1 + 24);
                __m256 _r14 = _mm256_load_ps(r1 + 32);
                __m256 _r15 = _mm256_load_ps(r1 + 40);

                _sum10 = _mm256_comp_fmadd_ps(_k00, _r10, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k00, _r11, _sum11);
                _sum12 = _mm256_comp_fmadd_ps(_k00, _r12, _sum12);
                _sum13 = _mm256_comp_fmadd_ps(_k00, _r13, _sum13);
                _sum00 = _mm256_comp_fmadd_ps(_k10, _r10, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k10, _r11, _sum01);
                _sum02 = _mm256_comp_fmadd_ps(_k10, _r12, _sum02);
                _sum03 = _mm256_comp_fmadd_ps(_k10, _r13, _sum03);

                _sum10 = _mm256_comp_fmadd_ps(_k01, _r11, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k01, _r12, _sum11);
                _sum12 = _mm256_comp_fmadd_ps(_k01, _r13, _sum12);
                _sum13 = _mm256_comp_fmadd_ps(_k01, _r14, _sum13);
                _sum00 = _mm256_comp_fmadd_ps(_k11, _r11, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k11, _r12, _sum01);
                _sum02 = _mm256_comp_fmadd_ps(_k11, _r13, _sum02);
                _sum03 = _mm256_comp_fmadd_ps(_k11, _r14, _sum03);

                _sum10 = _mm256_comp_fmadd_ps(_k02, _r12, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k02, _r13, _sum11);
                _sum12 = _mm256_comp_fmadd_ps(_k02, _r14, _sum12);
                _sum13 = _mm256_comp_fmadd_ps(_k02, _r15, _sum13);
                _sum00 = _mm256_comp_fmadd_ps(_k12, _r12, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k12, _r13, _sum01);
                _sum02 = _mm256_comp_fmadd_ps(_k12, _r14, _sum02);
                _sum03 = _mm256_comp_fmadd_ps(_k12, _r15, _sum03);

                __m256 _k20 = _mm256_load_ps(k0 + 48);
                __m256 _k21 = _mm256_load_ps(k0 + 56);
                __m256 _k22 = _mm256_load_ps(k0 + 64);

                __m256 _r20 = _mm256_load_ps(r2);
                __m256 _r21 = _mm256_load_ps(r2 + 8);
                __m256 _r22 = _mm256_load_ps(r2 + 16);
                __m256 _r23 = _mm256_load_ps(r2 + 24);
                __m256 _r24 = _mm256_load_ps(r2 + 32);
                __m256 _r25 = _mm256_load_ps(r2 + 40);

                _sum10 = _mm256_comp_fmadd_ps(_k10, _r20, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k10, _r21, _sum11);
                _sum12 = _mm256_comp_fmadd_ps(_k10, _r22, _sum12);
                _sum13 = _mm256_comp_fmadd_ps(_k10, _r23, _sum13);
                _sum00 = _mm256_comp_fmadd_ps(_k20, _r20, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k20, _r21, _sum01);
                _sum02 = _mm256_comp_fmadd_ps(_k20, _r22, _sum02);
                _sum03 = _mm256_comp_fmadd_ps(_k20, _r23, _sum03);

                _sum10 = _mm256_comp_fmadd_ps(_k11, _r21, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k11, _r22, _sum11);
                _sum12 = _mm256_comp_fmadd_ps(_k11, _r23, _sum12);
                _sum13 = _mm256_comp_fmadd_ps(_k11, _r24, _sum13);
                _sum00 = _mm256_comp_fmadd_ps(_k21, _r21, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k21, _r22, _sum01);
                _sum02 = _mm256_comp_fmadd_ps(_k21, _r23, _sum02);
                _sum03 = _mm256_comp_fmadd_ps(_k21, _r24, _sum03);

                _sum10 = _mm256_comp_fmadd_ps(_k12, _r22, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k12, _r23, _sum11);
                _sum12 = _mm256_comp_fmadd_ps(_k12, _r24, _sum12);
                _sum13 = _mm256_comp_fmadd_ps(_k12, _r25, _sum13);
                _sum00 = _mm256_comp_fmadd_ps(_k22, _r22, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k22, _r23, _sum01);
                _sum02 = _mm256_comp_fmadd_ps(_k22, _r24, _sum02);
                _sum03 = _mm256_comp_fmadd_ps(_k22, _r25, _sum03);

                __m256 _r30 = _mm256_load_ps(r3);
                __m256 _r31 = _mm256_load_ps(r3 + 8);
                __m256 _r32 = _mm256_load_ps(r3 + 16);
                __m256 _r33 = _mm256_load_ps(r3 + 24);
                __m256 _r34 = _mm256_load_ps(r3 + 32);
                __m256 _r35 = _mm256_load_ps(r3 + 40);

                _sum10 = _mm256_comp_fmadd_ps(_k20, _r30, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k20, _r31, _sum11);
                _sum12 = _mm256_comp_fmadd_ps(_k20, _r32, _sum12);
                _sum13 = _mm256_comp_fmadd_ps(_k20, _r33, _sum13);
                _sum10 = _mm256_comp_fmadd_ps(_k21, _r31, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k21, _r32, _sum11);
                _sum12 = _mm256_comp_fmadd_ps(_k21, _r33, _sum12);
                _sum13 = _mm256_comp_fmadd_ps(_k21, _r34, _sum13);
                _sum10 = _mm256_comp_fmadd_ps(_k22, _r32, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k22, _r33, _sum11);
                _sum12 = _mm256_comp_fmadd_ps(_k22, _r34, _sum12);
                _sum13 = _mm256_comp_fmadd_ps(_k22, _r35, _sum13);

                _mm256_store_ps(outptr0, _sum00);
                _mm256_store_ps(outptr0 + 8, _sum01);
                _mm256_store_ps(outptr0 + 16, _sum02);
                _mm256_store_ps(outptr0 + 24, _sum03);
                _mm256_store_ps(outptr1, _sum10);
                _mm256_store_ps(outptr1 + 8, _sum11);
                _mm256_store_ps(outptr1 + 16, _sum12);
                _mm256_store_ps(outptr1 + 24, _sum13);

                r0 += 32;
                r1 += 32;
                r2 += 32;
                r3 += 32;
                outptr0 += 32;
                outptr1 += 32;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _sum00 = _bias0;
                __m256 _sum01 = _bias0;
                __m256 _sum10 = _bias0;
                __m256 _sum11 = _bias0;

                __m256 _k00 = _mm256_load_ps(k0);
                __m256 _k01 = _mm256_load_ps(k0 + 8);
                __m256 _k02 = _mm256_load_ps(k0 + 16);

                __m256 _r00 = _mm256_load_ps(r0);
                __m256 _r01 = _mm256_load_ps(r0 + 8);
                __m256 _r02 = _mm256_load_ps(r0 + 16);
                __m256 _r03 = _mm256_load_ps(r0 + 24);

                _sum00 = _mm256_comp_fmadd_ps(_k00, _r00, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k00, _r01, _sum01);
                _sum00 = _mm256_comp_fmadd_ps(_k01, _r01, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k01, _r02, _sum01);
                _sum00 = _mm256_comp_fmadd_ps(_k02, _r02, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k02, _r03, _sum01);

                __m256 _k10 = _mm256_load_ps(k0 + 24);
                __m256 _k11 = _mm256_load_ps(k0 + 32);
                __m256 _k12 = _mm256_load_ps(k0 + 40);

                __m256 _r10 = _mm256_load_ps(r1);
                __m256 _r11 = _mm256_load_ps(r1 + 8);
                __m256 _r12 = _mm256_load_ps(r1 + 16);
                __m256 _r13 = _mm256_load_ps(r1 + 24);

                _sum00 = _mm256_comp_fmadd_ps(_k10, _r10, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k10, _r11, _sum01);
                _sum10 = _mm256_comp_fmadd_ps(_k00, _r10, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k00, _r11, _sum11);

                _sum00 = _mm256_comp_fmadd_ps(_k11, _r11, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k11, _r12, _sum01);
                _sum10 = _mm256_comp_fmadd_ps(_k01, _r11, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k01, _r12, _sum11);

                _sum00 = _mm256_comp_fmadd_ps(_k12, _r12, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k12, _r13, _sum01);
                _sum10 = _mm256_comp_fmadd_ps(_k02, _r12, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k02, _r13, _sum11);

                __m256 _k20 = _mm256_load_ps(k0 + 48);
                __m256 _k21 = _mm256_load_ps(k0 + 56);
                __m256 _k22 = _mm256_load_ps(k0 + 64);

                __m256 _r20 = _mm256_load_ps(r2);
                __m256 _r21 = _mm256_load_ps(r2 + 8);
                __m256 _r22 = _mm256_load_ps(r2 + 16);
                __m256 _r23 = _mm256_load_ps(r2 + 24);

                _sum00 = _mm256_comp_fmadd_ps(_k20, _r20, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k20, _r21, _sum01);
                _sum10 = _mm256_comp_fmadd_ps(_k10, _r20, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k10, _r21, _sum11);

                _sum00 = _mm256_comp_fmadd_ps(_k21, _r21, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k21, _r22, _sum01);
                _sum10 = _mm256_comp_fmadd_ps(_k11, _r21, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k11, _r22, _sum11);

                _sum00 = _mm256_comp_fmadd_ps(_k22, _r22, _sum00);
                _sum01 = _mm256_comp_fmadd_ps(_k22, _r23, _sum01);
                _sum10 = _mm256_comp_fmadd_ps(_k12, _r22, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k12, _r23, _sum11);

                __m256 _r30 = _mm256_load_ps(r3);
                __m256 _r31 = _mm256_load_ps(r3 + 8);
                __m256 _r32 = _mm256_load_ps(r3 + 16);
                __m256 _r33 = _mm256_load_ps(r3 + 24);

                _sum10 = _mm256_comp_fmadd_ps(_k20, _r30, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k20, _r31, _sum11);
                _sum10 = _mm256_comp_fmadd_ps(_k21, _r31, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k21, _r32, _sum11);
                _sum10 = _mm256_comp_fmadd_ps(_k22, _r32, _sum10);
                _sum11 = _mm256_comp_fmadd_ps(_k22, _r33, _sum11);

                _mm256_store_ps(outptr0, _sum00);
                _mm256_store_ps(outptr0 + 8, _sum01);
                _mm256_store_ps(outptr1, _sum10);
                _mm256_store_ps(outptr1 + 8, _sum11);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                r3 += 16;
                outptr0 += 16;
                outptr1 += 16;
            }
            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;

                __m256 _k00 = _mm256_load_ps(k0);
                __m256 _k01 = _mm256_load_ps(k0 + 8);
                __m256 _k02 = _mm256_load_ps(k0 + 16);

                __m256 _r00 = _mm256_load_ps(r0);
                __m256 _r01 = _mm256_load_ps(r0 + 8);
                __m256 _r02 = _mm256_load_ps(r0 + 16);

                _sum0 = _mm256_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k02, _r02, _sum0);

                __m256 _k10 = _mm256_load_ps(k0 + 24);
                __m256 _k11 = _mm256_load_ps(k0 + 32);
                __m256 _k12 = _mm256_load_ps(k0 + 40);

                __m256 _r10 = _mm256_load_ps(r1);
                __m256 _r11 = _mm256_load_ps(r1 + 8);
                __m256 _r12 = _mm256_load_ps(r1 + 16);

                _sum0 = _mm256_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k00, _r10, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k01, _r11, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k02, _r12, _sum1);

                __m256 _k20 = _mm256_load_ps(k0 + 48);
                __m256 _k21 = _mm256_load_ps(k0 + 56);
                __m256 _k22 = _mm256_load_ps(k0 + 64);

                __m256 _r20 = _mm256_load_ps(r2);
                __m256 _r21 = _mm256_load_ps(r2 + 8);
                __m256 _r22 = _mm256_load_ps(r2 + 16);

                _sum0 = _mm256_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k10, _r20, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k11, _r21, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k22, _r22, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k12, _r22, _sum1);

                __m256 _r30 = _mm256_load_ps(r3);
                __m256 _r31 = _mm256_load_ps(r3 + 8);
                __m256 _r32 = _mm256_load_ps(r3 + 16);

                _sum1 = _mm256_comp_fmadd_ps(_k20, _r30, _sum1);
                _sum1 = _mm256_comp_fmadd_ps(_k21, _r31, _sum1);
                _sum1 = _mm256_comp_fmadd_ps(_k22, _r32, _sum1);

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr1, _sum1);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                outptr0 += 8;
                outptr1 += 8;
            }

            r0 += 2 * 8 + w * 8;
            r1 += 2 * 8 + w * 8;
            r2 += 2 * 8 + w * 8;
            r3 += 2 * 8 + w * 8;

            outptr0 += outw * 8;
            outptr1 += outw * 8;
        }
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;
                __m256 _sum2 = _bias0;
                __m256 _sum3 = _bias0;

                __m256 _k00 = _mm256_load_ps(k0);
                __m256 _k01 = _mm256_load_ps(k0 + 8);
                __m256 _k02 = _mm256_load_ps(k0 + 16);

                __m256 _r00 = _mm256_load_ps(r0);
                __m256 _r01 = _mm256_load_ps(r0 + 8);
                __m256 _r02 = _mm256_load_ps(r0 + 16);
                __m256 _r03 = _mm256_load_ps(r0 + 24);
                __m256 _r04 = _mm256_load_ps(r0 + 32);
                __m256 _r05 = _mm256_load_ps(r0 + 40);

                _sum0 = _mm256_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k00, _r01, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k00, _r02, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k00, _r03, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k01, _r02, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k01, _r03, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k01, _r04, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k02, _r03, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k02, _r04, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k02, _r05, _sum3);

                __m256 _k10 = _mm256_load_ps(k0 + 24);
                __m256 _k11 = _mm256_load_ps(k0 + 32);
                __m256 _k12 = _mm256_load_ps(k0 + 40);

                __m256 _r10 = _mm256_load_ps(r1);
                __m256 _r11 = _mm256_load_ps(r1 + 8);
                __m256 _r12 = _mm256_load_ps(r1 + 16);
                __m256 _r13 = _mm256_load_ps(r1 + 24);
                __m256 _r14 = _mm256_load_ps(r1 + 32);
                __m256 _r15 = _mm256_load_ps(r1 + 40);

                _sum0 = _mm256_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k10, _r11, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k10, _r12, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k10, _r13, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k11, _r12, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k11, _r13, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k11, _r14, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k12, _r13, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k12, _r14, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k12, _r15, _sum3);

                __m256 _k20 = _mm256_load_ps(k0 + 48);
                __m256 _k21 = _mm256_load_ps(k0 + 56);
                __m256 _k22 = _mm256_load_ps(k0 + 64);

                __m256 _r20 = _mm256_load_ps(r2);
                __m256 _r21 = _mm256_load_ps(r2 + 8);
                __m256 _r22 = _mm256_load_ps(r2 + 16);
                __m256 _r23 = _mm256_load_ps(r2 + 24);
                __m256 _r24 = _mm256_load_ps(r2 + 32);
                __m256 _r25 = _mm256_load_ps(r2 + 40);

                _sum0 = _mm256_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k20, _r21, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k20, _r22, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k20, _r23, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k21, _r22, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k21, _r23, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k21, _r24, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k22, _r22, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k22, _r23, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k22, _r24, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k22, _r25, _sum3);

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);
                _mm256_store_ps(outptr0 + 16, _sum2);
                _mm256_store_ps(outptr0 + 24, _sum3);

                r0 += 32;
                r1 += 32;
                r2 += 32;
                outptr0 += 32;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;

                __m256 _k00 = _mm256_load_ps(k0);
                __m256 _k01 = _mm256_load_ps(k0 + 8);
                __m256 _k02 = _mm256_load_ps(k0 + 16);

                __m256 _r00 = _mm256_load_ps(r0);
                __m256 _r01 = _mm256_load_ps(r0 + 8);
                __m256 _r02 = _mm256_load_ps(r0 + 16);
                __m256 _r03 = _mm256_load_ps(r0 + 24);

                _sum0 = _mm256_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k00, _r01, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k01, _r02, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k02, _r03, _sum1);

                __m256 _k10 = _mm256_load_ps(k0 + 24);
                __m256 _k11 = _mm256_load_ps(k0 + 32);
                __m256 _k12 = _mm256_load_ps(k0 + 40);

                __m256 _r10 = _mm256_load_ps(r1);
                __m256 _r11 = _mm256_load_ps(r1 + 8);
                __m256 _r12 = _mm256_load_ps(r1 + 16);
                __m256 _r13 = _mm256_load_ps(r1 + 24);

                _sum0 = _mm256_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k10, _r11, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k11, _r12, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k12, _r13, _sum1);

                __m256 _k20 = _mm256_load_ps(k0 + 48);
                __m256 _k21 = _mm256_load_ps(k0 + 56);
                __m256 _k22 = _mm256_load_ps(k0 + 64);

                __m256 _r20 = _mm256_load_ps(r2);
                __m256 _r21 = _mm256_load_ps(r2 + 8);
                __m256 _r22 = _mm256_load_ps(r2 + 16);
                __m256 _r23 = _mm256_load_ps(r2 + 24);

                _sum0 = _mm256_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k20, _r21, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k21, _r22, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k22, _r22, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k22, _r23, _sum1);

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr0 += 16;
            }
            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;

                __m256 _k00 = _mm256_load_ps(k0);
                __m256 _k01 = _mm256_load_ps(k0 + 8);
                __m256 _k02 = _mm256_load_ps(k0 + 16);

                __m256 _r00 = _mm256_load_ps(r0);
                __m256 _r01 = _mm256_load_ps(r0 + 8);
                __m256 _r02 = _mm256_load_ps(r0 + 16);

                _sum0 = _mm256_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k02, _r02, _sum0);

                __m256 _k10 = _mm256_load_ps(k0 + 24);
                __m256 _k11 = _mm256_load_ps(k0 + 32);
                __m256 _k12 = _mm256_load_ps(k0 + 40);

                __m256 _r10 = _mm256_load_ps(r1);
                __m256 _r11 = _mm256_load_ps(r1 + 8);
                __m256 _r12 = _mm256_load_ps(r1 + 16);

                _sum0 = _mm256_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k12, _r12, _sum0);

                __m256 _k20 = _mm256_load_ps(k0 + 48);
                __m256 _k21 = _mm256_load_ps(k0 + 56);
                __m256 _k22 = _mm256_load_ps(k0 + 64);

                __m256 _r20 = _mm256_load_ps(r2);
                __m256 _r21 = _mm256_load_ps(r2 + 8);
                __m256 _r22 = _mm256_load_ps(r2 + 16);

                _sum0 = _mm256_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k22, _r22, _sum0);

                _mm256_store_ps(outptr0, _sum0);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr0 += 8;
            }

            r0 += 2 * 8;
            r1 += 2 * 8;
            r2 += 2 * 8;
        }
    }
}

static void convdw3x3s2_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2 * outw + w) * 8;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + g * 8) : _mm256_setzero_ps();

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);

        __m256 _k00 = _mm256_load_ps(k0);
        __m256 _k01 = _mm256_load_ps(k0 + 8);
        __m256 _k02 = _mm256_load_ps(k0 + 16);
        __m256 _k10 = _mm256_load_ps(k0 + 24);
        __m256 _k11 = _mm256_load_ps(k0 + 32);
        __m256 _k12 = _mm256_load_ps(k0 + 40);
        __m256 _k20 = _mm256_load_ps(k0 + 48);
        __m256 _k21 = _mm256_load_ps(k0 + 56);
        __m256 _k22 = _mm256_load_ps(k0 + 64);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;
                __m256 _sum2 = _bias0;
                __m256 _sum3 = _bias0;

                __m256 _r00 = _mm256_load_ps(r0);
                __m256 _r01 = _mm256_load_ps(r0 + 8);
                __m256 _r02 = _mm256_load_ps(r0 + 16);
                __m256 _r03 = _mm256_load_ps(r0 + 24);
                __m256 _r04 = _mm256_load_ps(r0 + 32);
                __m256 _r05 = _mm256_load_ps(r0 + 40);
                __m256 _r06 = _mm256_load_ps(r0 + 48);
                __m256 _r07 = _mm256_load_ps(r0 + 56);
                __m256 _r08 = _mm256_load_ps(r0 + 64);

                _sum0 = _mm256_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k00, _r02, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k00, _r04, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k00, _r06, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k01, _r03, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k01, _r05, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k01, _r07, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k02, _r04, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k02, _r06, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k02, _r08, _sum3);

                __m256 _r10 = _mm256_load_ps(r1);
                __m256 _r11 = _mm256_load_ps(r1 + 8);
                __m256 _r12 = _mm256_load_ps(r1 + 16);
                __m256 _r13 = _mm256_load_ps(r1 + 24);
                __m256 _r14 = _mm256_load_ps(r1 + 32);
                __m256 _r15 = _mm256_load_ps(r1 + 40);
                __m256 _r16 = _mm256_load_ps(r1 + 48);
                __m256 _r17 = _mm256_load_ps(r1 + 56);
                __m256 _r18 = _mm256_load_ps(r1 + 64);

                _sum0 = _mm256_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k10, _r12, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k10, _r14, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k10, _r16, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k11, _r13, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k11, _r15, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k11, _r17, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k12, _r14, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k12, _r16, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k12, _r18, _sum3);

                __m256 _r20 = _mm256_load_ps(r2);
                __m256 _r21 = _mm256_load_ps(r2 + 8);
                __m256 _r22 = _mm256_load_ps(r2 + 16);
                __m256 _r23 = _mm256_load_ps(r2 + 24);
                __m256 _r24 = _mm256_load_ps(r2 + 32);
                __m256 _r25 = _mm256_load_ps(r2 + 40);
                __m256 _r26 = _mm256_load_ps(r2 + 48);
                __m256 _r27 = _mm256_load_ps(r2 + 56);
                __m256 _r28 = _mm256_load_ps(r2 + 64);

                _sum0 = _mm256_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k20, _r22, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k20, _r24, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k20, _r26, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k21, _r23, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k21, _r25, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k21, _r27, _sum3);
                _sum0 = _mm256_comp_fmadd_ps(_k22, _r22, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k22, _r24, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_k22, _r26, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_k22, _r28, _sum3);

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);
                _mm256_store_ps(outptr0 + 16, _sum2);
                _mm256_store_ps(outptr0 + 24, _sum3);

                r0 += 2 * 32;
                r1 += 2 * 32;
                r2 += 2 * 32;
                outptr0 += 32;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;

                __m256 _r00 = _mm256_load_ps(r0);
                __m256 _r01 = _mm256_load_ps(r0 + 8);
                __m256 _r02 = _mm256_load_ps(r0 + 16);
                __m256 _r03 = _mm256_load_ps(r0 + 24);
                __m256 _r04 = _mm256_load_ps(r0 + 32);

                _sum0 = _mm256_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k00, _r02, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k01, _r03, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k02, _r02, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k02, _r04, _sum1);

                __m256 _r10 = _mm256_load_ps(r1);
                __m256 _r11 = _mm256_load_ps(r1 + 8);
                __m256 _r12 = _mm256_load_ps(r1 + 16);
                __m256 _r13 = _mm256_load_ps(r1 + 24);
                __m256 _r14 = _mm256_load_ps(r1 + 32);

                _sum0 = _mm256_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k10, _r12, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k11, _r13, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k12, _r12, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k12, _r14, _sum1);

                __m256 _r20 = _mm256_load_ps(r2);
                __m256 _r21 = _mm256_load_ps(r2 + 8);
                __m256 _r22 = _mm256_load_ps(r2 + 16);
                __m256 _r23 = _mm256_load_ps(r2 + 24);
                __m256 _r24 = _mm256_load_ps(r2 + 32);

                _sum0 = _mm256_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k20, _r22, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k21, _r23, _sum1);
                _sum0 = _mm256_comp_fmadd_ps(_k22, _r22, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_k22, _r24, _sum1);

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);

                r0 += 2 * 16;
                r1 += 2 * 16;
                r2 += 2 * 16;
                outptr0 += 16;
            }
            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;

                __m256 _r00 = _mm256_load_ps(r0);
                __m256 _r01 = _mm256_load_ps(r0 + 8);
                __m256 _r02 = _mm256_load_ps(r0 + 16);

                _sum0 = _mm256_comp_fmadd_ps(_k00, _r00, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k01, _r01, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k02, _r02, _sum0);

                __m256 _r10 = _mm256_load_ps(r1);
                __m256 _r11 = _mm256_load_ps(r1 + 8);
                __m256 _r12 = _mm256_load_ps(r1 + 16);

                _sum0 = _mm256_comp_fmadd_ps(_k10, _r10, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k11, _r11, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k12, _r12, _sum0);

                __m256 _r20 = _mm256_load_ps(r2);
                __m256 _r21 = _mm256_load_ps(r2 + 8);
                __m256 _r22 = _mm256_load_ps(r2 + 16);

                _sum0 = _mm256_comp_fmadd_ps(_k20, _r20, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k21, _r21, _sum0);
                _sum0 = _mm256_comp_fmadd_ps(_k22, _r22, _sum0);

                _mm256_store_ps(outptr0, _sum0);

                r0 += 2 * 8;
                r1 += 2 * 8;
                r2 += 2 * 8;
                outptr0 += 8;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

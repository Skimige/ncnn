// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

Reshape_x86::Reshape_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Reshape_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    // resolve out shape
    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (ndim == 1)
    {
        // flatten
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;

    const int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        int out_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
#if __AVX512F__
            out_elempack = outh % 16 == 0 ? 16 : outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#elif __AVX__
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            top_blob.dims = 2;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.cstep = top_blob.cstep * top_blob.elempack;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (out_elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 16;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 16 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 16 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 16 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + outw * (i * 16 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + outw * (i * 16 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + outw * (i * 16 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + outw * (i * 16 + 7);
                const float* ptr8 = (const float*)bottom_blob_flattened + outw * (i * 16 + 8);
                const float* ptr9 = (const float*)bottom_blob_flattened + outw * (i * 16 + 9);
                const float* ptra = (const float*)bottom_blob_flattened + outw * (i * 16 + 10);
                const float* ptrb = (const float*)bottom_blob_flattened + outw * (i * 16 + 11);
                const float* ptrc = (const float*)bottom_blob_flattened + outw * (i * 16 + 12);
                const float* ptrd = (const float*)bottom_blob_flattened + outw * (i * 16 + 13);
                const float* ptre = (const float*)bottom_blob_flattened + outw * (i * 16 + 14);
                const float* ptrf = (const float*)bottom_blob_flattened + outw * (i * 16 + 15);
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 15 < outw; j += 16)
                {
                    __m512 _row0 = _mm512_loadu_ps(ptr0);
                    __m512 _row1 = _mm512_loadu_ps(ptr1);
                    __m512 _row2 = _mm512_loadu_ps(ptr2);
                    __m512 _row3 = _mm512_loadu_ps(ptr3);
                    __m512 _row4 = _mm512_loadu_ps(ptr4);
                    __m512 _row5 = _mm512_loadu_ps(ptr5);
                    __m512 _row6 = _mm512_loadu_ps(ptr6);
                    __m512 _row7 = _mm512_loadu_ps(ptr7);
                    __m512 _row8 = _mm512_loadu_ps(ptr8);
                    __m512 _row9 = _mm512_loadu_ps(ptr9);
                    __m512 _rowa = _mm512_loadu_ps(ptra);
                    __m512 _rowb = _mm512_loadu_ps(ptrb);
                    __m512 _rowc = _mm512_loadu_ps(ptrc);
                    __m512 _rowd = _mm512_loadu_ps(ptrd);
                    __m512 _rowe = _mm512_loadu_ps(ptre);
                    __m512 _rowf = _mm512_loadu_ps(ptrf);

                    transpose16x16_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9, _rowa, _rowb, _rowc, _rowd, _rowe, _rowf);

                    _mm512_storeu_ps(outptr, _row0);
                    _mm512_storeu_ps(outptr + 16, _row1);
                    _mm512_storeu_ps(outptr + 16 * 2, _row2);
                    _mm512_storeu_ps(outptr + 16 * 3, _row3);
                    _mm512_storeu_ps(outptr + 16 * 4, _row4);
                    _mm512_storeu_ps(outptr + 16 * 5, _row5);
                    _mm512_storeu_ps(outptr + 16 * 6, _row6);
                    _mm512_storeu_ps(outptr + 16 * 7, _row7);
                    _mm512_storeu_ps(outptr + 16 * 8, _row8);
                    _mm512_storeu_ps(outptr + 16 * 9, _row9);
                    _mm512_storeu_ps(outptr + 16 * 10, _rowa);
                    _mm512_storeu_ps(outptr + 16 * 11, _rowb);
                    _mm512_storeu_ps(outptr + 16 * 12, _rowc);
                    _mm512_storeu_ps(outptr + 16 * 13, _rowd);
                    _mm512_storeu_ps(outptr + 16 * 14, _rowe);
                    _mm512_storeu_ps(outptr + 16 * 15, _rowf);

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    ptr3 += 16;
                    ptr4 += 16;
                    ptr5 += 16;
                    ptr6 += 16;
                    ptr7 += 16;
                    ptr8 += 16;
                    ptr9 += 16;
                    ptra += 16;
                    ptrb += 16;
                    ptrc += 16;
                    ptrd += 16;
                    ptre += 16;
                    ptrf += 16;
                    outptr += 256;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;
                    outptr[8] = *ptr8++;
                    outptr[9] = *ptr9++;
                    outptr[10] = *ptra++;
                    outptr[11] = *ptrb++;
                    outptr[12] = *ptrc++;
                    outptr[13] = *ptrd++;
                    outptr[14] = *ptre++;
                    outptr[15] = *ptrf++;

                    outptr += 16;
                }
            }
        }
#endif // __AVX512F__

        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 8;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 8 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 8 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 8 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + outw * (i * 8 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + outw * (i * 8 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + outw * (i * 8 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + outw * (i * 8 + 7);
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    __m256 _row0 = _mm256_loadu_ps(ptr0);
                    __m256 _row1 = _mm256_loadu_ps(ptr1);
                    __m256 _row2 = _mm256_loadu_ps(ptr2);
                    __m256 _row3 = _mm256_loadu_ps(ptr3);
                    __m256 _row4 = _mm256_loadu_ps(ptr4);
                    __m256 _row5 = _mm256_loadu_ps(ptr5);
                    __m256 _row6 = _mm256_loadu_ps(ptr6);
                    __m256 _row7 = _mm256_loadu_ps(ptr7);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr, _row0);
                    _mm256_storeu_ps(outptr + 8, _row1);
                    _mm256_storeu_ps(outptr + 16, _row2);
                    _mm256_storeu_ps(outptr + 24, _row3);
                    _mm256_storeu_ps(outptr + 32, _row4);
                    _mm256_storeu_ps(outptr + 40, _row5);
                    _mm256_storeu_ps(outptr + 48, _row6);
                    _mm256_storeu_ps(outptr + 56, _row7);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }
        }
#endif // __AVX__

        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 4;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 4 + 3);
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    __m128 _row0 = _mm_loadu_ps(ptr0);
                    __m128 _row1 = _mm_loadu_ps(ptr1);
                    __m128 _row2 = _mm_loadu_ps(ptr2);
                    __m128 _row3 = _mm_loadu_ps(ptr3);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr, _row0);
                    _mm_storeu_ps(outptr + 4, _row1);
                    _mm_storeu_ps(outptr + 8, _row2);
                    _mm_storeu_ps(outptr + 12, _row3);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }
#endif // __SSE2__
    }

    if (ndim == 3 || ndim == 4)
    {
        if (ndim == 3)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outc == 0)
                outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outh;
            if (outh == -1)
                outh = total / outc / outw;
            if (outc == -1)
                outc = total / outh / outw;

            outd = 1;
        }
        else // if (ndim == 4)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outd == 0)
                outd = bottom_blob.d;
            if (outc == 0)
                outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outd / outh;
            if (outh == -1)
                outh = total / outc / outd / outw;
            if (outd == -1)
                outd = total / outc / outh / outw;
            if (outc == -1)
                outc = total / outd / outh / outw;
        }

        int out_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
#if __AVX512F__
            out_elempack = outc % 16 == 0 ? 16 : outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#elif __AVX__
            out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.dims = ndim;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
        {
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        else // if (ndim == 4)
        {
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h * top_blob.d;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (out_elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 16;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 16 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 16 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 16 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + size * (q * 16 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + size * (q * 16 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + size * (q * 16 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + size * (q * 16 + 7);
                const float* ptr8 = (const float*)bottom_blob_flattened + size * (q * 16 + 8);
                const float* ptr9 = (const float*)bottom_blob_flattened + size * (q * 16 + 9);
                const float* ptra = (const float*)bottom_blob_flattened + size * (q * 16 + 10);
                const float* ptrb = (const float*)bottom_blob_flattened + size * (q * 16 + 11);
                const float* ptrc = (const float*)bottom_blob_flattened + size * (q * 16 + 12);
                const float* ptrd = (const float*)bottom_blob_flattened + size * (q * 16 + 13);
                const float* ptre = (const float*)bottom_blob_flattened + size * (q * 16 + 14);
                const float* ptrf = (const float*)bottom_blob_flattened + size * (q * 16 + 15);
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 15 < size; i += 16)
                {
                    __m512 _row0 = _mm512_loadu_ps(ptr0);
                    __m512 _row1 = _mm512_loadu_ps(ptr1);
                    __m512 _row2 = _mm512_loadu_ps(ptr2);
                    __m512 _row3 = _mm512_loadu_ps(ptr3);
                    __m512 _row4 = _mm512_loadu_ps(ptr4);
                    __m512 _row5 = _mm512_loadu_ps(ptr5);
                    __m512 _row6 = _mm512_loadu_ps(ptr6);
                    __m512 _row7 = _mm512_loadu_ps(ptr7);
                    __m512 _row8 = _mm512_loadu_ps(ptr8);
                    __m512 _row9 = _mm512_loadu_ps(ptr9);
                    __m512 _rowa = _mm512_loadu_ps(ptra);
                    __m512 _rowb = _mm512_loadu_ps(ptrb);
                    __m512 _rowc = _mm512_loadu_ps(ptrc);
                    __m512 _rowd = _mm512_loadu_ps(ptrd);
                    __m512 _rowe = _mm512_loadu_ps(ptre);
                    __m512 _rowf = _mm512_loadu_ps(ptrf);

                    transpose16x16_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9, _rowa, _rowb, _rowc, _rowd, _rowe, _rowf);

                    _mm512_storeu_ps(outptr, _row0);
                    _mm512_storeu_ps(outptr + 16, _row1);
                    _mm512_storeu_ps(outptr + 16 * 2, _row2);
                    _mm512_storeu_ps(outptr + 16 * 3, _row3);
                    _mm512_storeu_ps(outptr + 16 * 4, _row4);
                    _mm512_storeu_ps(outptr + 16 * 5, _row5);
                    _mm512_storeu_ps(outptr + 16 * 6, _row6);
                    _mm512_storeu_ps(outptr + 16 * 7, _row7);
                    _mm512_storeu_ps(outptr + 16 * 8, _row8);
                    _mm512_storeu_ps(outptr + 16 * 9, _row9);
                    _mm512_storeu_ps(outptr + 16 * 10, _rowa);
                    _mm512_storeu_ps(outptr + 16 * 11, _rowb);
                    _mm512_storeu_ps(outptr + 16 * 12, _rowc);
                    _mm512_storeu_ps(outptr + 16 * 13, _rowd);
                    _mm512_storeu_ps(outptr + 16 * 14, _rowe);
                    _mm512_storeu_ps(outptr + 16 * 15, _rowf);

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    ptr3 += 16;
                    ptr4 += 16;
                    ptr5 += 16;
                    ptr6 += 16;
                    ptr7 += 16;
                    ptr8 += 16;
                    ptr9 += 16;
                    ptra += 16;
                    ptrb += 16;
                    ptrc += 16;
                    ptrd += 16;
                    ptre += 16;
                    ptrf += 16;
                    outptr += 256;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;
                    outptr[8] = *ptr8++;
                    outptr[9] = *ptr9++;
                    outptr[10] = *ptra++;
                    outptr[11] = *ptrb++;
                    outptr[12] = *ptrc++;
                    outptr[13] = *ptrd++;
                    outptr[14] = *ptre++;
                    outptr[15] = *ptrf++;

                    outptr += 16;
                }
            }
        }
#endif // __AVX512F__

        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 8;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 8 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 8 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 8 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + size * (q * 8 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + size * (q * 8 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + size * (q * 8 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + size * (q * 8 + 7);
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m256 _row0 = _mm256_loadu_ps(ptr0);
                    __m256 _row1 = _mm256_loadu_ps(ptr1);
                    __m256 _row2 = _mm256_loadu_ps(ptr2);
                    __m256 _row3 = _mm256_loadu_ps(ptr3);
                    __m256 _row4 = _mm256_loadu_ps(ptr4);
                    __m256 _row5 = _mm256_loadu_ps(ptr5);
                    __m256 _row6 = _mm256_loadu_ps(ptr6);
                    __m256 _row7 = _mm256_loadu_ps(ptr7);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr, _row0);
                    _mm256_storeu_ps(outptr + 8, _row1);
                    _mm256_storeu_ps(outptr + 16, _row2);
                    _mm256_storeu_ps(outptr + 24, _row3);
                    _mm256_storeu_ps(outptr + 32, _row4);
                    _mm256_storeu_ps(outptr + 40, _row5);
                    _mm256_storeu_ps(outptr + 48, _row6);
                    _mm256_storeu_ps(outptr + 56, _row7);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }
        }
#endif // __AVX__

        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 4;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 4 + 3);
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __m128 _row0 = _mm_loadu_ps(ptr0);
                    __m128 _row1 = _mm_loadu_ps(ptr1);
                    __m128 _row2 = _mm_loadu_ps(ptr2);
                    __m128 _row3 = _mm_loadu_ps(ptr3);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr, _row0);
                    _mm_storeu_ps(outptr + 4, _row1);
                    _mm_storeu_ps(outptr + 8, _row2);
                    _mm_storeu_ps(outptr + 12, _row3);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }
#endif // __SSE2__

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob_flattened + size * q;
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __SSE2__
#if __AVX__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _v = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(outptr, _v);
                    ptr += 8;
                    outptr += 8;
                }
#endif
                for (; i + 3 < size; i += 4)
                {
                    __m128 _v = _mm_loadu_ps(ptr);
                    _mm_storeu_ps(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn

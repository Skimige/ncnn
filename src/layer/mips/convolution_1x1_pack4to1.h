// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void conv1x1s1_sgemm_pack4to1_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    const int size = w * h;

    Mat bottom_im2col = bottom_blob;
    bottom_im2col.w = size;
    bottom_im2col.h = 1;

    im2col_sgemm_pack4to1_msa(bottom_im2col, top_blob, kernel, _bias, opt);
}

static void conv1x1s2_sgemm_pack4to1_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 4;

    Mat bottom_blob_shrinked;
    bottom_blob_shrinked.create(outw, outh, channels, elemsize, elempack, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const float* r0 = bottom_blob.channel(p);
        float* outptr = bottom_blob_shrinked.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                v4f32 _val = (v4f32)__msa_ld_w(r0, 0);
                __msa_st_w((v4i32)_val, outptr, 0);

                r0 += 4 * 2;
                outptr += 4;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack4to1_msa(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}

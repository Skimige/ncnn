// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolutiondepthwise_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

#include "convolutiondepthwise_3x3.h"

#if __riscv_vector
#include "convolutiondepthwise_3x3_packn.h"
#include "convolutiondepthwise_5x5_packn.h"
#endif // __riscv_vector

ConvolutionDepthWise_riscv::ConvolutionDepthWise_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif

    activation = 0;
}

int ConvolutionDepthWise_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        // TODO implement int8
        return 0;
    }
#endif

#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            elempack = channels % packn == 0 ? packn : 1;
        }
#endif

#if __riscv_vector
        // packn
        if (elempack == packn)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_tm, packn, opt);
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            weight_data_tm = weight_data;
        }

        if (opt.lightmode)
            weight_data.release();

        return 0;
    }

    // group convolution
    create_group_ops(opt);

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int ConvolutionDepthWise_riscv::create_group_ops(const Option& opt)
{
    // create Convolution op for each group
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    for (int i = 0; i < (int)group_ops.size(); i++)
        delete group_ops[i];

    group_ops.clear();

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    group_ops.resize(group);

    for (int g = 0; g < group; g++)
    {
        Mat weight_data_g = weight_data.range(maxk * channels_g * num_output_g * g, maxk * channels_g * num_output_g).clone();
        Mat bias_data_g;
        if (bias_term)
            bias_data_g = bias_data.range(num_output_g * g, num_output_g);

        ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output_g); // num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, 0);  // pad_w
        pd.set(14, 0); // pad_h
        pd.set(5, bias_term);
        pd.set(6, maxk * channels_g * num_output_g); // weight_data_size
        pd.set(8, int8_scale_term);
        pd.set(9, activation_type);
        pd.set(10, activation_params);

        op->load_param(pd);

        // set weights
        if (bias_term)
        {
            ncnn::Mat weights[5];
            weights[0] = weight_data_g;
            weights[1] = bias_data_g;

#if NCNN_INT8
            if (int8_scale_term)
            {
                Mat weight_data_int8_scales_g(num_output_g);
                weight_data_int8_scales_g.fill(weight_data_int8_scales[g]);
                weights[2] = weight_data_int8_scales_g;
                weights[3] = bottom_blob_int8_scales.range(g, 1);
            }
            if (int8_scale_term > 100)
            {
                weights[4] = top_blob_int8_scales.range(g, 1);
            }
#endif

            op->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[4];
            weights[0] = weight_data_g;

#if NCNN_INT8
            if (int8_scale_term)
            {
                Mat weight_data_int8_scales_g(num_output_g);
                weight_data_int8_scales_g.fill(weight_data_int8_scales[g]);
                weights[1] = weight_data_int8_scales_g;
                weights[2] = bottom_blob_int8_scales.range(g, 1);
            }
            if (int8_scale_term > 100)
            {
                weights[3] = top_blob_int8_scales.range(g, 1);
            }
#endif

            op->load_model(ModelBinFromMatArray(weights));
        }

        op->create_pipeline(opt);

        group_ops[g] = op;
    }

    return 0;
}

int ConvolutionDepthWise_riscv::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    for (int i = 0; i < (int)group_ops.size(); i++)
    {
        group_ops[i]->destroy_pipeline(opt);
        delete group_ops[i];
    }
    group_ops.clear();

    return 0;
}

int ConvolutionDepthWise_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        Mat bottom_blob_unpacked = bottom_blob;
        if (bottom_blob.elempack != 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        }

        Mat bottom_blob_unpacked_fp32 = bottom_blob_unpacked;
        if (bottom_blob_unpacked.elembits() == 16)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            cast_float16_to_float32(bottom_blob_unpacked, bottom_blob_unpacked_fp32, opt_pack1);
        }

        Option opt_unpacked = opt;
        opt_unpacked.use_packing_layout = false;
        return ConvolutionDepthWise::forward_int8(bottom_blob_unpacked_fp32, top_blob, opt_unpacked);
    }
#endif

#if NCNN_ZFH
    int elembits = bottom_blob.elembits();

    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
                const int maxk = kernel_w * kernel_h;

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

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    float* outptr = top_blob.channel(g);
                    const float* kptr = (const float*)weight_data_tm + maxk * g * packn;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);

                            if (bias_term)
                            {
                                _sum = __riscv_vle32_v_f32m1((const float*)bias_data + g * packn, vl);
                            }

                            const float* sptr = m.row(i * stride_h) + j * stride_w * packn;

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + space_ofs[k] * packn, vl);
                                vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr + k * packn, vl);
                                _sum = __riscv_vfmacc_vv_f32m1(_sum, _val, _w, vl);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params, vl);

                            __riscv_vse32_v_f32m1(outptr + j * packn, _sum, vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
                const int maxk = kernel_w * kernel_h;

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

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < group; g++)
                {
                    float* outptr = top_blob.channel(g);
                    const float* kptr = (const float*)weight_data_tm + maxk * g;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                                sum = bias_data[g];

                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = (float)sptr[space_ofs[k]];
                                float w = (float)kptr[k];
                                sum += val * w;
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = sum;
                        }

                        outptr += outw;
                    }
                }
            }
        }

        return 0;
    }

    // group convolution
    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int g_elempack = 1;
    int out_g_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % packn == 0 ? packn : 1;
        out_g_elempack = num_output_g % packn == 0 ? packn : 1;
    }
#endif

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > g_elempack)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, 1, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (out_g_elempack < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    for (int g = 0; g < group; g++)
    {
        const Mat bottom_blob_bordered_g = bottom_blob_bordered_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
        Mat top_blob_g = top_blob_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

        const ncnn::Layer* op = group_ops[g];

        Option opt_g = opt;
        opt_g.blob_allocator = top_blob_unpacked.allocator;

        // forward
        op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
    }

    // packing
    if (out_g_elempack < out_elempack)
    {
        convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}

int ConvolutionDepthWise_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.c * _weight_data.elempack;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

#if NCNN_ZFH
    if (opt.use_fp16_storage && support_fp16_storage && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_float16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
    }
#endif // NCNN_ZFH

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

#if NCNN_ZFH
        if (opt.use_fp16_storage && support_fp16_storage && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_float16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
        }
#endif // NCNN_ZFH

        // bias_data_flattened as pack1
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::ConvolutionDepthWise);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(11, _kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(14, pad_top);
    pd.set(16, pad_bottom);
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(7, group);
    pd.set(8, int8_scale_term);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_flattened;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

} // namespace ncnn

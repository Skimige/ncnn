// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_requantize_pack1_oom(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size, int activation_type, float alpha, float beta)
{
    ncnn::ParamDict pd;
    pd.set(0, scale_in_data_size);
    pd.set(1, scale_out_data_size);
    pd.set(2, bias_data_size);

    ncnn::Mat activation_params(2);
    activation_params[0] = alpha;
    activation_params[1] = beta;
    pd.set(3, activation_type);
    pd.set(4, activation_params);

    std::vector<ncnn::Mat> weights(bias_data_size ? 3 : 2);
    weights[0] = RandomMat(scale_in_data_size);
    weights[1] = RandomMat(scale_out_data_size);
    if (bias_data_size)
        weights[2] = RandomMat(bias_data_size);

    Randomize(weights[0], 0.0001, 0.001);
    Randomize(weights[1], 10, 100);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING | TEST_LAYER_DISABLE_AUTO_INPUT_PACKING;
    int ret = test_layer_oom("Requantize", pd, weights, a, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_requantize_pack1_oom failed a.dims=%d a=(%d %d %d) scale_in_data_size=%d scale_out_data_size=%d bias_data_size=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.c, scale_in_data_size, scale_out_data_size, bias_data_size, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_requantize_pack1_oom(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size)
{
    return 0
           || test_requantize_pack1_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 0, 0.f, 0.f)
           || test_requantize_pack1_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 1, 0.f, 0.f)
           || test_requantize_pack1_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 2, RandomFloat(0, 1), 0.f)
           || test_requantize_pack1_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 3, RandomFloat(-1, 0), RandomFloat(0, 1))
           || test_requantize_pack1_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 4, 0.f, 0.f)
           || test_requantize_pack1_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 5, 0.f, 0.f);
}

static int test_requantize_pack8_oom(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size, int activation_type, float alpha, float beta)
{
    ncnn::ParamDict pd;
    pd.set(0, scale_in_data_size);
    pd.set(1, scale_out_data_size);
    pd.set(2, bias_data_size);

    ncnn::Mat activation_params(2);
    activation_params[0] = alpha;
    activation_params[1] = beta;
    pd.set(3, activation_type);
    pd.set(4, activation_params);

    std::vector<ncnn::Mat> weights(bias_data_size ? 3 : 2);
    weights[0] = RandomMat(scale_in_data_size);
    weights[1] = RandomMat(scale_out_data_size);
    if (bias_data_size)
        weights[2] = RandomMat(bias_data_size);

    Randomize(weights[0], 0.0001, 0.001);
    Randomize(weights[1], 10, 100);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING | TEST_LAYER_ENABLE_FORCE_INPUT_PACK8;
    int ret = test_layer_oom("Requantize", pd, weights, a, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_requantize_pack8_oom failed a.dims=%d a=(%d %d %d) scale_in_data_size=%d scale_out_data_size=%d bias_data_size=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.c, scale_in_data_size, scale_out_data_size, bias_data_size, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_requantize_pack8_oom(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size)
{
    return 0
           || test_requantize_pack8_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 0, 0.f, 0.f)
           || test_requantize_pack8_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 1, 0.f, 0.f)
           || test_requantize_pack8_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 2, RandomFloat(0, 1), 0.f)
           || test_requantize_pack8_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 3, RandomFloat(-1, 0), RandomFloat(0, 1))
           || test_requantize_pack8_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 4, 0.f, 0.f)
           || test_requantize_pack8_oom(a, scale_in_data_size, scale_out_data_size, bias_data_size, 5, 0.f, 0.f);
}

static int test_requantize_0()
{
    return 0
           || test_requantize_pack1_oom(RandomIntMat(7, 9, 12), 12, 12, 12)
           || test_requantize_pack1_oom(RandomIntMat(3, 5, 13), 13, 13, 13);
}

static int test_requantize_1()
{
    return 0
           || test_requantize_pack1_oom(RandomIntMat(17, 12), 12, 12, 12)
           || test_requantize_pack1_oom(RandomIntMat(19, 15), 15, 15, 15);
}

static int test_requantize_2()
{
    return test_requantize_pack1_oom(RandomIntMat(124), 1, 1, 1);
}

static int test_requantize_3()
{
    return 0
           || test_requantize_pack8_oom(RandomIntMat(5, 7, 24), 24, 24, 24)
           || test_requantize_pack8_oom(RandomIntMat(15, 24), 24, 24, 24)
           || test_requantize_pack8_oom(RandomIntMat(128), 1, 1, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_requantize_0()
           || test_requantize_1()
           || test_requantize_2()
           || test_requantize_3();
}

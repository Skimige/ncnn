// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

#include <float.h>

namespace pnnx {

class F_max_pool3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 kernel_size value=%kernel_size
prim::Constant          op_1        0 1 stride value=%stride
prim::Constant          op_2        0 1 padding value=%padding
prim::Constant          op_3        0 1 dilation value=%dilation
prim::Constant          op_4        0 1 ceil_mode value=%ceil_mode
aten::max_pool3d        op_5        6 1 input kernel_size stride padding dilation ceil_mode out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.max_pool3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        op->params["return_indices"] = false;
    }
};

class F_max_pool3d_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 8
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 kernel_size value=%kernel_size
prim::Constant          op_1        0 1 stride value=%stride
prim::Constant          op_2        0 1 padding value=%padding
prim::Constant          op_3        0 1 dilation value=%dilation
prim::Constant          op_4        0 1 ceil_mode value=%ceil_mode
aten::max_pool3d_with_indices op_5  6 2 input kernel_size stride padding dilation ceil_mode out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.max_pool3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        op->params["return_indices"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool3d, 120)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool3d_2, 120)

// https://github.com/pytorch/pytorch/blob/c263bd43e8e8502d4726643bc6fd046f0130ac0e/torch/onnx/symbolic_opset9.py#L1496
static int get_pool_ceil_padding(int w, int ksize, int stride, int pad)
{
    if (stride == 1)
        return 0;

    int ceiled_output_w = int(ceil((w + pad * 2 - ksize) / float(stride))) + 1;

    if ((ceiled_output_w - 1) * stride >= w + pad)
        ceiled_output_w = ceiled_output_w - 1;

    int ceil_pad = ksize - (w + pad * 2 - ((ceiled_output_w - 1) * stride + 1));

    return ceil_pad;
}

class F_max_pool3d_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
MaxPool                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.max_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (captured_params.find("op_0.kernel_shape") == captured_params.end())
            return false;

        if (captured_params.at("op_0.kernel_shape").type != 5 || captured_params.at("op_0.kernel_shape").ai.size() != 3)
            return false;

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            if (captured_params.at("op_0.dilations").type != 5 || captured_params.at("op_0.dilations").ai.size() != 3)
                return false;
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            if (captured_params.at("op_0.strides").type != 5 || captured_params.at("op_0.strides").ai.size() != 3)
                return false;
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            if (captured_params.at("op_0.pads").type != 5 || captured_params.at("op_0.pads").ai.size() != 6)
                return false;

            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            const int ceil_mode = captured_params.find("op_0.ceil_mode") != captured_params.end() ? captured_params.at("op_0.ceil_mode").i : 0;
            if (pads[0] != pads[3] || pads[1] != pads[4] || pads[2] != pads[5])
            {
                // ceil_mode for opset9
                const Operator* avgpool = matched_operators.at("op_0");
                const std::vector<int>& in_shape = avgpool->inputs[0]->shape;
                if (in_shape.size() < 2)
                    return false;

                const int ind = in_shape[in_shape.size() - 3];
                const int inh = in_shape[in_shape.size() - 2];
                const int inw = in_shape[in_shape.size() - 1];
                const int kd = captured_params.at("op_0.kernel_shape").ai[0];
                const int kh = captured_params.at("op_0.kernel_shape").ai[1];
                const int kw = captured_params.at("op_0.kernel_shape").ai[2];
                const int dd = captured_params.find("op_0.dilations") != captured_params.end() ? captured_params.at("op_0.dilations").ai[0] : 1;
                const int dh = captured_params.find("op_0.dilations") != captured_params.end() ? captured_params.at("op_0.dilations").ai[1] : 1;
                const int dw = captured_params.find("op_0.dilations") != captured_params.end() ? captured_params.at("op_0.dilations").ai[2] : 1;
                const int sd = captured_params.find("op_0.strides") != captured_params.end() ? captured_params.at("op_0.strides").ai[0] : 1;
                const int sh = captured_params.find("op_0.strides") != captured_params.end() ? captured_params.at("op_0.strides").ai[1] : 1;
                const int sw = captured_params.find("op_0.strides") != captured_params.end() ? captured_params.at("op_0.strides").ai[2] : 1;

                const int ked = dd * (kd - 1) + 1;
                const int keh = dh * (kh - 1) + 1;
                const int kew = dw * (kw - 1) + 1;

                int ceil_padd = get_pool_ceil_padding(ind, ked, sd, pads[0]);
                int ceil_padh = get_pool_ceil_padding(inh, keh, sh, pads[1]);
                int ceil_padw = get_pool_ceil_padding(inw, kew, sw, pads[2]);

                if (ceil_mode == 0 && pads[0] + ceil_padd == pads[3] && pads[1] + ceil_padh == pads[4] && pads[2] + ceil_padw == pads[5])
                {
                    // useless tail padding  :D
                }
                else
                {
                    return false;
                }
            }
        }

        if (captured_params.find("op_0.auto_pad") != captured_params.end())
        {
            if (captured_params.at("op_0.auto_pad").type != 4)
                return false;

            const std::string& auto_pad = captured_params.at("op_0.auto_pad").s;
            if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER")
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["kernel_size"] = captured_params.at("op_0.kernel_shape");

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            op->params["dilation"] = captured_params.at("op_0.dilations");
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            op->params["stride"] = captured_params.at("op_0.strides");
        }
        else
        {
            op->params["stride"] = {1, 1, 1};
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            op->params["padding"] = {pads[0], pads[1], pads[2]};
        }
        else
        {
            op->params["padding"] = {0, 0, 0};
        }

        if (captured_params.find("op_0.ceil_mode") != captured_params.end())
        {
            int ceil_mode = captured_params.at("op_0.ceil_mode").i;
            op->params["ceil_mode"] = (ceil_mode != 0);
        }
        else
        {
            op->params["ceil_mode"] = false;
        }

        // ceil_mode for opset9
        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            if (pads[0] != pads[3] || pads[1] != pads[4] || pads[2] != pads[5])
            {
                op->params["ceil_mode"] = true;
            }
        }

        op->params["return_indices"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool3d_onnx, 120)

class F_max_pool3d_onnx_1 : public F_max_pool3d_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 3
pnnx.Input              input       0 1 input
MaxPool                 op_0        1 2 input out indices %*=%*
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_max_pool3d_onnx::write(op, captured_params);

        op->params["return_indices"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool3d_onnx_1, 120)

class F_max_pool3d_onnx_pad : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
MaxPool                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
F.pad                   pad         1 1 input pad
F.max_pool3d            maxpool     1 1 pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.kernel_shape") == captured_params.end())
            return false;

        if (captured_params.at("op_0.kernel_shape").type != 5 || captured_params.at("op_0.kernel_shape").ai.size() != 3)
            return false;

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            if (captured_params.at("op_0.dilations").type != 5 || captured_params.at("op_0.dilations").ai.size() != 3)
                return false;
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            if (captured_params.at("op_0.strides").type != 5 || captured_params.at("op_0.strides").ai.size() != 3)
                return false;
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            if (captured_params.at("op_0.pads").type != 5 || captured_params.at("op_0.pads").ai.size() != 6)
                return false;

            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            if (pads[0] == pads[3] && pads[1] == pads[4] && pads[2] == pads[5])
                return false;
        }

        if (captured_params.find("op_0.auto_pad") != captured_params.end())
        {
            if (captured_params.at("op_0.auto_pad").type != 4)
                return false;

            const std::string& auto_pad = captured_params.at("op_0.auto_pad").s;
            if (auto_pad == "VALID")
                return false;
        }

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        Operator* op_pad = ops.at("pad");
        Operator* op_maxpool = ops.at("maxpool");

        op_pad->params["mode"] = "constant";
        op_pad->params["value"] = -FLT_MAX;

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            op_pad->params["pad"] = std::vector<int>{pads[2], pads[5], pads[1], pads[4], pads[0], pads[3]};
        }

        op_maxpool->params["kernel_size"] = captured_params.at("op_0.kernel_shape");

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            op_maxpool->params["dilation"] = captured_params.at("op_0.dilations");
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            op_maxpool->params["stride"] = captured_params.at("op_0.strides");
        }
        else
        {
            op_maxpool->params["stride"] = {1, 1, 1};
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            op_maxpool->params["padding"] = {pads[0], pads[1], pads[2]};
        }
        else
        {
            op_maxpool->params["padding"] = {0, 0, 0};
        }

        if (captured_params.find("op_0.ceil_mode") != captured_params.end())
        {
            int ceil_mode = captured_params.at("op_0.ceil_mode").i;
            op_maxpool->params["ceil_mode"] = (ceil_mode != 0);
        }
        else
        {
            op_maxpool->params["ceil_mode"] = false;
        }

        // ceil_mode for opset9
        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            if (pads[0] != pads[3] || pads[1] != pads[4] || pads[2] != pads[5])
            {
                op_maxpool->params["ceil_mode"] = true;
            }
        }

        // resolve auto_pad
        if (captured_params.find("op_0.auto_pad") != captured_params.end())
        {
            const std::string& auto_pad = captured_params.at("op_0.auto_pad").s;

            const int kernel_d = op_maxpool->params.at("kernel_size").ai[0];
            const int kernel_h = op_maxpool->params.at("kernel_size").ai[1];
            const int kernel_w = op_maxpool->params.at("kernel_size").ai[2];
            const int stride_d = op_maxpool->params.at("stride").ai[0];
            const int stride_h = op_maxpool->params.at("stride").ai[1];
            const int stride_w = op_maxpool->params.at("stride").ai[2];

            const int wpad = kernel_w - 1;
            const int hpad = kernel_h - 1;
            const int dpad = kernel_d - 1;

            if (auto_pad == "SAME_UPPER")
            {
                op_pad->params["pad"] = std::vector<int>{wpad / 2, wpad - wpad / 2, hpad / 2, hpad - hpad / 2, dpad / 2, dpad - dpad / 2};
            }
            if (auto_pad == "SAME_LOWER")
            {
                op_pad->params["pad"] = std::vector<int>{wpad - wpad / 2, wpad / 2, hpad - hpad / 2, hpad / 2, dpad - dpad / 2, dpad / 2};
            }

            if (stride_w != 1 || stride_w != 1 || stride_d != 1)
            {
                fprintf(stderr, "auto_pad %s with stride %d %d %d may lead to incorrect output shape\n", auto_pad.c_str(), stride_w, stride_h, stride_d);
            }
        }

        op_maxpool->params["return_indices"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool3d_onnx_pad, 120)

} // namespace pnnx

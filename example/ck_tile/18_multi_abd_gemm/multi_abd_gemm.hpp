// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/kernel/mutiple_d_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

struct AddAdd
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<ck_tile::half_t, float, float, float>(
        ck_tile::half_t& e, const float& c, const float& d0, const float& d1) const
    {
        const float x0_f = c + d0 + d1;

        e = ck_tile::type_convert<ck_tile::half_t>(x0_f);
    }
};

template <typename DataType>
struct GemmBasicTypeConfig;

template <>
struct GemmBasicTypeConfig<ck_tile::half_t>
{
    using ADataType         = ck_tile::half_t;
    using BDataType         = ck_tile::half_t;
    using D0DataType        = ck_tile::half_t;
    using D1DataType        = ck_tile::half_t;
    using DsDataType        = ck_tile::tuple<D0DataType, D1DataType>;
    using AccDataType       = float;
    using CDataType         = ck_tile::half_t;
};

using Types = GemmBasicTypeConfig<ck_tile::half_t>;

// Specific type aliases for easy access
using ADataType   = Types::ADataType;
using BDataType   = Types::BDataType;
using AccDataType = Types::AccDataType;
using D0DataType   = Types::D0DataType;
using D1DataType   = Types::D1DataType;
using DsDataType   = Types::DsDataType;
using CDataType   = Types::CDataType;

using multi_d_gemm_kargs = ck_tile::MultipleDGemmHostArgs<DsDataType::size()>;

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("Ms", "", "M dimensions - empty by default.")
        .insert("Ns", "", "N dimensions - empty by default.")
        .insert("Ks", "", "K dimensions - empty by default.")
        .insert("stride_As", "", "Tensor A strides - it is empty by default.")
        .insert("stride_Bs", "", "Tensor B strides - it is empty by default.")
        .insert("stride_Cs", "", "Tensor C strides - it is empty by default.")
        .insert("a_layout", "R", "A tensor data layout - Row by default.")
        .insert("b_layout", "R", "B tensor data layout - Row by default.")
        .insert("c_layout", "R", "C tensor data layout - Row by default.")
        .insert("validate", "1", "0. No validation, 1. Validation on CPU.")
        .insert("warmup", "10", "number of iterations before benchmark the kernel.")
        .insert("repeat", "100", "number of iterations to benchmark the kernel.")
        .insert("group_count", "16", "group count.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

float multiple_d_gemm(const multi_d_gemm_kargs &kargs,
                        [[maybe_unused]]ck_tile::element_wise::PassThrough& f_element_wise_a,
                        [[maybe_unused]]ck_tile::element_wise::PassThrough& f_element_wise_b,
                        [[maybe_unused]]AddAdd& f_element_wise_d,
                        [[maybe_unused]]const ck_tile::stream_config& s);

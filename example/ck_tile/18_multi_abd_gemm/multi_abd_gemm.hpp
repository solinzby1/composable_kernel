// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"

struct AddAdd
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float, float, float>(
        ck::half_t& e, const float& c, const float& d0, const float& d1) const
    {
        const float x0_f = c + d0 + d1;

        e = ck::type_convert<ck::half_t>(x0_f);
    }
};

template <typename DataType>
struct GemmBasicTypeConfig;

template <>
struct GemmBasicTypeConfig<ck_tile::half_t>
{
    using ADataType         = ck_tile::half_t;
    using BDataType         = ck_tile::half_t;
    using CShuffleDataType  = float;
    using D0DataType        = float;
    using D1DataType        = float;
    using DsDataType        = ck::Tuple<D0DataType, D1DataType>;
    using AccDataType       = float;
    using EDataType         = ck_tile::half_t;
};

using Types = GemmBasicTypeConfig<ck_tile::half_t>;

// Specific type aliases for easy access
using ADataType   = Types::ADataType;
using BDataType   = Types::BDataType;
using AccDataType = Types::AccDataType;
using DDataType   = Types::CDataType;
using EDataType   = Types::EDataType;

using grouped_gemm_kargs = ck_tile::GroupedGemmHostArgs;

// runtime args
struct mulit_d_args : public ck_tile::MultiDGemmHostArgs
{
};

struct

    auto
    create_args(int argc, char* argv[])
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

float multiple_d_gemm(const void* a_m_k_dev_buf,
                      const void* b_k_n_dev_buf,
                      std::array<const void*, 2> &d_m_n_dev_buf,
                      const void* e_m_n_dev_buf,
                          ck_tile::index_t M,
                          ck_tile::index_t N,
                          ck_tile::index_t K,
                          ck_tile::index_t StrideAs,
                          ck_tile::index_t StrideBs,
                          std::array<ck_tile::index_t, 2> StrideDs,
                          index_t StrideE,
                          ck_tile::PassThrough &a_element_op,
                          ck_tile::PassThrough &b_element_op,
                          ck_tile::AddAdd &cde_element_op, 
                          const ck_tile::stream_config& s;

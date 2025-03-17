// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <cstddef>

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename CLayout>
float invoke_multi_d_gemm(const void* a_m_k_dev_buf,
                          const void* b_k_n_dev_buf,
                          const void* d_m_n_dev_buf,
                          void* e_m_n_dev_buf,
                          ck_tile::index_t M,
                          ck_tile::index_t N,
                          ck_tile::index_t K,
                          ck_tile::index_t StrideA,
                          ck_tile::index_t StrideB,
                          ck_tile::index_t StrideD,
                          ck_tile::index_t StrideC,
                          ck_tile::element_wise::PassThrough& a_element_op,
                          ck_tile::element_wise::PassThrough& b_element_op,
                          AddAdd& cde_element_op)
{
    multi_d_gemm_kargs gemm_descs({a_m_k_dev_buf,
                                   b_k_n_dev_buf,
                                   d_m_n_dev_buf,
                                   e_m_n_dev_buf,
                                   M,
                                   N,
                                   K,
                                   StrideA,
                                   StrideB,
                                   StrideD,
                                   StrideC});

    using DsLayout = ck_tile::tuple<D0Layout, D1Layout>;
    float ave_time = multiple_d_gemm<ALayout, BLayout, DsLayout, CLayout>(
        gemm_descs,
        a_element_op,
        b_element_op,
        cde_element_op,
        ck_tile::stream_config{nullptr, true, 1, 100, 100});

    std::string op_name{"Multiple D Gemm"};

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < 1; ++j)
    {
        flop += std::size_t(2) * M * N * K;

        num_btype +=
            sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;
    }

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << std::left << "Perf: " << std::setw(10) << ave_time << " ms, " << std::setw(10)
              << tflops << " TFlops, " << std::setw(10) << gb_per_sec << " GB/s, " << op_name
              << std::endl;

    return ave_time;
}

template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename CLayout>
int run_multiple_d_gemm_example_with_layouts(const ALayout a_layout                  = ALayout{},
                                             const BLayout b_layout                  = BLayout{},
                                             const D0Layout d0_layout                = D0Layout{},
                                             const D1Layout d1_layout                = D1Layout{},
                                             [[maybe_unused]] const CLayout e_layout = CLayout{})
{
    ck_tile::index_t M = 3840;
    ck_tile::index_t N = 4096;
    ck_tile::index_t K = 4096;

    const ck_tile::index_t StrideA = K;
    const ck_tile::index_t StrideB = K;
    const ck_tile::index_t StrideD = K;
    const ck_tile::index_t StrideC = N;

    // Create host tensors
    ck_tile::HostTensor<ADataType> a_m_k_tesnor(f_host_tensor_descriptor(M, K, StrideA, a_layout));
    ck_tile::HostTensor<BDataType> b_k_n_tensors(f_host_tensor_descriptor(K, N, StrideB, b_layout));
    ck_tile::HostTensor<D0DataType> d0_m_n_tensors(
        f_host_tensor_descriptor(M, N, StrideD, d0_layout));
    ck_tile::HostTensor<D1DataType> d1_m_n_tensors(
        f_host_tensor_descriptor(M, N, StrideD, d1_layout));
    // ck_tile::HostTensor<CDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC,
    // CLayout{}));
    ck_tile::HostTensor<CDataType> c_m_n_device_result(
        f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    // Get RND data for a0, a1, bk, dk
    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k_tesnor);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n_tensors);
    ck_tile::FillUniformDistribution<D0DataType>{-5.f, 5.f}(d0_m_n_tensors);
    ck_tile::FillUniformDistribution<D1DataType>{-5.f, 5.f}(d1_m_n_tensors);

    // Fill in buffers for all tensors
    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k_tesnor.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n_tensors.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d0_m_n_dev_buf(d0_m_n_tensors.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d1_m_n_dev_buf(d1_m_n_tensors.get_element_space_size_in_bytes());
    ck_tile::DeviceMem e_m_n_dev_buf(c_m_n_device_result.get_element_space_size_in_bytes());

    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         if (i == 0 && j < N)
    //             d0_m_n_tensors.mData.data()[i * M + j] =
    //             ck_tile::type_convert<ck_tile::half_t>(j);
    //         else
    //             d0_m_n_tensors.mData.data()[i * M + j] =
    //             ck_tile::type_convert<ck_tile::half_t>(i);
    //     }
    // }
    // Cpy data to device
    a_m_k_dev_buf.ToDevice(a_m_k_tesnor.mData.data());
    b_k_n_dev_buf.ToDevice(b_k_n_tensors.mData.data());
    d0_m_n_dev_buf.ToDevice(d0_m_n_tensors.mData.data());
    d1_m_n_dev_buf.ToDevice(d1_m_n_tensors.mData.data());

    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         printf("%f ", static_cast<double>(d0_m_n_tensors.mData.data()[i * M + j]));
    //     }
    //     printf("\n");
    // }
    e_m_n_dev_buf.SetZero();
    c_m_n_device_result.SetZero();

    auto a_element_op   = ck_tile::element_wise::PassThrough{}; // c  : element_wise op: do nothing
    auto b_element_op   = ck_tile::element_wise::PassThrough{}; // b  : element_wise op  do nothing
    auto cde_element_op = AddAdd{}; // cde: element_wise op  send by user

    constexpr ck_tile::index_t NumDTensor = 2;
    auto d_ptr_array = std::array<const void*, NumDTensor>{d0_m_n_dev_buf.GetDeviceBuffer(),
                                                           d1_m_n_dev_buf.GetDeviceBuffer()};
    (void)(d_ptr_array);

    invoke_multi_d_gemm<ALayout, BLayout, D0Layout, D1Layout, CLayout>(
        a_m_k_dev_buf.GetDeviceBuffer(),
        b_k_n_dev_buf.GetDeviceBuffer(),
        d0_m_n_dev_buf.GetDeviceBuffer(),
        e_m_n_dev_buf.GetDeviceBuffer(),
        M,
        N,
        K,
        StrideA,
        StrideB,
        StrideD,
        StrideC,
        a_element_op,
        b_element_op,
        cde_element_op);

    e_m_n_dev_buf.FromDevice(c_m_n_device_result.data());
    bool pass{true};
    // printf("\n");
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         printf("%f ", static_cast<double>(c_m_n_device_result.mData.data()[i * M + j]));
    //     }
    //     printf("\n");
    // }
    // if(arg_parser.get_int("v") == 1)
    // {
    ck_tile::HostTensor<CDataType> c_m_n_host_ref(
        ck_tile::host_tensor_descriptor(M, N, StrideC, is_row_major(CLayout{})));
    c_m_n_host_ref.SetZero();

    ck_tile::reference_gemm_multiple_d<ADataType, BDataType, DDataType, AccDataType, CDataType>(
        a_m_k_tesnor,
        b_k_n_tensors,
        d0_m_n_tensors,
        c_m_n_host_ref,
        {},
        {},
        [&]([[maybe_unused]] const auto& c, [[maybe_unused]] const auto& d) {
            const float ft = c + d;
            return ft;
        });
    const float max_accumulated_value =
        *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
    const auto rtol_atol = calculate_rtol_atol(K, 1, max_accumulated_value);
    pass                 = ck_tile::check_err(c_m_n_device_result,
                              c_m_n_host_ref,
                              "Error: Incorrect results!",
                              rtol_atol.at(ck_tile::number<0>{}),
                              rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
              << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The CPU veification result is:" << (pass ? "correct" : "fail") << std::endl;
    // }

    return pass;
}

int run_multiple_d_gemm_example(int argc, char* argv[])
{
    (void)(argc);
    (void)(argv);
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    const std::string a_layout = "R";
    const std::string b_layout = "C";

    if(a_layout == "R" && b_layout == "C")
    {
        return run_multiple_d_gemm_example_with_layouts(Row{}, Col{}, Row{}, Row{}, Row{});
    }
    else if(a_layout == "R" && b_layout == "R")
    {
        return run_multiple_d_gemm_example_with_layouts(Row{}, Col{}, Row{}, Row{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}

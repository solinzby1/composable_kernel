// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <cstddef>

template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout>
float invoke_multi_d_gemm(const void* a_m_k_dev_buf,
                          const void* b_k_n_dev_buf,
                          std::array<const void*, 2>& d_m_n_dev_buf,
                          const void* e_m_n_dev_buf,
                          ck_tile::index_t M,
                          ck_tile::index_t N,
                          ck_tile::index_t K,
                          ck_tile::index_t StrideAs,
                          ck_tile::index_t StrideBs,
                          std::array<ck_tile::index_t, 2> StrideDs,
                          index_t StrideE,
                          ck_tile::PassThrough& a_element_op,
                          ck_tile::PassThrough& b_element_op,
                          ck_tile::AddAdd& cde_element_op)
{
    using DsLayout = ck::Tuple<D0Layout, D1Layout>;
    float ave_time = multiple_d_gemm<ALayout, BLayout, DsLayout, ELayout>(
        a_m_k_dev_buf,
        b_k_n_dev_buf,
        d_m_n_dev_buf,
        e_m_n_dev_buf,
        M,
        N,
        K,
        StrideAs,
        StrideBs,
        StrideDs,
        StrideE,
        a_element_op,
        b_element_op,
        cde_element_op,
        ck_tile::stream_config{nullptr, true, 1, 100, 100});

    std::string op_name{"Multiple D Gemm"};

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < group_count; ++j)
    {
        flop += std::size_t(2) * args[j].M * args[j].N * args[j].K;

        num_btype += sizeof(ADataType) * args[j].M * args[j].K +
                     sizeof(BDataType) * args[j].K * args[j].N +
                     sizeof(CDataType) * args[j].M * args[j].N;
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
          typename ELayout>
int run_grouped_gemm_example_with_layouts(const ALayout a_layout   = ALayout{},
                                          const BLayout b_layout   = BLayout{},
                                          const D0Layout d0_layout = D0Layout{},
                                          const D1Layout d1_layout = D1Layout{},
                                          const ELayout e_layout   = ELayout{})
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    constexpr ck_tile::index_t StrideA = K;
    constexpr ck_tile::index_t StrideB = K;
    constexpr ck_tile::index_t StrideD = K;
    constexpr ck_tile::index_t StrideE = N;

    // Create host tensors
    ck_tile::HostTensor<ADataType> a_m_k_tesnor(f_host_tensor_descriptor(M, K, StrideA, a_layout));
    ck_tile::HostTensor<BDataType> b_k_n_tensors(f_host_tensor_descriptor(K, N, StrideB, b_layout));
    ck_tile::HostTensor<D0DataType> d0_m_n_tensors(f_host_tensor_descriptor(M, N, StrideD, d0_layout));
    ck_tile::HostTensor<D1DataType> d1_m_n_tensors(f_host_tensor_descriptor(M, N, StrideD, d1_layout));
    ck_tile::HostTensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    ck_tile::HostTensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

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
    ck_tile::DeviceMem e_m_n_dev_buf(d1_m_n_tensors.get_element_space_size_in_bytes());

    // Cpy data to device
    a_m_k_dev_buf.ToDevice(a_m_k_tesnor.mData.data());
    b_k_n_dev_buf.ToDevice(b_k_n_tensors.mData.data());
    d0_m_n_dev_buf.ToDevice(d0_m_n_tensors.mData.data());
    d1_m_n_dev_buf.ToDevice(d1_m_n_tensors.mData.data());
    e_m_n_dev_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = ck_tile::PassThrough{}; // c  : element_wise op: do nothing
    auto b_element_op   = ck_tile::PassThrough{}; // b  : element_wise op  do nothing
    auto cde_element_op = ck_tile::AddAdd{};      // cde: element_wise op  send by user

    constexpr ck::index_t NumDTensor = 2;
    auto d_ptr_array = std::array<const void*, NumDTensor>{d0_m_n_dev_buf.GetDeviceBuffer(),
                                                           d1_m_n_dev_buf.GetDeviceBuffer()};

    invoke_multi_d_gemm<ALayout, BLayout, D0Layout, D1Layout, ELayout>(
        a_m_k_dev_buf.GetDeviceBuffer(),
        b_k_n_dev_buf.GetDeviceBuffer(),
        container_d,
        e_m_n_dev_buf.GetDeviceBuffer(),
        M,
        N,
        K,
        StrideA,
        StrideB,
        std::array<ck::index_t, NumDTensor>{StrideD, StrideD},
        StrideE,
        a_element_op,
        b_element_op,
        cde_element_op);

    bool pass{true};

    return pass;
}

int run_grouped_gemm_example(int argc, char* argv[])
{
    (void)(argc);
    (void)(argv);
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(a_layout == "R" && b_layout == "C")
    {
        return run_grouped_gemm_example_with_layouts(Row{}, Col{}, Row{}, Row{});
    }
    else if(a_layout == "R" && b_layout == "R")
    {
        return run_grouped_gemm_example_with_layouts(Row{}, Row{}, Row{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}

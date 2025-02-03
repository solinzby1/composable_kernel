// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm.hpp"
#include "utils.hpp"

namespace {

struct TileSettings {

};

struct WrapSettings {

};

struct WrapTileSettings {

};

class SeqNamespace { 
    protected:
    using Seq = ck_tile::sequence;
};

template <typename T = TileShape()> class TileShape : public SeqNamespace {
  using shape = Seq<T::M_Tile, T::N_Tile, T::K_Tile>;
};
template <typename T> class WrapShape : public SeqNamespace {
  using shape = Seq<T::M_Warp, T::N_Warp, T::K_Warp>;
};

template <typename T> class WrapTileShape : public SeqNamespace {
  using shape = Seq<T::M_Warp_Tile, T::N_Warp_Tile, T::K_Warp_Tile>
};

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout>
float multiple_d_gemm(const void* a_m_k_dev_buf,
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
                      ck_tile::AddAdd& cde_element_op,
                      const ck_tile::stream_config& s)
{
    using code_gen_tile_shape = ck_tile::TileGemmShape<
            TileShape<T>::shape,
            WrapShape<T>::shape,
            WrapTileShape<T>::shape
    >;

    using f_partitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

    template <typename ALayout, typename BLayout, typename CLayout>
    using CodegenGemmTraits = ck_tile::TileGemmTraits<GroupedGemmKernelParam::kPadM,
                                                    GroupedGemmKernelParam::kPadN,
                                                    GroupedGemmKernelParam::kPadK,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout>;

    using f_kernel = ck_tile::MultipleDGemmKernel<f_paritioner, f_pipeline, f_epilog>
    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
    }

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel with args:"
                  << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                  << std::endl;
    }

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}
#include "run_grouped_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_gemm_example(argc, argv); }

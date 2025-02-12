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

inline namespace {

/**
 * @brief Settings enum for Tile
 */
enum class TileSetting : int {
  M = 128,
  N = 128,
  K = 32,
};

/**
 * @brief Settings enum for Tile
 */
enum class WrapSetting : int {
  M = 2,
  N = 2,
  K = 1,
};

/**
 * @brief Settings enum for Tile
 */
enum class WrapTileSetting : int {
  M = 32,
  N = 32,
  K = 8,
};

/**
 * @brief The kPadM, kPadN, kPadK & kBlockPerCu 
 *        should also come from the Codegen part.
 */
enum class CodeGenPart : bool {
    kPadM = false;
    kPadN = false;
    kPadK = false;
};

class SequenceMapper {
  protected:
  using Seq = ck_tile::sequence; 
}
/**
 * @brief Wrapper to keep the shape for a given settings
 */
template <typename T = TileSetting> class TileShape : public SequenceMapper {
  using shape = Seq<T::M, T::N, T::K>;
};

/**
 * @brief Wrapper to keep the shape for a given settings
 */
template <typename T = WrapSetting> class WrapShape : public SequenceMapper {
  using shape = Seq<T::M, T::N, T::K>;
};

/**
 * @brief Wrapper to keep the shape for a given settings
 */
template <typename T = WrapTileSetting> class WrapTileShape : public SequenceMapper {
  using shape = Seq<T::M, T::N, T::K>;
};

/**
 * @brief Function to run multiple_gemm with multiple D
 * 
 */

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout>
auto multiple_d_gemm(const void* a_m_k_dev_buf,
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
                      const ck_tile::stream_config& s) -> float
{
  using f_code_gemm_traits 
        = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, DsLayout, ELayout>;
  using f_shape 
        = ck_tile::TileGemmShape<TileShape::shape, WrapShape::shape, WrapTileShape::shape>;

  // TODO(mozga-amd): MutlipleGemm Pipeline requires impl
  using f_code_gemm_pipeline = ck_tile::MutlipleGemmPipelineProblem<
            ADataType,
            BDataType, 
            DDataType, 
            AccDataType,
            f_element_wise_a, // Old ck works only when A element wise is PassThrough
            f_element_wise_b, // Old ck works only when B element wise is PassThrough
            f_element_wise_abd, // Old ck works for each of them
            f_shape, 
            f_code_gemm_traits>;

  using f_gemm_epilogue =
      ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<
            AccDataType, EDataType, ELayout, 
            CodegenPipelineProblem::kBlockSize,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            WrapSettings::M_Warp, WrapSettings::N_Warp,
            WrapTileSettings::M_Warp_Tile, WrapTileSettings::N_Warp_Tile,
            WrapTileSettings::K_Warp_Tile, f_code_gemm_pipeline::TransposeC>
        >;
    
    // 2D partitioner for that
    using f_partitioner = ck_tile::GemmTile1DPartitioner<f_shape>;
    using f_kernel = ck_tile::MultipleDGemmKernel<f_paritioner, f_code_gemm_pipeline, f_epilog>;

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

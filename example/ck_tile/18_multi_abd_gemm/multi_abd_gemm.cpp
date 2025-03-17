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
#include "multi_abd_gemm.hpp"
#include "utils.hpp"

/**
 * @brief Settings enum for Tile
 */
enum TileSetting
{
    M1 = 256,
    N1 = 256,
    K1 = 32,
};

/**
 * @brief Settings enum for Tile
 */
enum WrapSetting
{
    M2 = 2,
    N2 = 2,
    K2 = 1,
};

/**
 * @brief Settings enum for Tile
 */
enum WrapTileSetting
{
    M3 = 32,
    N3 = 32,
    K3 = 16,
};

/**
 * @brief The kPadM, kPadN, kPadK & kBlockPerCu
 *        should also come from the Codegen part.
 */
enum CodeGenPart
{
    kPadM = false,
    kPadN = false,
    kPadK = false,
};

/**
 * @brief Wrapper to keep the shape for a given settings
 */
template <typename T = TileSetting>
struct TileShape
{
    using shape = ck_tile::sequence<T::M1, T::N1, T::K1>;
};

/**
 * @brief Wrapper to keep the shape for a given settings
 */
template <typename T = WrapSetting>
struct WrapShape
{
    using shape = ck_tile::sequence<T::M2, T::N2, T::K2>;
};

/**
 * @brief Wrapper to keep the shape for a given settings
 */
template <typename T = WrapTileSetting>
struct WrapTileShape
{
    using shape = ck_tile::sequence<T::M3, T::N3, T::K3>;
};

/**
 * @brief Function to run multiple_gemm with multiple D
 *
 */
template <typename ALayout, typename BLayout, typename DLayout, typename CLayout>
auto multiple_d_gemm(const multi_d_gemm_kargs& args,
                     [[maybe_unused]] ck_tile::element_wise::PassThrough& f_element_wise_a,
                     [[maybe_unused]] ck_tile::element_wise::PassThrough& f_element_wise_b,
                     [[maybe_unused]] AddAdd& f_element_wise_d,
                     [[maybe_unused]] const ck_tile::stream_config& s) -> float
{
    // The kPadM, kPadN, kPadK & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr int kBlockPerCu = 1;

    // This part comes from the Codegen
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 128;
    constexpr ck_tile::index_t K_Tile = 32;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;
    //constexpr bool TransposeC = false;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 8;

    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    // ===============================================

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<CodegenGemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using CodegenGemmTraits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;
    using CodegenPipelineProblem = ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenGemmShape, CodegenGemmTraits>;
    using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

    using GemmEpilogue = ck_tile::MultipleDCShuffleEpilogue<
        ck_tile::MultipleDCShuffleEpilogueProblem<AccDataType,
                                                  CDataType,
                                                  DDataType,
                                                  DLayout,
                                                  CLayout,
                                                  AddAdd,
                                                  CodegenPipelineProblem::kBlockSize,
                                                  TilePartitioner::MPerBlock,
                                                  TilePartitioner::NPerBlock,
                                                  M_Warp,
                                                  N_Warp,
                                                  M_Warp_Tile,
                                                  N_Warp_Tile,
                                                  K_Warp_Tile,
                                                  CodegenPipelineProblem::TransposeC>>;

    // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
    // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
    using Kernel = ck_tile::MultipleDGemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;

    auto kargs = Kernel::MakeKernelArgs(args);

    const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
    constexpr dim3 blocks = Kernel::BlockSize();

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

#include "run_multi_abd_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_multiple_d_gemm_example(argc, argv); }

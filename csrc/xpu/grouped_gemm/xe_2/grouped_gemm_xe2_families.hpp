/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief The heavy half of the Xe2 MoE grouped GEMM: the kernel launcher template and the five
    weight-format families, as templates over the tile policy and the activation type.

    Included only by the per-variant translation units in split/ (one or two kernels each), never
    by the dispatch (grouped_gemm_xe2_interface.hpp). Before this split one translation unit
    instantiated every variant and needed more than 30 GB to compile.
*/
#pragma once

#include <torch/all.h>
#include "csrc/utils.h"

#include <cute/tensor.hpp>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"

#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"
#include "grouped_gemm_xe2_launch.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

// type tag to define a unique sycl kernel name
template <
    typename,
    typename,
    typename,
    typename,
    char,
    char,
    class,
    A_DTYPE,
    B_DTYPE>
class GemmCuteName;

template <
    char layoutA,
    char layoutB,
    class policy,
    A_DTYPE TENSOR_A_DTYPE,
    B_DTYPE TENSOR_B_DTYPE,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
void MoEGEMMLauncher(
    sycl::queue& stream,
    const ElementA* activations,
    const ElementB* weights,
    const ElementS* scales,
    const ElementBI* bias,
    ElementD* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* rows_per_expert,
    const int num_experts,
    const int group_size,
    int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  static constexpr int MaxThreadsPerSM = 512;

  TORCH_CHECK(
      MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0,
      "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup")

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(
      1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<GemmCuteName<
        ElementA,
        ElementB,
        ElementS,
        ElementD,
        layoutA,
        layoutB,
        policy,
        TENSOR_A_DTYPE,
        TENSOR_B_DTYPE>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoE::MoEGEMM<
              TENSOR_A_DTYPE,
              TENSOR_B_DTYPE,
              GmemTiledCopyA,
              GmemTiledCopyB,
              GmemTiledCopyD,
              layoutA,
              layoutB,
              'R'>(
              activations,
              weights,
              scales,
              bias,
              outputs,
              mma,
              rows_per_expert,
              num_experts,
              group_size,
              gemm_n,
              gemm_k,
              atomic_buffer,
              local_mem);
        });
  });
}

// ---- the five families -----------------------------------------------------------------------
// Each is the body of one of the launcher macros of the original interface, with the tensors
// already reduced to raw pointers (MoELaunchArgs). ScalarT is the activation / bias / output type.

// int4 or mxfp4 weights [E, N, K/2], activations 16-bit, scales 16-bit (int4) or E8M0 bytes (mxfp4)
template <class Policy, class ScalarT>
void w4a16_launch(const MoELaunchArgs& a) {
  if (a.is_int4) {
    MoEGEMMLauncher<'R', 'C', Policy, A_DTYPE::BITS16, B_DTYPE::INT4>(
        *a.queue,
        static_cast<const ScalarT*>(a.A),
        static_cast<const uint8_t*>(a.B),
        static_cast<const ScalarT*>(a.scales),
        static_cast<const ScalarT*>(a.bias),
        static_cast<ScalarT*>(a.D),
        a.N, a.K, a.rows_per_expert, a.num_experts, a.group_size, a.atomic_buffer);
  } else if (a.is_mxfp4) {
    MoEGEMMLauncher<'R', 'C', Policy, A_DTYPE::BITS16, B_DTYPE::MXFP4>(
        *a.queue,
        static_cast<const ScalarT*>(a.A),
        static_cast<const uint8_t*>(a.B),
        static_cast<const uint8_t*>(a.scales),
        static_cast<const ScalarT*>(a.bias),
        static_cast<ScalarT*>(a.D),
        a.N, a.K, a.rows_per_expert, a.num_experts, a.group_size, a.atomic_buffer);
  } else {
    TORCH_CHECK(false, "w4a16 grouped gemm: weights must be int4 or mxfp4");
  }
}

// fp8 weights (e4m3 or e5m2) [E, K, N], activations 16-bit; BD picks per-tensor / block / mx scaling,
// ScaleT is float for the first two and E8M0 bytes for mxfp8
template <class Policy, class ScalarT, B_DTYPE BD, class ScaleT>
void fp8_launch(const MoELaunchArgs& a) {
  if (a.B_dtype == c10::ScalarType::Float8_e4m3fn) {
    MoEGEMMLauncher<'R', 'R', Policy, A_DTYPE::BITS16, BD>(
        *a.queue,
        static_cast<const ScalarT*>(a.A),
        static_cast<const float_e4m3_t*>(a.B),
        static_cast<const ScaleT*>(a.scales),
        static_cast<const ScalarT*>(a.bias),
        static_cast<ScalarT*>(a.D),
        a.N, a.K, a.rows_per_expert, a.num_experts, a.group_size, a.atomic_buffer);
  } else if (a.B_dtype == c10::ScalarType::Float8_e5m2) {
    MoEGEMMLauncher<'R', 'R', Policy, A_DTYPE::BITS16, BD>(
        *a.queue,
        static_cast<const ScalarT*>(a.A),
        static_cast<const float_e5m2_t*>(a.B),
        static_cast<const ScaleT*>(a.scales),
        static_cast<const ScalarT*>(a.bias),
        static_cast<ScalarT*>(a.D),
        a.N, a.K, a.rows_per_expert, a.num_experts, a.group_size, a.atomic_buffer);
  } else {
    TORCH_CHECK(false, "fp8 grouped gemm requires B in {float8_e4m3fn, float8_e5m2}");
  }
}

// 16-bit weights [E, K, N] and activations of the same type, no scales
template <class Policy, class ScalarT>
void w16a16_launch(const MoELaunchArgs& a) {
  MoEGEMMLauncher<'R', 'R', Policy, A_DTYPE::BITS16, B_DTYPE::BITS16>(
      *a.queue,
      static_cast<const ScalarT*>(a.A),
      static_cast<const ScalarT*>(a.B),
      static_cast<const ScalarT*>(a.scales),
      static_cast<const ScalarT*>(a.bias),
      static_cast<ScalarT*>(a.D),
      a.N, a.K, a.rows_per_expert, a.num_experts, a.group_size, a.atomic_buffer);
}

}  // namespace MoE

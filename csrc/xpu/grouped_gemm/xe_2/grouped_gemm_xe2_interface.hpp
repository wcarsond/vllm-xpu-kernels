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
    \brief CUTLASS Intel BMG MoE API: argument checking and dispatch for the grouped GEMM.

    This file no longer includes cutlass. It validates the tensors, works out the weight format
    (the "family") and the tile policy, fills MoELaunchArgs and calls one of the launchers that the
    translation units in split/ define (see grouped_gemm_xe2_launch.hpp). Each of those compiles one
    or two kernels; the previous single-file form compiled all ~62 and needed over 30 GB.
*/
#pragma once

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "csrc/utils.h"
#include "grouped_gemm_xe2_launch.hpp"

namespace MoE {

at::Tensor cutlass_grouped_gemm_xe2_impl(
    at::Tensor& ptr_A,
    at::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor& ptr_D,
    at::Tensor& rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  auto A_dtype = ptr_A.dtype();
  auto B_dtype = ptr_B.dtype();
  bool is_weight_fp8 =
      ((B_dtype == at::kFloat8_e4m3fn) || (B_dtype == at::kFloat8_e5m2));
  bool is_B_int4 = (B_dtype == at::kChar) && ptr_scales.has_value();
  bool is_B_mxfp4 =
      (B_dtype == at::kFloat4_e2m1fn_x2) && ptr_scales.has_value();
  // MXFP8: FP8 weights + E8M0 (uint8 / float8_e8m0fnu) block scales, gs=32.
  bool is_B_mxfp8 = false;
  if (is_weight_fp8 && ptr_scales.has_value() && ptr_scales->dim() == 3) {
    auto sd = ptr_scales->dtype();
    is_B_mxfp8 = (sd == at::kByte) || (sd == at::kFloat8_e8m0fnu);
  }
  // Block-FP8: FP8 weights + float32 2D scales [E, K/128, N/128].
  bool is_B_block_fp8 = false;
  if (is_weight_fp8 && ptr_scales.has_value() && ptr_scales->dim() == 3 &&
      !is_B_mxfp8) {
    is_B_block_fp8 = (ptr_scales->dtype() == at::kFloat);
  }

  TORCH_CHECK(N % 8 == 0, "N must be divisible by 8");

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B.dim() == 3, "ptr_B must be 3D [num_experts, K, N]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, N]");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->dim() == 2, "ptr_bias must be 2D [num_experts, N]");
  }

  TORCH_CHECK(ptr_A.is_contiguous(), "ptr_A must be contiguous");
  TORCH_CHECK(ptr_B.is_contiguous(), "ptr_B must be contiguous");
  TORCH_CHECK(ptr_D.is_contiguous(), "ptr_D must be contiguous");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->is_contiguous(), "ptr_bias must be contiguous");
  }

  int A_total_M = ptr_A.size(0);
  int A_K = ptr_A.size(1);

  int B_E = ptr_B.size(0);
  int B_K = ptr_B.size(1);
  int B_N = ptr_B.size(2);
  if (is_B_int4 || is_B_mxfp4) {
    B_K = ptr_B.size(2) * 2;
    B_N = ptr_B.size(1);
  }

  int D_total_M = ptr_D.size(0);
  int D_N = ptr_D.size(1);
  int group_size = -1;
  int A_avg_M = A_total_M / num_experts;

  TORCH_CHECK(B_E == num_experts, "ptr_B.size(0) must match num_experts");
  TORCH_CHECK(A_total_M == D_total_M, "ptr_A.size(0) must match ptr_D.size(0)");
  TORCH_CHECK(A_K == B_K && B_K == K, "ptr_A.size(1) must match ptr_B.size(1)");
  TORCH_CHECK(B_N == D_N && D_N == N, "ptr_B.size(2) must match ptr_D.size(1)");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(
        ptr_bias->size(0) == num_experts,
        "ptr_bias.size(0) must match num_experts");
    TORCH_CHECK(ptr_bias->size(1) == N, "ptr_bias.size(1) must match N");
  }

  TORCH_CHECK(
      A_dtype == at::kBFloat16 || A_dtype == at::kHalf,
      "grouped gemm activations must be bfloat16 or float16");
  const bool a_bf16 = (A_dtype == at::kBFloat16);

  // Must be zeroed before kernel launch. Workgroup execution order
  // is non-deterministic, so initializing in the definition is the most
  // safe place to avoid any race condition as well as garbage data
  at::Tensor atomic_buffer =
      at::zeros({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));

  MoELaunchArgs a{};
  a.queue = &dpcpp_queue;
  a.A = ptr_A.data_ptr();
  a.B = ptr_B.data_ptr();
  a.scales = nullptr;
  a.bias = ptr_bias.has_value() ? ptr_bias->data_ptr() : nullptr;
  a.D = ptr_D.data_ptr();
  a.N = static_cast<int>(N);
  a.K = static_cast<int>(K);
  a.rows_per_expert = static_cast<const int*>(rows_per_expert.data_ptr());
  a.num_experts = static_cast<int>(num_experts);
  a.atomic_buffer = static_cast<int32_t*>(atomic_buffer.data_ptr());
  a.A_dtype = ptr_A.scalar_type();
  a.B_dtype = ptr_B.scalar_type();
  a.is_int4 = is_B_int4;
  a.is_mxfp4 = is_B_mxfp4;

  MoEFamily family;
  int policy;
  // Keeps a host-side rearranged scale tensor alive until the kernel has been submitted.
  at::Tensor scales_keep;

  if (is_B_int4 || is_B_mxfp4) {
    TORCH_CHECK(ptr_scales.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(
        ptr_scales->dim() == 3,
        "ptr_scales of int4 must be 3D [num_experts, group_num, N]");
    TORCH_CHECK(
        ptr_scales->size(0) == num_experts,
        "ptr_scales.size(0) of int4 must match num_experts");
    TORCH_CHECK(
        K % ptr_scales->size(2) == 0,
        "ptr_scales.size(2) of int4 must be divisible by K");
    TORCH_CHECK(
        ptr_scales->size(1) == N, "ptr_scales.size(1) of int4 must match N");
    int group_num = ptr_scales->size(2);
    group_size = K / group_num;

    TORCH_CHECK(
        group_size == 32 || group_size == 64 || group_size == 128 ||
            group_size == 256,
        "group_size must be 32, 64, 128 or 256");
    a.scales = ptr_scales->data_ptr();
    family = MoEFamily::W4A16;
    policy = (A_avg_M <= 4) ? 0 : (A_avg_M <= 8) ? 1 : (A_avg_M <= 128) ? 2 : 3;
  } else if (is_B_mxfp8) {
    TORCH_CHECK(ptr_scales.has_value(), "mxfp8 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(
        ptr_scales->dim() == 3,
        "ptr_scales of mxfp8 must be 3D [num_experts, K/32, N] "
        "or [num_experts, N, K/32]");
    TORCH_CHECK(
        ptr_scales->size(0) == num_experts,
        "ptr_scales.size(0) of mxfp8 must match num_experts");
    TORCH_CHECK(K % 32 == 0, "mxfp8 requires K divisible by 32");

    // Canonical kernel layout is [E, N, K/32] (same indexing as mxfp4 E8M0).
    // Accept [E, K/32, N] from moe_utils / vLLM and transpose once on host.
    scales_keep = *ptr_scales;
    if (scales_keep.scalar_type() == at::kFloat8_e8m0fnu) {
      scales_keep = scales_keep.view(at::kByte);
    }
    if (scales_keep.size(1) == (K / 32) && scales_keep.size(2) == N) {
      scales_keep = scales_keep.transpose(1, 2).contiguous();
    }
    TORCH_CHECK(
        scales_keep.size(1) == N && scales_keep.size(2) == (K / 32),
        "mxfp8 scales must be [E, N, K/32] after layout normalize; got [",
        scales_keep.size(0),
        ", ",
        scales_keep.size(1),
        ", ",
        scales_keep.size(2),
        "] with N=",
        N,
        " K=",
        K);
    group_size = 32;
    a.scales = scales_keep.data_ptr();
    family = MoEFamily::MXFP8;
    policy = (A_avg_M <= 8) ? 0 : (A_avg_M <= 32) ? 1 : 2;
  } else if (is_B_block_fp8) {
    TORCH_CHECK(
        ptr_scales.has_value(), "block-fp8 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(
        ptr_scales->dim() == 3,
        "ptr_scales of block-fp8 must be 3D [num_experts, K/128, N/128]");
    TORCH_CHECK(
        ptr_scales->size(0) == num_experts,
        "ptr_scales.size(0) of block-fp8 must match num_experts");
    TORCH_CHECK(K % 128 == 0, "block-fp8 requires K divisible by 128");
    TORCH_CHECK(N % 128 == 0, "block-fp8 requires N divisible by 128");
    TORCH_CHECK(
        ptr_scales->size(1) == (K / 128) && ptr_scales->size(2) == (N / 128),
        "block-fp8 scales must be [E, K/128, N/128]; got [",
        ptr_scales->size(0),
        ", ",
        ptr_scales->size(1),
        ", ",
        ptr_scales->size(2),
        "] with N=",
        N,
        " K=",
        K);
    TORCH_CHECK(
        ptr_scales->dtype() == at::kFloat, "block-fp8 scales must be float32");
    group_size = 128;
    a.scales = ptr_scales->data_ptr();
    family = MoEFamily::BLOCK_FP8;
    policy = (A_avg_M <= 8) ? 0 : (A_avg_M <= 32) ? 1 : 2;
  } else if (is_weight_fp8) {
    TORCH_CHECK(ptr_scales.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(
        ptr_scales->dim() == 1, "ptr_scales of fp8 must be 1D [num_experts]");
    TORCH_CHECK(
        ptr_scales->size(0) == num_experts,
        "ptr_scales.size(0) of fp8 must match num_experts");
    TORCH_CHECK(ptr_scales->dtype() == at::kFloat, "ptr_scales must be float");
    a.scales = ptr_scales->data_ptr();
    family = MoEFamily::FP8;
    policy = (A_avg_M <= 8) ? 0 : (A_avg_M <= 32) ? 1 : 2;
  } else {
    TORCH_CHECK(
        !ptr_scales.has_value(), "w16a16 grouped gemm must not have scales");
    TORCH_CHECK(
        B_dtype == A_dtype,
        "w16a16 grouped gemm: weights and activations must share a dtype");
    family = MoEFamily::W16A16;
    if (A_avg_M <= 8) {
      policy = 0;  // m_16
    } else if (A_avg_M <= 16) {
      policy = 1;  // m_32
    } else if (B_N <= 64) {
      policy = 2;  // n_64
    } else if (B_N <= 512) {
      policy = 3;  // n_128
    } else {
      policy = 4;  // full tile
    }
  }
  a.group_size = group_size;
  select_launcher(family, policy, a_bf16)(a);
  return ptr_D;
}

}  // namespace MoE

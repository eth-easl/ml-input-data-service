/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(HIFIMINI)

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/xtensa/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_conv.h"

namespace tflite {

void ConvEvalHifiMini(const ConvParams& params,
                      const int32_t* output_multiplier,
                      const int32_t* output_shift,
                      const RuntimeShape& input_shape, const int8_t* input_data,
                      const RuntimeShape& filter_shape,
                      const int8_t* filter_data, const RuntimeShape& bias_shape,
                      const int32_t* bias_data,
                      const RuntimeShape& output_shape, int8_t* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int batches = input_shape.Dims(0);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);

  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);

  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  ae_p24x2s input_offset_24x2 = AE_MOVPA24(input_offset);
  ae_q56s output_offset_56 = AE_CVTQ48A32S(output_offset);
  ae_q56s output_activation_min_56 = AE_CVTQ48A32S(output_activation_min);
  ae_q56s output_activation_max_56 = AE_CVTQ48A32S(output_activation_max);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          ae_q56s acc_56 = AE_ZEROQ56();

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; filter_x += 2) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);
              if (is_point_inside_image) {
                // Find current input index, minus 2 for Xtensa load
                // alignments:
                // TODO(b/147322595): Consider doing these offset calculations
                // with intrinsics:
                int input_idx =
                    ((batch * input_height + in_y) * input_width + in_x) *
                        input_depth * 2 -
                    2;
                const int8_t* input_vals_offset_ptr = input_data + input_idx;
                for (int i = 0; i < input_depth; i += 2) {
                  // Load signed 2x 8bit values and right shift into 24bit
                  // alignment:
                  ae_p24x2s input_vals_24x2;
                  AE_LP8X2F_IU(input_vals_24x2, input_vals_offset_ptr, 2);
                  input_vals_24x2 = AE_P24X2S_SRAI(input_vals_24x2, 16);

                  // Add input offset (24bit aligned):
                  input_vals_24x2 =
                      AE_P24S_ADDS_P24X2S(input_vals_24x2, input_offset_24x2);

                  // Find current filter index, minus 2 for Xtensa load
                  // alignments:
                  int filter_idx =
                      ((out_channel * filter_height + filter_y) * filter_width +
                       filter_x) *
                          filter_depth +
                      i - 2;
                  const int8_t* filter_vals_offset_ptr =
                      filter_data + filter_idx;

                  // Load signed 2x 8bit values and right shift into 24bit
                  // alignment:
                  ae_p24x2s filter_vals_24x2;
                  AE_LP8X2F_IU(filter_vals_24x2, filter_vals_offset_ptr, 2);
                  filter_vals_24x2 = AE_P24X2S_SRAI(filter_vals_24x2, 16);

                  // Multiply and accumulate into 48bit bit space:
                  AE_MULAAP24S_HH_LL(acc_56, filter_vals_24x2, input_vals_24x2);
                }
              }
            }
          }

          // Left shift from 48bit alignment to 32bit:
          acc_56 = AE_Q56S_SLAI(acc_56, 16);

          if (bias_data) {
            // Load and add bias at 32bit alignment:
            ae_q56s bias_56 = AE_CVTQ48A32S(bias_data[out_channel]);
            acc_56 = AE_ADDQ56(acc_56, bias_56);
          }

          // Shift from 32bit alignment to 24bit alignment and place back on
          // the PR register:
          acc_56 = AE_Q56S_SLAI(acc_56, 8);
          ae_p24x2s acc_24x2 = AE_TRUNCP24Q48(acc_56);

          // Apply quantized multiplier and accumulate result at 48bit
          // alignment. Convert the (unsigned) 32-bit multiplier down to a
          // 24-bit multiplier.
          acc_56 = MultiplyByQuantizedMultiplier(
              acc_24x2, output_multiplier[out_channel] >> 8,
              output_shift[out_channel]);

          // Add output offset, cap activation, and assign to the output:
          acc_56 = AE_ADDQ56(acc_56, output_offset_56);
          acc_56 = AE_MINQ56S(acc_56, output_activation_max_56);
          acc_56 = AE_MAXQ56S(acc_56, output_activation_min_56);

          int output_idx =
              ((batch * output_height + out_y) * output_width + out_x) *
                  output_depth +
              out_channel;
          output_data[output_idx] = static_cast<int8_t>(AE_TRUNCA32Q48(acc_56));
        }
      }
    }
  }
}

// TODO(b/154240772): Move shared code into common methods.
void Conv1x32Input32x32FilterHifiMini(
    const int input_offset, const int output_offset,
    const int quantized_activation_min, const int quantized_activation_max,
    const int32_t* output_multiplier, const int32_t* output_shift,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  ae_p24x2s input_offset_24x2 = AE_MOVPA24(input_offset);
  ae_q56s output_offset_56 = AE_CVTQ48A32S(output_offset);
  ae_q56s output_activation_max_56 = AE_CVTQ48A32S(quantized_activation_max);
  ae_q56s output_activation_min_56 = AE_CVTQ48A32S(quantized_activation_min);

  constexpr int kChannels = 32;
  constexpr int kFilterDepth = 32;
  for (int ch = 0; ch < kChannels; ch++) {
    ae_q56s acc_56 = AE_ZEROQ56();
    const int8_t* input_vals_ptr = input_data - 2;
    for (int i = 0; i < kFilterDepth; i += 2) {
      // Load signed 2x 8bit values and right shift into 24bit
      // alignment:
      ae_p24x2s input_vals_24x2;
      AE_LP8X2F_IU(input_vals_24x2, input_vals_ptr, 2);
      input_vals_24x2 = AE_P24X2S_SRAI(input_vals_24x2, 16);

      // Add input offset (24bit aligned):
      input_vals_24x2 = AE_P24S_ADDS_P24X2S(input_vals_24x2, input_offset_24x2);
      // Find current filter index, minus 2 for Xtensa load
      // alignments:
      const int filter_idx = ch * kFilterDepth + i - 2;
      const int8_t* filter_vals_offset_ptr = filter_data + filter_idx;

      // Load signed 2x 8bit values and right shift into 24bit
      // alignment:
      ae_p24x2s filter_vals_24x2;
      AE_LP8X2F_IU(filter_vals_24x2, filter_vals_offset_ptr, 2);
      filter_vals_24x2 = AE_P24X2S_SRAI(filter_vals_24x2, 16);

      // Multiply and accumulate into 48bit bit space:
      AE_MULAAP24S_HH_LL(acc_56, filter_vals_24x2, input_vals_24x2);
    }
    // Left shift from 48bit alignment to 32bit:
    acc_56 = AE_Q56S_SLAI(acc_56, 16);
    if (bias_data) {
      // Load and add bias at 32bit alignment:
      ae_q56s bias_56 = AE_CVTQ48A32S(bias_data[ch]);
      acc_56 = AE_ADDQ56(acc_56, bias_56);
    }

    // Shift from 32bit alignment to 24bit alignment and place back on
    // the PR register:
    acc_56 = AE_Q56S_SLAI(acc_56, 8);
    ae_p24x2s acc_24x2 = AE_TRUNCP24Q48(acc_56);

    // Apply quantized multiplier and accumulate result at 48bit alignment.
    // Convert the (unsigned) 32-bit multiplier down to a 24-bit multiplier.
    acc_56 = MultiplyByQuantizedMultiplier(acc_24x2, output_multiplier[ch] >> 8,
                                           output_shift[ch]);

    // Add output offset, cap activation, and assign to the output:
    acc_56 = AE_ADDQ56(acc_56, output_offset_56);
    acc_56 = AE_MINQ56S(acc_56, output_activation_max_56);
    acc_56 = AE_MAXQ56S(acc_56, output_activation_min_56);

    output_data[ch] = static_cast<int8_t>(AE_TRUNCA32Q48(acc_56));
  }
}

}  // namespace tflite
#endif  // defined(HIFIMINI)

#ifndef QANT_NATIVE_COMPUTING_TOOLKIT_AI_H
#define QANT_NATIVE_COMPUTING_TOOLKIT_AI_H

#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <stddef.h>
#include <stdint.h>

#include <dlpack/dlpack.h>

#ifdef __cplusplus
namespace qant_native_computing_toolkit::ai
{
    extern "C"
    {
#endif

        /**
         * @brief Element-wise addition. Supports batched input.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 2D tensor with data type bfloat16 and shape (n_batches, n_channels).
         * @param bias: A 1D tensor with data type bfloat16 and shape (n_channels)
         * @return A 2D tensor, same shape as inputs.
         */
        DLManagedTensorVersioned *add_bias_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                                 DLManagedTensorVersioned const *bias);

        /**
         * @brief Element-wise addition. Specifically used for conv2d.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param bias: A 1D tensor with data type bfloat16 and shape (n_channels)
         * @return A 4D tensor containing the result of the addition, same shape as the input features.
         */
        DLManagedTensorVersioned *add_bias_for_conv2d_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                                            DLManagedTensorVersioned const *bias);

        /**
         * @brief Performs a forward pass through a ReLU layer.
         *
         * For a mathematical definition, see
         * https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 1D tensor with data type bfloat16.
         * @return A 1D array containing the result of the ReLU operation.
         */
        DLManagedTensorVersioned *relu_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features);

        /**
         * @brief Performs a forward pass through a Sigmoid layer.
         *
         * For a mathematical definition, see
         * https://en.wikipedia.org/wiki/Sigmoid_function
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 1D tensor with data type bfloat16.
         * @return A 1D array containing the result of the Sigmoid operation with the same shape as the input.
         */
        DLManagedTensorVersioned *sigmoid_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features);

        /**
         * @brief Performs a forward pass through a Softmax layer.
         *
         * For a mathematical definition, see
         * https://en.wikipedia.org/wiki/Softmax_function
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 2D tensor with data type bfloat16 and shape (1, N).
         * @return A 2D tensor, containing the result of the Softmax operation with the same shape as the input.
         */
        DLManagedTensorVersioned *softmax_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features);

        /**
         * @brief Performs a forward pass through a convolution layer.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels_in, height, width)
         * @param kernels: A 4D tensor with data type bfloat16 and shape (n_channels_out,
         * n_channels_in, height, width)
         * @param padding: Amount of padding added to the input features.
         * @param stride: Step size for moving the filter window over the input
         * features.
         * @param dilation: Dilation ("zoom out") of the filter window.
         * @return A 4D tensor containing the result of the convolution with shape (n_batches, n_channels_out, height, width)
         */
        DLManagedTensorVersioned *conv_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                             DLManagedTensorVersioned const *kernels,
                                             size_t const padding,
                                             size_t const stride,
                                             size_t const dilation);

        /**
         * @brief Performs a forward pass through a transposed convolution layer.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels_in, height, width)
         * @param kernels: A 4D tensor with data type bfloat16 and shape (n_channels_in, n_channels_out, height, width)
         * @param padding: Amount of padding added to the input features.
         * @param stride: Step size for moving the filter window over the input
         * features.
         * @param dilation: Dilation ("zoom out") of the filter window.
         * @param output_padding: Padding for the returned features.
         * @return A 4D tensor containing the result of the transpose convolution with shape (n_batches, n_channels_out, height, width)
         */
        DLManagedTensorVersioned *conv_transpose_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                                       DLManagedTensorVersioned const *kernels,
                                                       size_t const padding,
                                                       size_t const stride,
                                                       size_t const dilation,
                                                       size_t const output_padding);

        /**
         * @brief Performs a forward pass through a batchnorm2d layer
         *
         * For a mathematical definition, see
         * https://en.wikipedia.org/wiki/Batch_normalization
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param means: A 1D tensor containing the mean values for each feature, with
         * data type bfloat16 and length n_channels
         * @param variances: A 1D tensor containing the variance values for each
         * feature, with data type bfloat16 and length n_channels
         * @param weights: A 1D tensor containing the parameter for each feature, with
         * data type bfloat16 and length n_channels
         * @param bias: A 1D tensor containing the bias value for each feature, with
         * data type bfloat16 and length n_channels
         * @param eps: A small value added to the variance for numerical stability.
         * @return A 4D tensor containing the result of the batchnorm operation with shape (n_batches, n_channels, height, width)
         */
        DLManagedTensorVersioned *batchnorm2d_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                                    DLManagedTensorVersioned const *means,
                                                    DLManagedTensorVersioned const *variances,
                                                    DLManagedTensorVersioned const *weights,
                                                    DLManagedTensorVersioned const *bias,
                                                    float const eps);

        /**
         * @brief Performs a forward pass through a maxpooling2d layer
         *
         * For a mathematical definition, see
         * https://en.wikipedia.org/wiki/Pooling_layer
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param kernel_height: Height of the pooling kernel.
         * @param kernel_width: Width of the pooling kernel.
         * @param padding: Amount of padding added to the input features.
         * @param stride: Step size for moving the filter window over the input
         * features.
         * @return A 4D tensor containing the result of the maxpool operation with shape
         * ([batches], n_channels, output_height, output_width).
         */
        DLManagedTensorVersioned *maxpool2d_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                                  size_t const kernel_height,
                                                  size_t const kernel_width,
                                                  size_t const padding,
                                                  size_t const stride);

        /**
         * @brief Performs a forward pass through a avgpooling2d layer
         *
         * For a mathematical definition, see
         * https://en.wikipedia.org/wiki/Pooling_layer
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param kernel_height: Height of the pooling kernel.
         * @param kernel_width: Width of the pooling kernel.
         * @param padding: Amount of padding added to the input features.
         * @param stride: Step size for moving the filter window over the input
         * features.
         * @param count_include_pad: When True, will include the zero-padding in the averaging calculation.
         * @return A 4D tensor with shape (n_batches, n_channels, height, width)
         */
        DLManagedTensorVersioned *avgpool2d_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                                  size_t const kernel_height,
                                                  size_t const kernel_width,
                                                  size_t const padding,
                                                  size_t const stride,
                                                  bool const count_include_pad);

        /**
         * @brief Performs a forward pass through an adaptive maxpooling2d layer.
         *
         * Only works for symmetric input and output sizes. Input size has to be integer multiple of output size.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param output_height: Height of the output.
         * @param output_width: Width of the output.
         * @return A 4D tensor with shape (n_batches, n_channels, output_height, output_width)
         */
        DLManagedTensorVersioned *adaptive_maxpool2d_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                                           size_t const output_height,
                                                           size_t const output_width);

        /**
         * @brief Performs a forward pass through an adaptive avgpooling2d layer.
         *
         * Only works for symmetric input and output sizes. Input size has to be integer multiple of output size.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param output_height: Height of the output.
         * @param output_width: Width of the output.
         * @return A 4D tensor with shape (n_batches, n_channels, output_height, output_width)
         */
        DLManagedTensorVersioned *adaptive_avgpool2d_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                                           size_t const output_height,
                                                           size_t const output_width);

        /**
         * @brief Calculates a Q.ANT version of a KAN layer (https://arxiv.org/abs/2404.19756) based on
         * @ref calc_scaled_periodic_nl_fprop "scaled_periodic_nl".
         *
         * The mathematical function is
         * @f[
         *     y_j = \sum_{i,l} \mathrm{tcos}\!\left(ks_l x_i + \phi_{jil}\right) ampls_{jil},
         * @f]
         * where @f$x_i@f$ is the input (vector), @f$ks_l@f$ is the frequency components,
         * @f$\phi_{jil}@f$ the phase offsets (tensor) and @f$ampls_{jil}@f$ the amplitude (tensor)
         * and @f$y_j@f$ output (vector). @f$\mathrm{tcos}@f$ denotes the cosine-related shape of the
         * periodic optical nonlinearity.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 2D input tensor with shape (n_batches, n_channels_in), with data type bfloat16.
         * @param phis: A 3D tensor of phase offsets, shape (n_channels_out, n_channels_in, n_ks), with data type bfloat16.
         * @param ampls: A 3D array of amplitudes, same shape as phis, with data type bfloat16.
         * @param ks: A 1D tensor of frequency components, shape (n_ks), with data type bfloat16.
         * @return The result of the KAN layer with shape (n_batches, n_channels_out), with data type bfloat16.
         */
        DLManagedTensorVersioned *calc_kan_layer_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                                       DLManagedTensorVersioned const *phis,
                                                       DLManagedTensorVersioned const *ampls,
                                                       DLManagedTensorVersioned const *ks);

#ifdef __cplusplus
    } // extern "C"
} // namespace qant_native_computing_toolkit::ai
#endif

#endif // QANT_NATIVE_COMPUTING_TOOLKIT_AI_H

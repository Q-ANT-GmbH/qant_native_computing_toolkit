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
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 2D tensor with data type bfloat16 and shape (n_batches, n_channels).
         * @param bias: A 1D tensor with data type bfloat16 and shape (n_channels)
         * @return A 2D tensor, same shape as inputs.
         */
        DLManagedTensorVersioned *add_bias_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
                                                 DLManagedTensorVersioned const *bias);

        /**
         * @brief Element-wise addition. Specifically used for conv2d.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param bias: A 1D tensor with data type bfloat16 and shape (n_channels)
         * @return A 4D tensor containing the result of the addition, same shape as the input features.
         */
        DLManagedTensorVersioned *add_bias_for_conv2d_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
                                                            DLManagedTensorVersioned const *bias);

        /**
         * @brief Performs a forward pass through a ReLU layer.
         *
         * For a mathematical definition, see
         * https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 1D tensor with data type bfloat16.
         * @return A 1D array containing the result of the ReLU operation.
         */
        DLManagedTensorVersioned *relu_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features);

        /**
         * @brief Performs a forward pass through a Sigmoid layer.
         *
         * For a mathematical definition, see
         * https://en.wikipedia.org/wiki/Sigmoid_function
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 1D tensor with data type bfloat16.
         * @return A 1D array containing the result of the Sigmoid operation with the same shape as the input.
         */
        DLManagedTensorVersioned *sigmoid_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features);

        /**
         * @brief Performs a forward pass through a Softmax layer.
         *
         * For a mathematical definition, see
         * https://en.wikipedia.org/wiki/Softmax_function
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 2D tensor with data type bfloat16 and shape (1, N).
         * @return A 2D tensor, containing the result of the Softmax operation with the same shape as the input.
         */
        DLManagedTensorVersioned *softmax_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features);

        /**
         * @brief Performs a forward pass through a convolution layer.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * in_channels, height, width)
         * @param weights: A 4D tensor with data type bfloat16 and shape (out_channels,
         * in_channels, height, width)
         * @param padding: Amount of padding added to the input features.
         * @param stride: Step size for moving the kernel over the input
         * features.
         * @param dilation: Dilation ("zoom out") of the kernel.
         * @return A 4D tensor containing the result of the convolution with shape (n_batches, out_channels, height_out, width_out)
         */
        DLManagedTensorVersioned *conv_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
                                             DLManagedTensorVersioned const *weights,
                                             size_t const padding,
                                             size_t const stride,
                                             size_t const dilation);

        /**
         * @brief Performs a forward pass through a transposed convolution layer.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * in_channels, height, width)
         * @param weights: A 4D tensor with data type bfloat16 and shape (in_channels, out_channels, height, width)
         * @param padding: Amount of padding added to the input features.
         * @param stride: Step size for moving the kernel over the input
         * features.
         * @param dilation: Dilation ("zoom out") of the kernel.
         * @param output_padding: Padding for the returned features.
         * @return A 4D tensor containing the result of the transpose convolution with shape (n_batches, out_channels, height_out, width_out)
         */
        DLManagedTensorVersioned *conv_transpose_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
                                                       DLManagedTensorVersioned const *weights,
                                                       size_t const padding,
                                                       size_t const stride,
                                                       size_t const dilation,
                                                       size_t const output_padding);

        /**
         * @brief Opaque handle to convolution weights that have been uploaded to device memory.
         *
         * Instances are created by upload_conv_weights() and must be released exactly once
         * with conv_weights_free(). Using a handle after it has been freed is undefined behavior.
         */
        typedef struct ConvWeightsHandle ConvWeightsHandle;

        /**
         * @brief Uploads convolution weights to device memory so they can be reused across
         * multiple calls to conv_fprop_preloaded_weights().
         *
         * @param npu_id The identifier of the NPU on which the weights are stored.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param weights: A 4D tensor with data type bfloat16 and shape (out_channels,
         * in_channels, height, width)
         * @return An owning pointer to the weights handle, or NULL on error. The caller must
         * release it with conv_weights_free().
         */
        ConvWeightsHandle *upload_conv_weights(const int32_t npu_id, DLManagedTensorVersioned const *weights);

        /**
         * @brief Frees convolution weights previously uploaded with upload_conv_weights().
         *
         * @param handle The handle to free. Must not be used afterwards.
         * @return 0 if successful, or a negative error code.
         */
        int conv_weights_free(ConvWeightsHandle *handle);

        /**
         * @brief Performs a forward pass through a convolution layer using weights already
         * uploaded to device memory. The computation runs on the NPU that holds the weights.
         *
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * in_channels, height, width)
         * @param weights: Handle to the weights, as returned by upload_conv_weights().
         * @param padding: Amount of padding added to the input features.
         * @param stride: Step size for moving the kernel over the input
         * features.
         * @param dilation: Dilation ("zoom out") of the kernel.
         * @return A 4D tensor containing the result of the convolution with shape (n_batches, out_channels, height_out, width_out)
         */
        DLManagedTensorVersioned *conv_fprop_preloaded_weights(DLManagedTensorVersioned const *features,
                                                               ConvWeightsHandle const *weights,
                                                               size_t const padding,
                                                               size_t const stride,
                                                               size_t const dilation);

        /**
         * @brief Opaque handle to transposed convolution weights that have been uploaded to
         * device memory.
         *
         * Instances are created by upload_conv_transpose_weights() and must be released exactly
         * once with conv_transpose_weights_free(). Using a handle after it has been freed is
         * undefined behavior.
         */
        typedef struct ConvTransposeWeightsHandle ConvTransposeWeightsHandle;

        /**
         * @brief Uploads transposed convolution weights to device memory so they can be reused
         * across multiple calls to conv_transpose_fprop_preloaded_weights().
         *
         * @param npu_id The identifier of the NPU on which the weights are stored.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param weights: A 4D tensor with data type bfloat16 and shape (in_channels, out_channels, height, width)
         * @return An owning pointer to the weights handle, or NULL on error. The caller must
         * release it with conv_transpose_weights_free().
         */
        ConvTransposeWeightsHandle *upload_conv_transpose_weights(const int32_t npu_id,
                                                                  DLManagedTensorVersioned const *weights);

        /**
         * @brief Frees transposed convolution weights previously uploaded with
         * upload_conv_transpose_weights().
         *
         * @param handle The handle to free. Must not be used afterwards.
         * @return 0 if successful, or a negative error code.
         */
        int conv_transpose_weights_free(ConvTransposeWeightsHandle *handle);

        /**
         * @brief Performs a forward pass through a transposed convolution layer using weights
         * already uploaded to device memory. The computation runs on the NPU that holds the weights.
         *
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * in_channels, height, width)
         * @param weights: Handle to the weights, as returned by upload_conv_transpose_weights().
         * @param padding: Amount of padding added to the input features.
         * @param stride: Step size for moving the kernel over the input
         * features.
         * @param dilation: Dilation ("zoom out") of the kernel.
         * @param output_padding: Padding for the returned features.
         * @return A 4D tensor containing the result of the transpose convolution with shape (n_batches, out_channels, height_out, width_out)
         */
        DLManagedTensorVersioned *conv_transpose_fprop_preloaded_weights(DLManagedTensorVersioned const *features,
                                                                         ConvTransposeWeightsHandle const *weights,
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
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
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
        DLManagedTensorVersioned *batchnorm2d_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
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
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param kernel_height: Height of the pooling kernel.
         * @param kernel_width: Width of the pooling kernel.
         * @param padding: Amount of padding added to the input features.
         * @param stride: Step size for moving the kernel over the input
         * features.
         * @return A 4D tensor containing the result of the maxpool operation with shape
         * ([batches], n_channels, output_height, output_width).
         */
        DLManagedTensorVersioned *maxpool2d_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
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
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param kernel_height: Height of the pooling kernel.
         * @param kernel_width: Width of the pooling kernel.
         * @param padding: Amount of padding added to the input features.
         * @param stride: Step size for moving the kernel over the input
         * features.
         * @param count_include_pad: When True, will include the zero-padding in the averaging calculation.
         * @return A 4D tensor with shape (n_batches, n_channels, height, width)
         */
        DLManagedTensorVersioned *avgpool2d_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
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
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param output_height: Height of the output.
         * @param output_width: Width of the output.
         * @return A 4D tensor with shape (n_batches, n_channels, output_height, output_width)
         */
        DLManagedTensorVersioned *adaptive_maxpool2d_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
                                                           size_t const output_height,
                                                           size_t const output_width);

        /**
         * @brief Performs a forward pass through an adaptive avgpooling2d layer.
         *
         * Only works for symmetric input and output sizes. Input size has to be integer multiple of output size.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 4D tensor with data type bfloat16 and shape (n_batches,
         * n_channels, height, width)
         * @param output_height: Height of the output.
         * @param output_width: Width of the output.
         * @return A 4D tensor with shape (n_batches, n_channels, output_height, output_width)
         */
        DLManagedTensorVersioned *adaptive_avgpool2d_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
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
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 2D input tensor with shape (n_batches, n_channels_in), with data type bfloat16.
         * @param phis: A 3D tensor of phase offsets, shape (n_channels_out, n_channels_in, n_ks), with data type bfloat16.
         * @param ampls: A 3D array of amplitudes, same shape as phis, with data type bfloat16.
         * @param ks: A 1D tensor of frequency components, shape (n_ks), with data type bfloat16.
         * @return The result of the KAN layer with shape (n_batches, n_channels_out), with data type bfloat16.
         */
        DLManagedTensorVersioned *calc_kan_layer_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
                                                       DLManagedTensorVersioned const *phis,
                                                       DLManagedTensorVersioned const *ampls,
                                                       DLManagedTensorVersioned const *ks);

#ifdef __cplusplus
    } // extern "C"
} // namespace qant_native_computing_toolkit::ai
#endif

#endif // QANT_NATIVE_COMPUTING_TOOLKIT_AI_H

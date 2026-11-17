#ifndef QANT_NATIVE_COMPUTING_TOOLKIT_NATIVE_H
#define QANT_NATIVE_COMPUTING_TOOLKIT_NATIVE_H

/**
 * @file
 * @brief Native math functionality.
 *
 * Functionality that is available directly on the NPU with no data processing on the host.
 */

#include <stdint.h>

#include <dlpack/dlpack.h>

#include "qant_native_computing_toolkit_generic.h"

#ifdef __cplusplus
namespace qant_native_computing_toolkit::native
{
    using ::qant_native_computing_toolkit::generic::QantMemoryHandle;

    extern "C"
    {
#endif

        /**
         * @brief Multiplies two 1D Tensors element-wise.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param us The first input tensor. Must be data type bfloat16, same
         * length as vs.
         * @param vs The second input tensor. Must be data type bfloat16, same
         * length as us.
         * @return A 1D tensor containing the elementwise product, same length as
         * input.
         */
        DLManagedTensorVersioned *mul_npu(const int32_t npu_id, DLManagedTensorVersioned const *us,
                                          DLManagedTensorVersioned const *vs);

        /**
         * @brief Calculates a scaled periodic nonlinearity pairwise for all elements of features @f$u@f$ and weights @f$v@f$.
         *
         * The output for each input pair is @f$w(u,v) = \mathrm{tcos}(u) \cdot v@f$, where @f$\mathrm{tcos}()@f$ is a @f$2\pi@f$ - periodic function
         * with values between @f$-1@f$ and @f$1@f$, similar to a cosine.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features: A 1D tensor with data type bfloat16.
         * @param weights: A 1D tensor with data type bfloat16, same length as features.
         * @return A 1D tensor containing the result, same shape as input features.
         */
        DLManagedTensorVersioned *calc_scaled_periodic_nl_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features, DLManagedTensorVersioned const *weights);

        /**
         * @brief Performs a linear forward propagation between input features and a
         * weight matrix.
         *
         * The operation computes <tt>output = features @ weights^T</tt>, i.e.:
         * @f[
         *   \mathrm{output}[i,\,j] = \sum_k \mathrm{features}[i,k]\cdot\mathrm{weights}[j,k]
         * @f]
         * Note that this is **not** a plain \c features \c @ \c weights — the weight
         * matrix is implicitly transposed, matching the convention where \p weights has
         * shape <tt>(n_channels_out, n_channels_in)</tt>.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param features A 2D tensor with data type bfloat16 and shape
         *        <tt>(n_batches, n_channels_in)</tt>.
         * @param weights A 2D tensor with data type bfloat16 and shape
         *        <tt>(n_channels_out, n_channels_in)</tt>.
         * @return A 2D tensor containing the result of the forward propagation with
         *         shape <tt>(n_batches, n_channels_out)</tt>.
         */
        DLManagedTensorVersioned *linear_fprop(const int32_t npu_id, DLManagedTensorVersioned const *features,
                                               DLManagedTensorVersioned const *weights);

        /**
         * @brief Same as linear_fprop(), but uses a weight matrix previously uploaded.
         *
         * Use in combination with npu_upload() or npu_upload_into() (data role Weights).
         * The operation is performed on the NPU the weights were uploaded to.
         *
         * @param features A 2D tensor with data type bfloat16 and shape
         *        <tt>(n_batches, n_channels_in)</tt>.
         * @param weights Handle to the weights with shape
         *        <tt>(n_channels_out, n_channels_in)</tt>, as returned by npu_upload().
         * @return A 2D tensor containing the result of the forward propagation with
         *         shape <tt>(n_batches, n_channels_out)</tt>.
         */
        DLManagedTensorVersioned *linear_fprop_preloaded_weights(DLManagedTensorVersioned const *features,
                                                                 QantMemoryHandle const *weights);

        /**
         * @brief Same as linear_fprop(), but features, weights and result all live in device
         * memory.
         *
         * The features and weights must have been uploaded with npu_upload() or
         * npu_upload_into() (data roles Features and Weights respectively). The result
         * handle is not allocated by this function: it must be allocated beforehand, e.g.
         * with npu_allocate(), on the same NPU and with data role Features.
         * The operation is performed on the NPU the weights were uploaded to.
         *
         * @param features Handle to the features with shape
         *        <tt>(n_batches, n_channels_in)</tt>.
         * @param weights Handle to the weights with shape
         *        <tt>(n_channels_out, n_channels_in)</tt>.
         * @param result Pre-allocated handle to receive the result with shape
         *        <tt>(n_batches, n_channels_out)</tt> and data role Features. Its previous
         *        contents are overwritten.
         * @return 0 if successful, or a negative error code.
         */
        int linear_fprop_preloaded(QantMemoryHandle const *features, QantMemoryHandle const *weights,
                                   QantMemoryHandle const *result);

        /**
         * @brief Performs a batched Discrete Fourier Transform (DFT) on complex input data.
         *
         * Each row along the batch dimension is transformed independently via a
         * forward FFT, complex-to-complex.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param data A 2D tensor of shape <tt>(n_dfts, dft_size)</tt> with data type
         *        bcomplex32 (<tt>kDLBcomplex</tt>, 32 bits, 1 lane): each element is a
         *        complex number stored as two bfloat16 values (real, imaginary).
         *        <tt>kDLBcomplex</tt> requires DLPack >= 1.4 headers.
         * @return A 2D tensor with the same shape and data type as \p data,
         *         containing the DFT result for each row.
         */
        DLManagedTensorVersioned *dft(const int32_t npu_id, DLManagedTensorVersioned const *data);

#ifdef __cplusplus
    } // extern "C"
} // namespace qant_native_computing_toolkit::native
#endif

#endif // QANT_NATIVE_COMPUTING_TOOLKIT_NATIVE_H

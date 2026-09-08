#ifndef QANT_NATIVE_COMPUTING_TOOLKIT_NATIVE_H
#define QANT_NATIVE_COMPUTING_TOOLKIT_NATIVE_H

#include <stdint.h>

#include <dlpack/dlpack.h>

#ifdef __cplusplus
namespace qant_native_computing_toolkit::native
{
    extern "C"
    {
#endif

        /**
         * @brief Multiplies two 1D Tensors element-wise.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param us The first input tensor. Must be data type bfloat16, same
         * length as vs.
         * @param vs The second input tensor. Must be data type bfloat16, same
         * length as us.
         * @return A 1D tensor containing the elementwise product, same length as
         * input.
         */
        DLManagedTensorVersioned *mul_npu(const uint32_t npu_id, DLManagedTensorVersioned const *us,
                                          DLManagedTensorVersioned const *vs);

        /**
         * @brief Calculates a scaled periodic nonlinearity pairwise for all elements of features @f$u@f$ and weights @f$v@f$.
         *
         * The output for each input pair is @f$w(u,v) = \mathrm{tcos}(u) \cdot v@f$, where @f$\mathrm{tcos}()@f$ is a @f$2\pi@f$ - periodic function
         * with values between @f$-1@f$ and @f$1@f$, similar to a cosine.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param features: A 1D tensor with data type bfloat16.
         * @param weights: A 1D tensor with data type bfloat16, same length as features.
         * @return A 1D tensor containing the result, same shape as input features.
         */
        DLManagedTensorVersioned *calc_scaled_periodic_nl_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features, DLManagedTensorVersioned const *weights);

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
         * @param features A 2D tensor with data type bfloat16 and shape
         *        <tt>(n_batches, n_channels_in)</tt>.
         * @param weights A 2D tensor with data type bfloat16 and shape
         *        <tt>(n_channels_out, n_channels_in)</tt>.
         * @return A 2D tensor containing the result of the forward propagation with
         *         shape <tt>(n_batches, n_channels_out)</tt>.
         */
        DLManagedTensorVersioned *linear_fprop(const uint32_t npu_id, DLManagedTensorVersioned const *features,
                                               DLManagedTensorVersioned const *weights);

        /**
         * @brief Multiplies two 1D Tensors of float32 element-wise.
         *
         * @deprecated Use mul_npu() with bfloat16 tensors instead.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param us The first input tensor. Must be 1D with data type float32, same
         * length as vs.
         * @param vs The second input tensor. Must be 1D with data type float32, same
         * length as us.
         * @return A 1D tensor containing the elementwise product, same length as
         * input.
         */
        DLManagedTensorVersioned *mul_npu_f32(const uint32_t npu_id, DLManagedTensorVersioned const *us,
                                              DLManagedTensorVersioned const *vs);

        /**
         * @brief Multiplies two 1D Tensors of int16 element-wise.
         *
         * @deprecated Use mul_npu() with bfloat16 tensors instead.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * @param us The first input tensor. Must be data type int16, same
         * length as vs.
         * @param vs The second input tensor. Must be data type int16, same
         * length as us.
         * @return A 1D tensor containing the elementwise product, same length as
         * input.
         */
        DLManagedTensorVersioned *mul_npu_i16(const uint32_t npu_id, DLManagedTensorVersioned const *us,
                                              DLManagedTensorVersioned const *vs);

#ifdef __cplusplus
    } // extern "C"
} // namespace qant_native_computing_toolkit::native
#endif

#endif // QANT_NATIVE_COMPUTING_TOOLKIT_NATIVE_H

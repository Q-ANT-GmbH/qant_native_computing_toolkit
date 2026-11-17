#ifndef QANT_NATIVE_COMPUTING_TOOLKIT_GENERIC_H
#define QANT_NATIVE_COMPUTING_TOOLKIT_GENERIC_H

#include <stddef.h>
#include <stdint.h>

#include <dlpack/dlpack.h>

// Sentinel npu_id for automatic NPU selection: the first enumerated NPU that
// supports the requested operation. Every npu_id parameter throughout this
// toolkit accepts this value. Passing an npu_id below this value is invalid
// and results in an error.
#define QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1)

// String-like parameters and return values are represented as arrays of char with this length.
#define QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH 1024

#ifdef __cplusplus
namespace qant_native_computing_toolkit::generic
{
    extern "C"
    {
#endif

        /**
         * @brief Initialises the Native Processing Unit.
         *
         * Optional, not necessary for computation.
         *
         * NPUs are automatically initialised on the first operation that uses the specific device.
         * Within one process, concurrent operation calls on a single NPU are possible,
         * this creates a first-come first-serve queue.
         * A process has an exclusive lock on each NPU it initialised, NPUs can be released explicitly
         * to free devices for use by a different process, see `release_npu`.
         *
         * @param id The identifier of the NPU to be initialised.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first
         * available NPU. Values below -1 are invalid.
         *
         * @return 0 if successful, -1 otherwise. The error message is printed to stderr.
         */
        int init_npu(const int32_t id);

        /**
         * @brief Releases an NPU and frees associated memory.
         *
         * Only relevant if multiple processes want to access the same device.
         *
         * @param id The identifier of the NPU to be released.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first
         * available NPU. Values below -1 are invalid.
         *
         * @return 0 if successful, -1 otherwise. The error message is printed to stderr.
         */
        int release_npu(const int32_t id);

        /**
         * @brief An operation an NPU can support, used to filter the available NPUs.
         */
        typedef enum QantDeviceCapability
        {
            Mul = 0,         // elementwise and matrix multiplication
            NonLin = 1,      // non-linear functions
            Dft = 2,         // discrete Fourier transform
            MemHandling = 3, // allocating/uploading/downloading NPU memory
        } QantDeviceCapability;

        /**
         * @brief Get the available NPUs that support all of the given operations.
         *
         * The NPUs are written sorted by id.
         *
         * @param operations: Array of the operations a listed NPU must support, length: n_operations.
         * May be NULL if n_operations is 0, in which case all available NPUs are listed.
         * @param n_operations: Length of operations.
         * @param idxs_out: Array to place the ids of the matching NPUs, length: capacity.
         * @param serials_out: Array to place the NPU serial numbers, length: capacity.
         * @param capacity: Length of idxs_out, serials_out. Should be chosen at least as large as the expected number of matching NPUs.
         * @param out_count: The number of matching NPUs.
         *
         * @return 0 if successful, -1 otherwise. The error message is printed to stderr.
         */
        int get_available_npus(
            const QantDeviceCapability *operations,
            size_t n_operations,
            uint32_t *idxs_out,
            char (*serials_out)[QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH],
            size_t capacity,
            size_t *out_count);

        /**
         * @brief Opaque handle to a block of device memory sized to hold a matrix.
         *
         * Instances are created by npu_allocate() and must be released exactly once
         * with npu_free(). Using a handle after it has been freed is undefined behavior.
         */
        typedef struct QantMemoryHandle QantMemoryHandle;

        /**
         * @brief Role of a block of device memory.
         */
        typedef enum QantDataRole
        {
            Weights,
            Features,
        } QantDataRole;

        /**
         * @brief Allocates a block of device memory sized to hold a (rows, cols) matrix.
         *
         * @param npu_id The identifier of the NPU on which the memory is allocated.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports memory handling. Values below -1 are invalid.
         * @param rows Number of rows of the matrix the memory is sized for.
         * @param cols Number of columns of the matrix the memory is sized for.
         * @param data_role Role of the memory block (Weights or Features).
         * @return An owning pointer to the allocated handle, or NULL on error. The
         * caller must release it with npu_free().
         */
        QantMemoryHandle *npu_allocate(const int32_t npu_id, const size_t rows,
                                       const size_t cols, const QantDataRole data_role);

        /**
         * @brief Frees a previously allocated block of device memory.
         *
         * @param handle The handle to free, as returned by npu_allocate(). Must not be
         * used again after this call.
         * @return 0 if successful, or a negative error code.
         */
        int npu_free(QantMemoryHandle *handle);

        /**
         * @brief Allocates a block of device memory and uploads a matrix to it.
         *
         * @param npu_id The identifier of the NPU on which the memory is allocated.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports memory handling. Values below -1 are invalid.
         * @param data A 2D tensor with shape (rows, cols) and data type bfloat16.
         * @param data_role Role of the memory block (Weights or Features).
         * @return An owning pointer to the allocated handle, or NULL on error. The
         * caller must release it with npu_free().
         */
        QantMemoryHandle *npu_upload(const int32_t npu_id, DLManagedTensorVersioned const *data,
                                     const QantDataRole data_role);

        /**
         * @brief Uploads a matrix to a previously allocated block of device memory.
         *
         * @param handle The handle to upload to, as returned by npu_allocate() or npu_upload().
         * @param data A 2D tensor with shape (rows, cols) matching the handle. Must be data type bfloat16.
         * @return 0 if successful, or a negative error code.
         */
        int npu_upload_into(QantMemoryHandle const *handle, DLManagedTensorVersioned const *data);

        /**
         * @brief Downloads the matrix currently stored in a block of device memory.
         *
         * @param handle The handle to download from, as returned by npu_allocate().
         * @return A 2D tensor with shape (rows, cols) and data type bfloat16, or NULL on error.
         */
        DLManagedTensorVersioned *npu_download(QantMemoryHandle const *handle);

#ifdef __cplusplus
    } // extern "C"
} // namespace qant_native_computing_toolkit::generic
#endif

#endif // QANT_NATIVE_COMPUTING_TOOLKIT_GENERIC_H

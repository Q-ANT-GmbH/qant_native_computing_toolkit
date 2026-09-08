#ifndef QANT_NATIVE_COMPUTING_TOOLKIT_GENERIC_H
#define QANT_NATIVE_COMPUTING_TOOLKIT_GENERIC_H

#include <stddef.h>
#include <stdint.h>

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
         * @param id The identifier of the NPU to be initialised.
         *
         * @return 0 if successful, or a negative error code.
         */
        int init_npu(const uint32_t id);

        /**
         * @brief Releases an NPU and frees associated memory.
         *
         * Only relevant if multiple users want to access the same device.
         *
         * @param id The identifier of the NPU to be released.
         *
         * @return 0 if successful, or a negative error code.
         */
        int release_npu(const uint32_t id);

#ifdef __cplusplus
    } // extern "C"
} // namespace qant_native_computing_toolkit::generic
#endif

#endif // QANT_NATIVE_COMPUTING_TOOLKIT_GENERIC_H

#ifndef QANT_NATIVE_COMPUTING_TOOLKIT_INFO_H
#define QANT_NATIVE_COMPUTING_TOOLKIT_INFO_H

#define QANT_NATIVE_COMPUTING_TOOLKIT_MAJOR_VERSION 2
#define QANT_NATIVE_COMPUTING_TOOLKIT_MINOR_VERSION 5

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qant_native_computing_toolkit_generic.h"

#ifdef __cplusplus
namespace qant_native_computing_toolkit::info
{
    using ::qant_native_computing_toolkit::generic::QantDeviceCapability;

    extern "C"
    {
#endif

        /**
         * @brief Enables logging.
         *
         * Log messages are written to stdout and to a file
         *
         * @param folder_name The folder where the log files are created. Individual
         * logfiles are created with a timestamp.
         * @param loglevel The logging level, from 0 = tracing to 4 = error only.
         * @return 0 if successful, error code otherwise
         */
        int setup_logging(char const *folder_name, int const loglevel);

        /**
         * @brief Gets detailed information about the Q.ANT native computing driver and its direct dependencies.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param output A preallocated memory to store the info string into.
         * @param len The maximum length of the string.
         * @return 0 if successful, -1 otherwise. The error message is printed to stderr.
         */
        int get_driver_info(const int32_t npu_id, char *output, size_t len);

        typedef struct
        {
            double temp_pd1;
            double temp_pd2;
            double temp_pd3;
            double temp_dac_1;
            double temp_dac_2;
            double voltage_1v8;
            double voltage_3v3;
            double voltage_6v0_1;
            double voltage_6v0_2;
            double voltage_6v2;
            double voltage_6v5;
            double voltage_8v0_1;
            double voltage_8v0_2;
            double voltage_12v;
            double voltage_12v5;
            double voltage_n12v5;
            double current_12v;
        } QantSensorInfo;

        /**
         * @brief Get detailed information about the NPU sensors, e.g. temperature.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param out A preallocated struct to store the sensor information into.
         * Left unchanged on failure.
         *
         * @return 0 if successful, -1 otherwise. The error message is printed to stderr.
         */
        int get_sensor_info(const int32_t npu_id, QantSensorInfo *out);

        typedef struct
        {
            char fw_version[QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH];
            char zephyr_version[QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH];
            char board_serial_no[QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH];
        } QantVersionInfo;

        /**
         * @brief Gets information about the firmware versions.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         * @param out A preallocated struct to store the version information into.
         * Left unchanged on failure.
         *
         * @return 0 if successful, -1 otherwise. The error message is printed to stderr.
         */
        int get_version_info(const int32_t npu_id, QantVersionInfo *out);

        /**
         * @brief Gets the operations supported by an NPU.
         *
         * @param npu_id The identifier of the NPU, e.g. one returned by get_available_npus.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first enumerated
         * NPU. Values below -1 are invalid.
         * @param capabilities_out: Array to place the supported operations, length: capacity.
         * At most 4 (the number of QantDeviceCapability values) are written.
         * @param capacity: Length of capabilities_out.
         * @param out_count: The number of supported operations. Also set if capacity is too small.
         *
         * @return 0 if successful, -1 otherwise. The error message is printed to stderr.
         */
        int get_capabilities(
            const int32_t npu_id,
            QantDeviceCapability *capabilities_out,
            size_t capacity,
            size_t *out_count);

        /**
         * @brief Reset the performance counters
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         *
         * @return 0 if successful, -1 otherwise. The error message is printed to stderr.
         */
        int reset_perf_counter(const int32_t npu_id);

        /**
         * @brief Struct to hold performance counter information
         */
        typedef struct
        {
            double timebased_counter; // Time since the last reset (in s)
            double
                stalling_counter; // Time since the last reset spent in idle mode (in s)
        } QantPerformanceCounterInfo;

        /**
         * @brief Gets the performance counters.
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         * QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID (-1) selects the first NPU that
         * supports this operation. Values below -1 are invalid.
         *
         * @param out A preallocated memory to store the counter info.
         * @return 0 if successful, -1 otherwise. The error message is printed to stderr.
         */
        int get_perf_counter(const int32_t npu_id, QantPerformanceCounterInfo *out);

#ifdef __cplusplus
    } // extern "C"
} // namespace qant_native_computing_toolkit::info
#endif

#endif // QANT_NATIVE_COMPUTING_TOOLKIT_INFO_H

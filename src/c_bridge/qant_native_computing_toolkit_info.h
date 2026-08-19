#ifndef QANT_NATIVE_COMPUTING_TOOLKIT_INFO_H
#define QANT_NATIVE_COMPUTING_TOOLKIT_INFO_H

#define QANT_NATIVE_COMPUTING_TOOLKIT_MAJOR_VERSION 2
#define QANT_NATIVE_COMPUTING_TOOLKIT_MINOR_VERSION 3

// String-like parameters and return values are represented as arrays of char with this length.
#define QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH 1024

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
namespace qant_native_computing_toolkit::info
{
    extern "C"
    {
#endif

        /**
         * @brief Enable logging. Log messages are written to stdout and to a file
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
         * @param output A preallocated memory to store the info string into.
         * @param len The maximum length of the string.
         * @return 0 if successful, negative error code otherwise.
         */
        int get_driver_info(const uint32_t npu_id, char *output, size_t len);

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
         *
         * @return The sensor information as a struct.
         */
        QantSensorInfo get_sensor_info(const uint32_t npu_id);

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
         *
         * @return The version information as a struct.
         */
        QantVersionInfo get_version_info(const uint32_t npu_id);

        /**
         * @brief Get all available NPUs
         *
         * @param idxs_out: Array to place the (pcie-enumeration) idxs of the available NPUs, length: capacity.
         * @param serials_out: Array to place the NPU serial numbers, length: capacity.
         * @param capacity: Length of idxs_out, serials_out. Should be chosen at least as large as the expected number of available NPUs.
         * @param out_count: The actual number of available NPUs.
         *
         * @return 0 if successful, and a negative error code in case of failure.
         */
        int get_available_npus(
            uint32_t *idxs_out,
            char (*serials_out)[QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH],
            size_t capacity,
            size_t *out_count);

        /**
         * @brief Reset the performance counters
         *
         * @param npu_id The identifier of the NPU on which to perform the operation.
         *
         * @return 0 if successful, and a negative error code in case of failure.
         */
        int reset_perf_counter(const uint32_t npu_id);

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
         *
         * @param out A preallocated memory to store the counter info.
         * @return 0 if successful, a non-zero code otherwise.
         */
        int get_perf_counter(const uint32_t npu_id, QantPerformanceCounterInfo *out);

#ifdef __cplusplus
    } // extern "C"
} // namespace qant_native_computing_toolkit::info
#endif

#endif // QANT_NATIVE_COMPUTING_TOOLKIT_INFO_H

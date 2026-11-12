pub mod bias;
mod c_bridge;
pub mod conv;
pub mod data_structures;
pub mod elementwise;
pub mod errors;
mod global_settings;
pub mod linear;
pub mod memory_management;
pub mod non_linear;
pub mod norm;
pub mod pooling;
#[cfg(feature = "python-bridge")]
mod python_bridge;
mod qant_driver_import;
pub(crate) mod utils;

pub use qant_driver_import::get_available_npus;
pub use qant_driver_import::get_device_info;
pub use qant_driver_import::get_perf_counter;
pub use qant_driver_import::get_version as get_driver_info;
pub use qant_driver_import::init_npu;
pub use qant_driver_import::release_npu;
pub use qant_driver_import::reset_perf_counter;
pub use qant_driver_import::setup_logging;

#[cfg(test)]
mod tests {
    use crate::qant_driver_import;

    #[test]
    fn init_release() {
        qant_driver_import::init_npu(0).unwrap();
        qant_driver_import::release_npu(0).unwrap();
    }

    #[test]
    fn driver_info() {
        let info = qant_driver_import::get_version(0).unwrap();
        println!("{info}");
        assert!(info.contains("commit"));
    }

    #[test]
    fn device_info() {
        let dev_info = qant_driver_import::get_device_info(0).unwrap();
        println!("{dev_info:?}");
        assert!(dev_info.sensor.voltage_12v >= 10.)
    }

    #[test]
    fn available_npus() {
        let available_npus = qant_driver_import::get_available_npus().unwrap();
        println!("{available_npus:?}");
        assert!(available_npus.keys().len() > 0)
    }

    #[test]
    fn perf_counters() {
        qant_driver_import::reset_perf_counter(0).unwrap();
        qant_driver_import::get_perf_counter(0).unwrap();
    }
}

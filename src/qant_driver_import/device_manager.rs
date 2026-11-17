//! Registry of the available NPUs and their capabilities, and the resolution
//! of public `npu_id`s to concrete NPUs.
use std::sync::LazyLock;

use super::{AUTO_NPU_ID, NPUBackend};
use crate::{errors::ToolkitError, toolkit_error};

/// An operation an NPU can support, used to filter the available NPUs.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "python-bridge",
    pyo3::pyclass(eq, frozen, hash, from_py_object)
)]
pub enum QantDeviceCapability {
    Mul = 0,
    NonLin = 1,
    Dft = 2,
    MemHandling = 3,
}

impl QantDeviceCapability {
    pub(crate) const ALL: [Self; 4] = [Self::Mul, Self::NonLin, Self::Dft, Self::MemHandling];
}

#[allow(clippy::upper_case_acronyms)]
pub(super) struct NPU {
    pub(super) id: u32,
    pub(super) backend: NPUBackend,
    serial: String,
    capabilities: Vec<QantDeviceCapability>,
}

impl NPU {
    /// Whether the NPU supports all of `ops`.
    pub(super) fn supports(&self, ops: &[QantDeviceCapability]) -> bool {
        ops.iter().all(|op| self.capabilities.contains(op))
    }
}

// The NPUs, sorted by id. Enumerated once on first use and never modified.
// A plain Vec is used instead of a HashMap: the registry only ever holds a
// handful of devices, and hashbrown's internal layout stores an interior
// (non-start-of-allocation) pointer for its bucket array, which makes
// valgrind's leak checker report this process-lifetime singleton as
// "possibly lost" instead of "still reachable" (a well known false positive,
// see https://github.com/rust-lang/hashbrown/issues/280).
pub(super) static DEVICES: LazyLock<Vec<NPU>> = LazyLock::new(enumerate_npus);

// FIXME: open for a better solution here!
#[allow(clippy::disallowed_macros)]
fn enumerate_npus() -> Vec<NPU> {
    let mut devices = Vec::new();
    for backend in [NPUBackend::Gen2, NPUBackend::Gen2p] {
        let npus = match backend.get_available_npus() {
            Ok(npus) => npus,
            Err(e) => {
                eprintln!("{backend:?} NPU enumeration failed: {e}");
                Vec::new()
            }
        };
        for (id, serial) in npus {
            match backend.get_capabilities(id) {
                Ok(capabilities) => devices.push(NPU {
                    id,
                    backend,
                    serial,
                    capabilities,
                }),
                Err(e) => eprintln!("{backend:?} NPU{id}: failed to get capabilities: {e}"),
            }
        }
    }
    devices.sort_by_key(|npu| npu.id);
    devices
}

/// Resolves a public, signed `npu_id` to a registered NPU.
///
/// `npu_id == AUTO_NPU_ID` (-1) resolves to the lowest-id NPU for which
/// `pred` holds; a non-negative id is looked up directly, regardless of
/// `pred` (unsupported operations are still rejected by the backend
/// dispatch). Errors for `npu_id < AUTO_NPU_ID` or if no NPU matches.
pub(super) fn resolve_npu(
    npu_id: i32,
    pred: impl Fn(&NPU) -> bool,
) -> Result<&'static NPU, ToolkitError> {
    match npu_id {
        AUTO_NPU_ID => DEVICES
            .iter()
            .find(|npu| pred(npu))
            .ok_or_else(|| toolkit_error!("No NPU available that supports this operation")),
        id if id >= 0 => npu_by_id(id as u32),
        _ => Err(toolkit_error!(
            "invalid npu_id {npu_id}: must be {AUTO_NPU_ID} (auto-select) or a non-negative NPU id"
        )),
    }
}

/// [`resolve_npu`] for operations every NPU supports: [`AUTO_NPU_ID`]
/// resolves to the lowest-id NPU.
pub(super) fn resolve_any(npu_id: i32) -> Result<&'static NPU, ToolkitError> {
    resolve_npu(npu_id, |_| true)
}

/// Looks up a registered NPU by its id, e.g. the one stored in a handle.
pub(super) fn npu_by_id(id: u32) -> Result<&'static NPU, ToolkitError> {
    DEVICES
        .iter()
        .find(|npu| npu.id == id)
        .ok_or_else(|| toolkit_error!("Could not acquire NPU{id}"))
}

/// Validates `npu_id` for operations that are not dispatched to any NPU, so
/// they accept the same ids as all other operations: [`AUTO_NPU_ID`]
/// (resolving to the first available NPU) or the id of an available NPU.
pub(crate) fn check_npu_id(npu_id: i32) -> Result<(), ToolkitError> {
    resolve_any(npu_id).map(|_| ())
}

/// Lists the available NPUs that support all of `ops`.
///
/// Returns `(npu_id, serial number)` pairs sorted by id. An empty `ops`
/// returns every available NPU.
pub fn get_available_npus(
    ops: &[QantDeviceCapability],
) -> Result<Vec<(u32, String)>, ToolkitError> {
    Ok(DEVICES
        .iter()
        .filter(|npu| npu.supports(ops))
        .map(|npu| (npu.id, npu.serial.clone()))
        .collect())
}

/// Returns the operations supported by an NPU.
///
/// `npu_id`: [`AUTO_NPU_ID`] (-1) selects the lowest-id NPU;
/// a non-negative id targets that NPU directly. Errors if `npu_id < -1`.
pub fn get_capabilities(npu_id: i32) -> Result<Vec<QantDeviceCapability>, ToolkitError> {
    Ok(resolve_any(npu_id)?.capabilities.clone())
}

#[cfg(test)]
mod tests {
    use super::QantDeviceCapability::*;
    use super::*;

    #[test]
    fn supports_requires_all_ops() {
        let npu = NPU {
            id: 0,
            backend: NPUBackend::Gen2,
            serial: String::new(),
            capabilities: vec![Mul, NonLin, MemHandling],
        };

        assert!(npu.supports(&[]));
        assert!(npu.supports(&[Mul]));
        assert!(npu.supports(&[MemHandling, Mul]));
        assert!(!npu.supports(&[Dft]));
        assert!(!npu.supports(&[Mul, Dft]));
    }

    // The following run against whatever NPUs are enumerated in this
    // environment (cpu-backend simulation or real hardware).

    #[test]
    fn devices_are_sorted_by_id() {
        assert!(DEVICES.windows(2).all(|w| w[0].id < w[1].id));
    }

    #[test]
    fn resolve_default_picks_first_matching_npu() {
        let ops_list: [&[QantDeviceCapability]; 4] = [&[], &[Mul], &[Dft], &[Mul, Dft]];
        for ops in ops_list {
            let expected = DEVICES
                .iter()
                .find(|npu| npu.supports(ops))
                .map(|npu| npu.id);
            let resolved = resolve_npu(AUTO_NPU_ID, |npu| npu.supports(ops))
                .ok()
                .map(|npu| npu.id);
            assert_eq!(resolved, expected);
        }
    }

    #[test]
    fn resolve_explicit_id_ignores_predicate() {
        for npu in DEVICES.iter() {
            assert_eq!(resolve_npu(npu.id as i32, |_| false).unwrap().id, npu.id);
        }
        assert!(resolve_npu(i32::MAX, |_| true).is_err());
    }

    #[test]
    fn resolve_rejects_ids_below_default() {
        assert!(resolve_npu(-2, |_| true).is_err());
        assert!(resolve_npu(i32::MIN, |_| true).is_err());
    }
}

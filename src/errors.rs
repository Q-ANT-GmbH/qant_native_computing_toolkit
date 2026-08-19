use std::error::Error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct ToolkitError {
    pub description: String,
}

impl fmt::Display for ToolkitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Toolkit Error: {}", self.description)
    }
}

// we implement this function with the same name as std::str::FromStr::from_str,
// but don't want to implement the trait as that would require the return type to be a Result<>
// our function can't fail and does not need to introduce this extra error handling
#[allow(clippy::should_implement_trait)]
impl ToolkitError {
    pub fn from_str(desc: &str) -> Self {
        Self {
            description: String::from(desc),
        }
    }
    pub fn from_format(args: fmt::Arguments) -> Self {
        ToolkitError {
            description: args.to_string(),
        }
    }
}

#[macro_export]
macro_rules! toolkit_error {
    ($($arg:tt)*) => {
        ToolkitError::from_format(format_args!($($arg)*))
    };
}

impl Error for ToolkitError {}

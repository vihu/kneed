pub mod knee_locator;
pub mod shape_detector;

#[cfg(any(all(feature = "testing", test), feature = "data-generator"))]
mod data_generator;

#[cfg(feature = "data-generator")]
pub use data_generator::DataGenerator;

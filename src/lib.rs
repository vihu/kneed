pub mod knee_locator;
pub mod shape_detector;

#[cfg(all(feature = "testing", test))]
mod data_generator;
#[cfg(feature = "data-generator")]
pub mod data_generator;

#[cfg(feature = "data-generator")]
pub use data_generator::DataGenerator;

#[cfg(test)]
mod shape_detector;

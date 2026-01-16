pub mod knee_locator;

#[cfg(feature = "data-generator")]
pub mod data_generator;

#[cfg(feature = "data-generator")]
pub use data_generator::DataGenerator;

#[cfg(test)]
mod shape_detector;

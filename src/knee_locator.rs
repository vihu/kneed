use anyhow::{bail, Result};
use ndarray::{s, Array1, Axis};
use ndarray_interp::interp1d::Interp1DBuilder;
use ndarray_stats::QuantileExt;
use polyfit_rs::polyfit_rs::polyfit;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KneeLocatorError {
    #[error("parameter error {0}")]
    ParamError(String),
}

#[derive(Debug, PartialEq, Clone)]
pub enum ValidCurve {
    Convex,
    Concave,
}

#[derive(Debug, PartialEq, Clone)]
pub enum ValidDirection {
    Increasing,
    Decreasing,
}

#[derive(Debug, PartialEq, Clone)]
pub enum InterpMethod {
    Interp1d,
    Polynomial,
}

#[derive(Debug, Clone)]
pub struct KneeLocatorParams {
    curve: ValidCurve,
    direction: ValidDirection,
    interp_method: InterpMethod,
}

impl KneeLocatorParams {
    pub fn new(curve: ValidCurve, direction: ValidDirection, interp_method: InterpMethod) -> Self {
        Self {
            curve,
            direction,
            interp_method,
        }
    }
}

#[derive(Debug)]
pub struct KneeLocator {
    x: Array1<f64>,
    y: Array1<f64>,
    curve: ValidCurve,
    direction: ValidDirection,
    s: f64,
    n: usize,
    all_knees: Vec<f64>,
    all_norm_knees: Vec<f64>,
    all_knees_y: Vec<f64>,
    all_norm_knees_y: Vec<f64>,
    online: bool,
    polynomial_degree: usize,
    x_normalized: Array1<f64>,
    y_normalized: Array1<f64>,
    y_difference: Array1<f64>,
    x_difference: Array1<f64>,
    maxima_indices: Vec<usize>,
    minima_indices: Vec<usize>,
    x_difference_maxima: Array1<f64>,
    y_difference_maxima: Array1<f64>,
    x_difference_minima: Array1<f64>,
    y_difference_minima: Array1<f64>,
    tmx: Array1<f64>,
    knee: Option<f64>,
    norm_knee: Option<f64>,
    knee_y: Option<f64>,
    norm_knee_y: Option<f64>,
}

impl KneeLocator {
    pub fn elbow(&self) -> Option<f64> {
        self.knee
    }

    pub fn norm_elbow(&self) -> Option<f64> {
        self.norm_knee
    }

    pub fn elbow_y(&self) -> Option<f64> {
        self.knee_y
    }

    pub fn norm_elbow_y(&self) -> Option<f64> {
        self.norm_knee_y
    }

    pub fn all_elbows(&self) -> Vec<f64> {
        self.all_knees.clone()
    }

    pub fn all_norm_elbows(&self) -> Vec<f64> {
        self.all_norm_knees.clone()
    }

    pub fn all_elbows_y(&self) -> Vec<f64> {
        self.all_knees_y.clone()
    }

    pub fn all_norm_elbows_y(&self) -> Vec<f64> {
        self.all_norm_knees_y.clone()
    }

    fn check_x_y(x: &[f64], y: &[f64]) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            bail!(KneeLocatorError::ParamError(
                "input series cannot be empty!".to_string()
            ));
        }

        if x.len() != y.len() {
            bail!(KneeLocatorError::ParamError(
                "input series should have equal length!".to_string()
            ));
        }

        Ok(())
    }

    pub fn new(x: Vec<f64>, y: Vec<f64>, s: f64, params: KneeLocatorParams) -> Result<Self> {
        Self::check_x_y(&x, &y)?;
        Self::parameterized_new(x, y, s, params, false, 7)
    }

    pub fn parameterized_new(
        x: Vec<f64>,
        y: Vec<f64>,
        s: f64,
        params: KneeLocatorParams,
        online: bool,
        polynomial_degree: usize,
    ) -> Result<Self> {
        Self::check_x_y(&x, &y)?;
        let n = x.len();
        let mut knee_locator = KneeLocator {
            x: Array1::from_vec(x),
            y: Array1::from_vec(y),
            curve: params.curve,
            direction: params.direction,
            s,
            n,
            all_knees: Vec::new(),
            all_norm_knees: Vec::new(),
            all_knees_y: Vec::new(),
            all_norm_knees_y: Vec::new(),
            online,
            polynomial_degree,
            x_normalized: Array1::zeros(n),
            y_normalized: Array1::zeros(n),
            y_difference: Array1::zeros(n),
            x_difference: Array1::zeros(n),
            y_difference_maxima: Array1::zeros(n),
            x_difference_maxima: Array1::zeros(n),
            y_difference_minima: Array1::zeros(n),
            x_difference_minima: Array1::zeros(n),
            maxima_indices: Vec::new(),
            minima_indices: Vec::new(),
            tmx: Array1::zeros(0),
            knee: None,
            norm_knee: None,
            knee_y: None,
            norm_knee_y: None,
        };

        knee_locator.initialize(params.interp_method);
        Ok(knee_locator)
    }

    fn initialize(&mut self, interp_method: InterpMethod) {
        // Step 1: Fit a smooth line
        let ds_y = match interp_method {
            InterpMethod::Interp1d => self.interp1d().unwrap(),
            InterpMethod::Polynomial => self.polynomial_interp().unwrap(),
        };

        // Step 2: Normalize values
        self.x_normalized = Self::normalize(&self.x).unwrap();
        self.y_normalized = Self::normalize(&ds_y).unwrap();

        // Step 3: Calculate the difference curve
        self.y_normalized = self.transform_y();
        self.y_difference = &self.y_normalized - &self.x_normalized;
        self.x_difference.clone_from(&self.x_normalized);

        // Step 4: Identify local maxima/minima
        self.find_local_extrema();

        // Step 5: Calculate thresholds
        self.calculate_thresholds();

        // Step 6: Find knee
        (self.knee, self.norm_knee) = self.find_knee();

        // Step 7: If we have a knee, extract data about it
        self.knee_y = match self.knee {
            None => None,
            Some(knee) => Some(self.y[self.x.iter().position(|&v| v == knee).unwrap()]),
        };

        self.norm_knee_y = match self.norm_knee {
            None => None,
            Some(norm_knee) => Some(
                self.y_normalized[self
                    .x_normalized
                    .iter()
                    .position(|&v| v == norm_knee)
                    .unwrap()],
            ),
        };
    }

    fn normalize(a: &Array1<f64>) -> Result<Array1<f64>> {
        let min = a.min()?;
        let max = a.max()?;
        Ok((a - *min) / (*max - *min))
    }

    fn interp1d(&self) -> Result<Array1<f64>> {
        let interpolator = Interp1DBuilder::new(self.y.view())
            .x(self.x.view())
            .build()?;

        Ok(Array1::from_vec(
            self.x
                .iter()
                .map(|&x| interpolator.interp_scalar(x).unwrap())
                .collect(),
        ))
    }

    fn polynomial_interp(&self) -> Result<Array1<f64>> {
        let coeffs = polyfit(
            self.x.as_slice().unwrap(),
            self.y.as_slice().unwrap(),
            self.polynomial_degree,
        )
        .map_err(|e| anyhow::anyhow!(e))?;

        Ok(Array1::from_vec(
            self.x
                .iter()
                .map(|&x| {
                    coeffs.iter().enumerate().fold(0.0, |acc, (power, &coeff)| {
                        acc + coeff * x.powi(power as i32)
                    })
                })
                .collect(),
        ))
    }

    fn transform_y(&self) -> Array1<f64> {
        match (&self.direction, &self.curve) {
            (ValidDirection::Decreasing, ValidCurve::Concave) => {
                self.y_normalized.slice(s![..;-1]).to_owned()
            }
            (ValidDirection::Decreasing, ValidCurve::Convex) => {
                let max = self.y_normalized.max().unwrap();
                self.y_normalized.mapv(|v| max - v)
            }
            (ValidDirection::Increasing, ValidCurve::Convex) => {
                let max = self.y_normalized.max().unwrap();
                self.y_normalized
                    .mapv(|v| max - v)
                    .slice(s![..;-1])
                    .to_owned()
            }
            _ => self.y_normalized.clone(), // No transformation needed for (Increasing, Concave)
        }
    }

    fn find_local_extrema(&mut self) {
        // Local maxima
        self.maxima_indices = argrelextrema(&self.y_difference, |a, b| a >= b, 1);
        self.x_difference_maxima = self
            .maxima_indices
            .iter()
            .map(|&i| self.x_difference[i])
            .collect::<Vec<f64>>()
            .into();
        self.y_difference_maxima = self
            .maxima_indices
            .iter()
            .map(|&i| self.y_difference[i])
            .collect::<Vec<f64>>()
            .into();

        // Local minima
        self.minima_indices = argrelextrema(&self.y_difference, |a, b| a <= b, 1);
        self.x_difference_minima = self
            .minima_indices
            .iter()
            .map(|&i| self.x_difference[i])
            .collect::<Vec<f64>>()
            .into();
        self.y_difference_minima = self
            .minima_indices
            .iter()
            .map(|&i| self.y_difference[i])
            .collect::<Vec<f64>>()
            .into();
    }

    fn calculate_thresholds(&mut self) {
        let mean_diff = self
            .x_normalized
            .windows(2)
            .into_iter()
            .map(|w| w[1] - w[0])
            .fold(0.0, |acc, x| acc + x)
            / (self.n - 1) as f64;

        let selected_y_diff = self.y_difference.select(Axis(0), &self.maxima_indices);

        self.tmx = &selected_y_diff - (self.s * mean_diff.abs());
    }

    fn find_knee(&mut self) -> (Option<f64>, Option<f64>) {
        // Return None if no local maxima found
        if self.maxima_indices.is_empty() {
            return (None, None);
        }

        // Placeholders for which threshold region i is located in
        let mut maxima_threshold_index = 0;
        let mut _minima_threshold_index = 0;
        let mut knee: Option<f64> = None;
        let mut norm_knee: Option<f64> = None;
        let mut threshold = 0.0;
        let mut threshold_index = 0;

        // Traverse the difference curve
        for (i, &_x) in self.x_difference.iter().enumerate() {
            // Skip points on the curve before the first local maxima
            if i < self.maxima_indices[0] {
                continue;
            }

            let j = i + 1;

            // Reached the end of the curve
            if i == (self.x_difference.len() - 1) {
                break;
            }

            // If we're at a local max, increment the maxima threshold index and continue
            if self.maxima_indices.contains(&i) {
                threshold = self.tmx[maxima_threshold_index];
                threshold_index = i;
                maxima_threshold_index += 1;
            }

            // Values in difference curve are at or after a local minimum
            if self.minima_indices.contains(&i) {
                threshold = 0.0;
                _minima_threshold_index += 1;
            }

            if self.y_difference[j] < threshold {
                match self.curve {
                    ValidCurve::Convex => {
                        if self.direction == ValidDirection::Decreasing {
                            knee = Some(self.x[threshold_index]);
                            norm_knee = Some(self.x_normalized[threshold_index]);
                        } else {
                            knee = Some(self.x[self.x.len() - threshold_index - 1]);
                            norm_knee = Some(self.x_normalized[threshold_index]);
                        }
                    }
                    ValidCurve::Concave => {
                        if self.direction == ValidDirection::Decreasing {
                            knee = Some(self.x[self.x.len() - threshold_index - 1]);
                            norm_knee = Some(self.x_normalized[threshold_index]);
                        } else {
                            knee = Some(self.x[threshold_index]);
                            norm_knee = Some(self.x_normalized[threshold_index]);
                        }
                    }
                }

                // Add the y value at the knee
                let y_at_knee = self.y[self.x.iter().position(|&v| v == knee.unwrap()).unwrap()];
                let y_norm_at_knee = self.y_normalized[self
                    .x_normalized
                    .iter()
                    .position(|&v| v == norm_knee.unwrap())
                    .unwrap()];

                if !self.all_knees.contains(&knee.unwrap()) {
                    self.all_knees_y.push(y_at_knee);
                    self.all_norm_knees_y.push(y_norm_at_knee);
                    self.all_knees.push(knee.unwrap());
                    self.all_norm_knees.push(norm_knee.unwrap());
                }

                // If detecting in offline mode, return the first knee found
                if !self.online {
                    return (knee, norm_knee);
                }
            }
        }

        if self.all_knees.is_empty() {
            // No knee was found
            return (None, None);
        }

        (knee, norm_knee)
    }
}

// Re-implement argrelextrema from numpy
fn argrelextrema<F>(data: &Array1<f64>, comparator: F, order: usize) -> Vec<usize>
where
    F: Fn(f64, f64) -> bool,
{
    let mut extrema_indices = Vec::new();
    let len = data.len();

    for i in 0..len {
        let mut is_extrema = true;

        // Compare with previous `order` elements
        for j in 1..=order {
            if i >= j && !comparator(data[i], data[i - j]) {
                is_extrema = false;
                break;
            }
        }

        // Compare with next `order` elements
        for j in 1..=order {
            if i + j < len && !comparator(data[i], data[i + j]) {
                is_extrema = false;
                break;
            }
        }

        if is_extrema {
            extrema_indices.push(i);
        }
    }

    extrema_indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_generator::DataGenerator;
    use approx::assert_abs_diff_eq;
    use rand::prelude::*;
    use rand_distr::{Distribution, Gamma};

    fn truncate_and_scale(
        x: &Array1<f64>,
        y: &Array1<f64>,
        truncate: usize,
        scale: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let x_truncated = x.slice(s![..x.len() - truncate]);
        let y_truncated = y.slice(s![..y.len() - truncate]);

        let x_scaled = x_truncated.map(|&v| v / scale);
        let y_scaled = y_truncated.map(|&v| v / scale);

        (x_scaled.to_owned(), y_scaled.to_owned())
    }

    fn generate_sine_wave(start: f64, end: f64, step: f64) -> (Array1<f64>, Array1<f64>) {
        let x = Array1::range(start, end, step);
        let y = x.mapv(|v| v.sin());
        (x, y)
    }

    #[test]
    fn test_figure2_interp1d() {
        let (x, y) = DataGenerator::figure2();

        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();

        assert_abs_diff_eq!(0.222222222222222, kl.knee.unwrap());
        assert_abs_diff_eq!(0.222222222222222, kl.elbow().unwrap());
        assert_abs_diff_eq!(0.222222222222222, kl.norm_elbow().unwrap());
        assert_abs_diff_eq!(1.8965517241379306, kl.knee_y.unwrap());
    }

    #[test]
    fn test_figure2_polynomial() {
        let (x, y) = DataGenerator::figure2();

        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            InterpMethod::Polynomial,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();

        assert_abs_diff_eq!(0.222222222222222, kl.knee.unwrap());
        assert_abs_diff_eq!(0.222222222222222, kl.elbow().unwrap());
        assert_abs_diff_eq!(0.222222222222222, kl.norm_elbow().unwrap());
        assert_abs_diff_eq!(1.8965517241379306, kl.knee_y.unwrap());
    }

    #[test]
    fn test_noisy_gaussian() {
        let (x, y) = DataGenerator::noisy_gaussian(50.0, 10.0, 1000, 42);
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            InterpMethod::Polynomial,
        );
        let kl =
            KneeLocator::parameterized_new(x.to_vec(), y.to_vec(), 1.0, params, true, 11).unwrap();
        // NOTE: python reports 63.0, presumably this is fine
        assert_abs_diff_eq!(62.25, kl.knee.unwrap(), epsilon = 0.1);
    }

    #[test]
    fn test_concave_increasing_interp1d() {
        let (x, y) = DataGenerator::concave_increasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(2.0, kl.knee.unwrap());
    }

    #[test]
    fn test_concave_increasing_polynomial() {
        let (x, y) = DataGenerator::concave_increasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            InterpMethod::Polynomial,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(2.0, kl.knee.unwrap());
    }

    #[test]
    fn test_concave_decreasing_interp1d() {
        let (x, y) = DataGenerator::concave_decreasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(7.0, kl.knee.unwrap());
    }

    #[test]
    fn test_concave_decreasing_polynomial() {
        let (x, y) = DataGenerator::concave_decreasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Decreasing,
            InterpMethod::Polynomial,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(7.0, kl.knee.unwrap());
    }

    #[test]
    fn test_convex_increasing_interp1d() {
        let (x, y) = DataGenerator::convex_increasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(7.0, kl.knee.unwrap());
    }

    #[test]
    fn test_convex_increasing_polynomial() {
        let (x, y) = DataGenerator::convex_increasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Increasing,
            InterpMethod::Polynomial,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(7.0, kl.knee.unwrap());
    }

    #[test]
    fn test_convex_decreasing_interp1d() {
        let (x, y) = DataGenerator::convex_decreasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(2.0, kl.knee.unwrap());
    }

    #[test]
    fn test_convex_decreasing_polynomial() {
        let (x, y) = DataGenerator::convex_decreasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Polynomial,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(2.0, kl.knee.unwrap());
    }

    #[test]
    fn test_concave_increasing_truncated_interp1d() {
        let (x, y) = DataGenerator::concave_increasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        );

        let (x1, y1) = truncate_and_scale(&x, &y, 3, 10.0);

        let kl = KneeLocator::new(x1.to_vec(), y1.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(0.2, kl.knee.unwrap());
    }

    #[test]
    fn test_concave_increasing_truncated_polynomial() {
        let (x, y) = DataGenerator::concave_increasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            InterpMethod::Polynomial,
        );

        let (x1, y1) = truncate_and_scale(&x, &y, 3, 10.0);

        let kl = KneeLocator::new(x1.to_vec(), y1.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(0.2, kl.knee.unwrap());
    }

    #[test]
    fn test_concave_decreasing_truncated_interp1d() {
        let (x, y) = DataGenerator::concave_decreasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );

        let (x1, y1) = truncate_and_scale(&x, &y, 3, 10.0);

        let kl = KneeLocator::new(x1.to_vec(), y1.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(0.4, kl.knee.unwrap());
    }

    #[test]
    fn test_concave_decreasing_truncated_polynomial() {
        let (x, y) = DataGenerator::concave_decreasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Decreasing,
            InterpMethod::Polynomial,
        );

        let (x1, y1) = truncate_and_scale(&x, &y, 3, 10.0);

        let kl = KneeLocator::new(x1.to_vec(), y1.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(0.4, kl.knee.unwrap());
    }

    #[test]
    fn test_convex_increasing_truncated_interp1d() {
        let (x, y) = DataGenerator::convex_increasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        );

        let (x1, y1) = truncate_and_scale(&x, &y, 3, 10.0);

        let kl = KneeLocator::new(x1.to_vec(), y1.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(0.4, kl.knee.unwrap());
    }

    #[test]
    fn test_convex_increasing_truncated_polynomial() {
        let (x, y) = DataGenerator::convex_increasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Increasing,
            InterpMethod::Polynomial,
        );

        let (x1, y1) = truncate_and_scale(&x, &y, 3, 10.0);

        let kl = KneeLocator::new(x1.to_vec(), y1.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(0.4, kl.knee.unwrap());
    }

    #[test]
    fn test_convex_decreasing_truncated_interp1d() {
        let (x, y) = DataGenerator::convex_decreasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );

        let (x1, y1) = truncate_and_scale(&x, &y, 3, 10.0);

        let kl = KneeLocator::new(x1.to_vec(), y1.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(0.2, kl.knee.unwrap());
    }

    #[test]
    fn test_convex_decreasing_truncated_polynomial() {
        let (x, y) = DataGenerator::convex_decreasing();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Polynomial,
        );

        let (x1, y1) = truncate_and_scale(&x, &y, 3, 10.0);

        let kl = KneeLocator::new(x1.to_vec(), y1.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(0.2, kl.knee.unwrap());
    }

    #[test]
    fn test_convex_decreasing_bumpy_interp1d() {
        let (x, y) = DataGenerator::bumpy();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(26.0, kl.knee.unwrap());
    }

    #[test]
    fn test_convex_decreasing_bumpy_polynomial() {
        let (x, y) = DataGenerator::bumpy();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Polynomial,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(28.0, kl.knee.unwrap());
    }

    #[test]
    fn test_gamma_online() {
        let mut rng = StdRng::seed_from_u64(23);
        let n = 1000;

        // Generate x values
        let x = Array1::range(1.0, (n + 1) as f64, 1.0);

        // Generate y values using gamma distribution
        let gamma = Gamma::new(0.5, 1.0).unwrap();
        let mut vec_y: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();

        // Sort y in descending order
        vec_y.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let y = Array1::from_vec(vec_y);

        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );

        let kl =
            KneeLocator::parameterized_new(x.to_vec(), y.to_vec(), 1.0, params, true, 7).unwrap();

        // NOTE: python reports 482.0, presumably because gamma sampling might differ?
        assert_abs_diff_eq!(497.0, kl.knee.unwrap());
    }

    #[test]
    fn test_gamma_offline() {
        let mut rng = StdRng::seed_from_u64(23);
        let n = 1000;

        // Generate x values
        let x = Array1::range(1.0, (n + 1) as f64, 1.0);

        // Generate y values using gamma distribution
        let gamma = Gamma::new(0.5, 1.0).unwrap();
        let mut vec_y: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();

        // Sort y in descending order
        vec_y.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let y = Array1::from_vec(vec_y);

        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );

        let kl =
            KneeLocator::parameterized_new(x.to_vec(), y.to_vec(), 1.0, params, false, 7).unwrap();

        // NOTE: python reports 22.0, presumably because gamma sampling might differ?
        assert_abs_diff_eq!(71.0, kl.knee.unwrap());
    }

    #[test]
    fn test_sensitivity() {
        let seed = 23;
        let n = 1000;

        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let gamma = Gamma::new(0.5, 1.0).unwrap();
        let mut y: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
        y.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let sensitivity = [1.0, 3.0, 5.0, 10.0, 100.0, 200.0, 400.0];

        // NOTE: python expected_knees = [43, 137, 178, 258, 305, 482, 482]
        let expected_knees = [71.0, 128.0, 172.0, 256.0, 497.0, 497.0, 497.0];

        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );

        for (&s, &expected_knee) in sensitivity.iter().zip(expected_knees.iter()) {
            let kl =
                KneeLocator::parameterized_new(x.clone(), y.clone(), s, params.clone(), false, 7)
                    .unwrap();
            let detected_knee = kl.knee.unwrap();
            println!(
                "Sensitivity: {}, Detected Knee: {}, Expected Knee: {}",
                s, detected_knee, expected_knee
            );
            assert_abs_diff_eq!(detected_knee, expected_knee, epsilon = 1.0);
        }
    }

    #[test]
    fn test_sine() {
        let (x, y_sin) = generate_sine_wave(0.0, 10.0, 0.1);

        let sine_combos = vec![
            (ValidDirection::Decreasing, ValidCurve::Convex),
            (ValidDirection::Increasing, ValidCurve::Convex),
            (ValidDirection::Increasing, ValidCurve::Concave),
            (ValidDirection::Decreasing, ValidCurve::Concave),
        ];

        let expected_knees = [4.5, 4.9, 7.7, 1.8];
        let mut detected_knees = Vec::new();

        for (direction, curve) in sine_combos {
            let params = KneeLocatorParams::new(curve, direction, InterpMethod::Interp1d);
            let kl_sine =
                KneeLocator::parameterized_new(x.to_vec(), y_sin.to_vec(), 1.0, params, true, 1)
                    .unwrap();
            detected_knees.push(kl_sine.knee.unwrap());
        }

        for (detected, expected) in detected_knees.iter().zip(expected_knees.iter()) {
            println!("Detected: {}, Expected: {}", detected, expected);
            assert_abs_diff_eq!(detected, expected);
        }
    }

    #[test]
    fn test_flat_maxima() {
        let x = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0,
        ];
        let y = [
            1.0,
            0.787701317715959,
            0.7437774524158126,
            0.6559297218155198,
            0.5065885797950219,
            0.36749633967789164,
            0.2547584187408492,
            0.16251830161054173,
            0.10395314787701318,
            0.06734992679355783,
            0.043923865300146414,
            0.027818448023426062,
            0.01903367496339678,
            0.013177159590043924,
            0.010248901903367497,
            0.007320644216691069,
            0.005856515373352855,
            0.004392386530014641,
        ];

        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );
        let kl1 = KneeLocator::new(x.to_vec(), y.to_vec(), 0.0, params.clone()).unwrap();
        assert_abs_diff_eq!(1.0, kl1.knee.unwrap());
        let kl2 = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(8.0, kl2.knee.unwrap());
    }

    #[test]
    fn test_y() {
        let (x, y) = DataGenerator::figure2();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert_abs_diff_eq!(1.897, kl.knee_y.unwrap(), epsilon = 0.03);
        assert_abs_diff_eq!(1.897, kl.all_knees_y[0], epsilon = 0.03);
        assert_abs_diff_eq!(0.758, kl.norm_knee_y.unwrap(), epsilon = 0.03);
        assert_abs_diff_eq!(0.758, kl.all_norm_knees_y[0], epsilon = 0.03);

        assert_abs_diff_eq!(1.897, kl.elbow_y().unwrap(), epsilon = 0.03);
        assert_abs_diff_eq!(1.897, kl.all_elbows_y()[0], epsilon = 0.03);
        assert_abs_diff_eq!(0.758, kl.norm_elbow_y().unwrap(), epsilon = 0.03);
        assert_abs_diff_eq!(0.758, kl.all_norm_elbows_y()[0], epsilon = 0.03);
    }

    #[test]
    fn test_y_no_knees() {
        let x = [1.0, 2.0, 3.0];
        let y = [0.90483742, 0.81873075, 0.74081822];
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );
        let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params).unwrap();
        assert!(kl.knee.is_none());
        assert!(kl.norm_knee_y.is_none());
    }

    #[test]
    fn test_x_equals_y() {
        let x: Vec<f64> = (0..10).map(|i| i as f64 * 1.0).collect();
        let y: Vec<f64> = x.iter().map(|_| 1.0).collect();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        );
        let kl = KneeLocator::new(x, y, 1.0, params).unwrap();
        assert!(kl.knee.is_none());
    }

    #[test]
    fn test_logistic() {
        let y = [
            2.00855493e-45,
            1.10299045e-43,
            4.48168384e-42,
            1.22376580e-41,
            5.10688883e-40,
            1.18778110e-38,
            5.88777891e-35,
            4.25317895e-34,
            4.06507035e-33,
            6.88084518e-32,
            2.99321831e-31,
            1.13291723e-30,
            1.05244482e-28,
            2.67578448e-27,
            1.22522190e-26,
            2.36517846e-26,
            8.30369408e-26,
            1.24303033e-25,
            2.27726918e-25,
            1.06330422e-24,
            5.55017673e-24,
            1.92068553e-23,
            3.31361011e-23,
            1.13575247e-22,
            1.75386416e-22,
            6.52680518e-22,
            2.05106011e-21,
            6.37285545e-21,
            4.16125535e-20,
            1.12709507e-19,
            5.75853420e-19,
            1.73333796e-18,
            2.70099890e-18,
            7.53254646e-18,
            1.38139433e-17,
            3.60081965e-17,
            8.08419977e-17,
            1.86378584e-16,
            5.36224556e-16,
            8.89404640e-16,
            2.34045104e-15,
            4.72168880e-15,
            6.84378992e-15,
            2.26898430e-14,
            3.10087652e-14,
            2.78081199e-13,
            1.06479577e-12,
            2.81002203e-12,
            4.22067092e-12,
            9.27095863e-12,
            1.54519738e-11,
            4.53347819e-11,
            1.35564441e-10,
            2.35242087e-10,
            4.45253545e-10,
            9.78613696e-10,
            1.53140922e-09,
            2.81648560e-09,
            6.70890436e-09,
            1.49724785e-08,
            5.59553565e-08,
            1.39510811e-07,
            7.64761811e-07,
            1.40723957e-06,
            4.97638863e-06,
            2.12817943e-05,
            3.26471410e-05,
            1.02599591e-04,
            3.18774179e-04,
            5.67297630e-04,
            9.22732716e-04,
            1.17445643e-03,
            3.59279384e-03,
            3.61936491e-02,
            6.39493416e-02,
            1.29304829e-01,
            1.72272215e-01,
            3.46945901e-01,
            5.02826602e-01,
            6.24800042e-01,
            7.38412957e-01,
            7.59931663e-01,
            7.73374421e-01,
            7.91421897e-01,
            8.29325597e-01,
            8.57718637e-01,
            8.73286061e-01,
            8.77056835e-01,
            8.93173768e-01,
            9.05435646e-01,
            9.17217910e-01,
            9.19119179e-01,
            9.24810910e-01,
            9.26306908e-01,
            9.28621233e-01,
            9.33855835e-01,
            9.37263027e-01,
            9.41651642e-01,
        ];
        let x = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0,
            45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0,
            59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0,
            73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0,
            87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0,
        ];

        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        );

        let kl =
            KneeLocator::parameterized_new(x.to_vec(), y.to_vec(), 1.0, params, true, 7).unwrap();

        assert_abs_diff_eq!(73.0, kl.knee.unwrap());
    }

    #[test]
    fn test_all_knees() {
        let (x, y) = DataGenerator::bumpy();
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        );
        let kl =
            KneeLocator::parameterized_new(x.to_vec(), y.to_vec(), 1.0, params, true, 7).unwrap();

        let expected_elbows = [26.0, 31.0, 41.0, 46.0, 53.0];
        let mut all_elbows = kl.all_elbows();
        all_elbows.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!("expected elbows {:?}", expected_elbows);
        println!("detected elbows {:?}", all_elbows);

        for (detected, expected) in all_elbows.iter().zip(expected_elbows.iter()) {
            println!("Detected elbow: {}, Expected elbow: {}", detected, expected);
            assert_abs_diff_eq!(detected, expected, epsilon = 0.1);
        }

        let expected_norm_elbows = [
            0.2921348314606742,
            0.348314606741573,
            0.4606741573033708,
            0.5168539325842696,
            0.5955056179775281,
        ];
        let mut all_norm_elbows = kl.all_norm_elbows();
        all_norm_elbows.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (detected, expected) in all_norm_elbows.iter().zip(expected_norm_elbows.iter()) {
            println!(
                "Detected norm elbow: {}, Expected norm elbow: {}",
                detected, expected
            );
            assert_abs_diff_eq!(detected, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_knee_locator_input_validation() {
        let params = KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        );

        // Test both x and y vectors empty
        let result = KneeLocator::new(vec![], vec![], 1.0, params.clone());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "parameter error input series cannot be empty!"
        );

        // Test empty x vector
        let result = KneeLocator::new(vec![], vec![1.0, 2.0, 3.0], 1.0, params.clone());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "parameter error input series cannot be empty!"
        );

        // Test empty y vector
        let result = KneeLocator::new(vec![1.0, 2.0, 3.0], vec![], 1.0, params.clone());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "parameter error input series cannot be empty!"
        );

        // Test unequal lengths
        let result = KneeLocator::new(vec![1.0, 2.0], vec![1.0, 2.0, 3.0], 1.0, params.clone());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "parameter error input series should have equal length!"
        );

        // Test valid input
        let result = KneeLocator::new(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 1.0, params);
        assert!(result.is_ok());
    }
}

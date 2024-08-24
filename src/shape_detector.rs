#![allow(dead_code)]

use crate::knee_locator::{ValidCurve, ValidDirection};

type Shape = (ValidDirection, ValidCurve);

/// Detect the direction and curve type of the line.
fn find_shape(x: &[f64], y: &[f64]) -> Shape {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");

    // Perform polynomial fitting
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    // Calculate indices for the middle 60% of the data
    let x1 = (x.len() as f64 * 0.2) as usize;
    let x2 = (x.len() as f64 * 0.8) as usize;

    // Calculate q
    let middle_x = &x[x1..x2];
    let middle_y = &y[x1..x2];
    let middle_y_mean: f64 = middle_y.iter().sum::<f64>() / middle_y.len() as f64;
    let middle_fitted_y_mean: f64 = middle_x
        .iter()
        .map(|&xi| xi * slope + intercept)
        .sum::<f64>()
        / middle_x.len() as f64;
    let q = middle_y_mean - middle_fitted_y_mean;

    // Use a small epsilon value to handle floating-point imprecision
    const EPSILON: f64 = 1e-10;

    // Determine direction and curve type
    if slope.abs() < EPSILON {
        // If slope is very close to zero, classify as decreasing and convex
        (ValidDirection::Decreasing, ValidCurve::Convex)
    } else if slope > 0.0 {
        if q >= 0.0 {
            (ValidDirection::Increasing, ValidCurve::Concave)
        } else {
            (ValidDirection::Increasing, ValidCurve::Convex)
        }
    } else if q > 0.0 {
        (ValidDirection::Decreasing, ValidCurve::Concave)
    } else {
        (ValidDirection::Decreasing, ValidCurve::Convex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "testing")]
    use crate::data_generator::DataGenerator;
    use crate::knee_locator::{ValidCurve, ValidDirection};

    #[test]
    fn test_curve_and_direction() {
        // Test case 1
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y1 = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        assert_eq!(
            find_shape(&x1, &y1),
            (ValidDirection::Increasing, ValidCurve::Convex)
        );

        // Test case 2
        let x2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y2 = vec![1.0, 1.5, 1.8, 1.9, 2.0];
        assert_eq!(
            find_shape(&x2, &y2),
            (ValidDirection::Increasing, ValidCurve::Concave)
        );

        // Test case 3
        let x3 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y3 = vec![15.0, 10.0, 6.0, 3.0, 1.0];
        assert_eq!(
            find_shape(&x3, &y3),
            (ValidDirection::Decreasing, ValidCurve::Convex)
        );

        // Test case 4
        let x4 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y4 = vec![2.0, 1.9, 1.8, 1.5, 1.0];
        assert_eq!(
            find_shape(&x4, &y4),
            (ValidDirection::Decreasing, ValidCurve::Concave)
        );

        // Test case 5
        let x5 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y5 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(
            find_shape(&x5, &y5),
            (ValidDirection::Increasing, ValidCurve::Concave)
        );

        // Test case 6
        let x6 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y6 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(
            find_shape(&x6, &y6),
            (ValidDirection::Decreasing, ValidCurve::Convex)
        );

        // Test case 7
        let x7 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y7 = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        assert_eq!(
            find_shape(&x7, &y7),
            (ValidDirection::Decreasing, ValidCurve::Convex)
        );

        // Test case 8
        let x8 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y8 = vec![1.1, 1.0, 1.2, 1.3, 1.25, 1.4, 1.5, 1.6, 1.7, 1.8];
        assert_eq!(
            find_shape(&x8, &y8),
            (ValidDirection::Increasing, ValidCurve::Convex)
        );
    }

    #[test]
    fn test_find_shape() {
        let (x, y) = DataGenerator::concave_increasing();
        assert_eq!(
            find_shape(&x, &y),
            (ValidDirection::Increasing, ValidCurve::Concave)
        );

        let (x, y) = DataGenerator::concave_decreasing();
        assert_eq!(
            find_shape(&x, &y),
            (ValidDirection::Decreasing, ValidCurve::Concave)
        );

        let (x, y) = DataGenerator::convex_increasing();
        assert_eq!(
            find_shape(&x, &y),
            (ValidDirection::Increasing, ValidCurve::Convex)
        );

        let (x, y) = DataGenerator::convex_decreasing();
        assert_eq!(
            find_shape(&x, &y),
            (ValidDirection::Decreasing, ValidCurve::Convex)
        );
    }
}

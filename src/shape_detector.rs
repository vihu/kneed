#![allow(dead_code)]

use crate::knee_locator::{ValidCurve, ValidDirection};
use ndarray::{s, stack, Array, Array1, Axis};
use ndarray_linalg::Solve;

type Shape = (ValidDirection, ValidCurve);

/// Detect the direction and curve type of the line.
fn find_shape(x: &Array1<f64>, y: &Array1<f64>) -> Shape {
    // Perform polynomial fitting
    let a = stack![Axis(1), x.mapv(|xi| xi), Array::ones(x.len())];
    let p = a.t().dot(&a).solve(&a.t().dot(y)).unwrap();
    // Calculate indices for the middle 60% of the data
    let x1 = (x.len() as f64 * 0.2) as usize;
    let x2 = (x.len() as f64 * 0.8) as usize;
    // Calculate q
    let middle_x = x.slice(s![x1..x2]);
    let middle_y = y.slice(s![x1..x2]);
    let q = middle_y.mean().unwrap() - (middle_x.mapv(|xi| xi * p[0] + p[1])).mean().unwrap();

    // Use a small epsilon value to handle floating-point imprecision
    const EPSILON: f64 = 1e-10;

    // Determine direction and curve type
    if p[0].abs() < EPSILON {
        // If slope is very close to zero, classify as decreasing and convex
        (ValidDirection::Decreasing, ValidCurve::Convex)
    } else if p[0] > 0.0 {
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
    use crate::{
        data_generator::DataGenerator,
        knee_locator::{ValidCurve, ValidDirection},
    };
    use ndarray::array;

    #[test]
    fn test_curve_and_direction() {
        // Test case 1
        let x1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y1 = array![1.0, 3.0, 6.0, 10.0, 15.0];
        assert_eq!(
            find_shape(&x1, &y1),
            (ValidDirection::Increasing, ValidCurve::Convex)
        );

        // Test case 2
        let x2 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y2 = array![1.0, 1.5, 1.8, 1.9, 2.0];
        assert_eq!(
            find_shape(&x2, &y2),
            (ValidDirection::Increasing, ValidCurve::Concave)
        );

        // Test case 3
        let x3 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y3 = array![15.0, 10.0, 6.0, 3.0, 1.0];
        assert_eq!(
            find_shape(&x3, &y3),
            (ValidDirection::Decreasing, ValidCurve::Convex)
        );

        // Test case 4
        let x4 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y4 = array![2.0, 1.9, 1.8, 1.5, 1.0];
        assert_eq!(
            find_shape(&x4, &y4),
            (ValidDirection::Decreasing, ValidCurve::Concave)
        );

        // Test case 5
        let x5 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y5 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(
            find_shape(&x5, &y5),
            (ValidDirection::Increasing, ValidCurve::Concave)
        );

        // Test case 6
        let x6 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y6 = array![5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(
            find_shape(&x6, &y6),
            (ValidDirection::Decreasing, ValidCurve::Convex)
        );

        // Test case 7
        let x7 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y7 = array![2.0, 2.0, 2.0, 2.0, 2.0];
        assert_eq!(
            find_shape(&x7, &y7),
            (ValidDirection::Decreasing, ValidCurve::Convex)
        );

        // Test case 8
        let x8 = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y8 = array![1.1, 1.0, 1.2, 1.3, 1.25, 1.4, 1.5, 1.6, 1.7, 1.8];
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

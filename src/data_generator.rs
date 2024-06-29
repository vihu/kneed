#![allow(dead_code)]

use ndarray::{array, Array1};
use rand::prelude::*;
use rand_distr::Normal;

pub struct DataGenerator;

impl DataGenerator {
    pub fn noisy_gaussian(mu: f64, sigma: f64, n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(mu, sigma).unwrap();

        let mut x: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        (Array1::from(x), Array1::from(y))
    }

    pub fn figure2() -> (Array1<f64>, Array1<f64>) {
        let x = Array1::linspace(0.0, 1.0, 10);
        let y = &(-1.0 / ((&x + 0.1) as Array1<f64>)) + 5.0;
        (x, y)
    }

    pub fn convex_increasing() -> (Array1<f64>, Array1<f64>) {
        let x = Array1::range(0.0, 10.0, 1.0);
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 40.0, 100.0];
        (x, y)
    }

    pub fn convex_decreasing() -> (Array1<f64>, Array1<f64>) {
        let x = Array1::range(0.0, 10.0, 1.0);
        let y = array![100.0, 40.0, 20.0, 15.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        (x, y)
    }

    pub fn concave_decreasing() -> (Array1<f64>, Array1<f64>) {
        let x = Array1::range(0.0, 10.0, 1.0);
        let y = array![99.0, 98.0, 97.0, 96.0, 95.0, 90.0, 85.0, 80.0, 60.0, 0.0];
        (x, y)
    }

    pub fn concave_increasing() -> (Array1<f64>, Array1<f64>) {
        let x = Array1::range(0.0, 10.0, 1.0);
        let y = array![0.0, 60.0, 80.0, 85.0, 90.0, 95.0, 96.0, 97.0, 98.0, 99.0];
        (x, y)
    }

    pub fn bumpy() -> (Array1<f64>, Array1<f64>) {
        let x = Array1::range(0.0, 90.0, 1.0);
        let y = array![
            7305.0, 6979.0, 6666.6, 6463.2, 6326.5, 6048.8, 6032.8, 5762.0, 5742.8, 5398.2, 5256.8,
            5227.0, 5001.7, 4942.0, 4854.2, 4734.6, 4558.7, 4491.1, 4411.6, 4333.0, 4234.6, 4139.1,
            4056.8, 4022.5, 3868.0, 3808.3, 3745.3, 3692.3, 3645.6, 3618.3, 3574.3, 3504.3, 3452.4,
            3401.2, 3382.4, 3340.7, 3301.1, 3247.6, 3190.3, 3180.0, 3154.2, 3089.5, 3045.6, 2989.0,
            2993.6, 2941.3, 2875.6, 2866.3, 2834.1, 2785.1, 2759.7, 2763.2, 2720.1, 2660.1, 2690.2,
            2635.7, 2632.9, 2574.6, 2556.0, 2545.7, 2513.4, 2491.6, 2496.0, 2466.5, 2442.7, 2420.5,
            2381.5, 2388.1, 2340.6, 2335.0, 2318.9, 2319.0, 2308.2, 2262.2, 2235.8, 2259.3, 2221.0,
            2202.7, 2184.3, 2170.1, 2160.0, 2127.7, 2134.7, 2102.0, 2101.4, 2066.4, 2074.3, 2063.7,
            2048.1, 2031.9
        ];
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    fn assert_array1_abs_diff_eq(arr1: &Array1<f64>, arr2: &Array1<f64>) {
        const EPSILON: f64 = 1e-7;

        assert_eq!(arr1.len(), arr2.len(), "Arrays must have the same length");

        for (a, b) in arr1.iter().zip(arr2.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_noisy_gaussian() {
        let (x, y) = DataGenerator::noisy_gaussian(50.0, 10.0, 100, 42);
        assert_eq!(x.len(), 100);
        assert_eq!(y.len(), 100);

        // Check if x is sorted
        assert!(x.windows(2).into_iter().all(|w| w[0] <= w[1]));

        // Check if y is a sequence from 0 to 0.99 with step 0.01
        let expected_y = Array1::linspace(0.0, 0.99, 100);
        assert_array1_abs_diff_eq(&y, &expected_y);

        // Check the range of x values
        assert!(x.iter().all(|&val| (20.0..=80.0).contains(&val)));
    }

    #[test]
    fn test_figure2() {
        let (x, y) = DataGenerator::figure2();
        assert_eq!(x.len(), 10);
        assert_eq!(y.len(), 10);
        assert_array1_abs_diff_eq(&x, &Array1::linspace(0.0, 1.0, 10));
        let expected_y = &(-1.0 / (&x + 0.1)) + 5.0;
        assert_array1_abs_diff_eq(&y, &expected_y);
    }

    #[test]
    fn test_convex_increasing() {
        let (x, y) = DataGenerator::convex_increasing();
        assert_eq!(x.len(), 10);
        assert_eq!(y.len(), 10);
        assert_eq!(x, Array1::range(0.0, 10.0, 1.0));
        assert_eq!(
            y,
            array![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 40.0, 100.0]
        );
    }

    #[test]
    fn test_convex_decreasing() {
        let (x, y) = DataGenerator::convex_decreasing();
        assert_eq!(x.len(), 10);
        assert_eq!(y.len(), 10);
        assert_eq!(x, Array1::range(0.0, 10.0, 1.0));
        assert_eq!(
            y,
            array![100.0, 40.0, 20.0, 15.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn test_concave_decreasing() {
        let (x, y) = DataGenerator::concave_decreasing();
        assert_eq!(x.len(), 10);
        assert_eq!(y.len(), 10);
        assert_eq!(x, Array1::range(0.0, 10.0, 1.0));
        assert_eq!(
            y,
            array![99.0, 98.0, 97.0, 96.0, 95.0, 90.0, 85.0, 80.0, 60.0, 0.0]
        );
    }

    #[test]
    fn test_concave_increasing() {
        let (x, y) = DataGenerator::concave_increasing();
        assert_eq!(x.len(), 10);
        assert_eq!(y.len(), 10);
        assert_eq!(x, Array1::range(0.0, 10.0, 1.0));
        assert_eq!(
            y,
            array![0.0, 60.0, 80.0, 85.0, 90.0, 95.0, 96.0, 97.0, 98.0, 99.0]
        );
    }

    #[test]
    fn test_bumpy() {
        let (x, y) = DataGenerator::bumpy();
        assert_eq!(x.len(), 90);
        assert_eq!(y.len(), 90);
        assert_eq!(x, Array1::range(0.0, 90.0, 1.0));
        let expected_y = array![
            7305.0, 6979.0, 6666.6, 6463.2, 6326.5, 6048.8, 6032.8, 5762.0, 5742.8, 5398.2, 5256.8,
            5227.0, 5001.7, 4942.0, 4854.2, 4734.6, 4558.7, 4491.1, 4411.6, 4333.0, 4234.6, 4139.1,
            4056.8, 4022.5, 3868.0, 3808.3, 3745.3, 3692.3, 3645.6, 3618.3, 3574.3, 3504.3, 3452.4,
            3401.2, 3382.4, 3340.7, 3301.1, 3247.6, 3190.3, 3180.0, 3154.2, 3089.5, 3045.6, 2989.0,
            2993.6, 2941.3, 2875.6, 2866.3, 2834.1, 2785.1, 2759.7, 2763.2, 2720.1, 2660.1, 2690.2,
            2635.7, 2632.9, 2574.6, 2556.0, 2545.7, 2513.4, 2491.6, 2496.0, 2466.5, 2442.7, 2420.5,
            2381.5, 2388.1, 2340.6, 2335.0, 2318.9, 2319.0, 2308.2, 2262.2, 2235.8, 2259.3, 2221.0,
            2202.7, 2184.3, 2170.1, 2160.0, 2127.7, 2134.7, 2102.0, 2101.4, 2066.4, 2074.3, 2063.7,
            2048.1, 2031.9
        ];
        assert_array1_abs_diff_eq(&y, &expected_y);
    }
}

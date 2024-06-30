## kneed

![build](https://github.com/vihu/kneed/actions/workflows/rust.yml/badge.svg)

This is a pure rust implementation of [Knee-point detection](https://raghavan.usc.edu//papers/kneedle-simplex11.pdf).

The code here aims to be a 1:1 match of [kneed](https://pypi.org/project/kneed/).

### Usage

General usage:

```rust
// Provide your x: Vec<f64> and y: Vec<f64>
let x = [1.0, 2.0, 3.0];
let y = [10.0, 20.0, 30.0];
let params = KneeLocatorParams::new(
    ValidCurve::Concave,
    ValidDirection::Increasing,
    InterpMethod::Interp1d,
);

// Instantiate KneeLocator
let kl = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params);

// After instantiation, you can invoke the following:
// kl.knee
// kl.knee_y
// kl.norm_knee
// kl.norm_knee_y
// kl.elbow()
// kl.norm_elbow()
// kl.elbow_y()
// kl.norm_elbow_y()
// kl.all_elbows()
// kl.all_norm_elbows()
// kl.all_elbows_y()
// kl.all_norm_elbows_y()
```

Example from the paper:

```rust
let (x, y) = DataGenerator::figure2();

let params = KneeLocatorParams::new(
    ValidCurve::Concave,
    ValidDirection::Increasing,
    InterpMethod::Interp1d,
);
let kneedle = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params);

assert_relative_eq!(0.222222222222222, kneedle.knee.unwrap());
assert_relative_eq!(1.8965517241379306, kneedle.knee_y.unwrap());
```

### Credits

All credit for the python implementation goes to [Kevin Arvai](https://github.com/arvkevi).

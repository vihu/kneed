## kneed

![build](https://github.com/vihu/kneed/actions/workflows/rust.yml/badge.svg)

This is a pure rust implementation of [Knee-point detection](https://raghavan.usc.edu//papers/kneedle-simplex11.pdf).

The code here aims to be a 1:1 match of [kneed](https://pypi.org/project/kneed/).

### Usage

#### Automatic shape detection (recommended)

The easiest way to use the library is with automatic shape detection, which analyzes your data
to determine the curve direction and type:

```rust
use kneed::knee_locator::KneeLocator;

let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
let y = vec![0.0, 60.0, 80.0, 85.0, 90.0, 95.0, 96.0, 97.0, 98.0, 99.0];

// Auto-detect curve shape and find the knee
let kl = KneeLocator::auto(x, y, 1.0).unwrap();

println!("Knee detected at x = {:?}", kl.knee);  // Some(2.0)
```

#### Manual parameter specification

For full control, you can manually specify the curve parameters:

```rust
use kneed::knee_locator::{KneeLocator, KneeLocatorParams, ValidCurve, ValidDirection, InterpMethod};

let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

let params = KneeLocatorParams::new(
    ValidCurve::Concave,
    ValidDirection::Increasing,
    InterpMethod::Interp1d,
);

let kl = KneeLocator::new(x, y, 1.0, params).unwrap();

// Available methods after instantiation:
// kl.knee           - The detected knee point (x-value)
// kl.knee_y         - The y-value at the knee
// kl.norm_knee      - Normalized knee x-value
// kl.norm_knee_y    - Normalized knee y-value
// kl.elbow()        - Alias for knee
// kl.all_elbows()   - All detected elbows (in online mode)
```

#### Shape detection utility

You can also use the shape detection function directly:

```rust
use kneed::shape_detector::find_shape;
use kneed::knee_locator::{ValidDirection, ValidCurve};

let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![1.0, 1.5, 1.8, 1.9, 2.0];

let (direction, curve) = find_shape(&x, &y);
assert_eq!(direction, ValidDirection::Increasing);
assert_eq!(curve, ValidCurve::Concave);
```

### Example from the paper

```rust
use kneed::knee_locator::{KneeLocator, KneeLocatorParams, ValidCurve, ValidDirection, InterpMethod};

// Figure 2 data from the Kneedle paper
let x: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
let y: Vec<f64> = x.iter().map(|&xi| -1.0 / (xi + 0.1) + 5.0).collect();

let params = KneeLocatorParams::new(
    ValidCurve::Concave,
    ValidDirection::Increasing,
    InterpMethod::Interp1d,
);
let kneedle = KneeLocator::new(x, y, 1.0, params).unwrap();

assert!((kneedle.knee.unwrap() - 0.222222222222222).abs() < 1e-10);
assert!((kneedle.knee_y.unwrap() - 1.8965517241379306).abs() < 1e-10);
```

### Credits

All credit for the python implementation goes to [Kevin Arvai](https://github.com/arvkevi).

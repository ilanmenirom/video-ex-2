# Shared starting point for HW2 (2026/2027)

This package implements exactly three functions in the original 2026/2027
`lucas_kanade.py` skeleton:

1. `build_pyramid(image, num_levels)`
2. `lucas_kanade_step(I1, I2, window_size)`
3. `warp_image(image, u, v)`

The other functions remain the supplied exercise stubs. This is a common base
for the two partners, not a completed assignment or a complete stabilization
pipeline. Both partners should start from the same file.

## Use

- Extract the ZIP and put `lucas_kanade.py` beside the original assignment's
  main scripts and input files. If you already wrote part of your solution,
  copy only these three function definitions into your existing file.
- Keep the assignment's constants and imports. No extra runtime dependency is
  needed beyond its original `requirements.txt`.
- Fill in your IDs as instructed by the assignment.
- Run the one-step portion of `main_river.py` up to the supplied "ONE STEP
  LUCAS KANADE ENDS HERE" marker. The full LK and video sections still need
  implementation by the partners.

## Changes relative to the old solution

- Pyramid: use the supplied `PYRAMID_FILTER`, symmetric-boundary convolution,
  and `[::2, ::2]` decimation. Do not use OpenCV pyramid functions. The result
  has `num_levels + 1` images and starts with an independent copy of the input.
- LK step: use the supplied, unscaled derivative filters on I2. Cast both
  images to float64 BEFORE computing I2 - I1, avoiding uint8 wraparound.
  Solve each complete window using `np.linalg.lstsq`; retain zero flow for
  rank-deficient windows or a failed solve. Never stop processing a row
  because an earlier pixel had zero/undefined flow. Uncomputed borders stay
  zero. `du` is horizontal (columns), `dv` is vertical (rows).
- Warp: sample the source image at `(column + u, row + v)` using the required
  SciPy `griddata` with linear interpolation and NaN fill. Resize each flow
  component with `cv2.resize` when needed and scale horizontal values by the
  width ratio, vertical values by the height ratio. Replace NaN samples with
  the original image at those output coordinates. Use array operations rather
  than per-pixel Python loops for preparing the grids and filling holes.

All three functions preserve their inputs. Output flow and warped-image
arrays use floating point; do not cast them to uint8 during estimation.

## Verification

From this directory, after installing the original requirements:

```bash
python verify_basic_functions.py
python verify_basic_functions.py --data-dir /path/to/original/ex2
```

Verified on 2026-11-20 with Python 3.12.14, NumPy 2.2.3, SciPy 1.15.2,
OpenCV 4.11.0 and tqdm 4.67.1. The dependency versions match the supplied
requirements for these packages; the assignment specifies Python 3.11,
which was not available in the verification runtime. Syntax was also checked
against Python 3.11's grammar.

- 16 synthetic tests passed: odd-sized pyramid levels, filter coefficients,
  symmetric edge behavior, isolated copies, no pyrDown call, known exact
  linearized motion, singular windows followed by valid windows, uint8 input,
  sampling direction, fractional shifts, different flow/image resolutions,
  independent axis scaling, and input preservation.
- Physical small translations in horizontal, vertical and combined directions
  all showed lower alignment error after a single step.
- The supplied river images (252 x 316), window_size=5: true float MSE fell
  from 364.802807 to 298.714754, an 18.1161% reduction after one step.
- The supplied main_river.py one-step portion ran and generated its PNG/GIF
  outputs. An AST comparison confirmed that only these three functions were
  changed relative to the original skeleton.
- The full optical-flow/stabilization pipeline and its performance targets
  have not been implemented or verified in this package.

### MSE arithmetic issue in the supplied main scripts

The original main_river.py calculates the original-image MSE directly on uint8
arrays, so subtraction/squaring wraps around. For the supplied river images
it prints 46.847472 before alignment, although the true float MSE is
364.802807. The warped image is floating point, so its reported MSE of
298.714754 is not affected by that wraparound. Comparing those two printed
numbers directly therefore gives a misleading conclusion.

For a separate diagnostic, convert BOTH images before subtraction:

```python
first = first_image.astype(np.float64)
second = second_image.astype(np.float64)
mse = ((first - second) ** 2).mean()
```

The supplied video MSE function has the same uint8 arithmetic issue. The
assignment's published thresholds/reference numbers come from its own
evaluation context; do not replace them with the float-MSE figures above.
This package does not change either main script or its evaluation function.

## Provenance

Adapted with corrections from the older shared student solution:
https://drive.google.com/file/d/1J5cT1c6rCXppga3GvfwYNI8nyj1NndJj/view

Matched against the HW2_2026.pdf instructions and the original
lucas_kanade.py skeleton supplied with the current assignment:
https://moodle.tau.ac.il/mod/assign/view.php?id=94705

No claim is made that the old solution's grade or correctness was verified.

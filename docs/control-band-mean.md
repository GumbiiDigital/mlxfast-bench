# Control band mean

## The problem

The candidate leg runs the submitted engine. A control leg runs the reference engine on the same box.

The control leg shows whether the box is healthy.

Some control legs are noisy. One slow leg can refuse a job, even when the other control legs are normal.

Calibration is stored timing from healthy control runs. The control band sets the permitted difference from that timing.

## The switch

`control_band_mode` selects one of two rules:

- `per_leg` checks each control leg by itself. The first failed check refuses the job.
- `mean_of_legs` checks the arithmetic mean after all pairs finish.

A pair is one control leg followed by one candidate leg for the same prompt.

A prompt is the input that the engine processes.

An axis is one timing measure: prefill or decode.

The arithmetic mean is the sum of the values divided by the number of values.

Each control leg gives a prefill ratio and a decode ratio. A ratio of 1.00 equals the calibration mean.

Each ratio uses the calibration for its prompt. benchd calculates one arithmetic mean for each axis.

Prefill processes the input. Decode creates the output tokens.

The paired path refuses a fixture that does not declare bands.

The CUDA track fixture declares a 3% decode band and a 5% prefill band.

## Turn it on

`mean_of_legs` is off by default. If the field is absent, benchd uses `per_leg`.

Add this field to the top-level JSON object of the track fixture in the engine repository.

```json
{
  "control_band_mode": "mean_of_legs"
}
```

## A refusal

If either mean exceeds its upper limit, benchd refuses the job with `SERIAL-CONTROL-LEGS-OUTSIDE-BAND`.

The refusal names the axis, measured ratio, limit, and calibration box. The job seals no score.

When the mean check refuses a job, benchd seals the last pair's timings as the baseline leg figures.

It does not seal the mean timings.

In `mean_of_legs`, every candidate leg runs before the mean check.

A candidate start-up, measurement, or strict-token error therefore takes priority over the mean refusal.

The mean refusal comes before the final floor, correctness, and token checks.

## Trade-off

Each pair still uses its own control leg as the score denominator.

A slow control leg that passes the mean check increases that pair's speedup.

With `official_pair_combine` set to `mean`, this also increases the sealed score.

The mean check does not enforce the lower limit. Thus, a fast control leg can offset a slow control leg.

`per_leg` checks each control leg by itself. A fast control leg cannot offset a slow control leg.

With one pair, `mean_of_legs` uses the same band limits and verdict as `per_leg`.

It still runs the candidate before the band check.

A failed one-pair check uses `SERIAL-CONTROL-LEG-OUTSIDE-BAND`.

## Evidence

Tests cover the default, both modes, invalid values, prompt rotation, both axes, and one-pair behavior.

Fifteen mutation checks each inserted a deliberate fault. Every fault caused a test failure.

### Offline replay

An offline replay scored 30 recorded jobs on one CUDA box both ways. Each job had three control legs. The candidate was the unchanged engine. The replay looks only at the control legs.

With the stored three-pass calibration, `per_leg` refused 5 of 30 jobs. `mean_of_legs` refused 0 of 30 jobs.

With a new six-pass calibration, both rules refused 0 of 30 jobs.

### Real jobs

Real jobs ran on one CUDA box with this benchd. They compared the unchanged dev engine with the serial control.

- Six pairs: 6 of 6 jobs passed, 0 jobs were refused, and each job took 7.8 minutes.
- Nine pairs: 6 of 6 jobs passed, 0 jobs were refused, and each job took 11.6 minutes.

These tests used one box and small samples. They do not predict the result for every box.

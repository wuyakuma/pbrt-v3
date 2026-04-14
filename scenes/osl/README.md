# OSL Regression Scenes

These scenes are used for Phase 1 OSL regression checks.

## Render Commands

Use an OSL-enabled build:

- `pbrt scenes/osl/closure-basic.pbrt`
- `pbrt scenes/osl/groupspec-basic.pbrt`

## Baseline Workflow

1. Render each scene and keep generated EXR output.
2. Convert EXR to PNG using `imgtool` for easier diffs.
3. Compare against baseline names listed in `baselines.json`.
4. If behavior changes intentionally, regenerate baselines and update this folder.

## Notes

- These baselines intentionally track:
  - shader/group/layer assignment behavior,
  - output fallback behavior,
  - closure-path stability for `Material "osl"`.

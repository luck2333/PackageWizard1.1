# MPD side-view matching bypass

## Background
`MPD` orchestrates per-view pairing of ruler lines and OCR annotations using `match_pairs_data`. For the side view, recent runs showed desired OCR boxes being dropped during the matching step.

## Change
- For the side view only, the `match_pairs_data` call inside `MPD` is skipped. The original `side_ocr_data` now flows directly into `match_pairs_data_angle`, preserving the detected boxes while still allowing angle-based processing.

## Expected behavior
- Inputs and outputs of `MPD` remain unchanged in shape and type.
- Side-view OCR entries are no longer filtered by the ruler-annotation matcher; downstream logic receives the raw side-view OCR detections (plus any angle-based augmentation).

## Files touched
- `package_core/PackageExtract/get_pairs_data_present5_test.py`: side-view `match_pairs_data` invocation commented out/bypassed.

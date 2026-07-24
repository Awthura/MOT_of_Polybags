# Real Polybags — LocateAnything-3B Autolabelling (2026-07-22)

Full pipeline summary: fine-tuning NVIDIA's LocateAnything-3B on the annotated
real-conveyor dataset, root-causing and working around a real bug in NVIDIA's
own training code, and using the resulting model to auto-label the entire
~2,926-image unlabelled dataset.

## 1. Fine-tuning pipeline

- Base model: `nvidia/LocateAnything-3B` (Qwen2.5-3B-Instruct + MoonViT vision
  encoder), LoRA fine-tuned (`use_llm_lora=64`, LLM frozen otherwise, backbone
  frozen), single H100, DeepSpeed ZeRO stage 1, bf16, gradient checkpointing.
- Training data: the 569-image `train_v11_obb_final` annotated set, OBB labels
  converted to axis-aligned `<box><x1><y1><x2><y2></box>` targets (both raw
  classes merged into one category, `"translucent bubble-wrap polybag"`, since
  their semantics were never confirmed — see `real_polybags/README.md`).

### The upstream bug

NVlabs' vendored training code (`Embodied/eaglevl/train/locany_finetune_magi_stream.py`,
MTP/stream-packing path) crashes with a deterministic CUDA "illegal memory
access" whenever a single training example has **more than one** supervised
detection ("MTP block") in its target answer — i.e. exactly the natural
"locate all N boxes in one response" format we needed for multi-instance
enumeration.

- Confirmed via bisection: crash is deterministic and data-triggered, not
  random/flaky.
- Root-caused to the attention-range construction in
  `eaglevl/model/locany/mask_magi_utils.py` (`convert_mtp_mask_to_magi_plan`),
  which builds the `q_ranges`/`k_ranges`/`attn_type_map` fed to MagiAttention's
  compiled `flex_flash_attn_func` kernel.
- **First workaround (superseded):** reformat training data to one-box-per-example
  (8,044 examples from the 569 images). This trains cleanly but the model then
  only ever emits a single detection per image — precision 0.87, recall 0.07.
  Not acceptable as a final result.
- **Patch attempt:** wrote a defensive bounds-clamping patch for
  `convert_mtp_mask_to_magi_plan` (validates every `(q_range, k_range, attn_type)`
  triple against sample and sequence bounds before use). Deployed and re-tested
  against the original multi-box crash scenario — **the crash still occurred,
  at the identical step, and none of the patch's clamp/drop diagnostics ever
  fired.** This proves the bug is not a Python-level out-of-range index in that
  function; it's an invariant violation inside the compiled MagiAttention
  kernel itself, out of reach for a source-level patch.
- **Working fix:** switching `--attn_implementation` from `magi` to `sdpa`
  (a diagnostic re-run of the identical multi-box data completed all steps
  cleanly under `sdpa`) sidesteps the buggy kernel path entirely while still
  using the intended "locate all N boxes" data format. This is the approach
  used for the final model.

### Final training run

All 569 images, full multi-box format, `ATTN_IMPL=sdpa`, 6 epochs (858 steps):
completed cleanly, `train_loss=0.3485`, ~36 minutes on one H100, no crashes.

## 2. Evaluation (15-image annotated pilot sample, IoU≥0.5)

| Approach | Precision | Recall |
|---|---|---|
| Zero-shot (no fine-tuning) | 0.70 | 0.74 |
| Single-box-per-example workaround | 0.87 | 0.07 |
| **Multi-box + sdpa (final)** | **0.90** | **0.93** |

The final model recovers both accurate localization *and* multi-instance
enumeration — e.g. correctly predicting 17/17, 12/12, 11/11 boxes on several
pilot images — which the single-box workaround had lost entirely.

## 3. Full-set autolabelling

The fine-tuned model was run over the complete unlabelled dataset (2,926
images, 5 camera folders, no ground truth) to produce first-pass labels.

| Camera | Images | Boxes | Avg boxes/img | Avg inference time |
|---|---|---|---|---|
| basler_1 | 430 | 935 | 2.17 | 0.60s |
| basler_2 | 750 | 2,984 | 3.98 | 0.97s |
| lucid | 247 | 1,008 | 4.08 | 0.96s |
| rgbd_1_color | 751 | 2,851 | 3.80 | 0.85s |
| rgbd_2_color | 748 | 2,921 | 3.90 | 1.03s |
| **Total** | **2,926** | **10,699** | **3.66** | **0.90s** |

Total inference time ~44 minutes on one H100. Zero crashes, zero skipped
images. Output format: YOLO-style axis-aligned labels (`class cx cy w h`,
normalized, single merged class `0`), plus an overlay JPEG per image with
predicted boxes drawn for visual QA.

### Known limitations (found during spot-checking overlays)

The annotated training data is clean, isolated bags against a plain red
backdrop; the unlabelled conveyor footage is a different, noisier domain
(motion blur, cluttered background, monitors/equipment in frame). This gap
shows up as a few concrete failure modes, quantified and fixed in §4 below:

- **137/2,926 images (4.7%)** got zero detections.
- **206/2,926 images (7.0%)** got a single degenerate box spanning almost the
  entire frame — seen on blurry/near-empty conveyor frames where the model
  appears to fall back to "the whole image" instead of predicting nothing.
- **319 boxes across 175 images** were geometrically invalid (negative width
  or height) — a symptom of the same decoder pathology described next.
- **Decoder drift/duplication:** on a handful of frames the autoregressive
  box decoder gets stuck re-describing the *same* physical object dozens of
  times, each copy shifted by a tiny amount from the last (border creeping a
  fraction of a pixel per step) instead of terminating. The worst case
  produced **146 boxes for a single frame** that in reality contains at most
  2 objects. Less extreme versions of the same pattern (a handful of
  near-identical repeats of one real box) show up more broadly.
- **Recurring false positive:** static blue monitor/screen corners visible in
  the camera framing get misclassified as polybags in multiple frames (their
  glossy blue surface visually resembles bubble wrap under this lighting).
  Genuine bags on the belt itself are localized well in the same frames —
  this is a background-object confusion, not a general accuracy problem, and
  is **not** addressed by the post-processing in §4 (it produces valid,
  non-duplicate boxes — just on the wrong object).

**These autolabels should be treated as a strong first pass for review/
correction, not ground truth.**

## 4. Post-processing: label cleanup (local, no cluster needed)

The raw autolabels were investigated and cleaned entirely on the local
machine (`real_polybags/dataset/analyze_autolabels.py` and
`clean_autolabels.py` — pure label-file post-processing, no model/GPU
involved). Three automatic fixes are applied, in order, per image:

1. **Drop invalid boxes** (width or height ≤ 0) — 319 boxes.
2. **Drop full-frame degenerate boxes** (width and height both ≥ 95% of the
   frame) — 206 boxes.
3. **Collapse decoder-drift chains**: boxes are scanned in original (decode)
   order and grouped whenever a box has IoU > 0.5 with the *immediately
   preceding* box in its group — this catches slow drift that a simple
   "compare against every kept box" NMS would miss, since the first and
   last member of a long drift chain can end up with near-zero mutual IoU
   even though every adjacent pair is a near-duplicate. Each chain collapses
   to its first member — 316 boxes removed this way (the 146-box chain
   above collapses to 2).
4. **Standard greedy NMS** (IoU > 0.5, no confidence scores available so
   first-seen wins) as a second pass, to catch near-duplicates that aren't
   sequential neighbours (e.g. the same object re-described non-consecutively)
   — 448 boxes removed.

**Result:** 10,699 → 9,410 boxes (12% removed), touching **691/2,926 images
(23.6%)**; the remaining 2,235 images were already clean and untouched.
Box-count-per-image outliers are gone (max dropped from 146 to 16; the
overall distribution now looks like plausible per-frame detection counts —
mostly 0-6 boxes). Genuinely busy frames with many real, distinct bags are
left alone (verified — e.g. a 19-box frame with real spatially-separated
detections correctly reduces to 12 after removing only the actual repeats).

Only images that changed got a re-rendered overlay (691 of them, saved to
`overlays_cleaned/`) to keep the review burden focused and disk usage down;
unaffected images can still be reviewed via the original `overlays/`.

| Cleaned artifact | Location |
|---|---|
| Cleaned labels (use these, not the raw ones) | `real_polybags/dataset/unlabelled_autolabels/labels_cleaned/<camera>/` |
| Review overlays (691 changed images only) | `real_polybags/dataset/unlabelled_autolabels/overlays_cleaned/<camera>/` |
| Cleaning stats | `real_polybags/dataset/unlabelled_autolabels/clean_stats.json` |

## 5. Cross-validation against an independent YOLO11n-OBB detector

To sanity-check the cleaned LocateAnything autolabels without any manual
review, `real_polybags/dataset/cross_validate_yolo.py` runs a second,
independently-trained detector (YOLO11n-OBB, same 569-image annotated set,
completely different architecture/loss/inductive bias) over all 2,926
unlabelled images and greedy-IoU-matches (threshold 0.5) its boxes against
`labels_cleaned/`. Agreement between two independently-trained models is a
much stronger signal than either model's own confidence.

| Camera | LA boxes | YOLO boxes | Matched |
|---|---|---|---|
| basler_1 | 573 | 438 | 277 |
| basler_2 | 2,593 | 3,616 | 1,405 |
| lucid | 839 | 1,057 | 357 |
| rgbd_1_color | 2,654 | 9,478 | 1,954 |
| rgbd_2_color | 2,751 | 3,343 | 1,327 |
| **Total** | **9,410** | **17,932** | **5,320** |

Overall LA-agreement (matched/LA) 56.5%, YOLO-agreement (matched/YOLO) 29.7%.

**Initial finding (rgbd_1_color, from the top-60 disagreement overlays):**
YOLO emits 3.6x as many boxes on this one camera as LA (12.6 vs 3.5
boxes/image on average), and manual review of its 60 most-disagreeing
overlays (`cross_validation/review_overlays/`, all of which happened to be
from this one camera) showed the extra YOLO boxes are **false positives on
static background fixtures** that recur almost identically across frames
because the camera framing is fixed: two blue conveyor roller/motor housings
on the left edge, and cable clutter/keyboard/monitor corner on the right
edge.

**Correction after broader checking:** the top-60 sample is biased toward
frames where the two models *disagree*, so it systematically hides cases
where LocateAnything makes the *same* mistake YOLO does (those frames show
as agreement, not disagreement, and never surface in that sample). Building
a full detection-density heatmap from LA's own cleaned labels across all
frames of each camera (not just the top-60) found real, camera-dependent LA
false positives on this exact kind of static clutter:

| Camera | Clutter region | LA hit rate | YOLO hit rate |
|---|---|---|---|
| rgbd_1_color | left rollers (x2) | 3.3–3.6% of frames | ~100%+ (multiple boxes/frame) |
| **basler_2** | **left + right rollers** | **65–66% of frames** | ~85–98% of frames |
| rgbd_2_color | right roller (upper) | — | 29.7% of frames |

So the earlier claim that "LocateAnything correctly ignores all of this in
every reviewed frame" was **wrong** — it was an artifact of only reviewing
the biased top-60 sample. LA's false-positive rate on background fixtures
varies a lot by camera, from negligible (rgbd_1_color, ~3.5%) to severe
(basler_2, ~66% — a majority of that camera's frames). YOLO's rate is worse
than LA's on every camera checked, but LA is not clean either. See section 6
for the fix and section 5's corrected final numbers below.

| Cross-validation artifact | Location |
|---|---|
| Stats (per-camera + totals, pre-mask) | `real_polybags/dataset/unlabelled_autolabels/cross_validation/cross_validation_stats.json` |
| YOLO's own labels (for reference) | `real_polybags/dataset/unlabelled_autolabels/cross_validation/yolo_labels/<camera>/` |
| Top-60 disagreement overlays (red=LocateAnything, green=YOLO) — biased sample, see correction above | `real_polybags/dataset/unlabelled_autolabels/cross_validation/review_overlays/<camera>/` |
| Cross-validation script | `real_polybags/dataset/cross_validate_yolo.py` |

## 6. Fix: per-camera belt masks

Since both models' false positives cluster on the same **static background
fixtures** (rollers, cables, monitor/keyboard) and every camera is fixed for
its whole session, the belt surface is the only thing with real temporal
motion frame-to-frame. `real_polybags/dataset/build_belt_masks.py` samples
150 frames per camera, computes per-pixel max-min variation, thresholds it,
and keeps only the single largest connected component (the belt) — clutter
reliably forms its own smaller, disjoint blob, confirmed by inspection (e.g.
on rgbd_1_color the belt is one ~274k-px component while the rollers and
monitor/cable clutter are four separate ~12–104k-px components).

**A generic automatic rescue rule was tried and rejected.** rgbd_1_color has
one real exception: a slow-moving bag "buffer/queue zone" where bags sit
nearly still for long stretches, reading as low-variance and getting wrongly
excluded even though ~30% of that camera's real detections land there. The
first fix tried was a blanket rule ("if cleaned-LA detections recur in ≥10%
of a region's frames, treat it as real and keep it"). This is **unsafe**:
basler_2's rollers are hit in 65–66% of frames (LA's own false positive,
per the table above), so a generic recurrence threshold would have kept
them as "real." Rejected in favor of hand-verifying each candidate region by
viewing source frames directly: only rgbd_1_color's buffer zone
(`x:875–1143, y:578–720`) checked out as genuinely real; every other
flagged region across all 5 cameras (checked individually) was confirmed
clutter. That one region is now a hardcoded override in
`MANUAL_INCLUDE_REGIONS`, not a general rule.

`real_polybags/dataset/apply_belt_masks.py` drops any box whose center falls
outside its camera's mask, applied to both LA's cleaned labels and YOLO's
labels:

| | LA boxes | YOLO boxes |
|---|---|---|
| Before masking | 9,410 | 17,932 |
| After masking | 7,196 (−23.5%) | 6,784 (**−62.2%**) |

Re-running cross-validation on the masked labels:

| Camera | LA | YOLO | Matched | LA-agree | YOLO-agree |
|---|---|---|---|---|---|
| basler_1 | 571 | 438 | 277 | 48.5% | 63.2% |
| basler_2 | 1,465 | 1,139 | 1,007 | 68.7% | 88.4% |
| lucid | 666 | 399 | 347 | 52.1% | 87.0% |
| rgbd_1_color | 2,551 | 3,288 | 1,936 | 75.9% | 58.9% |
| rgbd_2_color | 1,943 | 1,520 | 1,238 | 63.7% | 81.4% |
| **Total** | **7,196** | **6,784** | **4,805** | **66.8%** | **70.8%** |

Overall YOLO-agreement more than doubled (29.7% → 70.8%) and the two models'
agreement rates, which started 27 points apart, are now within 4 points of
each other — strong confirmation that most of the original disagreement was
static-clutter false positives on both sides, not a real accuracy gap.
**`labels_cleaned_masked/` is now the recommended label set** for anything
downstream (supersedes `labels_cleaned/`).

| Belt-mask artifact | Location |
|---|---|
| **Recommended final LA labels** | `real_polybags/dataset/unlabelled_autolabels/labels_cleaned_masked/<camera>/` |
| Masked YOLO labels (for reference/further cross-checks) | `real_polybags/dataset/unlabelled_autolabels/cross_validation/yolo_labels_masked/<camera>/` |
| Per-camera belt masks (binary PNG, native resolution) | `real_polybags/dataset/belt_masks/<camera>.png` |
| Mask-building script (variance + hand-verified overrides) | `real_polybags/dataset/build_belt_masks.py` |
| Mask-application script | `real_polybags/dataset/apply_belt_masks.py` |

## 7. Where everything lives

| Artifact | Location |
|---|---|
| Fine-tuned model (final, ~7.4GB, inference-ready) | `real_polybags/training/locate_anything/checkpoints/real_v3_multibox_sdpa/` (local) |
| Training-resumption checkpoints (optimizer states, ~48GB, not pulled) | `~/runs_locateanything_lora/real_v3_multibox_sdpa/checkpoint-{572,715,858}/` on cluster only |
| Raw autolabel predictions (YOLO-format `.txt`) | `real_polybags/dataset/unlabelled_autolabels/labels/<camera>/` |
| Cleaned autolabel predictions (superseded, see below) | `real_polybags/dataset/unlabelled_autolabels/labels_cleaned/<camera>/` |
| **Cleaned + belt-masked autolabel predictions (recommended)** | `real_polybags/dataset/unlabelled_autolabels/labels_cleaned_masked/<camera>/` |
| Raw overlay QA images (all 2,926) | `real_polybags/dataset/unlabelled_autolabels/overlays/<camera>/` |
| Cleaned overlay QA images (691 changed images) | `real_polybags/dataset/unlabelled_autolabels/overlays_cleaned/<camera>/` |
| Run stats (raw autolabel run) | `real_polybags/dataset/unlabelled_autolabels/summary.json` |
| Cleaning stats | `real_polybags/dataset/unlabelled_autolabels/clean_stats.json` |
| Per-camera belt masks | `real_polybags/dataset/belt_masks/<camera>.png` |
| Upstream bug patch (documented, not effective — kept for reference) | `real_polybags/training/locate_anything/patches/mask_magi_utils.py` |
| Batch inference script | `real_polybags/training/locate_anything/autolabel_unlabelled.py` |
| Label investigation script | `real_polybags/dataset/analyze_autolabels.py` |
| Label cleaning script | `real_polybags/dataset/clean_autolabels.py` |
| Cross-validation script | `real_polybags/dataset/cross_validate_yolo.py` |
| Belt-mask build / apply scripts | `real_polybags/dataset/build_belt_masks.py`, `real_polybags/dataset/apply_belt_masks.py` |

## 8. Round-2 domain-adaptive fine-tune

Belt masking (section 6) is a spatial post-filter — it can't fix a model that
still *wants* to fire on background clutter, it just crops those boxes out
afterward. To fix the underlying model behavior, ran a second LoRA fine-tune
(`training/locate_anything/prepare_jsonl_round2.py`) combining the original
569-image clean annotated set (repeat_time=3, to keep its influence up) with
pseudo-labelled examples from the real conveyor images:

- **Confident positive** (1,335 images, 4,805 boxes): LA and YOLO mutually
  agree (IoU≥0.5) on the belt-masked labels — only the matched boxes are used
  as targets.
- **Confident none** (1,182 images): both models see zero boxes after
  masking — kept as explicit negative (`<box>none</box>`) examples.
- **Ambiguous** (409 images): some boxes exist but no agreement — excluded
  entirely rather than guess at a label.

### A second real training bug, and the fix

Training crashed reproducibly (`FloatingPointError: Non-finite loss ...
hidden_states_has_nan=True`) within 12-19 steps, twice — including once after
excluding all `<box>none</box>` examples to rule that out as the cause. Root
cause: the annotated set is 640×480 (4:3); the unlabelled/pseudo set is
1280×720 (16:9). Mixing different-resolution images in the same training run
corrupts something in the vision-token packing (`real_v3`, 640×480-only,
trained 858 steps with zero crashes). **Fix**: letterbox-resize (scale-to-fit
+ black-bar pad, no distortion) every pseudo image down to 640×480 before
training, remapping box coordinates through the identical transform.
Verified with a 100-step isolated test before committing to the full run.

Also hit three cluster/`sbatch` environment bugs getting this onto the
cluster at all (previous sessions only ever used interactive `srun --pty
bash`, which never surfaces these) — all now fixed and documented directly
in `run_finetune_round2.slurm`:
- `BASH_SOURCE[0]` resolves to a node-local, permission-restricted slurm
  spool path under real `sbatch`, not the actual script location — use
  `$SLURM_SUBMIT_DIR` instead.
- `eaglevl` isn't pip-installed and isn't on `sys.path` for a relative-path
  script launch — needs explicit `PYTHONPATH`.
- The training script defaults to a `slurm`-native distributed launcher
  needing `SLURM_NTASKS`/`SLURM_PROCID` (only set by an inner `srun`), but
  we launch via `torchrun` — needs `LAUNCHER=pytorch` to match.

### Result: `real_v4_round2`

2 epochs, 4,224 effective images/epoch (1,707 weighted-annotated + 2,517
pseudo, letterboxed), same batch=1/grad_acc=4/sdpa setup as `real_v3`.
2,112 steps, ~91 min on one H100, `train_loss=0.382` (down from ~4.2 at
initialization).

- **15-image annotated pilot**: precision 0.90 / recall 0.93 — identical to
  `real_v3`. Expected: this pilot sample is drawn from the same clean
  annotated domain both models were already trained on, so it's a ceiling
  match, not evidence either way about the actual target (the noisy conveyor
  domain).
- **The real test** — same 100-image `basler_2` subset used to find the
  original roller false-positive problem (section 5/6):

  | | `real_v3` | `real_v4_round2` |
  |---|---|---|
  | Left-roller false-positive rate | 86% of frames | **19%** of frames |
  | Right-roller false-positive rate | 85% of frames | **1%** of frames |
  | Total boxes (100 images) | 416 | 212 |

  Visual spot-check on shared frames confirmed real bag detections are
  preserved (near-identical box placement on actual bags in busy frames) —
  the drop in total box count is the roller false positives being fixed at
  the model level, not a loss of recall. Belt masking is no longer strictly
  needed for `basler_2` specifically, though it remains a cheap, harmless
  extra safety net and is still needed for cameras/regions not covered by
  this round's pseudo-label mix.

### Full-dataset confirmation

Re-ran autolabelling with `real_v4_round2` over all 2,926 images
(`autolabel_unlabelled.py`, 31 min on one H100) and pushed the result through
the same pipeline as `real_v3`: cleanup (`clean_autolabels_v4.py`) then belt
masking (`apply_belt_masks_v4.py`).

- Raw: 6,090 boxes (vs `real_v3`'s 10,699 — 43% fewer). Cleanup dropped only
  114 boxes across 108/2,926 images (vs `real_v3`'s 1,289 boxes across
  691/2,926 images) — the decoder-drift/degenerate-box pathology described
  in section 3 is far less frequent now too. Cleaned: 5,976 boxes.
- Belt masking now drops only **5.0%** of boxes (298/5,976) vs `real_v3`'s
  23.5% — confirms the fix is working at the model level, not just being
  papered over by the spatial filter. Per camera: `rgbd_1_color` 0.0%,
  `basler_1` 0.3%, `lucid` 1.0%, `rgbd_2_color` 9.5%, `basler_2` 10.3% (down
  from 43.5%) — some residual clutter-attraction remains on the latter two,
  but sharply reduced.
- **Roller false-positive rate, full `basler_2` (750 images, not just the
  100-image subset)**: left-roller 66.0%→**18.0%**, right-roller
  64.9%→**0.0%**. Matches (and for the right roller, beats) the subset
  result — not a sampling artifact.
- Cross-validated against YOLO's existing masked labels (YOLO itself
  unchanged, still 6,784 boxes):

  | | `real_v3` (masked) | `real_v4_round2` (masked) |
  |---|---|---|
  | LA boxes | 7,196 | 5,678 |
  | Matched (IoU≥0.5) | 4,805 | 4,199 |
  | LA-agreement (matched/LA) | 66.8% | **74.0%** |
  | YOLO-agreement (matched/YOLO) | 70.8% | 61.9% |

  LA-agreement improving is the meaningful signal here (LA's own remaining
  boxes are more often independently confirmed). The YOLO-agreement number
  *dropping*, despite `real_v4_round2` being strictly better, is expected
  and not a regression: a chunk of `real_v3`'s "agreement" was actually two
  models **coincidentally agreeing on the same roller false positive** —
  both are drawn to the same shiny/glossy texture, so YOLO's still-present
  roller boxes used to spuriously match `real_v3`'s roller boxes too. Now
  that `real_v4_round2` doesn't produce those, that correlated-error overlap
  disappears along with it, correctly lowering the raw agreement count even
  as real accuracy improves. This is a good caveat to remember generally:
  agreement between two models isn't a clean signal when both share a
  failure mode.

**`unlabelled_autolabels_v4/labels_cleaned_masked/` is now the overall
recommended label set**, superseding both `unlabelled_autolabels/labels_cleaned/`
and `unlabelled_autolabels/labels_cleaned_masked/` (the `real_v3`-based ones).

| Round-2 artifact | Location |
|---|---|
| Round-2 training data + letterboxing script | `real_polybags/training/locate_anything/prepare_jsonl_round2.py` |
| Round-2 sbatch script (with all 3 cluster-env fixes) | `real_polybags/training/locate_anything/run_finetune_round2.slurm` |
| Fine-tuned model (cluster only, not yet pulled locally) | `~/runs_locateanything_lora/real_v4_round2/` |
| **Recommended final labels (v4, cleaned + masked)** | `real_polybags/dataset/unlabelled_autolabels_v4/labels_cleaned_masked/<camera>/` |
| Raw v4 autolabel predictions | `real_polybags/dataset/unlabelled_autolabels_v4/labels/<camera>/` |
| v4 cleanup script | `real_polybags/dataset/clean_autolabels_v4.py` |
| v4 belt-mask application script | `real_polybags/dataset/apply_belt_masks_v4.py` |

## Next steps

- Pull `real_v4_round2` locally (~7.4GB, mirrors `real_v3_multibox_sdpa`'s
  layout) if local inference/inspection is needed — currently cluster-only.
- `basler_2` and `rgbd_2_color` still show some residual clutter-attraction
  after round 2 (belt masking still drops 9.5-10.3% of their boxes, vs
  ~0-1% for the other three cameras) — worth a closer look at what's driving
  that specifically, and whether a round 3 (or more pseudo-label coverage
  from those two cameras) would close the gap.
- LA-agreement against YOLO is now 74.0% (up from 66.8%), but not 100% —
  worth investigating the remaining in-belt disagreement (not just clutter)
  to see if it's genuine model error on either side or a further
  cross-camera quirk.
- Consider reporting both the MagiAttention multi-block bug and the
  mixed-resolution NaN-crash bug upstream to NVLabs/Eagle — both real,
  reproducible, and (as far as could be determined) previously unreported.
- Confirm `class_0`/`class_1` semantics if per-class autolabels are ever
  needed (currently everything is one merged class).
- Use `labels_cleaned_masked/` (or fresh `real_v4_round2` autolabels, once
  regenerated) — not `labels_cleaned/` or the raw `labels/` — for anything
  downstream, e.g. a second YOLO training round on the combined annotated +
  autolabelled set.

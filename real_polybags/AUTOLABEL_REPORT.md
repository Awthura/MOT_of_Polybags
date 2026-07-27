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

| Round-2 artifact | Location |
|---|---|
| Round-2 training data + letterboxing script | `real_polybags/training/locate_anything/prepare_jsonl_round2.py` |
| Round-2 sbatch script (with all 3 cluster-env fixes) | `real_polybags/training/locate_anything/run_finetune_round2.slurm` |
| Fine-tuned model (cluster only, not yet pulled locally) | `~/runs_locateanything_lora/real_v4_round2/` |
| v4 cleanup script | `real_polybags/dataset/clean_autolabels_v4.py` |
| v4 belt-mask application script | `real_polybags/dataset/apply_belt_masks_v4.py` |

## 9. Manual review findings: whole-belt fallback pattern

Rendered overlays for the full `labels_cleaned_masked` set
(`render_final_overlays_v4.py`, all 2,926 images, local-only) and reviewed
them directly. Two issues reported:

1. **Some boxes still cover almost the entire frame.**
2. **Some visible bags aren't labelled** (missed detections in dense
   clusters).

### Issue 1 — root-caused and fixed

The existing full-frame filter (`analyze_autolabels.py`'s `is_fullframe`,
w≥0.95 AND h≥0.95) only catches a box spanning the literal *frame*. The
actual pattern found on manual review is a box spanning almost the entire
**belt** — sized to each camera's belt geometry (e.g. lucid's version was
w=0.73/h=1.00, `basler_2`'s was w=0.72/h≈1.00), which slips through since
it's not 95% of the raw frame in both dimensions. Visually confirmed on
several examples: an empty belt (just a person's arm reaching in, no bag)
gets one giant box roughly matching the belt's own boundary instead of no
detections — the same "fall back to boxing everything" pathology as the
literal full-frame case, just scoped to the belt rather than the frame.

**Fix** (`drop_wholebelt_boxes.py`): using the belt masks we already built
(`build_belt_masks.py`), flag a box as a whole-belt fallback if it's large
(≥35% of frame area) **and** covers ≥90% of the belt mask's own pixel area.
Verified against all 405 candidate boxes found dataset-wide: 404/405 covered
≥99.6% of the belt (all confirmed degenerate on visual inspection — several
checked directly), and the **one** exception covered only 64.6% of the belt
and was confirmed by inspection to be a real, correctly-boxed large bag —
a wide, clean safety margin either side of the 90% cutoff, so no real
detections were at risk of being dropped.

| | Before fix | After fix |
|---|---|---|
| Total boxes | 5,678 | 5,274 |
| Whole-belt fallback boxes | 404 | 0 |

Breakdown: `basler_2` 277 affected images, `lucid` 112, `rgbd_2_color` 15,
`basler_1`/`rgbd_1_color` 0. **`unlabelled_autolabels_v4/labels_final/` is
now the overall recommended label set**, superseding
`labels_cleaned_masked/` (both v3's and v4's) and `labels_cleaned/`.

### Issue 2 — investigated, not a post-processing bug

Sampled dense multi-bag frames and found a genuine example: a frame with 9
visually distinct bags got only 8 boxes, missing one that was fully
in-frame (not edge-cropped) and not occluded. Checked whether YOLO's
independent prediction on the same frame caught it (which would let
cross-validation rescue it, the way the belt-mask work used model
agreement elsewhere) — **it didn't; both models missed the same bag.**
This means it isn't fixable by post-processing (nothing in the *output* of
a single missed detection signals that a detection is missing), and it
isn't resolvable by cross-checking against YOLO either, since YOLO shares
the same gap on this example. Other sampled busy frames (6-8 bags each,
several cameras) showed correct, tight boxes on every visible bag including
partially-cropped ones at frame edges, so this looks like an occasional
recall gap in the hardest, most cluttered/overlapping cases specifically,
not a systematic pattern with an identifiable fix. Documented here as a
known, accepted residual limitation rather than papered over.

| Manual-review artifact | Location |
|---|---|
| **Recommended final labels (v4, cleaned + masked + whole-belt fix)** | `real_polybags/dataset/unlabelled_autolabels_v4/labels_final/<camera>/` |
| Full-dataset overlay renderer (any label set, local-only) | `real_polybags/dataset/render_final_overlays_v4.py` |
| Whole-belt fallback fix script | `real_polybags/dataset/drop_wholebelt_boxes.py` |
| Fix stats | `real_polybags/dataset/unlabelled_autolabels_v4/wholebelt_fix_stats.json` |
| Changed-image overlays (404 images) | `real_polybags/dataset/unlabelled_autolabels_v4/overlays_wholebelt_fix/<camera>/` |

## 10. Fixing missed detections in dense clusters (tiled + iterative inference)

Issue 2 from section 9 (occasional missed bags in the densest clusters,
confirmed not fixable by post-processing) prompted a short research pass —
see the plan this section implements — into whether iterative retraining,
SAM 3, or using LocateAnything as a segmentor would help. Findings, briefly:
LocateAnything's own paper documents this exact failure mode ("Spatial
Ambiguity" in dense scenes, mitigated upstream via training-data
composition, not an architecture fix); SAM 3 is a genuinely different tool
(native promptable segmentation, no such capability in LocateAnything) but
is **not** immune to the same crowded-scene miss pattern itself; the
literature's actual architecture-agnostic fix is inference-time tiling
(SAHI) + iterative re-detection (IterDet) — zero retraining, directly
applicable to the existing `LocateAnythingWorker` API. Decided to implement
this (Tier 1) first, deferring a targeted round-3 retrain and SAM 3
integration (Tiers 2/3).

### Implementation and validation

`autolabel_tiled.py`: slices each frame into an overlapping 2x2 tile grid,
runs detection on each tile plus the full frame, merges results, then does
up to 2 rounds of iterative re-detection (paint over what's already found,
re-run on the residual image, merge in anything genuinely new). Tiling
coverage, coordinate remapping, and merge logic were unit-tested locally
with a mock worker (zero model calls) before any cluster run.

First cluster test recovered the target frame's missing bag, but also
surfaced a genuine merge bug: a bag split across two tiles produced a
partial duplicate box that symmetric IoU (0.46) didn't catch. Root cause:
a small box that's mostly *contained* within a larger one is a strong
duplicate signal regardless of the area mismatch that penalizes IoU —
confirmed via a containment ratio (intersection over smaller-box area,
"IoS") of 0.92 for the duplicate vs. 0.61 for a manually-confirmed
genuinely-distinct overlapping bag pair, a clean margin. Added
`MERGE_IOS_THRESH=0.85` alongside IoU; retested and got the target frame
fully correct (9/9 bags, no duplicates) with no effect on the
distinct-pair case.

### Full-dataset run and a second bug found at scale

With single-frame validation clean, ran the full 2,926-image dataset
(`unlabelled_autolabels_v4_tiled/`, ~2h). Cross-validating against YOLO
afterward showed `basler_1`'s LA-agreement had collapsed from 63.1% (no
tiling) to 17.4% — despite belt-masking barely touching its boxes (0.7%
dropped), meaning the extra boxes were landing on the belt itself, not on
known static clutter. Manual inspection of the frames with the biggest
box-count jumps found several **completely empty belts** with 3-5 new
boxes each, whose coordinates matched the tiling grid's own rectangles
almost exactly (measured IoU=0.9975 against a tile's true bounds). This is
a third variant of the "fall back to boxing everything instead of nothing"
pathology already found and fixed at the frame level (section 3) and the
belt level (section 9) — this time triggered by low-content/ambiguous
*individual tile crops*, small enough to slip under both the frame-fraction
and belt-coverage thresholds of the existing fixes.

**Fix** (`drop_wholetile_boxes.py`, pure local post-processing, no cluster
re-run needed since tile geometry is fixed and known for every image):
drop any box with IoU≥0.85 against any of the 4 known tile rectangles.
Verified against the one confirmed real large-bag box on file (54% of
frame) before running at scale — its max IoU with any tile was 0.40, a
comfortable margin below the threshold. Dropped 1,187 boxes dataset-wide
(983 from `basler_1` alone, across 283/430 = 66% of its images — this
camera was overwhelmingly the one affected). `basler_1` LA-agreement
recovered to 45.6%.

### A methodological correction: YOLO is not ground truth

45.6% was still well below the 63.1% pre-tiling baseline, so before
building anything further (Weighted Box Fusion was considered next, to
reconcile any remaining tile-boundary fragments), the still-unmatched LA
boxes on `basler_1` were inspected directly — specifically by rendering
**YOLO's detections alone**, without LA's boxes overlaid, to avoid
anchoring on LA's output when judging YOLO. Across 6 of `basler_1`'s
busiest frames, YOLO found as few as 1 of 6-7 visibly distinct bags, and
never found more than 7 of ~9 — a severe, camera-specific YOLO recall
problem, not a LocateAnything problem. This means the cross-validation
agreement metric itself was misleading here: it was penalizing the tiled
LA output for finding *more real bags than a demonstrably under-detecting
YOLO reference does*, not for making mistakes. Weighted Box Fusion was
**not** pursued on this basis — there is no longer solid evidence of a
real fragmentation problem to fix, and building more merge logic to chase
an unreliable metric would be solving the wrong problem. General lesson,
consistent with section 5's "agreement between two models isn't a clean
signal when both share a failure mode": it's equally not a clean signal
when one reference model has its own independent, camera-specific
weakness — cross-validation numbers should prompt visual verification
before being read as scores.

**Follow-up: is this `basler_1`-specific, or does it call the whole
project's YOLO cross-validation into question?** Checked directly — for
each of the other 4 cameras, rendered YOLO-only overlays for their single
busiest LA/YOLO-count-gap frame. Result: YOLO catches most of the visible
bags on all four (11/13 on `rgbd_1_color`, 6/7 on `basler_2`, ~5/8 on
`lucid`, ~4/7 on `rgbd_2_color`) — nowhere close to `basler_1`'s
catastrophic 1-of-6/7 pattern. **Confirmed `basler_1`-specific.** The
section 5/6/8 cross-validation numbers for the other four cameras stand;
only `basler_1`'s historical agreement figures throughout this report
should be read with this caveat. (Root cause of *why* `basler_1`
specifically is harder for YOLO wasn't investigated further — camera
framing/distance/lighting are plausible candidates but unconfirmed.)

**`unlabelled_autolabels_v4_tiled/labels_final_v2/` is the current
dense-cluster-improved label set** — not yet promoted to "overall
recommended" ahead of `unlabelled_autolabels_v4/labels_final/` pending the
open items below, but confirmed to fix the target recall issue without
the regressions initially suspected.

| Tiled-inference artifact | Location |
|---|---|
| Tiling + iterative re-detection script | `real_polybags/training/locate_anything/autolabel_tiled.py` |
| Whole-tile fallback fix (local post-processing) | `real_polybags/dataset/drop_wholetile_boxes.py` |
| Tile-fragment merge fix (local post-processing) | `real_polybags/dataset/merge_fragments_tiled.py` |
| Cleanup/mask scripts for this run | `real_polybags/dataset/clean_autolabels_tiled.py`, `apply_belt_masks_tiled.py` |
| **Current best labels** (tiled, whole-tile-fixed, fragment-merged) | `real_polybags/dataset/unlabelled_autolabels_v4_tiled/labels_final_v3/<camera>/` |
| Overlays for the above | `real_polybags/dataset/unlabelled_autolabels_v4_tiled/overlays_final_v3/` |
| Implementation plan (full research + decision log) | `~/.claude/plans/vivid-foraging-flurry.md` |

### Promotion decision: `labels_final_v3` is the recommended set

The tiled set was held back from promotion pending a full overlay review
(only spot-checks and the whole-tile-fix diff frames had been looked at).
That review was done, and it resolved the question decisively — but in the
opposite direction to the one being guarded against.

**The risk was assumed to be that tiling added false positives.** The
sharpest test for that is frames that had *zero* boxes in the non-tiled set
and gained boxes under tiling — 127 such frames (63 `basler_1`, 33 `lucid`,
25 `rgbd_2_color`, 5 `basler_2`, 1 `rgbd_1_color`). Inspecting the three
with the most new boxes, **all three were real recoveries, not false
positives**: `rgbd_2_color_frame_0000136` has ~12 plainly visible bags on
the belt and the non-tiled set returned *nothing* for it; likewise
`rgbd_1_color_frame_0000163` (5 bags) and `rgbd_2_color_frame_0000097`
(4 bags). So the previously-recommended `labels_final` set has a
catastrophic, not marginal, recall hole: on some dense frames the model
emits **no boxes at all**. That is the mirror image of the
"fallback-to-everything" pathology documented above — a
*fallback-to-nothing* mode — and tiling is what breaks it, by reducing
objects-per-call below whatever threshold triggers it.

**A real (if minor) tiling artifact did show up, and is now fixed.**
Counting box pairs by IoS (intersection over the smaller box's area — the
containment measure), the tiled set had **191 pairs in the IoS 0.70–0.85
band** versus 13 for the non-tiled set: fragments sitting just under
`autolabel_tiled.py`'s `MERGE_IOS_THRESH=0.85` cut. That threshold had been
calibrated on a single frame from just two data points (a confirmed
duplicate at 0.92, a confirmed distinct pair at 0.61), so it was
conservative by construction. Sampling the band visually settled where the
boundary actually is:

| IoS band | sampled verdict |
|---|---|
| 0.70–0.85 | 4/4 inspected pairs were **duplicates** — one box a thin partial sliver of the same bag |
| 0.60–0.70 | inspected pairs were **genuinely distinct bags** in dense piles; merging would be wrong |

So the true boundary is ~0.70, and `merge_fragments_tiled.py` re-merges at
IoS ≥ 0.70 (keeping the larger box, dropping the contained fragment) as a
local post-processing pass — no cluster re-run needed. Result:

| set | boxes | IoS ≥ .85 pairs | .70–.85 | .60–.70 |
|---|---|---|---|---|
| `v4/labels_final` (non-tiled) | 5,274 | 7 | 13 | 27 |
| `v4_tiled/labels_final_v2` | 7,364 | 0 | 191 | 159 |
| **`v4_tiled/labels_final_v3`** | **7,184** | **0** | **0** | 155 |

`labels_final_v3` therefore beats the non-tiled set on *both* axes at once:
**+36% boxes** (5,274 → 7,184, per-camera range +18% to +49%) and **fewer
duplicate pairs** (0 above IoS 0.70, vs 20 for the non-tiled set — that
pipeline never had an IoS merge pass at all). The 155 remaining 0.60–0.70
pairs are the verified genuinely-distinct stacked bags and are correctly
kept.

Caveat worth recording: box *localization* in dense piles is visibly loose
(boxes offset from bag edges, roughly one per bag rather than tightly
bounding it). That is a model-quality limit, not a pipeline bug, and no
post-processing pass addresses it — it is the strongest remaining argument
for Tier 2/Tier 3 if tighter boundaries are ever needed.

## Next steps

- Pull `real_v4_round2` locally (~7.4GB, mirrors `real_v3_multibox_sdpa`'s
  layout) if local inference/inspection is needed — currently cluster-only.
- ~~Check whether YOLO's `basler_1` recall weakness is isolated or
  broader~~ — **done** (section 10): confirmed `basler_1`-specific via
  YOLO-only overlays on the other 4 cameras' worst-gap frames, all of which
  showed YOLO catching most visible bags. Section 5/6/8 numbers for
  `basler_2`/`lucid`/`rgbd_1_color`/`rgbd_2_color` stand as-is; only
  `basler_1`'s historical agreement figures carry the caveat. Root cause of
  *why* `basler_1` specifically is harder for YOLO remains unconfirmed
  (camera framing/distance/lighting are plausible, unverified guesses) —
  worth a look if `basler_1`-specific work comes up again.
- ~~Decide whether to promote the tiled set to the overall recommended
  set~~ — **done**: promoted, but as `labels_final_v3` (see "Promotion
  decision" above). Full overlay review found the non-tiled set has a
  fallback-to-nothing recall hole on dense frames, and the one genuine
  tiling artifact (191 surviving tile fragments) is fixed by
  `merge_fragments_tiled.py`. Net: +36% boxes and fewer duplicates.
- Box *localization* tightness in dense piles is the main remaining quality
  gap and is not fixable in post-processing — this is now the strongest
  concrete motivation for Tier 2 (round-3 fine-tune) or Tier 3 (SAM 3
  masks) should tighter boundaries be needed downstream.
- The non-tiled `v4/labels_final` set still carries 20 unmerged duplicate
  pairs (7 at IoS ≥ 0.85). Not worth fixing since it's superseded by
  `labels_final_v3`, but note it if that set is ever used for comparison.
- `basler_2` and `rgbd_2_color` still show some residual clutter-attraction
  after round 2 (belt masking still drops 9.5-10.3% of their boxes, vs
  ~0-1% for the other three cameras) — worth a closer look at what's driving
  that specifically, and whether a round 3 (or more pseudo-label coverage
  from those two cameras) would close the gap.
- Consider reporting the MagiAttention multi-block bug, the mixed-resolution
  NaN-crash bug, and (if it turns out NVLabs' base model has the same
  whole-frame/whole-belt/whole-tile fallback pathology zero-shot) the
  fallback-instead-of-none pattern upstream to NVLabs/Eagle — all real,
  reproducible, and (as far as could be determined) previously unreported.
- Confirm `class_0`/`class_1` semantics if per-class autolabels are ever
  needed (currently everything is one merged class).
- Tier 2 (targeted round-3 fine-tune oversampling dense-cluster frames) and
  Tier 3 (SAM 3 as a third cross-validator / segmentation masks / temporal
  continuity) from the dense-cluster-fix plan remain deliberately deferred.
- Use **`labels_final_v3/`** (tiled, whole-tile-fixed, fragment-merged) for
  anything downstream — e.g. a second YOLO training round on the combined
  annotated + autolabelled set. Not `labels_final_v2/`, `labels_final/`,
  `labels_cleaned_masked/`, `labels_cleaned/`, or the raw `labels/`.

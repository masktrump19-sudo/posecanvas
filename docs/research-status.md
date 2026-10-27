# Research status

PoseCanvas is a course study of pose-guided anime still-image generation, built around FLUX.1. This page separates the report's design from the current repository implementation.

## What is in the checkout

| Component | Source | Current state |
| --- | --- | --- |
| FLUX text-to-image and image-to-image interfaces | `demo_gr.py`, `demo_st.py` | Inherited base-model demos; they do not call the custom pose adapter. |
| Pose and prop preprocessing | `src/flux/data/preprocess.py` | Keypoint processing, skeleton rendering, and supplied prop-contour rendering. This is not an automatic prop detector. |
| Dataset loading | `src/flux/data/dataset.py` | Image / skeleton / caption loading from local JSON or JSONL. |
| Visual Context Adapter | `src/flux/modules/adapter.py` | Convolutional skeleton encoder producing feature tokens. |
| Skeleton-weighted loss | `src/flux/modules/loss.py` | Mask-weighted MSE utilities and related losses. |
| Custom pose training | `src/flux/train.py` | Scaffold: empty data loader; prediction, loss, backward pass and optimizer update are commented out. |
| Custom pose inference | `src/flux/inference.py` | Scaffold: incomplete component initialization; generation and decoding are commented out. It does not save a generated image. |
| Adapter-to-backbone integration | `src/flux/model.py` | `Flux.forward` has no adapter-feature input yet. |

## How to read the images

The gallery comes from the project technical report provided by the maintainer. It shows selected qualitative examples, not a new run of this checkout and not an independently reproduced benchmark. The comparison figure keeps the original `Baseline (SDXL)` and `Ours (Flux.1)` labels. It does not establish a general performance ranking.

The research design uses a pose map, optional prop outlines, text conditioning and a FLUX.1 backbone. The adapter and weighted-loss modules are useful starting points, but the complete report workflow cannot currently be reproduced by running the custom training or inference scripts unchanged. Video generation and temporal consistency are outside the implemented scope.

## Integration work that would make a useful contribution

1. Connect adapter feature tokens to explicit injection points in `Flux.forward` and the denoising loop.
2. Wire the real data loader, pretrained component loading, flow-matching target and optimizer steps into training.
3. Save and reload compatible adapter checkpoints, then provide a small pose-conditioned inference example.
4. Record seeds, prompts, model identifiers and environment details for reproducible comparisons.

The base FLUX demos can be explored separately using the README instructions. This documentation update was checked against source code; no large-model inference or training run was performed.

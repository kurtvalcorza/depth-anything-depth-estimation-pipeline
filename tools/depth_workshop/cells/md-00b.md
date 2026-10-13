### Key terms

| Term | Meaning in this notebook |
|---|---|
| **Depth / metres** | Distance from the camera to the surface seen at a pixel, in metres. "Distance" can mean either the length of the viewing ray to the surface or the distance along the camera's forward (optical) axis; the two agree at the image centre and differ towards the edges. This notebook has not verified which convention the pinned DIODE references or ZoeDepth's outputs use. |
| **Inverse depth** | `1 / depth`. Larger values mean nearer surfaces. Depth Anything predicts a *relative* inverse depth: its scale and offset are unknown, so it is not in metres. |
| **Reference depth / sensor mask** | The measured depth from the dataset's scanner, and a true/false map of the pixels where the scanner returned a usable value. |
| **AbsRel** | Mean of `abs(predicted − reference) / reference` over the scored pixels; lower is better. |
| **Delta-1** | Fraction of scored pixels whose prediction is within a factor of 1.25 of the reference; higher is better. |
| **Informative image** | An image with at least 1,000 scored pixels covering at least 1% of the image. Other images are still scored and listed, but left out of every mean and count. |
| **Affine alignment** | Fitting a scale and an offset per image so a relative prediction best matches that image's reference. It uses the answer, so it is an evaluation device, not a calibration. |
| **Backbone** | The large first part of the network that turns pixels into general visual features. It stays frozen (unchanged) here. |
| **Neck and head** | The smaller later layers that combine those features and produce the depth map (the DPT neck and head in Depth Anything). Only these are trained here. |
| **Adapter** | The saved file of trained neck/head weights; it is applied on top of the unchanged base model when reloaded. |
| **AdamW** | The optimisation algorithm that updates the trainable weights from the training loss. |
| **Scale-and-shift-invariant loss** | A training loss that first aligns the prediction's scale and offset to the target, so the model is trained on relative structure rather than metres. |
| **Epoch / selection** | One pass over the training views; selection keeps the epoch (0 = untrained) with the lowest validation AbsRel. |
| **Reload parity** | Loading the saved adapter in a new process and checking that it reproduces the same predictions; it shows the file is complete, not that the model is accurate. |

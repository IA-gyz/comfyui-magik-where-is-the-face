# Magik Where Is The Face

Detect female people, detect faces inside their bounding boxes, and report every
original-image quadrant intersected by a face box. No sampling or VAE is used.

## Install

Extract the `magik_where_is_the_face` folder into `ComfyUI/custom_nodes/` and
restart ComfyUI. Requires ComfyUI, Impact Pack, and Impact Subpack for the
Ultralytics Detector Provider. NumPy and Pillow are already used by ComfyUI.
Place both YOLO models in `ComfyUI/models/ultralytics/bbox/`.

## Connect

1. Connect the image to **Magik Where Is The Face → image**.
2. Add **Ultralytics Detector Provider**, select your female-only person model,
   and connect its **BBOX_DETECTOR** output to **female_body_detector**.
3. Add a second provider, select a face-only model suitable for your image style,
   and connect its **BBOX_DETECTOR** output to **face_detector**.
4. Connect **preview** to Preview Image if you want to inspect results.

Both stages use dilation 0 and crop factor **1.0**, fixed in code. The person
image is cropped to its bbox (rounded outward to integer pixels), not an
expanded detailer crop. Face bboxes are translated back to original-image
coordinates; quadrants always refer to the original image.

## Outputs and controls

- **quadrants**: a native ComfyUI INT list via OUTPUT_IS_LIST. Values are unique
  and sorted: 1 top left, 2 top right, 3 bottom right, 4 bottom left.
  Example: `[1, 2]`. No faces produces an empty list. Connecting this to a
  standard INT input makes ComfyUI execute the downstream node per integer;
  a list-aware receiver is needed to consume the entire list at once.
- **details_json**: STRING containing the union of quadrants and per-image
  quadrants, person boxes, face boxes, confidences, and source-person indices.
  Use this if a downstream tool needs a serialized list or batch boundaries.
- **preview**: IMAGE batch. Cyan = female-person boxes; green = face boxes;
  white midlines; yellow quadrant numbers = occupied quadrants.
- **body_threshold / face_threshold**: independent confidence cutoffs, both 0.5.
- **body_min_size / face_min_size**: detector drop_size, defaults 10 and 2.
  Impact's Ultralytics detector keeps boxes whose two dimensions are strictly
  greater than drop_size.

Any positive-area bbox intersection counts. A box touching a midline without
crossing it does not count on the other side. Bboxes include background around
faces: this reports bbox overlap, not exact facial-pixel overlap.

For image batches, the INT list is the union across images; details_json preserves
each image's result and preview preserves batch order. Face detections repeated
inside overlapping person boxes can appear more than once in JSON/preview;
quadrants remain unique.

## Important limitation

The female-only model supplies the female classification. The second detector
finds all faces inside each female-person crop. If a male face lies inside that
same rectangle (overlapping people, crowded scenes), it may also be counted.
This spatial two-stage pipeline cannot guarantee male exclusion in such cases.
Detector misclassification and missed detections also affect results.

## Implementation contract for maintainers

Calls Impact-compatible `detector.detect(image, threshold, dilation, crop_factor,
drop_size=...)`, receiving `(image_shape, SEG_items)`. SEG.bbox is XYXY in the
input image's coordinate system. SEG.crop_region is deliberately not used for
quadrant assignment. No dependency on Impact's internal imports or model loader.
The node accepts ComfyUI RGB tensor batches [B,H,W,3] and returns an IMAGE tensor
on the input device and dtype. Preview creation copies pixels through CPU/Pillow.
Two detector calls are not concurrent: one person pass per image and one face
pass per surviving person. The node does not mutate input images or detector
configuration.

Validation: synthetic detector tests cover coordinate translation, quadrant
boundaries, clipping, multiple bodies, no detections, crop_factor=1.0 and batches.
Actual YOLO model inference must be verified in your ComfyUI installation.

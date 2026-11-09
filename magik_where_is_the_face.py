"""Two-stage Impact Pack BBOX_DETECTOR pipeline; coordinates are XYXY."""
import json
import math

import numpy as np
from PIL import Image, ImageDraw

CROP_FACTOR = 1.0
DILATION = 0
MIN_DROP_SIZE = 1


def clipped_box(box, width, height):
    values = [float(v) for v in box]
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError("Detector returned an invalid XYXY bounding box")
    x1, y1, x2, y2 = values
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def quadrants_for_box(box, width, height):
    """Count positive-area intersection, not mere contact with a midline."""
    x1, y1, x2, y2 = box
    mx, my = width / 2, height / 2
    regions = [(0, 0, mx, my), (mx, 0, width, my),
               (mx, my, width, height), (0, my, mx, height)]
    return [i for i, (a, b, c, d) in enumerate(regions, 1)
            if min(x2, c) > max(x1, a) and min(y2, d) > max(y1, b)]


class MagikWhereIsTheFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "female_body_detector": ("BBOX_DETECTOR",),
            "face_detector": ("BBOX_DETECTOR",),
            "body_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "face_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "body_min_size": ("INT", {"default": 10, "min": 1, "max": 16384}),
            "face_min_size": ("INT", {"default": 2, "min": 1, "max": 16384}),
        }}

    RETURN_TYPES = ("INT", "STRING", "IMAGE")
    RETURN_NAMES = ("quadrants", "details_json", "preview")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "locate"
    CATEGORY = "Magik/Detection"
    DESCRIPTION = ("Detect female people, then faces inside each person box. "
                   "Quadrants: 1 top left, 2 top right, 3 bottom right, 4 bottom left. "
                   "For batches, quadrants is the union; details_json holds per-image results.")

    def locate(self, image, female_body_detector, face_detector,
               body_threshold=0.5, face_threshold=0.5,
               body_min_size=10, face_min_size=2):
        if len(image.shape) != 4 or image.shape[0] < 1 or image.shape[-1] != 3:
            raise ValueError("Expected a nonempty ComfyUI RGB IMAGE batch [B,H,W,3]")
        height, width = int(image.shape[1]), int(image.shape[2])
        if height < 1 or width < 1:
            raise ValueError("Image width and height must be positive")
        records, previews, union = [], [], set()
        for index in range(image.shape[0]):
            frame = image[index:index + 1]
            bodies = female_body_detector.detect(
                frame, body_threshold, DILATION, CROP_FACTOR, drop_size=body_min_size)
            frame_quadrants, faces, body_boxes = set(), [], []
            for body in bodies[1]:
                box = clipped_box(body.bbox, width, height)
                if box is None:
                    continue
                # Outward pixel rounding preserves the entire detected person.
                x1, y1 = math.floor(box[0]), math.floor(box[1])
                x2, y2 = math.ceil(box[2]), math.ceil(box[3])
                body_boxes.append([x1, y1, x2, y2])
                crop = frame[:, y1:y2, x1:x2, :].contiguous()
                detections = face_detector.detect(
                    crop, face_threshold, DILATION, CROP_FACTOR, drop_size=face_min_size)
                for face in detections[1]:
                    local = clipped_box(face.bbox, x2 - x1, y2 - y1)
                    if local is None:
                        continue
                    global_box = [local[0] + x1, local[1] + y1,
                                  local[2] + x1, local[3] + y1]
                    quadrants = quadrants_for_box(global_box, width, height)
                    frame_quadrants.update(quadrants)
                    faces.append({"bbox": global_box, "confidence": float(face.confidence),
                                  "body_index": len(body_boxes) - 1, "quadrants": quadrants})
                del crop, detections
            union.update(frame_quadrants)
            records.append({"image_index": index, "quadrants": sorted(frame_quadrants),
                            "female_body_boxes": body_boxes, "faces": faces})
            pixels = np.uint8(np.clip(frame[0].detach().cpu().numpy(), 0, 1) * 255)
            canvas = Image.fromarray(pixels)
            draw = ImageDraw.Draw(canvas)
            line_width = max(1, round(min(width, height) / 300))
            draw.line([(width / 2, 0), (width / 2, height - 1)], fill="white", width=line_width)
            draw.line([(0, height / 2), (width - 1, height / 2)], fill="white", width=line_width)
            for q, xy in enumerate([(5, 5), (width // 2 + 5, 5),
                                    (width // 2 + 5, height // 2 + 5), (5, height // 2 + 5)], 1):
                draw.text(xy, str(q), fill="yellow" if q in frame_quadrants else "white", stroke_width=1, stroke_fill="black")
            for box in body_boxes:
                draw.rectangle(box, outline="cyan", width=line_width)
            for face in faces:
                draw.rectangle(face["bbox"], outline="lime", width=line_width + 1)
            previews.append(np.asarray(canvas, dtype=np.float32) / 255.0)
            del bodies
        preview = image.new_tensor(np.stack(previews))
        details = {"quadrants": sorted(union), "images": records,
                   "numbering": {"1": "top left", "2": "top right", "3": "bottom right", "4": "bottom left"}}
        return sorted(union), json.dumps(details, ensure_ascii=False), preview

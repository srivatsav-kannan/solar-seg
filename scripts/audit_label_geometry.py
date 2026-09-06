"""Read-only geometry audit of all supplied training annotations; no test labels."""

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

from solarseg.data import CompetitionData, annotation_rle, sha256
from solarseg.metrics import overlap
from solarseg.provenance import utc_now

root = Path(__file__).resolve().parents[1]
data = CompetitionData(root / "data/raw")
areas, bbox_errors, relative_area_errors = [], [], []
short_polygons = nonfinite = outside = empty = 0
for annotation in data.raw["annotations"]:
    for polygon in annotation["segmentation"]:
        points = np.asarray(polygon, dtype=float)
        short_polygons += int(len(points) < 6 or len(points) % 2 != 0)
        nonfinite += int(not np.isfinite(points).all())
        outside += int(((points < 0) | (points > 2048)).any())
    rle = annotation_rle(annotation)
    area = float(coco_mask.area(rle))
    areas.append(area)
    empty += int(area == 0)
    bbox_errors.append(float(np.abs(coco_mask.toBbox(rle) - annotation["bbox"]).max()))
    if area:
        relative_area_errors.append(abs(float(annotation["area"]) - area) / area)

images_with_overlap = pairs_with_overlap = pairs_iou_over_half = 0
for stem in data.by_stem:
    for rles in data.ground_truth(stem).values():
        matrix = overlap(rles, rles)
        upper = matrix[np.triu_indices(len(rles), 1)]
        images_with_overlap += int((upper > 0).any())
        pairs_with_overlap += int((upper > 0).sum())
        pairs_iou_over_half += int((upper > 0.5).sum())

result = {
    "created_at": utc_now(),
    "annotation_sha256": sha256(data.annotation_path),
    "annotations": len(areas),
    "malformed_polygon_arrays": short_polygons,
    "nonfinite_polygon_arrays": nonfinite,
    "polygon_arrays_with_out_of_canvas_vertices": outside,
    "empty_rasterized_instances": empty,
    "raster_area_quantiles_0_25_50_75_100": np.quantile(areas, [0, .25, .5, .75, 1]).tolist(),
    "bbox_max_coordinate_error_quantiles_50_95_100": np.quantile(bbox_errors, [.5, .95, 1]).tolist(),
    "bbox_errors_greater_than_two_pixels": int((np.asarray(bbox_errors) > 2).sum()),
    "raster_instances_below_area_400": int((np.asarray(areas) < 400).sum()),
    "relative_supplied_area_error_quantiles_50_95_100": np.quantile(relative_area_errors, [.5, .95, 1]).tolist(),
    "annotator_records_with_internal_overlap": images_with_overlap,
    "internal_instance_pairs_with_overlap": pairs_with_overlap,
    "internal_instance_pairs_with_iou_over_half": pairs_iou_over_half,
    "scope": "All approved training annotation geometry; no labels changed and no model selected",
    "interpretation": "Polygon rasterization defines masks. Small bbox differences can reflect geometry conventions; larger discrepancies require inspection. Derive future detector boxes from the actual masks.",
}
(root / "reports/label-geometry.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

from ultralytics import YOLO
import cvat_sdk.auto_annotation as cvataa
from PIL import Image

# Load seg model
_model = YOLO("your-seg-model.pt")  # Ensure this is a segmentation model

# CVAT specification (specify that we will use polygons)
spec = cvataa.DetectionFunctionSpec(
    labels=[cvataa.label_spec(name, id) for id, name in _model.names.items()],
)

def _yolo_to_cvat(results):
    """
    Converts YOLOv8-seg results into a format understandable by CVAT (polygons).
    """
    for result in results:
        if result.masks is not None:  # Check if there are masks
            for mask, label in zip(result.masks.xy, result.boxes.cls):
                # The mask contains a list of (x, y) coordinates for the polygon
                # Convert to a flat list: [x1, y1, x2, y2, ...]
                polygon_points = [float(coord) for point in mask for coord in point]
                yield cvataa.polygon(int(label.item()), polygon_points)

def detect(context, image):
    """
    Function for model inference on an image.
    """
    conf_threshold = 0.5 if context.conf_threshold is None else context.conf_threshold
    # Add retina_masks=True for more accurate masks
    return list(_yolo_to_cvat(_model.predict(source=image, verbose=False, conf=conf_threshold, retina_masks=True)))

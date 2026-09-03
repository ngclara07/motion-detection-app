# this is the core implementation; it contains the computer-vision algorithm, independent of Streamlit

from __future__ import annotations

import json
import math
import subprocess
import time

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch

from rfdetr.assets.coco_classes import COCO_CLASSES
from trackers import ByteTrackTracker


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class PipelineConfig:
    """
    Configuration for Exercise 1.1 moving-vehicle detection.

    Defaults preserve the parameters used in the Jupyter
    Notebook implementation.
    """

    # General processing
    processing_width: int = 960

    # RF-DETR
    confidence_threshold: float = 0.50
    allowed_vehicle_classes: tuple[str, ...] = (
        "car",
        "truck",
        "bus",
    )

    # Frame differencing
    frame_difference_threshold: int = 18
    gaussian_blur_size: tuple[int, int] = (5, 5)
    frame_difference_dilation_iterations: int = 2

    # MOG2
    background_history: int = 500
    background_variance_threshold: int = 28
    background_detect_shadows: bool = False
    background_warmup_frames: int = 45
    background_warmup_learning_rate: float = 0.04
    background_learning_rate: float = 0.001

    # Motion validation
    minimum_box_area: int = 800
    box_inset_ratio: float = 0.12
    minimum_frame_diff_occupancy: float = 0.015
    minimum_background_occupancy: float = 0.020
    minimum_combined_occupancy: float = 0.010
    minimum_largest_component_area: int = 80

    # Morphology
    morph_kernel_size: tuple[int, int] = (5, 5)
    morph_open_iterations: int = 1
    morph_close_iterations: int = 2
    morph_dilate_iterations: int = 1

    # ROI
    roi_top: float = 0.47

    # Trajectory confirmation
    track_history_length: int = 8
    minimum_track_observations: int = 4
    minimum_track_displacement: float = 8.0

    # ByteTrack
    tracker_lost_buffer: int = 30

    # Behaviour
    suppress_detections_during_warmup: bool = True


@dataclass
class FrameResult:
    """
    Output generated for one processed frame.
    """

    frame_number: int
    total_frames: int

    annotated_frame: np.ndarray
    frame_difference_mask: np.ndarray
    background_mask: np.ndarray
    combined_motion_mask: np.ndarray

    confirmed_rows: list[dict[str, Any]]

    active_vehicle_count: int
    unique_vehicle_count: int

    processing_fps: float
    warmup: bool

    progress: float


# ============================================================
# BASIC HELPERS
# ============================================================


def resize_to_max_width(
    image: np.ndarray,
    maximum_width: int,
) -> np.ndarray:
    """
    Resize an image while preserving its aspect ratio.
    """

    if image.shape[1] <= maximum_width:
        return image

    scale = maximum_width / image.shape[1]

    new_height = int(
        round(image.shape[0] * scale)
    )

    return cv2.resize(
        image,
        (maximum_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def get_class_name(class_id: int) -> str:
    """
    Convert a COCO class identifier into a lowercase class name.
    """

    class_id = int(class_id)

    if isinstance(COCO_CLASSES, dict):

        # Support integer and string dictionary keys.
        value = COCO_CLASSES.get(class_id)

        if value is None:
            value = COCO_CLASSES.get(str(class_id))

        if value is None:
            return f"class_{class_id}"

        return str(value).lower()

    try:
        return str(
            COCO_CLASSES[class_id]
        ).lower()

    except (IndexError, KeyError, TypeError):
        return f"class_{class_id}"


def empty_detections(
    detections: sv.Detections,
) -> sv.Detections:
    """
    Return a zero-length Detections object while preserving
    the original structure.
    """

    if len(detections) == 0:
        return detections

    mask = np.zeros(
        len(detections),
        dtype=bool,
    )

    return detections[mask]


# ============================================================
# PREPROCESSING
# ============================================================


def preprocess_grayscale(
    frame_bgr: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """
    Convert BGR frame to blurred grayscale.
    """

    gray = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        config.gaussian_blur_size,
        0,
    )

    return gray


def calculate_frame_difference(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """
    Consecutive-frame differencing.
    """

    absolute_difference = cv2.absdiff(
        previous_gray,
        current_gray,
    )

    _, difference_mask = cv2.threshold(
        absolute_difference,
        config.frame_difference_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    difference_mask = cv2.dilate(
        difference_mask,
        kernel,
        iterations=(
            config.frame_difference_dilation_iterations
        ),
    )

    return difference_mask


# ============================================================
# ROI
# ============================================================


def create_roi_polygon(
    frame_width: int,
    frame_height: int,
    roi_top: float,
) -> np.ndarray:
    """
    Construct the normalized Main Street ROI.

    Default:
        upper boundary = 47% down the image
        bottom boundary = image bottom
    """

    polygon = np.array(
        [
            [0.00, roi_top],
            [1.00, roi_top],
            [1.00, 1.00],
            [0.00, 1.00],
        ],
        dtype=np.float32,
    )

    polygon[:, 0] *= frame_width - 1
    polygon[:, 1] *= frame_height - 1

    return np.round(
        polygon
    ).astype(np.int32)


def create_roi_mask(
    frame_shape: tuple[int, ...],
    polygon: np.ndarray,
) -> np.ndarray:

    mask = np.zeros(
        frame_shape[:2],
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [polygon],
        255,
    )

    return mask


def point_inside_roi(
    point: tuple[float, float],
    roi_polygon: np.ndarray,
) -> bool:

    return (
        cv2.pointPolygonTest(
            roi_polygon,
            (
                float(point[0]),
                float(point[1]),
            ),
            False,
        )
        >= 0
    )


def detection_bottom_centre(
    box: np.ndarray,
) -> tuple[float, float]:

    x1, y1, x2, y2 = box

    return (
        float((x1 + x2) / 2.0),
        float(y2),
    )


# ============================================================
# SEMANTIC VEHICLE FILTERING
# ============================================================


def filter_vehicle_classes(
    detections: sv.Detections,
    config: PipelineConfig,
) -> sv.Detections:
    """
    Keep only selected motor-vehicle classes.
    """

    if len(detections) == 0:
        return detections

    if detections.class_id is None:
        return empty_detections(detections)

    allowed = {
        name.lower()
        for name in config.allowed_vehicle_classes
    }

    keep_mask = np.array(
        [
            get_class_name(class_id) in allowed
            for class_id in detections.class_id
        ],
        dtype=bool,
    )

    return detections[keep_mask]


# ============================================================
# MOTION VALIDATION
# ============================================================


def crop_box_region(
    mask: np.ndarray,
    box: np.ndarray,
    inset_ratio: float = 0.0,
) -> np.ndarray | None:

    height, width = mask.shape[:2]

    x1, y1, x2, y2 = np.round(
        box
    ).astype(int)

    box_width = max(
        1,
        x2 - x1,
    )

    box_height = max(
        1,
        y2 - y1,
    )

    inset_x = int(
        box_width * inset_ratio
    )

    inset_y = int(
        box_height * inset_ratio
    )

    x1 = max(
        0,
        x1 + inset_x,
    )

    y1 = max(
        0,
        y1 + inset_y,
    )

    x2 = min(
        width,
        x2 - inset_x,
    )

    y2 = min(
        height,
        y2 - inset_y,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return mask[
        y1:y2,
        x1:x2,
    ]


def mask_occupancy(
    mask_region: np.ndarray | None,
) -> float:

    if (
        mask_region is None
        or mask_region.size == 0
    ):
        return 0.0

    return (
        float(
            cv2.countNonZero(mask_region)
        )
        / float(mask_region.size)
    )


def largest_component_area(
    mask_region: np.ndarray | None,
) -> int:

    if (
        mask_region is None
        or mask_region.size == 0
    ):
        return 0

    num_labels, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_region,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return 0

    return int(
        stats[
            1:,
            cv2.CC_STAT_AREA,
        ].max()
    )


def filter_motion_validated_vehicles(
    detections: sv.Detections,
    frame_difference_mask: np.ndarray,
    background_mask: np.ndarray,
    combined_motion_mask: np.ndarray,
    roi_polygon: np.ndarray,
    config: PipelineConfig,
) -> sv.Detections:
    """
    Retain only semantically detected vehicles that contain
    sufficient frame-difference and MOG2 motion evidence.
    """

    if len(detections) == 0:
        return detections

    keep: list[bool] = []

    for box in detections.xyxy:

        x1, y1, x2, y2 = box

        box_area = max(
            0.0,
            float(
                (x2 - x1)
                * (y2 - y1)
            ),
        )

        if (
            box_area
            < config.minimum_box_area
        ):
            keep.append(False)
            continue

        road_contact_point = (
            detection_bottom_centre(box)
        )

        if not point_inside_roi(
            road_contact_point,
            roi_polygon,
        ):
            keep.append(False)
            continue

        diff_region = crop_box_region(
            frame_difference_mask,
            box,
            config.box_inset_ratio,
        )

        background_region = (
            crop_box_region(
                background_mask,
                box,
                config.box_inset_ratio,
            )
        )

        combined_region = (
            crop_box_region(
                combined_motion_mask,
                box,
                config.box_inset_ratio,
            )
        )

        diff_occupancy = mask_occupancy(
            diff_region
        )

        background_occupancy = (
            mask_occupancy(
                background_region
            )
        )

        combined_occupancy = (
            mask_occupancy(
                combined_region
            )
        )

        component_area = (
            largest_component_area(
                combined_region
            )
        )

        is_moving = (
            diff_occupancy
            >= config.minimum_frame_diff_occupancy

            and background_occupancy
            >= config.minimum_background_occupancy

            and combined_occupancy
            >= config.minimum_combined_occupancy

            and component_area
            >= config.minimum_largest_component_area
        )

        keep.append(is_moving)

    return detections[
        np.asarray(
            keep,
            dtype=bool,
        )
    ]


# ============================================================
# MORPHOLOGICAL MOTION FUSION
# ============================================================


def refine_motion_mask(
    mask: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        config.morph_kernel_size,
    )

    refined = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=(
            config.morph_open_iterations
        ),
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=(
            config.morph_close_iterations
        ),
    )

    refined = cv2.dilate(
        refined,
        kernel,
        iterations=(
            config.morph_dilate_iterations
        ),
    )

    return refined


# ============================================================
# VISUALISATION
# ============================================================


def draw_dashed_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    colour: tuple[int, int, int] = (
        40,
        55,
        245,
    ),
    thickness: int = 2,
    dash_length: int = 18,
    gap_length: int = 10,
) -> None:

    points = polygon.reshape(
        -1,
        2,
    )

    for i in range(len(points)):

        start = points[i].astype(
            np.float32
        )

        end = points[
            (i + 1) % len(points)
        ].astype(np.float32)

        vector = end - start

        length = float(
            np.linalg.norm(vector)
        )

        if length <= 0:
            continue

        direction = vector / length

        distance = 0.0

        while distance < length:

            dash_start = (
                start
                + direction * distance
            )

            dash_end = (
                start
                + direction
                * min(
                    distance + dash_length,
                    length,
                )
            )

            cv2.line(
                image,
                tuple(
                    np.round(
                        dash_start
                    ).astype(int)
                ),
                tuple(
                    np.round(
                        dash_end
                    ).astype(int)
                ),
                colour,
                thickness,
                cv2.LINE_AA,
            )

            distance += (
                dash_length
                + gap_length
            )

    label_position = (
        int(points[0][0] + 12),
        max(
            24,
            int(points[0][1] - 12),
        ),
    )

    cv2.putText(
        image,
        "MAIN STREET ROI",
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        colour,
        2,
        cv2.LINE_AA,
    )


def draw_track_trajectory(
    image: np.ndarray,
    history: deque,
) -> None:

    points = list(history)

    if len(points) < 2:
        return

    for index in range(
        1,
        len(points),
    ):

        p1 = tuple(
            np.round(
                points[index - 1]
            ).astype(int)
        )

        p2 = tuple(
            np.round(
                points[index]
            ).astype(int)
        )

        cv2.line(
            image,
            p1,
            p2,
            (255, 210, 30),
            2,
            cv2.LINE_AA,
        )


def draw_detection(
    image: np.ndarray,
    box: np.ndarray,
    label: str,
) -> None:

    x1, y1, x2, y2 = np.round(
        box
    ).astype(int)

    colour = (
        70,
        220,
        110,
    )

    # Bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        colour,
        2,
    )

    # Corner accents
    corner = 12
    thickness = 3

    cv2.line(
        image,
        (x1, y1),
        (x1 + corner, y1),
        colour,
        thickness,
    )

    cv2.line(
        image,
        (x1, y1),
        (x1, y1 + corner),
        colour,
        thickness,
    )

    cv2.line(
        image,
        (x2, y1),
        (x2 - corner, y1),
        colour,
        thickness,
    )

    cv2.line(
        image,
        (x2, y1),
        (x2, y1 + corner),
        colour,
        thickness,
    )

    # Label dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        label,
        font,
        0.48,
        1,
    )

    label_top = max(
        0,
        y1 - text_height - baseline - 12,
    )

    label_bottom = y1

    cv2.rectangle(
        image,
        (
            x1,
            label_top,
        ),
        (
            x1 + text_width + 14,
            label_bottom,
        ),
        (7, 18, 28),
        -1,
    )

    cv2.rectangle(
        image,
        (
            x1,
            label_top,
        ),
        (
            x1 + text_width + 14,
            label_bottom,
        ),
        colour,
        1,
    )

    cv2.putText(
        image,
        label,
        (
            x1 + 7,
            y1 - 7,
        ),
        font,
        0.48,
        colour,
        1,
        cv2.LINE_AA,
    )


def create_diagnostic_view(
    frame_difference_mask: np.ndarray,
    background_mask: np.ndarray,
    combined_motion_mask: np.ndarray,
) -> np.ndarray:

    diff_bgr = cv2.cvtColor(
        frame_difference_mask,
        cv2.COLOR_GRAY2BGR,
    )

    bg_bgr = cv2.cvtColor(
        background_mask,
        cv2.COLOR_GRAY2BGR,
    )

    fused_bgr = cv2.cvtColor(
        combined_motion_mask,
        cv2.COLOR_GRAY2BGR,
    )

    diagnostic = np.hstack(
        [
            diff_bgr,
            bg_bgr,
            fused_bgr,
        ]
    )

    width = (
        frame_difference_mask.shape[1]
    )

    labels = [
        (
            "FRAME DIFFERENCE",
            10,
        ),
        (
            "MOG2 FOREGROUND",
            width + 10,
        ),
        (
            "FUSED MOTION",
            2 * width + 10,
        ),
    ]

    for text, x in labels:

        cv2.rectangle(
            diagnostic,
            (x - 5, 5),
            (x + 220, 35),
            (5, 15, 25),
            -1,
        )

        cv2.putText(
            diagnostic,
            text,
            (x, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (40, 220, 255),
            1,
            cv2.LINE_AA,
        )

    return diagnostic


# ============================================================
# VIDEO METADATA
# ============================================================


def get_video_metadata(
    video_path: Path | str,
) -> dict[str, Any]:

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"OpenCV could not open video:\n"
            f"{video_path}"
        )

    try:

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

    finally:
        capture.release()

    if (
        not math.isfinite(fps)
        or fps <= 0
    ):
        fps = 30.0

    duration = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


# ============================================================
# BROWSER VIDEO TRANSCODING
# ============================================================


def transcode_h264(
    input_path: Path,
) -> Path:
    """
    Convert OpenCV's MP4V output to browser-friendly H.264.

    If imageio-ffmpeg or H.264 conversion is unavailable,
    the original MP4 is returned.
    """

    try:
        import imageio_ffmpeg
    except ImportError:
        return input_path

    if not input_path.is_file():
        return input_path

    output_path = input_path.with_name(
        input_path.stem
        + "_browser.mp4"
    )

    try:

        ffmpeg = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if (
            output_path.is_file()
            and output_path.stat().st_size > 0
        ):
            return output_path

    except Exception:
        pass

    return input_path


# ============================================================
# COMPLETE EXERCISE 1.1 PIPELINE
# ============================================================


class VehicleDetectionPipeline:
    """
    Hybrid moving-vehicle detection pipeline:

        frame difference
              +
             MOG2
              ↓
         fused motion
              +
          Main Street ROI
              +
          RF-DETR Nano
              ↓
        motion validation
              ↓
         ByteTrackTracker
              ↓
       trajectory confirmation
    """

    def __init__(
        self,
        model,
        config: PipelineConfig,
    ) -> None:

        self.model = model
        self.config = config

        self.track_histories = (
            defaultdict(
                lambda: deque(
                    maxlen=(
                        self.config
                        .track_history_length
                    )
                )
            )
        )

        self.seen_track_ids: set[int] = (
            set()
        )

        self.track_classes: dict[
            int,
            str,
        ] = {}

        self.track_confidences = (
            defaultdict(list)
        )

        self.history: list[
            dict[str, Any]
        ] = []

        self.evidence_paths: list[
            Path
        ] = []

        self.diagnostic_paths: list[
            Path
        ] = []

        self.output_video_path: (
            Path | None
        ) = None

        self.raw_output_video_path: (
            Path | None
        ) = None

        self.summary: dict[
            str,
            Any,
        ] = {}

    # --------------------------------------------------------

    def _create_background_subtractor(
        self,
    ):

        return (
            cv2.createBackgroundSubtractorMOG2(
                history=(
                    self.config
                    .background_history
                ),
                varThreshold=(
                    self.config
                    .background_variance_threshold
                ),
                detectShadows=(
                    self.config
                    .background_detect_shadows
                ),
            )
        )

    # --------------------------------------------------------

    def _confirm_tracks(
        self,
        tracked_detections: sv.Detections,
    ) -> tuple[
        sv.Detections,
        dict[int, float],
    ]:

        if len(tracked_detections) == 0:
            return (
                tracked_detections,
                {},
            )

        if (
            tracked_detections.tracker_id
            is None
        ):
            return (
                empty_detections(
                    tracked_detections
                ),
                {},
            )

        keep: list[bool] = []

        displacement_by_id: dict[
            int,
            float,
        ] = {}

        for box, tracker_id in zip(
            tracked_detections.xyxy,
            tracked_detections.tracker_id,
        ):

            if tracker_id is None:
                keep.append(False)
                continue

            tracker_id = int(
                tracker_id
            )

            # Current trackers package uses -1 for
            # detections not assigned to a valid track.
            if tracker_id < 0:
                keep.append(False)
                continue

            x1, y1, x2, y2 = box

            centre = np.array(
                [
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                ],
                dtype=np.float32,
            )

            history = (
                self.track_histories[
                    tracker_id
                ]
            )

            history.append(centre)

            if (
                len(history)
                < self.config
                .minimum_track_observations
            ):
                keep.append(False)
                continue

            start = np.asarray(
                history[0],
                dtype=np.float32,
            )

            end = np.asarray(
                history[-1],
                dtype=np.float32,
            )

            displacement = float(
                np.linalg.norm(
                    end - start
                )
            )

            displacement_by_id[
                tracker_id
            ] = displacement

            confirmed = (
                displacement
                >= self.config
                .minimum_track_displacement
            )

            keep.append(confirmed)

        keep_mask = np.asarray(
            keep,
            dtype=bool,
        )

        return (
            tracked_detections[
                keep_mask
            ],
            displacement_by_id,
        )

    # --------------------------------------------------------

    def _create_rows(
        self,
        confirmed: sv.Detections,
        displacement_by_id: dict[
            int,
            float,
        ],
    ) -> list[dict[str, Any]]:

        rows: list[
            dict[str, Any]
        ] = []

        if len(confirmed) == 0:
            return rows

        for (
            box,
            class_id,
            confidence,
            tracker_id,
        ) in zip(
            confirmed.xyxy,
            confirmed.class_id,
            confirmed.confidence,
            confirmed.tracker_id,
        ):

            if tracker_id is None:
                continue

            tracker_id = int(
                tracker_id
            )

            if tracker_id < 0:
                continue

            class_name = get_class_name(
                class_id
            )

            confidence_value = (
                float(confidence)
                if confidence is not None
                else 0.0
            )

            displacement = (
                displacement_by_id.get(
                    tracker_id,
                    0.0,
                )
            )

            self.seen_track_ids.add(
                tracker_id
            )

            self.track_classes.setdefault(
                tracker_id,
                class_name,
            )

            self.track_confidences[
                tracker_id
            ].append(
                confidence_value
            )

            rows.append(
                {
                    "Track ID": (
                        f"#{tracker_id:03d}"
                    ),
                    "Class": (
                        class_name.title()
                    ),
                    "Confidence": (
                        round(
                            confidence_value,
                            3,
                        )
                    ),
                    "Displacement (px)": (
                        round(
                            displacement,
                            1,
                        )
                    ),
                    "Status": "Moving",
                    "_id": tracker_id,
                    "_box": box,
                }
            )

        return rows

    # --------------------------------------------------------

    def _annotate_frame(
        self,
        frame_bgr: np.ndarray,
        roi_polygon: np.ndarray,
        rows: list[dict[str, Any]],
    ) -> np.ndarray:

        annotated = frame_bgr.copy()

        draw_dashed_polygon(
            annotated,
            roi_polygon,
        )

        for row in rows:

            tracker_id = row["_id"]
            box = row["_box"]

            draw_track_trajectory(
                annotated,
                self.track_histories[
                    tracker_id
                ],
            )

            label = (
                f"{row['Class'].upper()}  "
                f"{row['Track ID']}  "
                f"{row['Confidence']:.2f}"
            )

            draw_detection(
                annotated,
                box,
                label,
            )

        return annotated

    # --------------------------------------------------------

    def process_video(
        self,
        video_path: Path | str,
        output_directory: Path | str,
        save_output_video: bool = True,
        save_evidence: bool = True,
    ):
        """
        Process the source video and yield FrameResult instances.
        """

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save reproducibility configuration.
        with open(
            output_directory
            / "run_config.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(self.config),
                file,
                indent=2,
            )

        metadata = get_video_metadata(
            video_path
        )

        source_fps = float(
            metadata["fps"]
        )

        total_frames = int(
            metadata["total_frames"]
        )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():

            raise RuntimeError(
                "Could not open input video:\n"
                f"{video_path}"
            )

        background_subtractor = (
            self._create_background_subtractor()
        )

        tracker = ByteTrackTracker(
            frame_rate=max(
                1.0,
                source_fps,
            ),
            lost_track_buffer=(
                self.config
                .tracker_lost_buffer
            ),
        )

        self.track_histories.clear()
        self.seen_track_ids.clear()
        self.track_classes.clear()
        self.track_confidences.clear()
        self.history.clear()
        self.evidence_paths.clear()
        self.diagnostic_paths.clear()

        raw_output_path = (
            output_directory
            / "exercise_1_1_detected_moving_vehicles.mp4"
        )

        self.raw_output_video_path = (
            raw_output_path
        )

        video_writer = None

        evidence_frame_numbers = {
            int(total_frames * 0.25),
            int(total_frames * 0.50),
            int(total_frames * 0.75),
        }

        processing_start = (
            time.perf_counter()
        )

        frames_processed = 0

        try:

            success, first_frame = (
                capture.read()
            )

            if not success:

                raise RuntimeError(
                    "Could not read the "
                    "first video frame."
                )

            first_frame = (
                resize_to_max_width(
                    first_frame,
                    self.config
                    .processing_width,
                )
            )

            previous_gray = (
                preprocess_grayscale(
                    first_frame,
                    self.config,
                )
            )

            (
                frame_height,
                frame_width,
            ) = first_frame.shape[:2]

            roi_polygon = (
                create_roi_polygon(
                    frame_width,
                    frame_height,
                    self.config.roi_top,
                )
            )

            roi_mask = create_roi_mask(
                first_frame.shape,
                roi_polygon,
            )

            # Initial full background learning.
            background_subtractor.apply(
                first_frame,
                learningRate=1.0,
            )

            if save_output_video:

                fourcc = (
                    cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )
                )

                video_writer = (
                    cv2.VideoWriter(
                        str(
                            raw_output_path
                        ),
                        fourcc,
                        source_fps,
                        (
                            frame_width,
                            frame_height,
                        ),
                    )
                )

                if (
                    not video_writer
                    .isOpened()
                ):
                    raise RuntimeError(
                        "Could not create "
                        "output MP4 video."
                    )

            frame_number = 1

            while True:

                success, frame_bgr = (
                    capture.read()
                )

                if not success:
                    break

                frame_start = (
                    time.perf_counter()
                )

                frame_number += 1
                frames_processed += 1

                frame_bgr = (
                    resize_to_max_width(
                        frame_bgr,
                        self.config
                        .processing_width,
                    )
                )

                # --------------------------------
                # I. PREPROCESSING
                # --------------------------------

                current_gray = (
                    preprocess_grayscale(
                        frame_bgr,
                        self.config,
                    )
                )

                # --------------------------------
                # II. FRAME DIFFERENCING
                # --------------------------------

                frame_difference_mask = (
                    calculate_frame_difference(
                        previous_gray,
                        current_gray,
                        self.config,
                    )
                )

                frame_difference_mask = (
                    cv2.bitwise_and(
                        frame_difference_mask,
                        roi_mask,
                    )
                )

                # --------------------------------
                # III. MOG2
                # --------------------------------

                warmup = (
                    frame_number
                    <= self.config
                    .background_warmup_frames
                )

                if warmup:

                    learning_rate = (
                        self.config
                        .background_warmup_learning_rate
                    )

                else:

                    learning_rate = (
                        self.config
                        .background_learning_rate
                    )

                background_mask = (
                    background_subtractor.apply(
                        frame_bgr,
                        learningRate=(
                            learning_rate
                        ),
                    )
                )

                _, background_mask = (
                    cv2.threshold(
                        background_mask,
                        200,
                        255,
                        cv2.THRESH_BINARY,
                    )
                )

                background_mask = (
                    cv2.bitwise_and(
                        background_mask,
                        roi_mask,
                    )
                )

                # --------------------------------
                # IV. MOTION FUSION
                # --------------------------------

                combined_motion_mask = (
                    cv2.bitwise_and(
                        frame_difference_mask,
                        background_mask,
                    )
                )

                combined_motion_mask = (
                    refine_motion_mask(
                        combined_motion_mask,
                        self.config,
                    )
                )

                # Keep dilation strictly inside ROI.
                combined_motion_mask = (
                    cv2.bitwise_and(
                        combined_motion_mask,
                        roi_mask,
                    )
                )

                # --------------------------------
                # V. RF-DETR NANO
                # --------------------------------

                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                with torch.inference_mode():

                    try:

                        detections = (
                            self.model.predict(
                                frame_rgb,
                                threshold=(
                                    self.config
                                    .confidence_threshold
                                ),
                                include_source_image=False,
                            )
                        )

                    except TypeError:

                        # Compatibility with versions
                        # without include_source_image.
                        detections = (
                            self.model.predict(
                                frame_rgb,
                                threshold=(
                                    self.config
                                    .confidence_threshold
                                ),
                            )
                        )

                vehicle_detections = (
                    filter_vehicle_classes(
                        detections,
                        self.config,
                    )
                )

                # --------------------------------
                # VI. MOTION VALIDATION
                # --------------------------------

                if (
                    warmup
                    and self.config
                    .suppress_detections_during_warmup
                ):

                    motion_validated = (
                        empty_detections(
                            vehicle_detections
                        )
                    )

                else:

                    motion_validated = (
                        filter_motion_validated_vehicles(
                            detections=(
                                vehicle_detections
                            ),
                            frame_difference_mask=(
                                frame_difference_mask
                            ),
                            background_mask=(
                                background_mask
                            ),
                            combined_motion_mask=(
                                combined_motion_mask
                            ),
                            roi_polygon=(
                                roi_polygon
                            ),
                            config=(
                                self.config
                            ),
                        )
                    )

                # --------------------------------
                # VII. BYTETRACK
                # --------------------------------

                tracked_vehicles = (
                    tracker.update(
                        motion_validated
                    )
                )

                (
                    confirmed_vehicles,
                    displacement_by_id,
                ) = self._confirm_tracks(
                    tracked_vehicles
                )

                rows = self._create_rows(
                    confirmed_vehicles,
                    displacement_by_id,
                )

                # --------------------------------
                # VIII. VISUALISATION
                # --------------------------------

                annotated_frame = (
                    self._annotate_frame(
                        frame_bgr,
                        roi_polygon,
                        rows,
                    )
                )

                if (
                    save_output_video
                    and video_writer is not None
                ):
                    video_writer.write(
                        annotated_frame
                    )

                # --------------------------------
                # IX. EVIDENCE
                # --------------------------------

                if (
                    save_evidence
                    and frame_number
                    in evidence_frame_numbers
                ):

                    evidence_path = (
                        output_directory
                        / (
                            "evidence_frame_"
                            f"{frame_number}.jpg"
                        )
                    )

                    cv2.imwrite(
                        str(evidence_path),
                        annotated_frame,
                    )

                    self.evidence_paths.append(
                        evidence_path
                    )

                    diagnostic = (
                        create_diagnostic_view(
                            frame_difference_mask,
                            background_mask,
                            combined_motion_mask,
                        )
                    )

                    diagnostic_path = (
                        output_directory
                        / (
                            "diagnostic_frame_"
                            f"{frame_number}.jpg"
                        )
                    )

                    cv2.imwrite(
                        str(diagnostic_path),
                        diagnostic,
                    )

                    self.diagnostic_paths.append(
                        diagnostic_path
                    )

                # --------------------------------
                # PERFORMANCE
                # --------------------------------

                elapsed = max(
                    1e-9,
                    time.perf_counter()
                    - frame_start,
                )

                processing_fps = (
                    1.0 / elapsed
                )

                self.history.append(
                    {
                        "frame": frame_number,
                        "time_seconds": (
                            frame_number
                            / source_fps
                        ),
                        "active_vehicles": (
                            len(rows)
                        ),
                        "unique_vehicles": (
                            len(
                                self.seen_track_ids
                            )
                        ),
                        "processing_fps": (
                            processing_fps
                        ),
                    }
                )

                progress = (
                    frame_number
                    / max(
                        1,
                        total_frames,
                    )
                )

                # Remove private UI-only data.
                public_rows = []

                for row in rows:

                    public_rows.append(
                        {
                            key: value
                            for key, value
                            in row.items()
                            if not key.startswith(
                                "_"
                            )
                        }
                    )

                yield FrameResult(
                    frame_number=frame_number,
                    total_frames=total_frames,
                    annotated_frame=(
                        annotated_frame
                    ),
                    frame_difference_mask=(
                        frame_difference_mask
                    ),
                    background_mask=(
                        background_mask
                    ),
                    combined_motion_mask=(
                        combined_motion_mask
                    ),
                    confirmed_rows=(
                        public_rows
                    ),
                    active_vehicle_count=(
                        len(rows)
                    ),
                    unique_vehicle_count=(
                        len(
                            self.seen_track_ids
                        )
                    ),
                    processing_fps=(
                        processing_fps
                    ),
                    warmup=warmup,
                    progress=min(
                        1.0,
                        progress,
                    ),
                )

                previous_gray = (
                    current_gray
                )

        finally:

            capture.release()

            if video_writer is not None:
                video_writer.release()

        # ====================================================
        # FINALISE OUTPUT
        # ====================================================

        elapsed_total = max(
            1e-9,
            time.perf_counter()
            - processing_start,
        )

        average_processing_fps = (
            frames_processed
            / elapsed_total
        )

        if (
            save_output_video
            and raw_output_path.is_file()
        ):

            self.output_video_path = (
                transcode_h264(
                    raw_output_path
                )
            )

        class_counts = Counter(
            self.track_classes.values()
        )

        all_confidences = [
            confidence
            for values
            in self.track_confidences.values()
            for confidence in values
        ]

        average_confidence = (
            float(
                np.mean(
                    all_confidences
                )
            )
            if all_confidences
            else 0.0
        )

        max_active = max(
            (
                record[
                    "active_vehicles"
                ]
                for record
                in self.history
            ),
            default=0,
        )

        self.summary = {
            "source_fps": source_fps,
            "source_frames": total_frames,
            "frames_processed": (
                frames_processed
            ),
            "processing_seconds": (
                elapsed_total
            ),
            "average_processing_fps": (
                average_processing_fps
            ),
            "unique_vehicles": (
                len(
                    self.seen_track_ids
                )
            ),
            "maximum_active_vehicles": (
                max_active
            ),
            "average_confidence": (
                average_confidence
            ),
            "class_counts": dict(
                class_counts
            ),
            "output_video": (
                str(
                    self.output_video_path
                )
                if self.output_video_path
                else None
            ),
            "raw_output_video": (
                str(
                    self.raw_output_video_path
                )
                if self.raw_output_video_path
                else None
            ),
            "evidence_paths": [
                str(path)
                for path
                in self.evidence_paths
            ],
            "diagnostic_paths": [
                str(path)
                for path
                in self.diagnostic_paths
            ],
        }

        with open(
            output_directory
            / "run_summary.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.summary,
                file,
                indent=2,
            )

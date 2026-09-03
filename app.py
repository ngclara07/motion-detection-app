# this is the complete interactive UI
# command to launch streamlit app: python -m streamlit run app.py

from __future__ import annotations

import os
import tempfile

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch

from rfdetr import RFDETRNano

from vehicle_pipeline import (
    PipelineConfig,
    VehicleDetectionPipeline,
    get_video_metadata,
)


# ============================================================
# APPLICATION PATHS
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

MEDIA_DIR = BASE_DIR / "media"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"

# SAMPLE_VIDEO = (
#     MEDIA_DIR
#     / "Traffic_Laramie_1.mp4"
# )

SAMPLE_VIDEO = (
    MEDIA_DIR
    / "Traffic_Laramie_1_demo.mp4"
)

OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================


st.set_page_config(
    page_title=(
        "Traffic Vision Lab | "
        "Exercise 1.1"
    ),
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================


style_path = (
    ASSETS_DIR
    / "styles.css"
)

if style_path.is_file():

    st.markdown(
        "<style>"
        + style_path.read_text(
            encoding="utf-8"
        )
        + "</style>",
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL LOADING
# ============================================================


@st.cache_resource(
    show_spinner=False,
)
def load_detection_model():
    """
    Load RF-DETR Nano exactly once per Streamlit session.
    """

    cpu_count = os.cpu_count() or 1

    torch.set_num_threads(
        min(
            2,
            max(1, cpu_count - 1),
        )
    )

    model = RFDETRNano()

    return model


# ============================================================
# FILE HANDLING
# ============================================================


def save_uploaded_video(
    uploaded_file,
) -> Path:

    suffix = (
        Path(
            uploaded_file.name
        ).suffix
        or ".mp4"
    )

    temporary_file = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        )
    )

    temporary_file.write(
        uploaded_file.getbuffer()
    )

    temporary_file.close()

    return Path(
        temporary_file.name
    )


# ============================================================
# HELPER UI FUNCTIONS
# ============================================================


def display_video_information(
    metadata: dict,
):

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Resolution",
        (
            f"{metadata['width']} × "
            f"{metadata['height']}"
        ),
    )

    col2.metric(
        "Source FPS",
        f"{metadata['fps']:.2f}",
    )

    col3.metric(
        "Frames",
        f"{metadata['total_frames']:,}",
    )

    duration = (
        metadata[
            "duration_seconds"
        ]
    )

    minutes = int(
        duration // 60
    )

    seconds = int(
        duration % 60
    )

    col4.metric(
        "Duration",
        f"{minutes:02d}:{seconds:02d}",
    )


def display_final_summary(
    summary: dict,
    history: list[dict],
):

    st.markdown(
        "### Run summary"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Unique Moving Vehicles",
        summary.get(
            "unique_vehicles",
            0,
        ),
    )

    c2.metric(
        "Max Concurrent",
        summary.get(
            "maximum_active_vehicles",
            0,
        ),
    )

    c3.metric(
        "Average Confidence",
        (
            f"{summary.get('average_confidence', 0) * 100:.1f}%"
        ),
    )

    c4.metric(
        "Average Processing FPS",
        (
            f"{summary.get('average_processing_fps', 0):.2f}"
        ),
    )

    if history:

        history_df = pd.DataFrame(
            history
        )

        st.markdown(
            "#### Moving vehicles over time"
        )

        st.line_chart(
            history_df.set_index(
                "time_seconds"
            )[
                [
                    "active_vehicles",
                    "unique_vehicles",
                ]
            ],
            use_container_width=True,
        )

        st.markdown(
            "#### Processing throughput"
        )

        st.line_chart(
            history_df.set_index(
                "time_seconds"
            )[
                [
                    "processing_fps",
                ]
            ],
            use_container_width=True,
        )

    class_counts = summary.get(
        "class_counts",
        {},
    )

    if class_counts:

        class_df = pd.DataFrame(
            {
                "Vehicle class": [
                    key.title()
                    for key
                    in class_counts
                ],
                "Vehicles": list(
                    class_counts.values()
                ),
            }
        )

        st.markdown(
            "#### Vehicle classes"
        )

        st.bar_chart(
            class_df.set_index(
                "Vehicle class"
            ),
            use_container_width=True,
        )


def display_saved_results(
    result_data: dict,
):

    summary = result_data.get(
        "summary",
        {},
    )

    history = result_data.get(
        "history",
        [],
    )

    output_path_text = (
        summary.get(
            "output_video"
        )
    )

    if output_path_text:

        output_path = Path(
            output_path_text
        )

        if output_path.is_file():

            st.markdown(
                "### Final annotated video"
            )

            st.video(
                str(output_path)
            )

            with open(
                output_path,
                "rb",
            ) as file:

                st.download_button(
                    label=(
                        "⬇ Download annotated video"
                    ),
                    data=file,
                    file_name=(
                        "exercise_1_1_"
                        "detected_moving_vehicles.mp4"
                    ),
                    mime="video/mp4",
                    use_container_width=True,
                )

    display_final_summary(
        summary,
        history,
    )


# ============================================================
# HEADER
# ============================================================


header_left, header_right = (
    st.columns(
        [5, 1.2]
    )
)

with header_left:

    st.markdown(
        """
        <div class="eyebrow">
            CM3065 · INTELLIGENT SIGNAL PROCESSING
        </div>

        <div class="main-title">
            Traffic Vision Lab
        </div>

        <div class="main-subtitle">
            Hybrid moving-vehicle detection and
            multi-object tracking
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:

    st.markdown(
        """
        <div class="system-status">
            <span class="status-dot"></span>
            SYSTEM READY
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PIPELINE DESCRIPTION
# ============================================================


st.markdown(
    """
    <div class="pipeline-strip">

    <span>FRAME Δ</span>
    <b>＋</b>
    <span>MOG2</span>
    <b>→</b>
    <span>MOTION FUSION</span>
    <b>→</b>
    <span>RF-DETR NANO</span>
    <b>→</b>
    <span>BYTETRACK</span>
    <b>→</b>
    <span>TRAJECTORY CONFIRMATION</span>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:

    st.markdown(
        "## Analysis Control"
    )

    st.caption(
        "Exercise 1.1 · "
        "Moving Vehicle Detection"
    )

    # --------------------------------------------------------
    # VIDEO SOURCE
    # --------------------------------------------------------

    st.markdown(
        "### Video Source"
    )

    source_options = [
        "Upload video"
    ]

    if SAMPLE_VIDEO.is_file():
        source_options.insert(
            0,
            "Traffic_Laramie_1.mp4",
        )

    source_choice = st.radio(
        "Input",
        source_options,
        label_visibility="collapsed",
    )

    uploaded_video = None

    if source_choice == "Upload video":

        uploaded_video = (
            st.file_uploader(
                "Upload traffic recording",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                ],
            )
        )

    st.divider()

    # --------------------------------------------------------
    # RF-DETR
    # --------------------------------------------------------

    with st.expander(
        "RF-DETR Nano",
        expanded=True,
    ):

        confidence_threshold = (
            st.slider(
                "Detection confidence",
                min_value=0.10,
                max_value=0.95,
                value=0.50,
                step=0.05,
            )
        )

        allowed_classes = (
            st.multiselect(
                "Vehicle classes",
                options=[
                    "car",
                    "truck",
                    "bus",
                    "motorcycle",
                ],
                default=[
                    "car",
                    "truck",
                    "bus",
                ],
            )
        )

    # --------------------------------------------------------
    # FRAME DIFFERENCE
    # --------------------------------------------------------

    with st.expander(
        "Frame Differencing"
    ):

        frame_difference_threshold = (
            st.slider(
                "Difference threshold",
                min_value=1,
                max_value=100,
                value=18,
            )
        )

        frame_difference_dilation = (
            st.slider(
                "Dilation iterations",
                min_value=0,
                max_value=5,
                value=2,
            )
        )

    # --------------------------------------------------------
    # MOG2
    # --------------------------------------------------------

    with st.expander(
        "MOG2 Background Model"
    ):

        background_history = (
            st.slider(
                "History",
                min_value=50,
                max_value=1000,
                value=500,
                step=50,
            )
        )

        background_variance = (
            st.slider(
                "Variance threshold",
                min_value=5,
                max_value=100,
                value=28,
            )
        )

        warmup_frames = (
            st.slider(
                "Warm-up frames",
                min_value=0,
                max_value=150,
                value=45,
            )
        )

        suppress_warmup = (
            st.checkbox(
                (
                    "Suppress detections "
                    "during warm-up"
                ),
                value=True,
            )
        )

    # --------------------------------------------------------
    # MOTION VALIDATION
    # --------------------------------------------------------

    with st.expander(
        "Motion Validation"
    ):

        minimum_box_area = (
            st.slider(
                "Minimum box area",
                min_value=100,
                max_value=5000,
                value=800,
                step=100,
            )
        )

        min_diff_occupancy = (
            st.slider(
                "Frame Δ occupancy",
                min_value=0.001,
                max_value=0.100,
                value=0.015,
                step=0.001,
                format="%.3f",
            )
        )

        min_bg_occupancy = (
            st.slider(
                "MOG2 occupancy",
                min_value=0.001,
                max_value=0.100,
                value=0.020,
                step=0.001,
                format="%.3f",
            )
        )

        min_combined_occupancy = (
            st.slider(
                "Fused occupancy",
                min_value=0.001,
                max_value=0.100,
                value=0.010,
                step=0.001,
                format="%.3f",
            )
        )

        minimum_component_area = (
            st.slider(
                (
                    "Largest component "
                    "area"
                ),
                min_value=10,
                max_value=500,
                value=80,
                step=10,
            )
        )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    with st.expander(
        "Tracking"
    ):

        track_history_length = (
            st.slider(
                "Trajectory history",
                min_value=4,
                max_value=30,
                value=8,
            )
        )

        minimum_observations = (
            st.slider(
                "Minimum observations",
                min_value=2,
                max_value=15,
                value=4,
            )
        )

        minimum_displacement = (
            st.slider(
                "Minimum displacement",
                min_value=1.0,
                max_value=50.0,
                value=8.0,
                step=1.0,
            )
        )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    with st.expander(
        "Main Street ROI"
    ):

        roi_top = (
            st.slider(
                "Upper ROI boundary",
                min_value=0.20,
                max_value=0.80,
                value=0.47,
                step=0.01,
            )
        )

        st.caption(
            (
                "0.47 reproduces the "
                "notebook's lower-road ROI."
            )
        )

    # --------------------------------------------------------
    # APPLICATION
    # --------------------------------------------------------

    with st.expander(
        "Application"
    ):

        processing_width = (
            st.select_slider(
                "Processing width",
                options=[
                    640,
                    768,
                    854,
                    960,
                    1040,
                ],
                value=960,
            )
        )

        display_stride = (
            st.slider(
                "UI refresh every N frames",
                min_value=1,
                max_value=15,
                value=3,
            )
        )

        save_output_video = (
            st.checkbox(
                "Save annotated video",
                value=True,
            )
        )

        save_evidence = (
            st.checkbox(
                "Save evidence frames",
                value=True,
            )
        )

    st.divider()

    run_button = st.button(
        "▶  RUN VEHICLE ANALYSIS",
        type="primary",
        use_container_width=True,
    )

    if st.button(
        "Clear previous results",
        use_container_width=True,
    ):
        st.session_state.pop(
            "last_run",
            None,
        )


# ============================================================
# BUILD CONFIGURATION
# ============================================================


config = PipelineConfig(
    processing_width=(
        processing_width
    ),
    confidence_threshold=(
        confidence_threshold
    ),
    allowed_vehicle_classes=tuple(
        allowed_classes
    ),
    frame_difference_threshold=(
        frame_difference_threshold
    ),
    frame_difference_dilation_iterations=(
        frame_difference_dilation
    ),
    background_history=(
        background_history
    ),
    background_variance_threshold=(
        background_variance
    ),
    background_warmup_frames=(
        warmup_frames
    ),
    minimum_box_area=(
        minimum_box_area
    ),
    minimum_frame_diff_occupancy=(
        min_diff_occupancy
    ),
    minimum_background_occupancy=(
        min_bg_occupancy
    ),
    minimum_combined_occupancy=(
        min_combined_occupancy
    ),
    minimum_largest_component_area=(
        minimum_component_area
    ),
    track_history_length=(
        track_history_length
    ),
    minimum_track_observations=(
        minimum_observations
    ),
    minimum_track_displacement=(
        minimum_displacement
    ),
    roi_top=(
        roi_top
    ),
    suppress_detections_during_warmup=(
        suppress_warmup
    ),
)


# ============================================================
# MAIN TABS
# ============================================================


(
    detection_tab,
    diagnostics_tab,
    analytics_tab,
    evidence_tab,
    configuration_tab,
) = st.tabs(
    [
        "🎥 Detection",
        "🔬 Motion Diagnostics",
        "📊 Analytics",
        "🖼 Evidence",
        "⚙ Configuration",
    ]
)


# ============================================================
# DETECTION TAB PLACEHOLDERS
# ============================================================


with detection_tab:

    st.markdown(
        "### Intelligent Traffic Monitor"
    )

    telemetry_1, telemetry_2, telemetry_3, telemetry_4 = (
        st.columns(4)
    )

    active_metric = (
        telemetry_1.empty()
    )

    unique_metric = (
        telemetry_2.empty()
    )

    fps_metric = (
        telemetry_3.empty()
    )

    frame_metric = (
        telemetry_4.empty()
    )

    live_frame_placeholder = (
        st.empty()
    )

    status_placeholder = (
        st.empty()
    )

    progress_placeholder = (
        st.empty()
    )

    st.markdown(
        "#### Active tracks"
    )

    track_table_placeholder = (
        st.empty()
    )

    final_video_placeholder = (
        st.container()
    )


# ============================================================
# DIAGNOSTIC TAB
# ============================================================


with diagnostics_tab:

    st.markdown(
        "### Classical Motion Analysis"
    )

    st.caption(
        (
            "A vehicle must exhibit both "
            "short-term frame change and "
            "foreground evidence."
        )
    )

    diag_1, diag_2, diag_3 = (
        st.columns(3)
    )

    with diag_1:

        st.markdown(
            "**Frame Difference**"
        )

        difference_placeholder = (
            st.empty()
        )

    with diag_2:

        st.markdown(
            "**MOG2 Foreground**"
        )

        background_placeholder = (
            st.empty()
        )

    with diag_3:

        st.markdown(
            "**Fused Motion**"
        )

        combined_placeholder = (
            st.empty()
        )


# ============================================================
# ANALYTICS TAB
# ============================================================


with analytics_tab:

    analytics_placeholder = (
        st.container()
    )


# ============================================================
# EVIDENCE TAB
# ============================================================


with evidence_tab:

    evidence_placeholder = (
        st.container()
    )


# ============================================================
# CONFIGURATION TAB
# ============================================================


with configuration_tab:

    st.markdown(
        "### Current Experimental Configuration"
    )

    configuration_df = (
        pd.DataFrame(
            [
                {
                    "Parameter": key,
                    "Value": str(value),
                }
                for key, value
                in asdict(config).items()
            ]
        )
    )

    st.dataframe(
        configuration_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        (
            "Every analysis run saves this "
            "configuration as run_config.json "
            "for reproducibility."
        )
    )


# ============================================================
# PROCESS VIDEO
# ============================================================


if run_button:

    source_path = None
    temporary_upload = None

    # --------------------------------------------------------
    # VALIDATE SOURCE
    # --------------------------------------------------------

    if (
        source_choice
        == "Traffic_Laramie_1.mp4"
    ):

        source_path = (
            SAMPLE_VIDEO
        )

    elif uploaded_video is None:

        st.error(
            (
                "Please upload a video "
                "before starting analysis."
            )
        )

        st.stop()

    else:

        temporary_upload = (
            save_uploaded_video(
                uploaded_video
            )
        )

        source_path = (
            temporary_upload
        )

    try:

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        metadata = (
            get_video_metadata(
                source_path
            )
        )

        with detection_tab:

            display_video_information(
                metadata
            )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        status_placeholder.info(
            "Loading RF-DETR Nano..."
        )

        with st.spinner(
            "Initialising RF-DETR Nano..."
        ):

            model = (
                load_detection_model()
            )

        status_placeholder.success(
            (
                "RF-DETR Nano ready. "
                "Starting analysis..."
            )
        )

        # ----------------------------------------------------
        # CREATE OUTPUT RUN DIRECTORY
        # ----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        run_directory = (
            OUTPUTS_DIR
            / f"run_{timestamp}"
        )

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # INITIALISE PIPELINE
        # ----------------------------------------------------

        pipeline = (
            VehicleDetectionPipeline(
                model=model,
                config=config,
            )
        )

        progress_bar = (
            progress_placeholder.progress(
                0.0
            )
        )

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        for result in pipeline.process_video(
            video_path=source_path,
            output_directory=(
                run_directory
            ),
            save_output_video=(
                save_output_video
            ),
            save_evidence=(
                save_evidence
            ),
        ):

            # Update the browser only every N frames
            # to avoid UI rendering becoming the bottleneck.
            should_refresh = (
                result.frame_number
                % display_stride
                == 0
                or result.progress >= 0.999
            )

            if not should_refresh:
                continue

            frame_rgb = cv2.cvtColor(
                result.annotated_frame,
                cv2.COLOR_BGR2RGB,
            )

            live_frame_placeholder.image(
                frame_rgb,
                use_container_width=True,
            )

            active_metric.metric(
                "Moving Now",
                result.active_vehicle_count,
            )

            unique_metric.metric(
                "Unique Tracks",
                result.unique_vehicle_count,
            )

            fps_metric.metric(
                "Processing FPS",
                (
                    f"{result.processing_fps:.2f}"
                ),
            )

            frame_metric.metric(
                "Frame",
                (
                    f"{result.frame_number:,}"
                    f" / "
                    f"{result.total_frames:,}"
                ),
            )

            if result.warmup:

                status_placeholder.warning(
                    (
                        "MOG2 background "
                        "model warming up"
                    )
                )

            else:

                status_placeholder.success(
                    (
                        "● ANALYSIS ACTIVE"
                    )
                )

            progress_bar.progress(
                result.progress
            )

            if (
                result.confirmed_rows
            ):

                track_table_placeholder.dataframe(
                    pd.DataFrame(
                        result.confirmed_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                track_table_placeholder.info(
                    (
                        "No confirmed moving "
                        "vehicles in this frame."
                    )
                )

            difference_placeholder.image(
                result.frame_difference_mask,
                clamp=True,
                use_container_width=True,
            )

            background_placeholder.image(
                result.background_mask,
                clamp=True,
                use_container_width=True,
            )

            combined_placeholder.image(
                result.combined_motion_mask,
                clamp=True,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        progress_bar.progress(
            1.0
        )

        status_placeholder.success(
            (
                "✓ Vehicle analysis complete"
            )
        )

        summary = pipeline.summary

        last_run = {
            "summary": summary,
            "history": pipeline.history,
            "run_directory": str(
                run_directory
            ),
        }

        st.session_state[
            "last_run"
        ] = last_run

        # ----------------------------------------------------
        # FINAL VIDEO
        # ----------------------------------------------------

        with final_video_placeholder:

            output_text = (
                summary.get(
                    "output_video"
                )
            )

            if output_text:

                output_path = Path(
                    output_text
                )

                if output_path.is_file():

                    st.markdown(
                        "### Annotated Output"
                    )

                    st.video(
                        str(
                            output_path
                        )
                    )

                    with open(
                        output_path,
                        "rb",
                    ) as file:

                        st.download_button(
                            (
                                "⬇ Download "
                                "annotated MP4"
                            ),
                            data=file,
                            file_name=(
                                "exercise_1_1_"
                                "detected_moving_"
                                "vehicles.mp4"
                            ),
                            mime="video/mp4",
                            type="primary",
                            use_container_width=True,
                        )

        # ----------------------------------------------------
        # ANALYTICS
        # ----------------------------------------------------

        with analytics_placeholder:

            display_final_summary(
                summary,
                pipeline.history,
            )

        # ----------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------

        with evidence_placeholder:

            st.markdown(
                "### Saved Evidence Frames"
            )

            evidence_paths = [
                Path(path)
                for path
                in summary.get(
                    "evidence_paths",
                    [],
                )
            ]

            diagnostic_paths = [
                Path(path)
                for path
                in summary.get(
                    "diagnostic_paths",
                    [],
                )
            ]

            if evidence_paths:

                cols = st.columns(
                    min(
                        3,
                        len(
                            evidence_paths
                        ),
                    )
                )

                for index, path in enumerate(
                    evidence_paths
                ):

                    if path.is_file():

                        cols[
                            index
                            % len(cols)
                        ].image(
                            str(path),
                            caption=(
                                path.stem
                            ),
                            use_container_width=True,
                        )

            if diagnostic_paths:

                st.markdown(
                    "### Diagnostic Evidence"
                )

                for path in diagnostic_paths:

                    if path.is_file():

                        st.image(
                            str(path),
                            caption=(
                                path.stem
                            ),
                            use_container_width=True,
                        )

        st.toast(
            (
                "Exercise 1.1 "
                "processing completed."
            ),
            icon="✅",
        )

    except Exception as error:

        status_placeholder.error(
            "Analysis failed."
        )

        st.exception(
            error
        )

    finally:

        if (
            temporary_upload
            is not None
            and temporary_upload.exists()
        ):

            try:
                temporary_upload.unlink()
            except OSError:
                pass


# ============================================================
# INITIAL / PREVIOUS RUN STATE
# ============================================================


elif (
    "last_run"
    in st.session_state
):

    with analytics_placeholder:

        display_saved_results(
            st.session_state[
                "last_run"
            ]
        )

else:

    active_metric.metric(
        "Moving Now",
        "—",
    )

    unique_metric.metric(
        "Unique Tracks",
        "—",
    )

    fps_metric.metric(
        "Processing FPS",
        "—",
    )

    frame_metric.metric(
        "Frame",
        "—",
    )

    live_frame_placeholder.markdown(
        """
        <div class="empty-monitor">

            <div class="monitor-icon">
                ◉
            </div>

            <div class="monitor-title">
                Traffic analysis ready
            </div>

            <div class="monitor-text">
                Select a video source and press
                <strong>Run Vehicle Analysis</strong>.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    status_placeholder.info(
        (
            "Waiting for a traffic "
            "video analysis run."
        )
    )

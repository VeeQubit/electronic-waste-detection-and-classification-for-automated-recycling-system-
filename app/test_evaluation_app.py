from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

TEST_IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo_dataset"
    / "images"
    / "test"
)

TEST_LABELS_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo_dataset"
    / "labels"
    / "test"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Waste YOLO26n Test Evaluation",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Electronic Waste Detection & Classification")
st.subheader("YOLO26n Test Dataset Evaluation")

st.write(
    """
    This application evaluates the trained YOLO26n model on images
    from the independent test dataset.

    For each selected image, the system displays:
    - Ground-truth annotations
    - Model predictions
    - Predicted class
    - Confidence score
    - True Positives (TP)
    - False Positives (FP)
    - False Negatives (FN)
    - IoU-based matching information
    """
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        f"Model not found:\n\n{MODEL_PATH}\n\n"
        "Download best.pt from Google Drive and place it "
        "inside the models folder."
    )
    st.stop()


if not TEST_IMAGES_DIR.exists():
    st.error(
        f"Test image directory not found:\n\n{TEST_IMAGES_DIR}"
    )
    st.stop()


if not TEST_LABELS_DIR.exists():
    st.error(
        f"Test label directory not found:\n\n{TEST_LABELS_DIR}"
    )
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO(str(MODEL_PATH))


model = load_model()

class_names = model.names


# ============================================================
# GET TEST IMAGES
# ============================================================

image_extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png"
]

test_images = []

for extension in image_extensions:
    test_images.extend(
        TEST_IMAGES_DIR.glob(extension)
    )

test_images = sorted(test_images)


if not test_images:
    st.error("No test images were found.")
    st.stop()


# ============================================================
# IOU FUNCTION
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(image_path):

    label_path = TEST_LABELS_DIR / (
        image_path.stem + ".txt"
    )

    if not label_path.exists():
        return []

    image = cv2.imread(str(image_path))

    if image is None:
        return []

    height, width = image.shape[:2]

    ground_truth = []

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1]) * width
            y_center = float(values[2]) * height
            box_width = float(values[3]) * width
            box_height = float(values[4]) * height

            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            ground_truth.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "box": [x1, y1, x2, y2]
                }
            )

    return ground_truth


# ============================================================
# MATCH PREDICTIONS WITH GROUND TRUTH
# ============================================================

def evaluate_image(
    image_path,
    confidence_threshold,
    iou_threshold
):

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=confidence_threshold,
        verbose=False
    )

    result = results[0]

    predictions = []

    if result.boxes is not None:

        for box in result.boxes:

            coordinates = box.xyxy[0].cpu().numpy()

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "confidence": confidence,
                    "box": coordinates.tolist()
                }
            )

    ground_truth = load_ground_truth(image_path)

    matched_gt = set()

    true_positives = []
    false_positives = []

    # Highest-confidence predictions first
    predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    for prediction in predictions:

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(ground_truth):

            if gt_index in matched_gt:
                continue

            if (
                prediction["class_id"]
                != gt["class_id"]
            ):
                continue

            iou = calculate_iou(
                prediction["box"],
                gt["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):

            matched_gt.add(best_gt_index)

            true_positives.append(
                {
                    "class": prediction["class_name"],
                    "confidence": prediction["confidence"],
                    "iou": best_iou
                }
            )

        else:

            false_positives.append(
                {
                    "class": prediction["class_name"],
                    "confidence": prediction["confidence"],
                    "iou": best_iou
                }
            )

    false_negatives = []

    for gt_index, gt in enumerate(ground_truth):

        if gt_index not in matched_gt:

            false_negatives.append(
                {
                    "class": gt["class_name"]
                }
            )

    return {
        "result": result,
        "ground_truth": ground_truth,
        "predictions": predictions,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives
    }


# ============================================================
# DRAW GROUND TRUTH
# ============================================================

def draw_ground_truth(
    image_path,
    ground_truth
):

    image = cv2.imread(str(image_path))

    for gt in ground_truth:

        x1, y1, x2, y2 = map(
            int,
            gt["box"]
        )

        label = (
            f"GT: {gt['class_name']}"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Evaluation Settings")

number_of_images = st.sidebar.slider(
    "Number of test images",
    min_value=20,
    max_value=25,
    value=20
)

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "IoU threshold",
    min_value=0.30,
    max_value=0.90,
    value=0.50,
    step=0.05
)

selection_mode = st.sidebar.radio(
    "Image selection",
    [
        "Random test images",
        "Select specific images"
    ]
)


# ============================================================
# IMAGE SELECTION
# ============================================================

if selection_mode == "Random test images":

    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42
    )

    random.seed(seed)

    selected_images = random.sample(
        test_images,
        min(
            number_of_images,
            len(test_images)
        )
    )

else:

    selected_names = st.sidebar.multiselect(
        "Select test images",
        [
            image.name
            for image in test_images
        ],
        max_selections=15
    )

    selected_images = [
        TEST_IMAGES_DIR / name
        for name in selected_names
    ]


# ============================================================
# RUN EVALUATION
# ============================================================

run_evaluation = st.sidebar.button(
    "Run Evaluation",
    type="primary"
)


if run_evaluation:

    st.session_state["selected_images"] = selected_images
    st.session_state["run"] = True


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.get("run", False):

    selected_images = st.session_state[
        "selected_images"
    ]

    if len(selected_images) == 0:

        st.warning(
            "Please select at least one image."
        )

        st.stop()

    st.header(
        f"Evaluation Results — "
        f"{len(selected_images)} Images"
    )

    all_results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------

    for index, image_path in enumerate(
        selected_images,
        start=1
    ):

        evaluation = evaluate_image(
            image_path,
            confidence_threshold,
            iou_threshold
        )

        all_results.append(evaluation)

        tp = len(evaluation["tp"])
        fp = len(evaluation["fp"])
        fn = len(evaluation["fn"])

        total_tp += tp
        total_fp += fp
        total_fn += fn

        st.subheader(
            f"{index}. {image_path.name}"
        )

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # PREDICTION IMAGE
        # ----------------------------------------------------

        with col1:

            st.markdown(
                "**YOLO26n Prediction**"
            )

            annotated = evaluation[
                "result"
            ].plot()

            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                annotated,
                use_container_width=True
            )

        # ----------------------------------------------------
        # GROUND TRUTH IMAGE
        # ----------------------------------------------------

        with col2:

            st.markdown(
                "**Ground Truth**"
            )

            gt_image = draw_ground_truth(
                image_path,
                evaluation["ground_truth"]
            )

            st.image(
                gt_image,
                use_container_width=True
            )

        # ----------------------------------------------------
        # IMAGE METRICS
        # ----------------------------------------------------

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Ground Truth",
            len(evaluation["ground_truth"])
        )

        m2.metric(
            "Predictions",
            len(evaluation["predictions"])
        )

        m3.metric(
            "True Positives",
            tp
        )

        m4.metric(
            "FP / FN",
            f"{fp} / {fn}"
        )

        # ----------------------------------------------------
        # DETECTION INFORMATION
        # ----------------------------------------------------

        rows = []

        for item in evaluation["tp"]:

            rows.append(
                {
                    "Status": "Correct",
                    "Class": item["class"],
                    "Confidence": round(
                        item["confidence"],
                        3
                    ),
                    "IoU": round(
                        item["iou"],
                        3
                    )
                }
            )

        for item in evaluation["fp"]:

            rows.append(
                {
                    "Status": "False Positive",
                    "Class": item["class"],
                    "Confidence": round(
                        item["confidence"],
                        3
                    ),
                    "IoU": round(
                        item["iou"],
                        3
                    )
                }
            )

        for item in evaluation["fn"]:

            rows.append(
                {
                    "Status": "False Negative",
                    "Class": item["class"],
                    "Confidence": None,
                    "IoU": None
                }
            )

        if rows:

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No detections or ground-truth objects "
                "were present in this image."
            )

        st.divider()

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    st.header("Overall Selected-Test-Image Results")

    precision = (
        total_tp
        / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "True Positives",
        total_tp
    )

    c2.metric(
        "False Positives",
        total_fp
    )

    c3.metric(
        "False Negatives",
        total_fn
    )

    c4.metric(
        "F1 Score",
        f"{f1:.3f}"
    )

    st.write(
        f"**Precision:** {precision:.3f}"
    )

    st.write(
        f"**Recall:** {recall:.3f}"
    )

    st.write(
        f"**IoU matching threshold:** "
        f"{iou_threshold:.2f}"
    )

    # ========================================================
# PER-CLASS EVALUATION SUMMARY
# ========================================================

st.header("Per-Class Evaluation Summary")

class_statistics = {}

# Collect all classes that appear in the selected images
for evaluation in all_results:

    for item in evaluation["tp"]:
        class_name = item["class"]

        if class_name not in class_statistics:
            class_statistics[class_name] = {
                "TP": 0,
                "FP": 0,
                "FN": 0
            }

        class_statistics[class_name]["TP"] += 1

    for item in evaluation["fp"]:
        class_name = item["class"]

        if class_name not in class_statistics:
            class_statistics[class_name] = {
                "TP": 0,
                "FP": 0,
                "FN": 0
            }

        class_statistics[class_name]["FP"] += 1

    for item in evaluation["fn"]:
        class_name = item["class"]

        if class_name not in class_statistics:
            class_statistics[class_name] = {
                "TP": 0,
                "FP": 0,
                "FN": 0
            }

        class_statistics[class_name]["FN"] += 1


# --------------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------------

class_rows = []

for class_name in sorted(class_statistics.keys()):

    tp = class_statistics[class_name]["TP"]
    fp = class_statistics[class_name]["FP"]
    fn = class_statistics[class_name]["FN"]

    precision_class = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall_class = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1_class = (
        2 * precision_class * recall_class
        / (precision_class + recall_class)
        if (precision_class + recall_class) > 0
        else 0
    )

    class_rows.append(
        {
            "Class": class_name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": round(precision_class, 3),
            "Recall": round(recall_class, 3),
            "F1": round(f1_class, 3)
        }
    )


# --------------------------------------------------------
# DISPLAY TABLE
# --------------------------------------------------------

if class_rows:

    per_class_df = pd.DataFrame(class_rows)

    st.dataframe(
        per_class_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No class-level detections were available "
        "for the selected images."
    )
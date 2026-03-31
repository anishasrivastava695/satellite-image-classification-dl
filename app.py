from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from utils.image_enhancement import (
    EnhancementConfig,
    ImageEnhancementError,
    convert_bgr_to_rgb as enh_bgr_to_rgb,
    full_enhancement_pipeline,
    get_histogram_data,
    process_uploaded_image,
)
from utils.feature_extraction import (
    EdgeConfig,
    FeatureExtractionError,
    apply_gabor_filters,
    convert_bgr_to_rgb as feat_bgr_to_rgb,
    detect_brisk_features,
    detect_canny_edges,
    detect_laplacian_edges,
    detect_orb_features,
    detect_sift_features,
    detect_sobel_edges,
    extract_hog_features,
    find_contours,
    segment_image_threshold,
)
from utils.classification import (
    ClassificationError,
    PredictionConfig,
    format_prediction_for_streamlit,
    get_bar_chart_data,
    get_probability_table,
    infer_model_target_size,
    predict_single_image,
    safe_load_model,
)

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Satellite Image Land Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Project Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "land_classifier_model.keras"
TRAIN_SCRIPT_PATH = BASE_DIR / "train_model.py"

CLASS_LABELS = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 34px;
            font-weight: 700;
            color: #1f4e79;
            margin-bottom: 10px;
        }
        .section-title {
            font-size: 24px;
            font-weight: 600;
            color: #1f4e79;
            margin-top: 16px;
            margin-bottom: 8px;
        }
        .info-box {
            background-color: #f5f9ff;
            padding: 14px;
            border-radius: 10px;
            border-left: 6px solid #1f4e79;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_classifier_model():
    """
    Load saved classifier model once and cache it.
    """
    if not MODEL_PATH.exists():
        return None, None

    model = safe_load_model(str(MODEL_PATH))
    target_size = infer_model_target_size(model)
    return model, target_size


def clear_model_cache() -> None:
    """
    Clear cached model so fresh model is reloaded after training.
    """
    load_classifier_model.clear()


def train_and_save_model_from_app() -> bool:
    """
    Import and run training pipeline from train_model.py

    Returns:
        True if model saved successfully, else False
    """
    try:
        import train_model as trainer
        trainer.MODEL_DIR = str(MODEL_DIR)
        trainer.MODEL_PATH = str(MODEL_PATH)
        trainer.LABELS_PATH = str(MODEL_DIR / "class_labels.json")

        with st.spinner("Training model... Please wait."):
            trainer.train_model()

        if MODEL_PATH.exists():
            clear_model_cache()
            return True

        return False

    except Exception as exc:
        st.error(f"Training failed: {exc}")
        return False


def encode_image_for_download(image: np.ndarray, extension: str = ".png") -> bytes:
    """
    Convert image to downloadable bytes.
    """
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise ValueError("Failed to encode image.")
    return buffer.tobytes()


def histogram_to_dataframe(hist_data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert histogram dict to pandas DataFrame.
    """
    df = pd.DataFrame({"intensity": np.arange(256)})
    for key, value in hist_data.items():
        df[key] = value
    return df


def display_color_or_gray(image: np.ndarray, caption: str) -> None:
    """
    Display image properly depending on channels.
    """
    if image.ndim == 2:
        st.image(image, caption=caption, use_container_width=True, clamp=True)
    else:
        st.image(enh_bgr_to_rgb(image), caption=caption, use_container_width=True)


def show_model_status() -> tuple[Optional[Any], Optional[tuple[int, int]]]:
    """
    Show model availability in sidebar and return model + target_size.
    """
    model, target_size = load_classifier_model()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Status")

    if model is None:
        st.sidebar.error("Model not found")
        st.sidebar.caption("Train and save model first.")
        st.sidebar.caption(f"Expected location:")
        st.sidebar.code(str(MODEL_PATH))
    else:
        st.sidebar.success("Saved model loaded")
        st.sidebar.caption(f"Path: {MODEL_PATH}")
        st.sidebar.caption(f"Input size: {target_size[0]} x {target_size[1]}")

    return model, target_size


# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.title("Controls")

st.sidebar.subheader("Image Enhancement")
resize_width = st.sidebar.slider("Resize Width", 128, 1024, 512, 32)
apply_clahe = st.sidebar.checkbox("Apply CLAHE", value=True)
clahe_clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.1)
clahe_grid = st.sidebar.selectbox("CLAHE Grid Size", [4, 8, 12, 16], index=1)

apply_denoise = st.sidebar.checkbox("Apply Denoising", value=True)
apply_brightness_contrast = st.sidebar.checkbox("Adjust Brightness / Contrast", value=False)
brightness = st.sidebar.slider("Brightness", -100, 100, 0, 1)
contrast = st.sidebar.slider("Contrast", -100, 100, 0, 1)
apply_gamma = st.sidebar.checkbox("Apply Gamma Correction", value=False)
gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
apply_sharpen = st.sidebar.checkbox("Apply Sharpening", value=True)
sharpen_strength = st.sidebar.slider("Sharpen Strength", 0.1, 3.0, 1.0, 0.1)

st.sidebar.subheader("Feature Extraction")
low_threshold = st.sidebar.slider("Canny Low Threshold", 0, 255, 100)
high_threshold = st.sidebar.slider("Canny High Threshold", 0, 255, 200)
show_sift = st.sidebar.checkbox("Try SIFT", value=True)
show_brisk = st.sidebar.checkbox("Show BRISK", value=True)
show_gabor = st.sidebar.checkbox("Show Gabor Texture", value=True)
contour_threshold = st.sidebar.slider("Contour Threshold", 0, 255, 127)
min_contour_area = st.sidebar.slider("Min Contour Area", 1, 500, 10)
segmentation_threshold = st.sidebar.slider("Segmentation Threshold", 0, 255, 127)

st.sidebar.subheader("Classification")
top_k = st.sidebar.slider("Top K Predictions", 1, 10, 3)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.0, 0.01)
normalize = st.sidebar.checkbox("Normalize for Model", value=True)

model, target_size = show_model_status()

# ----------------------------
# Main Header
# ----------------------------
st.markdown(
    '<div class="main-title">Satellite Image Enhancement, Feature Extraction, and Land Classification</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="info-box">
    Upload one image and run the complete pipeline:
    <b>Enhancement → Feature Extraction → Land Classification</b>
    <br><br>
    Model location:
    <b>{MODEL_PATH}</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Model Training Section
# ----------------------------
st.markdown('<div class="section-title">Model Setup</div>', unsafe_allow_html=True)

if model is None:
    st.warning("No saved model found.")
    st.write("Click the button below to train the model using `train_model.py` and save it automatically.")

    if st.button("Train Model and Save", type="primary", use_container_width=True):
        success = train_and_save_model_from_app()
        if success:
            st.success(f"Model trained and saved successfully at:\n{MODEL_PATH}")
            st.rerun()
        else:
            st.error("Model training completed but saved model was not found.")
else:
    st.success("Model is ready for classification.")

# ----------------------------
# Upload Section
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a satellite / land image",
    type=["jpg", "jpeg", "png"],
)

run_pipeline = st.button("Run Full Pipeline", type="primary", use_container_width=True)

# ----------------------------
# Main Pipeline
# ----------------------------
if uploaded_file is not None:
    try:
        original_image = process_uploaded_image(uploaded_file)

        st.markdown('<div class="section-title">Uploaded Image</div>', unsafe_allow_html=True)
        display_color_or_gray(original_image, "Uploaded Image")

        if run_pipeline:
            # Refresh model after any possible training
            model, target_size = load_classifier_model()

            # --------------------------------
            # 1. Image Enhancement
            # --------------------------------
            enhancement_config = EnhancementConfig(
                resize_width=resize_width,
                keep_aspect_ratio=True,
                apply_clahe=apply_clahe,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid_size=(clahe_grid, clahe_grid),
                apply_denoise=apply_denoise,
                apply_sharpen=apply_sharpen,
                sharpen_strength=sharpen_strength,
                apply_gamma=apply_gamma,
                gamma=gamma,
                apply_brightness_contrast=apply_brightness_contrast,
                brightness=brightness,
                contrast=contrast,
            )

            enhancement_results = full_enhancement_pipeline(
                original_image,
                config=enhancement_config,
            )

            enhanced_image = enhancement_results["final"]

            st.markdown('<div class="section-title">1. Image Enhancement</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                display_color_or_gray(enhancement_results["original"], "Original")
            with c2:
                display_color_or_gray(enhancement_results["final"], "Enhanced")

            with st.expander("View Intermediate Enhancement Outputs", expanded=False):
                e1, e2, e3 = st.columns(3)
                with e1:
                    display_color_or_gray(enhancement_results["resized"], "Resized")
                    display_color_or_gray(enhancement_results["contrast_enhanced"], "Contrast Enhanced")
                with e2:
                    display_color_or_gray(enhancement_results["denoised"], "Denoised")
                    display_color_or_gray(enhancement_results["brightness_contrast_adjusted"], "Brightness / Contrast")
                with e3:
                    display_color_or_gray(enhancement_results["gamma_corrected"], "Gamma Corrected")
                    display_color_or_gray(enhancement_results["sharpened"], "Sharpened")

            hist_col1, hist_col2 = st.columns(2)
            with hist_col1:
                st.markdown("**Original Histogram**")
                original_hist_df = histogram_to_dataframe(
                    get_histogram_data(enhancement_results["original"])
                )
                st.line_chart(original_hist_df.set_index("intensity"))
            with hist_col2:
                st.markdown("**Enhanced Histogram**")
                enhanced_hist_df = histogram_to_dataframe(
                    get_histogram_data(enhancement_results["final"])
                )
                st.line_chart(enhanced_hist_df.set_index("intensity"))

            # --------------------------------
            # 2. Feature Extraction
            # --------------------------------
            st.markdown('<div class="section-title">2. Feature Extraction</div>', unsafe_allow_html=True)

            edge_config = EdgeConfig(
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                aperture_size=3,
                l2gradient=False,
            )

            canny_edges = detect_canny_edges(enhanced_image, config=edge_config)
            sobel_results = detect_sobel_edges(enhanced_image)
            laplacian_edges = detect_laplacian_edges(enhanced_image)

            st.subheader("Edge Detection")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                st.image(canny_edges, caption="Canny", use_container_width=True, clamp=True)
            with f2:
                st.image(sobel_results["sobel_x"], caption="Sobel X", use_container_width=True, clamp=True)
            with f3:
                st.image(sobel_results["sobel_y"], caption="Sobel Y", use_container_width=True, clamp=True)
            with f4:
                st.image(laplacian_edges, caption="Laplacian", use_container_width=True, clamp=True)

            st.subheader("Keypoint Detection")
            orb_result = detect_orb_features(enhanced_image)
            k1, k2 = st.columns(2)
            with k1:
                st.image(
                    feat_bgr_to_rgb(orb_result["keypoint_image"]),
                    caption=f"ORB Keypoints ({orb_result['num_keypoints']})",
                    use_container_width=True,
                )
            with k2:
                if show_sift:
                    try:
                        sift_result = detect_sift_features(enhanced_image)
                        if sift_result["keypoint_image"].ndim == 2:
                            st.image(
                                sift_result["keypoint_image"],
                                caption=f"SIFT Keypoints ({sift_result['num_keypoints']})",
                                use_container_width=True,
                                clamp=True,
                            )
                        else:
                            st.image(
                                feat_bgr_to_rgb(sift_result["keypoint_image"]),
                                caption=f"SIFT Keypoints ({sift_result['num_keypoints']})",
                                use_container_width=True,
                            )
                    except Exception as exc:
                        st.warning(f"SIFT not available: {exc}")

            if show_brisk:
                brisk_result = detect_brisk_features(enhanced_image)
                st.subheader("BRISK Features")
                b1, b2 = st.columns([2, 1])
                with b1:
                    st.image(
                        feat_bgr_to_rgb(brisk_result["keypoint_image"]),
                        caption="BRISK Keypoints",
                        use_container_width=True,
                    )
                with b2:
                    st.metric("BRISK Keypoints", brisk_result["num_keypoints"])

            st.subheader("HOG Features")
            hog_result = extract_hog_features(enhanced_image)
            h1, h2 = st.columns([2, 1])
            with h1:
                st.image(
                    hog_result["resized_image"],
                    caption="HOG Input",
                    use_container_width=True,
                    clamp=True,
                )
            with h2:
                st.metric("HOG Feature Length", hog_result["feature_length"])

            st.subheader("Contours and Segmentation")
            contour_result = find_contours(
                enhanced_image,
                threshold=contour_threshold,
                min_area=float(min_contour_area),
                draw_contours=True,
            )
            segmentation_result = segment_image_threshold(
                enhanced_image,
                threshold=segmentation_threshold,
            )

            cs1, cs2, cs3 = st.columns(3)
            with cs1:
                st.image(
                    contour_result["binary"],
                    caption="Binary",
                    use_container_width=True,
                    clamp=True,
                )
            with cs2:
                st.image(
                    feat_bgr_to_rgb(contour_result["contour_image"]),
                    caption=f"Contours ({contour_result['num_contours']})",
                    use_container_width=True,
                )
            with cs3:
                st.image(
                    segmentation_result["segmented"],
                    caption="Threshold Segmentation",
                    use_container_width=True,
                    clamp=True,
                )

            if show_gabor:
                st.subheader("Texture Analysis with Gabor Filters")
                gabor_responses = apply_gabor_filters(enhanced_image)
                gabor_cols = st.columns(len(gabor_responses))
                for idx, (name, gabor_image) in enumerate(gabor_responses.items()):
                    with gabor_cols[idx]:
                        st.image(
                            gabor_image,
                            caption=name,
                            use_container_width=True,
                            clamp=True,
                        )

            # --------------------------------
            # 3. Land Classification
            # --------------------------------
            st.markdown('<div class="section-title">3. Land Classification</div>', unsafe_allow_html=True)

            if model is None or target_size is None:
                st.error(
                    "Saved model not found. Please train model first. Expected location:\n"
                    f"{MODEL_PATH}"
                )
            else:
                prediction_config = PredictionConfig(
                    target_size=target_size,
                    normalize=normalize,
                    color_mode="rgb",
                    confidence_threshold=confidence_threshold,
                    top_k=top_k,
                )

                prediction = predict_single_image(
                    model=model,
                    image=enhanced_image,
                    class_labels=CLASS_LABELS,
                    config=prediction_config,
                )

                p1, p2, p3 = st.columns(3)
                with p1:
                    st.metric("Predicted Class", prediction["predicted_label"])
                with p2:
                    st.metric("Confidence", f"{prediction['confidence_percent']:.2f}%")
                with p3:
                    st.metric("Accepted", "Yes" if prediction["accepted"] else "No")

                with st.expander("Detailed Prediction Summary", expanded=False):
                    formatted = format_prediction_for_streamlit(prediction)
                    for key, value in formatted.items():
                        st.write(f"**{key}:** {value}")

                st.subheader("Top Predictions")
                top_predictions_df = pd.DataFrame(get_probability_table(prediction))
                st.dataframe(top_predictions_df, use_container_width=True)

                st.subheader("Probability Distribution")
                chart_data = get_bar_chart_data(prediction, CLASS_LABELS)
                chart_df = pd.DataFrame(
                    {
                        "Class": chart_data["labels"],
                        "Confidence (%)": chart_data["probabilities_percent"],
                    }
                )
                st.bar_chart(chart_df.set_index("Class"))

                st.success(
                    f"The image is classified as **{prediction['predicted_label']}** "
                    f"with confidence **{prediction['confidence_percent']:.2f}%**."
                )

            # --------------------------------
            # Downloads
            # --------------------------------
            st.markdown('<div class="section-title">Download Outputs</div>', unsafe_allow_html=True)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    label="Download Enhanced Image",
                    data=encode_image_for_download(enhanced_image, ".png"),
                    file_name="enhanced_image.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with d2:
                st.download_button(
                    label="Download Canny Edge Image",
                    data=encode_image_for_download(canny_edges, ".png"),
                    file_name="canny_edges.png",
                    mime="image/png",
                    use_container_width=True,
                )

            # --------------------------------
            # Summary
            # --------------------------------
            st.markdown('<div class="section-title">Pipeline Summary</div>', unsafe_allow_html=True)

            summary = {
                "enhancement": {
                    "resize_width": resize_width,
                    "clahe": apply_clahe,
                    "denoise": apply_denoise,
                    "brightness_contrast": apply_brightness_contrast,
                    "gamma_correction": apply_gamma,
                    "sharpen": apply_sharpen,
                },
                "feature_extraction": {
                    "orb_keypoints": orb_result["num_keypoints"],
                    "hog_feature_length": hog_result["feature_length"],
                    "contours_found": contour_result["num_contours"],
                },
            }

            if model is not None and target_size is not None:
                summary["classification"] = {
                    "model_path": str(MODEL_PATH),
                    "model_input_size": {
                        "width": target_size[0],
                        "height": target_size[1],
                    },
                    "predicted_label": prediction["predicted_label"],
                    "confidence_percent": prediction["confidence_percent"],
                }

            st.json(summary)

    except ImageEnhancementError as exc:
        st.error(f"Image enhancement error: {exc}")
    except FeatureExtractionError as exc:
        st.error(f"Feature extraction error: {exc}")
    except ClassificationError as exc:
        st.error(f"Classification error: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

else:
    st.info("Upload an image first, then click 'Run Full Pipeline'.")
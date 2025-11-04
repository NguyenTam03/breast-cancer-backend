"""
Machine Learning Model Service for Breast Cancer Prediction
Supports two pathways:
- Direct CNN classifier on images (existing)
- GWO pipeline: CNN feature extractor + GWO-selected features + small classifier
"""

import os
import io
import time
import json
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Tuple, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreastCancerPredictor:
    def __init__(self, model_path: str = "models\model_gwo_selected_feature.h5"):
        """
        Initialize the breast cancer predictor with the trained CNN model
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.input_shape = (224, 224, 3)  # Standard CNN input shape
        self.class_names = ["BENIGN", "MALIGNANT"]
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained CNN model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at: {self.model_path}")
                return False
            
            # Load the model
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            logger.info(f"Model loaded successfully from: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess the input image for model prediction
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            np.ndarray: Preprocessed image array ready for prediction
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((self.input_shape[0], self.input_shape[1]))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Normalize pixel values to [0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """
        Make prediction on the input image
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            Dict containing prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please check model file.")
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction
            start_time = time.time()
            predictions = self.model.predict(processed_image)
            processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            
            # Get prediction probability
            confidence = float(predictions[0][0])
            
            # Determine class based on threshold (0.5)
            predicted_class = "MALIGNANT" if confidence > 0.5 else "BENIGN"
            
            # If predicting malignant, confidence is the raw score
            # If predicting benign, confidence is 1 - raw_score
            final_confidence = confidence if predicted_class == "MALIGNANT" else (1 - confidence)
            
            return {
                "prediction": predicted_class,
                "confidence": round(final_confidence, 3),
                "processing_time": processing_time,
                "raw_score": round(confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict containing model information
        """
        if not self.is_loaded:
            return {"status": "Model not loaded"}
        
        return {
            "status": "Model loaded",
            "input_shape": self.input_shape,
            "model_path": self.model_path,
            "classes": self.class_names,
            "model_summary": str(self.model.summary()) if self.model else None
        }

class GWOPredictor:
    """
    Predictor that uses CNN as feature extractor + GWO-selected features classifier.

    Artifacts expected:
    - CNN base: models/breast_cancer_cnn_model.h5 (or overridden)
    - GWO classifier: models/model_gwo_selected_feature.h5 or models/model_gwo.h5
    - Selected indices: gwo_selected_indices.npy (np.int32 array)
    - Feature extractor spec (optional): feature_extractor_spec.json
        {
          "cnn_model_path": "models/breast_cancer_cnn_model.h5",
          "layer_ref": {"type": "index", "value": -4} | {"type": "name", "value": "flatten_2"},
          "input_from_first_layer": true,
          "target_size": [224,224,3],
          "normalization": "scale_0_1"
        }
    """

    def __init__(
        self,
        cnn_model_path: str = "models/breast_cancer_cnn_model.h5",
        gwo_model_path: Optional[str] = None,
        selected_idx_path: str = "models/gwo_selected_indices.npy",
        extractor_spec_path: str = "models/feature_extractor_spec.json",
    ) -> None:
        self.cnn_model_path = self._resolve_first_existing([
            cnn_model_path,
            os.path.join("models", "breast_cancer_cnn_model.h5"),
            os.path.abspath(os.path.join("models", "breast_cancer_cnn_model.h5")),
            os.path.abspath(os.path.join("..", "models", "breast_cancer_cnn_model.h5")),
        ])
        self.gwo_model_path = self._resolve_first_existing([
            gwo_model_path or os.path.join("models", "model_gwo_selected_feature.h5"),
            os.path.join("models", "model_gwo.h5"),
            os.path.abspath(os.path.join("models", "model_gwo_selected_feature.h5")),
            os.path.abspath(os.path.join("models", "model_gwo.h5")),
        ])
        self.selected_idx_path = self._resolve_first_existing([
            selected_idx_path,
            os.path.join("models", "gwo_selected_indices.npy"),
            os.path.abspath(os.path.join("models", "gwo_selected_indices.npy")),
            os.path.abspath("gwo_selected_indices.npy"),
        ])
        self.extractor_spec_path = self._resolve_first_existing([
            extractor_spec_path,
            os.path.join("models", "feature_extractor_spec.json"),
            os.path.abspath(os.path.join("models", "feature_extractor_spec.json")),
        ], must_exist=False)

        self.cnn_model = None
        self.gwo_model = None
        self.feature_extractor = None
        self.selected_idx = None
        self.is_loaded = False
        self.input_shape = (224, 224, 3)
        self.class_names = ["BENIGN", "MALIGNANT"]

        self._load()

    def _resolve_first_existing(self, candidates, must_exist: bool = True) -> Optional[str]:
        for p in candidates:
            if p and os.path.exists(p):
                return p
        if must_exist:
            return candidates[0]  # return first, will fail later with clear error
        return None

    def _load(self) -> bool:
        try:
            
            if not os.path.exists(self.cnn_model_path):
                logger.error(f"CNN base model not found at: {self.cnn_model_path}")
                return False
            if not os.path.exists(self.gwo_model_path):
                logger.error(f"GWO classifier model not found at: {self.gwo_model_path}")
                return False
            if not os.path.exists(self.selected_idx_path):
                logger.error(f"GWO selected indices not found at: {self.selected_idx_path}")
                return False

            # Load base CNN
            self.cnn_model = tf.keras.models.load_model(self.cnn_model_path)

            # Determine extractor config
            layer_ref = {"type": "index", "value": -4}
            input_from_first = True
            target_size = list(self.input_shape)
            if self.extractor_spec_path and os.path.exists(self.extractor_spec_path):
                try:
                    with open(self.extractor_spec_path, "r", encoding="utf-8") as f:
                        spec = json.load(f)
                    layer_ref = spec.get("layer_ref", layer_ref)
                    input_from_first = spec.get("input_from_first_layer", input_from_first)
                    target_size = spec.get("target_size", target_size)
                except Exception as e:
                    logger.warning(f"Failed to read extractor spec: {e}. Using defaults.")

            self.input_shape = tuple(target_size)

            # Build feature extractor
            input_tensor = self.cnn_model.layers[0].input if input_from_first else self.cnn_model.input
            if layer_ref.get("type") == "name":
                feat_layer = self.cnn_model.get_layer(name=layer_ref.get("value", "flatten_2"))
            else:
                # default index -4
                feat_layer = self.cnn_model.get_layer(index=int(layer_ref.get("value", -4)))
            # Build feature extractor using tf.keras
            KModel = tf.keras.Model
            self.feature_extractor = KModel(inputs=input_tensor, outputs=feat_layer.output)

            # Load selected indices
            self.selected_idx = np.load(self.selected_idx_path)
            if self.selected_idx.ndim != 1:
                self.selected_idx = self.selected_idx.reshape(-1)

            # Load GWO classifier
            # Load GWO classifier using tf.keras
            self.gwo_model = tf.keras.models.load_model(self.gwo_model_path)

            self.is_loaded = True
            logger.info(
                f"GWO pipeline loaded. CNN: {self.cnn_model_path}, GWO: {self.gwo_model_path}, indices: {self.selected_idx_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading GWO pipeline: {e}")
            self.is_loaded = False
            return False

    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((self.input_shape[1], self.input_shape[0]))
            arr = np.array(image).astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            return arr
        except Exception as e:
            logger.error(f"Error preprocessing image (GWO): {e}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def predict(self, image_data: bytes) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("GWO pipeline is not loaded. Check model and indices files.")
        try:
            x = self.preprocess_image(image_data)  # (1, H, W, 3)

            # Extract features
            feat = self.feature_extractor.predict(x)
            feat = feat.reshape(1, -1)

            # Select GWO features
            feat_sel = feat[:, self.selected_idx]

            # Predict
            start = time.time()
            prob = float(self.gwo_model.predict(feat_sel)[0, 0])
            #predict model Cnn không co feat_sel
            # prob = float(self.cnn_model.predict(x)[0, 0])
            processing_time = int((time.time() - start) * 1000)

            predicted_class = "MALIGNANT" if prob > 0.5 else "BENIGN"
            final_confidence = prob if predicted_class == "MALIGNANT" else (1 - prob)

            return {
                "prediction": predicted_class,
                "confidence": round(final_confidence, 3),
                "processing_time": processing_time,
                "raw_score": round(prob, 3)
            }
        except Exception as e:
            logger.error(f"Error during GWO prediction: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")


# Global GWO model instance
_predictor_instance = None

def get_predictor() -> GWOPredictor:
    """
    Get the global GWO predictor instance (singleton pattern)
    
    Returns:
        GWOPredictor: The GWO predictor instance
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = GWOPredictor()
    return _predictor_instance

def predict_breast_cancer(image_data: bytes) -> Dict[str, Any]:
    """
    Convenience function to predict breast cancer from image data using GWO model
    
    Args:
        image_data: Raw image data in bytes
        
    Returns:
        Dict containing prediction results
    """
    predictor = get_predictor()
    return predictor.predict(image_data)

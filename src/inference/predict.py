import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def load_artifacts(artifacts_dir="artifacts"):
    paths = {
        "cnn": os.path.join(artifacts_dir, "cnn_model.h5"),
        "ada": os.path.join(artifacts_dir, "ada_model.pkl"),
        "kmeans": os.path.join(artifacts_dir, "kmeans.pkl"),
    }
    models_loaded = {}
    if os.path.isfile(paths["cnn"]):
        models_loaded["cnn"] = load_model(paths["cnn"])
    if os.path.isfile(paths["ada"]):
        models_loaded["ada"] = joblib.load(paths["ada"])
    if os.path.isfile(paths["kmeans"]):
        models_loaded["kmeans"] = joblib.load(paths["kmeans"])
    return models_loaded


def preprocess_image(img_path, target_size=(224,224)):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img)/255.0
    return np.expand_dims(arr, axis=0)


def predict_single(img_path, artifacts_dir="artifacts"):
    """Return probability estimates (0-1 and percentage) from CNN + AdaBoost and an ensemble average.

    Ensemble = mean of available model probabilities. This is a simple combination,
    not calibrated for medical decision-making. For educational/demo use only.
    """
    models_loaded = load_artifacts(artifacts_dir)
    missing = [m for m in ("cnn", "ada") if m not in models_loaded]
    if missing:
        raise RuntimeError(f"Required models missing: {missing}. Train first.")
    cnn = models_loaded["cnn"]
    ada = models_loaded["ada"]
    arr = preprocess_image(img_path, target_size=cnn.input_shape[1:3])
    # CNN probability (risk class assumed to be 1)
    prob_cnn = float(cnn.predict(arr, verbose=0)[0][0])
    # Feature extraction
    from tensorflow.keras import models as kmodels
    feat_extractor = kmodels.Model(inputs=cnn.input, outputs=cnn.layers[-2].output)
    feats = feat_extractor.predict(arr, verbose=0)
    ada_pred_prob = float(ada.predict_proba(feats)[0][1])
    # Simple ensemble (average)
    ensemble_prob = (prob_cnn + ada_pred_prob) / 2.0
    def pct(x):
        return round(x * 100.0, 2)
    risk_category = (
        "LOW" if ensemble_prob < 0.33 else
        ("MODERATE" if ensemble_prob < 0.66 else "HIGH")
    )
    return {
        "cnn_probability_risk": prob_cnn,
        "cnn_risk_percent": pct(prob_cnn),
        "ada_probability_risk": ada_pred_prob,
        "ada_risk_percent": pct(ada_pred_prob),
        "ensemble_probability_risk": ensemble_prob,
        "ensemble_risk_percent": pct(ensemble_prob),
        "ensemble_risk_bucket": risk_category,
        "disclaimer": "Prototype model. Probabilities are NOT medical advice."
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict heart attack risk from a retinal image")
    parser.add_argument("image", help="Path to retinal image")
    parser.add_argument("--artifacts", default="artifacts", help="Directory with saved models")
    args = parser.parse_args()
    res = predict_single(args.image, artifacts_dir=args.artifacts)
    # Pretty print
    print("Ensemble Risk: {}% ({} category)".format(res["ensemble_risk_percent"], res["ensemble_risk_bucket"]))
    print("Details:")
    for k in ["cnn_risk_percent", "ada_risk_percent"]:
        print(f"  {k}: {res[k]}%")
    print("Raw JSON:")
    print(res)

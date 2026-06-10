"""Batch prediction endpoint — file upload (.npz or .csv)."""

import csv
import io

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

router = APIRouter(prefix="/api")


@router.post("/predict/batch")
async def predict_batch(request: Request, file: UploadFile = File(...)):
    """Upload .npz or .csv file for batch prediction.

    NPZ files should contain 'X' array (n_samples, 36) and optionally 'y' labels.
    CSV files should have 36 feature columns (with optional 'label' first column).
    """
    service = request.app.state.prediction_service
    content = await file.read()

    if file.filename and file.filename.endswith(".npz"):
        data = np.load(io.BytesIO(content))
        X = data.get("X")
        if X is None:
            raise HTTPException(400, "NPZ file must contain an 'X' array")
        y = data.get("y", None)

    elif file.filename and file.filename.endswith(".csv"):
        text = content.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            raise HTTPException(400, "CSV file is empty")

        # Detect if first column is a label (non-numeric header or integer labels)
        try:
            float(rows[0][0])
            has_header = False
        except ValueError:
            has_header = True

        data_rows = rows[1:] if has_header else rows
        if not data_rows:
            raise HTTPException(400, "CSV file has no data rows")

        # Check if first column is label (integer) or feature (float with decimals)
        first_vals = [r[0] for r in data_rows[:5]]
        has_label_col = all(v.strip().isdigit() for v in first_vals) and len(data_rows[0]) == 37

        if has_label_col:
            y = np.array([int(r[0]) for r in data_rows])
            X = np.array([[float(v) for v in r[1:]] for r in data_rows])
        else:
            y = None
            X = np.array([[float(v) for v in r] for r in data_rows])

    else:
        raise HTTPException(400, "Unsupported file format. Use .npz or .csv")

    if X.ndim != 2 or X.shape[1] != 36:
        raise HTTPException(400, f"Expected 36 features per sample, got shape {X.shape}")

    result = service.predict_batch(X, y)
    return result

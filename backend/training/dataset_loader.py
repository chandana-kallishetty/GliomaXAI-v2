"""
dataset_loader.py
=================
Modular dataset loader for BraTS multi-modal MRI structure.
Handles parsing patient folders and missing modality fallbacks.
"""
import os
import numpy as np
from pathlib import Path

def parse_brats_folder(patient_dir: Path):
    """
    Parses a BraTS patient directory to find T1, T1CE, T2, and FLAIR files.
    """
    modalities = {"t1": None, "t1ce": None, "t2": None, "flair": None, "seg": None}
    
    if not patient_dir.exists() or not patient_dir.is_dir():
        return modalities

    for f in os.listdir(patient_dir):
        f_lower = f.lower()
        if "t1." in f_lower or "t1_" in f_lower:
            modalities["t1"] = patient_dir / f
        elif "t1ce" in f_lower:
            modalities["t1ce"] = patient_dir / f
        elif "t2" in f_lower:
            modalities["t2"] = patient_dir / f
        elif "flair" in f_lower:
            modalities["flair"] = patient_dir / f
        elif "seg" in f_lower:
            modalities["seg"] = patient_dir / f
            
    return modalities

def validate_dataset(data_dir: Path):
    """Validates the structure of the training dataset."""
    valid_patients = []
    if data_dir.exists():
        for patient_folder in data_dir.iterdir():
            if patient_folder.is_dir():
                mods = parse_brats_folder(patient_folder)
                if all(mods[m] is not None for m in ["t1", "t1ce", "t2", "flair"]):
                    valid_patients.append(patient_folder)
    return valid_patients

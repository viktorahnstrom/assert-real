"""
Forensic feature extraction for deepfake detection.

Provides pixel-level signal extraction per facial region using three
complementary forensic signals: Laplacian sharpness, FFT high-frequency
energy, and Error Level Analysis (ELA).
"""

from app.services.forensics.report import ForensicsReport, extract

__all__ = ["ForensicsReport", "extract"]

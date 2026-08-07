# Spectrum Image → MGF

**Convert mass spectrum images into MGF files with visual validation.**

Spectrum Image → MGF is a Streamlit application designed to recover centroid mass spectra from figures, screenshots, publications, and other graphical sources.

The application detects spectral peaks, estimates their relative intensities, reads or calibrates their **m/z values from the image**, generates a standard `.mgf` file, and reconstructs the resulting spectrum for visual and numerical validation.

> **Important principle:** the application does not chemically guess m/z values.
> m/z values must be read from the figure, obtained from its calibrated x-axis, or manually verified by the user.

---

## Features

* Upload mass spectrum images (`PNG`, `JPG`, `TIFF`, `BMP`)
* Automatic detection of centroid/stick peaks
* Relative intensity estimation from peak heights
* OCR-based reading of printed m/z values
* Automatic calibration of the m/z axis
* Manual two-point calibration when needed
* Editable peak table for verification
* MGF generation and download
* Reconstruction of the spectrum from the generated MGF
* Visual comparison between source and reconstructed spectra
* Cosine similarity calculation
* Peak-position error diagnostics

---

## Workflow

```text
Mass spectrum image
        │
        ▼
   Peak detection
        │
        ├──► Relative intensity
        │
        ▼
   m/z extraction
        │
        ├── OCR peak labels
        ├── OCR axis calibration
        └── Manual calibration
        │
        ▼
   User verification
        │
        ▼
     MGF file
        │
        ▼
Spectrum reconstruction
        │
        ▼
 Visual + numerical validation
```

---

## Why m/z is not inferred

Peak intensity can reasonably be estimated from the graphical height of a peak.

m/z cannot.

The horizontal position of a peak only represents a meaningful m/z value when the numerical scale of the figure is known.

Spectrum Image → MGF therefore obtains m/z values only from:

1. printed peak labels detected by OCR;
2. numerical labels on the m/z axis;
3. explicit axis calibration;
4. manual user correction.

The software does **not** assign m/z values based on expected fragmentation, chemical plausibility, neighboring peaks, or database knowledge.

---

## MGF output

The resulting spectrum is exported in standard Mascot Generic Format:

```text
BEGIN IONS
TITLE=Example_spectrum
PEPMASS=500.200000
CHARGE=1+
119.035200 34.6000
136.061800 100.0000
152.056700 22.4000
END IONS
```

`TITLE`, `PEPMASS`, and `CHARGE` can be defined before export.

---

## Validation

The generated MGF is parsed again and reconstructed as a spectrum.

The application then projects each MGF m/z value back onto the coordinate system of the original figure:

```text
m/z → expected x position in the image
```

This allows comparison between:

**the peaks actually detected in the image**

and

**the peaks reconstructed from the generated MGF**

A wrongly assigned m/z therefore shifts the reconstructed peak and decreases agreement with the original spectrum.

The application reports:

* cosine similarity;
* individual peak positional errors (`Δx`);
* median absolute positional error;
* maximum absolute positional error.

This provides an additional quality-control step before using the digitized spectrum.

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY>
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

Tesseract is required for automatic reading of numerical labels.

#### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### Windows

Install Tesseract OCR and make sure it is available in the system `PATH`.

If necessary, its location can be explicitly configured:

```python
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
```

---

## Run locally

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## Streamlit Community Cloud

The repository includes:

```text
app.py
requirements.txt
packages.txt
```

The `packages.txt` file contains:

```text
tesseract-ocr
```

allowing the OCR system dependency to be installed during deployment.

---

## Recommended input

Best results are obtained with:

* centroid/stick mass spectra;
* high-resolution images;
* white or light backgrounds;
* dark spectral peaks;
* clearly visible horizontal axes;
* readable m/z labels;
* linear m/z scales;
* limited overlapping text or annotations.

Whenever the original digital mass spectrum is available, it should be preferred over image digitization.

This application is intended primarily for cases where the **graphical spectrum is the available source**.

---

## Limitations

OCR results are not ground truth and should always be inspected before scientific use.

The accuracy of m/z values obtained from axis calibration is limited by the resolution and quality of the original figure.

Factors such as image scaling, compression, line thickness, rasterization, and poorly defined axes can affect the recovered spectrum.

The current version is primarily designed for **linear centroid/stick spectra**.

---

## Scientific use

The software follows a deliberately conservative workflow:

```text
IMAGE MEASUREMENT
       ↓
Peak position + relative intensity

READING / CALIBRATION
       ↓
Graphical position → m/z

USER VERIFICATION
       ↓
Verified peak list

MGF GENERATION
       ↓
Digital spectrum

VALIDATION
       ↓
MGF projected back onto source image
```

The goal is to **digitize the information contained in the figure**, not to generate what the spectrum is expected to contain.

---

## Future development

Planned or potential improvements include:

* per-peak m/z confidence scores;
* automatic classification as `Reliable`, `Review`, or `Unreadable`;
* improved OCR for mass spectral labels;
* multipoint axis calibration;
* automatic plot-boundary detection;
* rotated-label recognition;
* nonlinear axis support;
* external MGF/reference-spectrum comparison;
* ppm and Da peak-matching tolerances;
* conventional and modified cosine similarity;
* mirror plots;
* batch image processing;
* extraction/validation reports;
* uncertainty estimates for image-derived m/z values.

---

## Disclaimer

Spectrum Image → MGF is intended as a scientific digitization and data-recovery tool.

Automatically extracted spectra should be visually verified before use in spectral libraries, compound annotation, database submission, or downstream scientific analyses.

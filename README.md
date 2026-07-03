# istanbul-layered-gradient
Reproducibility materials for a manuscript under review at Environment and Planning B
# Istanbul as a Layered Gradient

Reproducibility materials for the manuscript:

> "Istanbul as a Layered Gradient: A Structural Analysis of 
> Street-Scale Urban Fabric in a Sample Guided by Classical 
> Urban Safety Theories"
> 
> Currently under peer review at *Environment and Planning B: 
> Urban Analytics and City Science*.
> # Istanbul as a Layered Gradient

Reproducibility materials for the manuscript:

> "Istanbul as a Layered Gradient: A Structural Analysis of 
> Street-Scale Urban Fabric in a Sample Guided by Classical 
> Urban Safety Theories"
> 
> Currently under peer review at *Environment and Planning B: 
> Urban Analytics and City Science*.

## Contents

- **Analysis notebook** (`istanbul_analysis.ipynb`) — full pipeline 
  from data collection to five robustness tests and sensitivity analysis.
- **Master dataset** — 74 streets × 12 features (semantic segmentation 
  pixel-share vectors), cluster assignments (main and sensitivity), 
  and safety labels.

## Requirements

Python 3.9+

Key dependencies:
- TensorFlow 2.x (for DeepLabv3+ inference)
- scikit-learn (K-means, GMM, cophenetic)
- hdbscan
- pandas, numpy, scipy
- matplotlib, seaborn

## Reproducing the analysis

1. Open `istanbul_analysis.ipynb` in Google Colab or Jupyter
2. Set your Google Maps Static API key as an environment variable:
3. 3. Run all cells sequentially

## Data note

Original Google Street View panoramas are not redistributed due to 
Google's terms of service. The data collection cell in the notebook 
reproduces the image set given a valid Google Maps Static API key.

Neighborhood-level safety scores were derived from Bilen (2016), a 
publicly available doctoral thesis in Turkish.

## License

MIT License for code. Data licensed under CC-BY 4.0 (except Google 
Street View imagery, subject to Google's terms of service).

## Citation

Citation will be added upon acceptance.

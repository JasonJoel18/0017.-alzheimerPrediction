# 🧠 Uncertainty-Aware Deep Learning for Early Detection of Alzheimer's Disease Using MRI Image Data

## 📖 Overview
This project leverages advanced deep learning techniques with uncertainty estimation to aid in the early detection of Alzheimer's disease using MRI data. It includes a full pipeline from data preprocessing to deployment, with a user-friendly web interface for practical use. Designed for both research and clinical applications.

---

## ✨ Features
- **🔄 Advanced Data Processing**
  - Robust preprocessing pipeline for MRI images.
  - Support for both full datasets and test subsets.
  - Automated workflows for seamless data preparation.

- **🤖 Machine Learning Model**
  - State-of-the-art uncertainty estimation.
  - Comprehensive training and evaluation pipelines.
  - Prediction with confidence scores.

- **📊 Visualization Tools**
  - Uncertainty visualization for better interpretability.
  - Publication-quality figure generation.
  - Interactive, web-based visualizations.

- **🌐 Web Interface**
  - Built using Flask for simplicity and speed.
  - Intuitive prediction and visualization tools.
  - Real-time uncertainty analysis.

## Project Structure

    ├── data              <- Main dataset directory containing MRI images
    │   └── external      <- Data from Kaggle sources
    │
    ├── data2             <- Subset of data for testing with smaller dataset
    │
    ├── docs              <- Documentation and research outputs
    │   ├── figures       <- Generated graphics and figures
    │   │   ├── fig1.png
    │   │   ├── fig2.jpeg
    │   │   └── ... (visualization outputs)
    │   ├── base_model    <- Kaggle notebook outputs for baseline model
    │   ├── proposed_model<- Kaggle notebook outputs for proposed model
    │   ├── AlzheimerPrediction.tex
    │   └── wordcount.log
    │
    ├── models            <- Trained and serialized models
    │   └── .gitkeep
    │
    ├── references        <- Reference materials
    │   └── folder_structure.txt
    │
    ├── reports           <- Generated analysis reports
    │
    ├── src               <- Source code
    │   ├── data          <- Scripts for dataset preparation
    │   │   └── make_dataset.py
    │   ├── models        <- Model training and prediction scripts
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   ├── visualization <- Visualization scripts
    │   │   ├── visualize.py           <- Results visualization with uncertainty
    │   │   └── visualize_for_paper.py <- Thesis paper figure generation
    │   ├── scripts       <- Web application files
    │   │   └── app.py    <- Flask web application
    │   ├── static        <- Static files for web app (CSS, output images)
    │   └── index.html    <- Web application interface
    │
    ├── README.md         <- Project documentation
    ├── environment2.yml  <- Environment configuration
    └── alzheimerPrediction.code-workspace

## Installation

1. Clone the repository
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create a virtual environment (recommended)
- On Mac/Linux:
```bash
python -m venv venv
```
- On Windowsx:
```bash
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

```bash
python src/data/make_dataset.py
```

### Model Training
```bash
python src/models/train_model.py
```

### Model Prediction
```bash
python src/models/predict_model.py
```

### Visualization
```bash
python src/visualization/visualize.py`
```
### Web Application
```bash
python scripts/app.py`
```

Then navigate to `http://localhost:5000` in your web browser.

## Development and Testing

- Use the `data2` directory for testing with a smaller dataset
- Run tests before submitting any changes
- Follow the existing code style and documentation patterns

## Documentation

- Detailed documentation is available in the `docs` directory
- LaTeX files contain comprehensive methodology and results
- Kaggle notebook outputs are stored in `docs/base_model` and `docs/proposed_model`

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Run tests
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

## Contact

jasonjoel188@gmail.com

## Acknowledgments

- [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) for project structure inspiration.

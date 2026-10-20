# 🧠 Deep Learning Practical Exam — Set C

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Project](https://img.shields.io/badge/Project-Practical%20Exam-2F6FEB)

> **Deep Learning Practical Exam · Set C**

A reproducible, notebook-based practical exploring five core deep-learning and computer-vision workflows: ANN classification, OpenCV image processing, CNN digit recognition, GRU time-series forecasting, and pretrained Transformer inference. Each section loads data, performs the task, evaluates the output, and saves relevant results.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Datasets](#-datasets)
- [Workflow and Models](#-workflow-and-models)
- [Results Summary](#-results-summary)
- [Key Visualizations](#-key-visualizations)
- [Repository Structure](#-repository-structure)
- [Installation and Usage](#-installation-and-usage)
- [Outputs and Saved Models](#-outputs-and-saved-models)
- [Demonstration Video](#-demonstration-video)
- [Author](#-author)

---

## 📌 Project Overview

This project brings together five standalone practical exercises in one Jupyter notebook. It focuses on understanding the end-to-end workflow—not only building models, but also preparing data, selecting suitable metrics, visualising results, and explaining limitations.

The work follows the Set C project instructions and includes a results summary, a written reflection, saved visualisations, and trained model files.

## 🎯 Objectives

- Train and evaluate a multi-class Artificial Neural Network (ANN) on tabular data.
- Apply OpenCV colour-space conversion, colour masking, morphology, Sobel edge detection, and Haar-cascade face detection.
- Train a Convolutional Neural Network (CNN) for handwritten-digit recognition and compare it with an ANN baseline.
- Forecast daily minimum temperatures with a GRU and compare it with a naive baseline.
- Use pretrained Hugging Face pipelines for fill-mask prediction, extractive question answering, and text generation.
- Save results and models and document the practical findings.

---

## 📂 Datasets

| Step | Dataset / Input | Task |
|---|---|---|
| **1 · ANN** | [Wine Recognition dataset](https://archive.ics.uci.edu/dataset/109/wine), loaded through `sklearn.datasets.load_wine` | Three-class classification using 178 samples and 13 chemical features |
| **2 · OpenCV** | Four sample images from `scikit-image` plus `data/images/RedFlower.jpg` | Colour processing, edge detection, and face detection |
| **3 · CNN** | [MNIST handwritten digits](https://keras.io/api/datasets/mnist/), using a 20,000-image training subset and 4,000-image test subset | Ten-class digit classification; CNN compared with an ANN baseline |
| **4 · GRU** | [Daily Minimum Temperatures in Melbourne](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv) | Predict the next day's minimum temperature from the previous 30 days |
| **5 · Transformers** | Five India-related text paragraphs, ten question-answering questions, five masked sentences, and three generation prompts | Fill-mask, question answering, and text generation using pretrained models |

### Pretrained Transformer models

- Fill-mask: [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)
- Question answering: [`distilbert-base-cased-distilled-squad`](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad)
- Text generation: [`distilgpt2`](https://huggingface.co/distilbert/distilgpt2)

---

## 🔄 Workflow and Models

1. **ANN — Wine classification:** inspect and scale the features, train a dense neural network, and review accuracy, classification metrics, and the confusion matrix.
2. **OpenCV — Image processing:** resize images, convert to HSV, isolate a colour, clean the mask with morphology, calculate Sobel edges, and run Haar-cascade face detection.
3. **CNN — Digit recognition:** normalise MNIST images, train a CNN and a dense ANN baseline, compare test accuracy and parameter counts, and inspect feature maps and incorrect predictions.
4. **GRU — Temperature forecasting:** create 30-day sequences, train a GRU without shuffling the time series, and compare its predictions with a naive forecast.
5. **Transformers — Text inference:** run fill-mask, question-answering, and text-generation pipelines; manually check QA answers and visualise confidence scores.
6. **Comparison and reflection:** display the saved results summary and reflect on model trade-offs and limitations.

---

## 📊 Results Summary

The following values are from the executed notebook's `outputs/results_summary.csv` as shown in the results table.

| Mini-project | Dataset / Input | Metric | Recorded result |
|---|---|---|---:|
| ANN | Wine Recognition — 178 rows, 13 features | Test accuracy | **94.44%** |
| OpenCV | Five images | Faces detected | **1** in `astronaut`; `0` in `coffee`, `chelsea`, `rocket`, and `red_flower` |
| CNN | MNIST — 20,000 training / 4,000 test images | Test accuracy | **97.55%** |
| GRU | Melbourne daily minimum temperatures | RMSE | **2.23 °C** |
| Transformers | Five paragraphs / ten QA questions | Manually checked QA answers | **7/10 correct** |

**How to interpret the table:** These tasks have different objectives and metrics, so their results should not be ranked directly against one another. The CNN accuracy is for digit classification; the GRU RMSE measures temperature prediction error in degrees Celsius; and the Transformer QA result is based on manual checking. The OpenCV count reports detections, not a classification accuracy.

---

## 🖼️ Key Visualizations

The notebook saves its figures in `outputs/`. The images below are linked to the generated filenames in this repository. Run the notebook to regenerate them if needed.

### 1. OpenCV image-processing grid

![OpenCV image-processing grid](outputs/opencv_processing_grid.png)

**Interpretation:** The grid shows the five input images across the original, HSV, red-colour mask, morphology-cleaned mask, and Sobel-edge views. It demonstrates how each operation changes the image representation and highlights different visual information.

### 2. ANN training and validation accuracy

![ANN training and validation accuracy](outputs/ann_accuracy.png)

**Interpretation:** The training and validation curves show how the ANN's classification accuracy changes over 60 epochs. Comparing the two curves helps assess learning progress and whether validation performance follows training performance.

### 3. CNN confusion matrix

![CNN confusion matrix](outputs/cnn_confusion_matrix.png)

**Interpretation:** The diagonal cells represent correctly classified digits, while off-diagonal cells show misclassifications. The matrix helps identify which digit classes the CNN confuses, complementing the recorded test accuracy of **97.55%**.

### 4. GRU actual-versus-predicted temperatures

![GRU actual versus predicted temperatures](outputs/rnn_actual_vs_predicted.png)

**Interpretation:** The plot compares actual minimum temperatures with the GRU's predictions across the displayed test period. The recorded RMSE is **2.23 °C**; use the notebook's naive-baseline comparison to judge whether the GRU adds predictive value.

### 5. Transformer question-answering confidence

![Transformer question-answering confidence](outputs/transformer_qa_confidence.png)

**Interpretation:** Each bar represents a question's confidence score, with green and red indicating answers marked correct and incorrect during manual checking. The run recorded **7 correct answers out of 10**. A high confidence score should not be treated as proof that an answer is factually correct.

---

## 🗂️ Repository Structure

```text
DeepLearning_Practical_Exam_Set_C/
├── DeepLearning_Practical_Exam_SetC.ipynb
├── README.md
├── requirements.txt
├── summary_report.md
├── ann_wine_model.h5
├── cnn_mnist_model.h5
├── rnn_temperature_model.h5
├── data/
│   └── images/
│       └── RedFlower.jpg
└── outputs/
    ├── ann_accuracy.png
    ├── ann_confusion_matrix.png
    ├── ann_loss.png
    ├── ann_wine_class_distribution.png
    ├── opencv_processing_grid.png
    ├── opencv_face_detections.png
    ├── cnn_accuracy.png
    ├── cnn_confusion_matrix.png
    ├── cnn_feature_maps.png
    ├── cnn_loss.png
    ├── cnn_mnist_samples.png
    ├── cnn_wrong_predictions.png
    ├── rnn_actual_vs_predicted.png
    ├── rnn_loss.png
    ├── rnn_temperature_1981.png
    ├── rnn_temperature_full_series.png
    ├── transformer_qa_confidence.png
    └── results_summary.csv
```

*The output folder may contain additional generated OpenCV face-detection images. Keep the filenames consistent with the files actually present in your repository.*

---

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/PareeSojitra0803/deep-learning-practical-exam-set-c.git
cd deep-learning-practical-exam-set-c
```

If your repository uses a different URL or capitalisation, replace the URL and folder name with the exact values shown on GitHub.

### 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Python 3.11 is recommended for the environment used with this project. Check `requirements.txt` if you have pinned package versions.

### 4. Launch Jupyter

```bash
python -m jupyter lab
```

Open `DeepLearning_Practical_Exam_SetC.ipynb`, select the project's `.venv` kernel, and run the notebook from top to bottom.

**Internet access is needed** to download the Melbourne temperature CSV and pretrained Hugging Face models on first use. MNIST may also download on first use. Model downloads can take time.

---

## 💾 Outputs and Saved Models

After successful execution, the notebook generates:

- `ann_wine_model.h5` — trained ANN for Wine Recognition.
- `cnn_mnist_model.h5` — trained CNN for MNIST digit recognition.
- `rnn_temperature_model.h5` — trained GRU temperature forecasting model.
- `outputs/` — evaluation plots, OpenCV processing and face-detection images, and `results_summary.csv`.
- `summary_report.md` — written practical summary.

The notebook's results summary is saved to `outputs/results_summary.csv`. Re-run the relevant cells if you change the models or datasets so the figures and summary reflect the latest run.

---

## 📝 Reflection Highlights

- The ANN achieved **94.44%** test accuracy on the Wine Recognition dataset.
- The CNN achieved **97.55%** test accuracy on the selected MNIST subset.
- The GRU recorded an RMSE of **2.23 °C**; its value should be considered alongside the naive baseline.
- The Transformer question-answering pipeline returned **7/10** answers marked correct during manual checking.
- Classical OpenCV operations provide an interpretable sequence of image transformations, while Haar-cascade face detection can miss faces or produce false positives depending on image conditions.
- Generative model output should be verified before being used as factual information.

---

## 🎥 Demonstration Video


**Video link:** https://drive.google.com/file/d/17CWNy86AhkEWCKF5EwGp2diNiFS5hhyz/view?usp=sharing

---

## 👩‍💻 Author

***Paree Sojitra***  

- **GitHub:** [PareeSojitra0803](https://github.com/PareeSojitra0803)
- **Repository:** [Deep Learning Practical Exam](https://github.com/PareeSojitra0803/deep-learning-practical-exam-set-c)

---

*Academic practical project completed for Red & White Skill Education. Results are reported from the notebook output supplied for this project.*

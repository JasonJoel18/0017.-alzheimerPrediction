
import os
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from PIL import Image
from tensorflow.keras.callbacks import ReduceLROnPlateau
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from rich import print
import time

warnings.filterwarnings(action="ignore")


def visualize_predictions_with_uncertainty(
    images, actual_labels, predicted_labels, uncertainty, class_labels
):
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), dpi=300)
    plt.subplots_adjust(hspace=0.5, wspace=0.2)

    for i in range(10):
        ax_img = axes[i // 5, i % 5]
        ax_bar = ax_img.inset_axes([0.0, -0.25, 1.0, 0.1])

        img = images[i]
        img = np.clip(img, 0, 255).astype(np.uint8)
        ax_img.imshow(img)
        ax_img.axis("off")

        actual_label = class_labels[actual_labels[i]]
        predicted_label = class_labels[predicted_labels[i]]

        # Calculate uncertainty and confidence
        if isinstance(uncertainty[i], np.ndarray):
            uncertainty_value = uncertainty[i].flatten()[0]
        else:
            uncertainty_value = uncertainty[i]
        confidence = 1 - uncertainty_value

        title_color = "green" if actual_labels[i] == predicted_labels[i] else "red"
        ax_img.set_title(
            f"Actual: {actual_label}\nPred: {predicted_label}",
            fontsize=10,
            loc="center",
            color=title_color,
            fontweight="bold",
        )
        colors = ["#FF4C4C", "#FFD34F", "#4CFF4C"]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_gradient", ["#FF4C4C", "#FFD34F", "#FFD34F", "#4CFF4C"], N=100
        )
        ax_bar.imshow(np.linspace(0, 1, 100).reshape(1, -1), aspect="auto", cmap=cmap)
        ax_bar.plot([confidence * 100, confidence * 100], [0, 1], color="black", lw=2)
        ax_img.text(
            0.5,
            -0.1,
            f"{confidence * 100:.2f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=12,
            fontweight="bold",
            transform=ax_img.transAxes,
        )

        ax_bar.set_xlim(0, 100)
        ax_bar.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(
        "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/reports/figures/predictions_with_uncertainty.png",
        dpi=300,
        bbox_inches="tight",
    )


visualize_predictions_with_uncertainty(
    sample_images,
    actual_labels_indices,
    predicted_labels,
    uncertainty[:10],
    cls_lbl,
)


import pandas as pd

timing_df = pd.DataFrame(timing_results)
print("\n[bold green]Timing Results for Each Stage[/bold green]")
print(timing_df)

timing_df.to_csv("timing_results.csv", index=False)
# ==================================



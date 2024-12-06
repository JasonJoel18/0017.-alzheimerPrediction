``` mermaid
flowchart LR
    A[Start] --> B{Import Libraries}
    B --> C{Define Constants & Parameters<br/>(IMAGE_SIZE, BATCH_SIZE, AUTOTUNE)}
    C --> D{Load Image Paths & Labels<br/>(from Directories)}
    D --> E{Create DataFrame<br/>(img_path, lbl)}
    E --> F{Split Data<br/>(Train-Test-Validation)}
    F --> G{Create Datasets<br/>(with Preprocessing)}
    G --> H{"Preprocess Image<br/>(with OpenCV)"<br/>- Read Image<br/>- Grayscale Conversion<br/>- Thresholding<br/>- Contour Detection<br/>- Cropping<br/>- Resizing<br/>- MobilenetV2 Preprocessing"}
    H --> G
    G --> I{Build Model<br/>(Xception)}
    I --> J{Compile Model<br/>(Adamax, Categorical Crossentropy)}
    J --> K{Train Model<br/>(Early Stopping)}
    K --> L{Evaluate Model<br/>(Test Loss, Test Accuracy)}
    K --> M{Visualize Training History<br/>(Accuracy, Loss Plots)}
    L --> N{Predict on Test Data}
    N --> O{Visualize Predictions<br/>(Images with Predicted/Actual Labels)}
    L --> P{Evaluate Metrics<br/>(Accuracy, Classification Report, Confusion Matrix)}
    P --> Q{Visualize Confusion Matrix}
    M --> Q
    N --> Q
    Q --> R[End]
    ```
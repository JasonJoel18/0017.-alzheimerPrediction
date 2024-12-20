y_true = test.classes
y_true = np.array(y_true)
y_pred = np.argmax(mean_predictions, axis=1)


report = classification_report(
    y_true, y_pred, target_names=class_labels, output_dict=True
)
class_metrics = pd.DataFrame(report).transpose()
class_accuracy = [
    np.sum(y_pred == idx) / len(y_true) for idx in range(len(class_labels))
]
class_metrics.loc[class_labels, "accuracy"] = class_accuracy
accuracy_macro = np.mean(class_accuracy)
accuracy_micro = np.sum(y_pred == y_true) / len(y_true)
accuracy_weighted = np.average(
    class_accuracy, weights=[np.sum(y_true == i) for i in range(len(class_labels))]
)
class_metrics.loc["macro avg", "accuracy"] = accuracy_macro
class_metrics.loc["micro avg", "accuracy"] = accuracy_micro
class_metrics.loc["weighted avg", "accuracy"] = accuracy_weighted
precision_micro = precision_score(y_true, y_pred, average="micro")
recall_micro = recall_score(y_true, y_pred, average="micro")
f1_micro = f1_score(y_true, y_pred, average="micro")
class_metrics.loc["micro avg", "precision"] = precision_micro
class_metrics.loc["micro avg", "recall"] = recall_micro
class_metrics.loc["micro avg", "f1-score"] = f1_micro

print("\nEach Class Metrics")
display(class_metrics)
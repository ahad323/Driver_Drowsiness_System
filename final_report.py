import matplotlib.pyplot as plt
import numpy as np

# ======================
# SAMPLE VALUES (replace with your real results)
# ======================

eye_metrics = {
    "Accuracy": 0.92,
    "Precision": 0.90,
    "Recall": 0.93,
    "F1": 0.91,
    "AUC": 0.94
}

yawn_metrics = {
    "Accuracy": 0.89,
    "Precision": 0.87,
    "Recall": 0.88,
    "F1": 0.87,
    "AUC": 0.90
}

labels = list(eye_metrics.keys())

eye_values = list(eye_metrics.values())
yawn_values = list(yawn_metrics.values())

x = np.arange(len(labels))

# ======================
# 📊 COMPARISON BAR CHART
# ======================
plt.figure(figsize=(8,5))
plt.bar(x - 0.2, eye_values, 0.4, label="Eye Model")
plt.bar(x + 0.2, yawn_values, 0.4, label="Yawn Model")

plt.xticks(x, labels)
plt.ylim(0, 1.1)
plt.title("Model Performance Comparison")
plt.legend()
plt.savefig("results/metrics_comparison.png")
plt.close()

# ======================
# 📊 FINAL SUMMARY PLOT (MEAN SCORE)
# ======================
eye_avg = np.mean(eye_values)
yawn_avg = np.mean(yawn_values)

plt.figure()
plt.bar(["Eye Model", "Yawn Model"], [eye_avg, yawn_avg])
plt.ylim(0,1)
plt.title("Overall Model Score")
plt.savefig("results/overall_score.png")
plt.close()

print("Final report graphs saved ✅")
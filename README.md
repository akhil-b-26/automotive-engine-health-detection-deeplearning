# 🚗 Automotive Vehicle Engine Health Detection using Deep Learning

This project leverages deep learning to build a binary classification system that detects whether an automotive engine is **Healthy** or **Unhealthy** based on real-time sensor data. The model is trained on a labeled dataset sourced from **Kaggle** and achieves high predictive accuracy.

---

## 🎯 Objective

Develop a reliable classification system that uses sensor data to determine engine health status, enabling proactive maintenance and reducing unexpected breakdowns.

---

## 🔍 Motivation

Modern vehicles generate vast amounts of sensor data, yet manual diagnostics remain slow and reactive. This project explores how deep learning can automate fault detection, improve predictive maintenance, and enhance overall vehicle reliability.

---

## 🛠️ Technologies Used

* **Language**: Python
* **Frameworks/Libraries**:

  * `Pandas`, `NumPy` – Data manipulation
  * `Matplotlib`, `Seaborn` – Visualization
  * `Scikit-learn` – Preprocessing, evaluation
  * `TensorFlow`, `Keras` – Deep learning
* **Platform**: Google Colab / Jupyter Notebook

---

## 📁 Dataset

* **Source**: [Kaggle](https://www.kaggle.com)
* **File**: `engine_health_dataset_corrected_new.csv`

### Features Used:

| Feature                | Description                       |
| ---------------------- | --------------------------------- |
| Engine rpm             | Engine rotational speed (in RPM)  |
| Lubricant oil pressure | Engine oil pressure               |
| Fuel pressure          | Fuel system pressure              |
| Coolant pressure       | Cooling system pressure           |
| Lubricant oil temp     | Oil temperature (in °C)           |
| Coolant temp           | Coolant temperature (in °C)       |
| Engine Condition       | Target (0: Unhealthy, 1: Healthy) |

---

## 📊 Exploratory Data Analysis (EDA)

* Histograms per feature categorized by engine condition
* Correlation heatmap to identify feature relationships

---

## ⚙️ Data Preprocessing

* Train/Validation/Test Split: 76% / 8% / 16%
* Standardized using `StandardScaler`
* Label encoded binary target (0 = Unhealthy, 1 = Healthy)

---

## 🧠 Model Architecture

A feedforward neural network designed for binary classification:

```python
model = Sequential([
    Dense(128, input_dim=6, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam
* **Epochs**: 50
* **Batch Size**: 32
* **Activations**: ReLU (hidden), Sigmoid (output)

---

## 📈 Model Evaluation

The model's performance was evaluated using accuracy, precision, recall, and a confusion matrix.

### Final Evaluation Metrics:

```text
Accuracy : 99.49%
Precision: 99.29%
Recall   : 100.00%
```

These results indicate strong classification performance and excellent generalization to unseen data.

---

## 🔍 Prediction Function

A utility to test new sensor inputs:

```python
def check_engine_condition(new_input):
    new_input = scaler.transform([new_input])
    prediction = model.predict(new_input)
    return "Healthy" if prediction[0] >= 0.5 else "Unhealthy"
```

**Example:**

```python
new_input = [2166, 26.6, 40.4, 14.9, 187.9, 204.3]
print(check_engine_condition(new_input))
# Output: The engine is: Healthy
```

---

## 🚀 Future Enhancements

* Multiclass prediction for specific fault types
* Integration into vehicle embedded systems
* Deployment with mobile/web dashboards for real-time use

---

## 📂 Project Structure

```
├── Engine_Health_Detection__DL_PROJECT.ipynb
├── README.md
```

---

## 📜 License

Distributed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

* Kaggle for dataset resources
* TensorFlow/Keras for deep learning tools

---

### 💡 Contributions welcome!

Feel free to fork the repo, raise issues, or submit pull requests to improve the system further.

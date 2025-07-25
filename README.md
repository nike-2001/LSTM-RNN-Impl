# 🧠 Next Word Prediction Using LSTM and GRU

## 📌 Project Overview
This project implements a deep learning model to predict the **next word in a text sequence** using **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) neural networks. The dataset used is Shakespeare's **Hamlet**, accessed via the NLTK library. The entire pipeline—from preprocessing to training and deployment—is included, with a **Streamlit web app** for real-time predictions.

## 🗂️ Project Structure

```
LSTM_RNN/
│
├── app.py                    # Streamlit web app
├── data_preprocessing.py     # Script for data preprocessing
├── model_lstm.py            # LSTM model implementation
├── model_gru.py             # GRU model implementation
├── utils.py                 # Utility functions
├── hamlet_sequences.pkl     # Preprocessed sequence data
├── requirements.txt         # Required libraries
└── README.md               # Project description
```

## 📥 1. Data Collection
- **Dataset**: *Hamlet* by William Shakespeare
- **Source**: NLTK's Gutenberg corpus
- **Purpose**: Provides a challenging and rich dataset for sequence-based word prediction.

```python
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

text = gutenberg.raw('shakespeare-hamlet.txt')
```

## 🧹 2. Data Preprocessing
**Steps**:
- Tokenize the text
- Convert tokens into sequences
- Pad the sequences for uniform input shape
- Save preprocessed data (`hamlet_sequences.pkl`)
- Split into training and testing datasets

## 🧠 3. Model Building

### 🔹 LSTM Model Architecture
- Embedding layer to convert word indices to dense vectors
- Two stacked LSTM layers to capture temporal dependencies
- Dense output layer with Softmax activation for word prediction

### 🔸 GRU Model Architecture
- Same structure as LSTM, but replaces LSTM layers with GRU layers

## 🏋️ 4. Model Training
- Trained using `categorical_crossentropy` loss
- Optimized using Adam optimizer
- Early stopping implemented to prevent overfitting

## 📊 5. Model Evaluation
- Evaluate both LSTM and GRU models with test inputs
- Metrics: prediction accuracy, loss trends
- Sample output predictions included for comparison

## 🚀 6. Deployment with Streamlit
A simple **Streamlit web interface** allows users to:
- Input a sequence of words
- Get a predicted **next word**

**Example**:
```
Input: "To be or not"
Output: "to"
```

## ⚙️ Environment Setup

### ✅ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### ✅ Install Dependencies
```bash
pip install -r requirements.txt
```

## 📦 Requirements
```txt
nltk
tensorflow
streamlit
numpy
pandas
```

⚠️ **Note**: Make sure to run the following to download the dataset:
```python
import nltk
nltk.download('gutenberg')
```

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LSTM_RNN
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('gutenberg')
   ```

4. **Run data preprocessing**
   ```bash
   python data_preprocessing.py
   ```

5. **Train models**
   ```bash
   python model_lstm.py
   python model_gru.py
   ```

6. **Launch Streamlit app**
   ```bash
   streamlit run app.py
   ```

## 🔜 Next Steps
- Download and preprocess text
- Build and train LSTM model
- Build and train GRU model
- Evaluate predictions
- Deploy with Streamlit

## 📝 Key Takeaways
- Demonstrates next word prediction with LSTM and GRU networks
- Uses Shakespeare's *Hamlet* as a complex text corpus
- Full pipeline: preprocessing → training → evaluation → deployment
- Real-time interaction via a Streamlit web application

## 🔧 Usage
Once the Streamlit app is running, simply:
1. Enter a sequence of words in the input field
2. Click "Predict Next Word"
3. View the predicted next word based on the trained model

## 📈 Model Performance
Both LSTM and GRU models are trained and evaluated for comparison:
- **LSTM**: Better at capturing long-term dependencies
- **GRU**: Faster training with comparable performance

## 🤝 Contributing
Feel free to fork this project and submit pull requests for improvements!

## 📄 License
This project is open source and available under the [MIT License](LICENSE).

## 📌 Conclusion
This end-to-end NLP project showcases how deep learning can be used for next word prediction. With both **LSTM** and **GRU** models implemented and deployed, this project serves as a practical example of sequence modeling and real-world application via **Streamlit**.

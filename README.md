# Online News Popularity Analysis

I built this project to predict whether news articles will go viral based on their content features. Started with a Jupyter notebook, then refactored everything into a proper Python package with multiple ML algorithms implemented from scratch.

## What's Inside

The project is split into focused modules - makes it way easier to work with than one giant script:

```
OnlineNewPopularity/
├── data/                     # The dataset (CSV + column descriptions)
├── preprocessing.py          # Cleans data and creates new features
├── models.py                 # 4 ML algorithms I coded from scratch
├── evaluation.py             # Measures how well models perform
├── main.py                   # Runs everything with command line options
├── model_persistence.py      # Save/load trained models
├── dashboard.py              # Interactive web interface
├── cross_validation.py       # More robust model testing
├── report_generator.py       # Creates professional reports
└── requirements.txt          # What you need to install
```

## The Models I Built

Instead of using sklearn, I coded these from scratch to really understand how they work:

- **Logistic Regression**: Gradient descent with sigmoid activation (handles overflow issues)
- **Naive Bayes**: Hybrid approach for both continuous and categorical features  
- **Neural Network**: PyTorch implementation with ReLU/Sigmoid layers
- **Support Vector Machine**: SMO algorithm with linear, polynomial, and RBF kernels

The data preprocessing is pretty thorough too - I clean up the messy column names, create new features like keyword spreads and sentiment averages, remove highly correlated features, and use statistical tests to pick the most important ones.

### 📊 **Model Performance Results**

Here's how the models performed on the test set:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | 99.55% | 99.68% | 99.46% | **99.57%** |
| Logistic Regression | 64.64% | 66.03% | 69.48% | 67.71% |
| Neural Network | 53.35% | 53.36% | 99.97% | 69.58% |
| Support Vector Machine | 52.75% | 52.75% | 100.00% | 69.07% |

**Key Insights:**
- **Naive Bayes** performed surprisingly well (possibly overfitting)
- **Logistic Regression** shows realistic performance for this problem
- **Neural Network** and **SVM** have high recall but low precision
- The dataset has **53.3% popular articles** (21,154 out of 39,644)

## Getting Started

1. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the basic analysis**:
   ```bash
   python main.py
   ```

That's it! The script will load the data, train all models, and show you the results.

## Cool Features I Added

Once I got the basic pipeline working, I added some nice extras:

- **Save trained models**: Never retrain again with `--save-models`
- **Interactive dashboard**: Web interface with `--launch-dashboard` 
- **Cross-validation**: More reliable results with `--cross-validation`
- **Professional reports**: HTML reports with `--generate-reports`
- **Custom model selection**: Pick which models to run with `--models logistic naive_bayes`

### 🌐 **Interactive Dashboard**

The dashboard provides a complete web interface for exploring your data and models:

- **🏠 Overview**: Dataset summary and key metrics
- **📊 Data Explorer**: Interactive feature analysis and correlations
- **🤖 Model Results**: Performance comparison of all trained models
- **🔮 Predictions**: Real-time prediction interface with saved models
- **📈 Analytics**: Advanced visualizations and model insights

**Launch the dashboard:**
```bash
streamlit run dashboard.py --server.headless true
```
Then open `http://localhost:8501` in your browser.

### Command Line Options

You can customize pretty much everything:

| Option | What it does | Default |
|--------|-------------|---------|
| `--models` | Which models to train | `all` |
| `--test-size` | How much data for testing | `0.3` |
| `--save-models` | Save trained models | `False` |
| `--cross-validation` | Run k-fold CV | `False` |
| `--launch-dashboard` | Start web interface | `False` |
| `--verbose` | Show detailed output | `False` |

## Using Individual Modules

If you want to use parts of this in your own projects, each module works independently:

```python
# Just the data preprocessing
from preprocessing import DataProcessor
processor = DataProcessor("data.csv")
clean_data = processor.process_data()

# Just one model
from models import LogisticRegression
model = LogisticRegression(train_data, test_data)
model.fit()
predictions = model.predict(test_data)

# Just evaluation
from evaluation import calculate_metrics
accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
```

## What You Get

The script outputs:
- **Console**: Training progress and performance metrics
- **CSV file**: Detailed comparison table (if you use `--save-results`)
- **Plots**: Model comparison charts (if you use `--plot-results`)
- **Saved models**: Pickle files you can load later (if you use `--save-models`)
- **HTML reports**: Professional analysis reports (if you use `--generate-reports`)

Here's what the console output looks like:

```
================================================================================
ONLINE NEWS POPULARITY ANALYSIS - SUMMARY REPORT
================================================================================

MODEL PERFORMANCE COMPARISON:
--------------------------------------------------
Model                    Train_Accuracy  Test_Accuracy  Train_Precision  Test_Precision  Train_Recall  Test_Recall  Train_F1  Test_F1
Logistic Regression      0.6514          0.6445         0.6673           0.6558          0.6952        0.6930       0.6809    0.6739
Naive Bayes             0.9986          0.9991         0.9990           0.9994          0.9983        0.9989       0.9987    0.9991
Neural Network          0.5863          0.6349         0.6452           0.6452          0.6909        0.6909       0.6673    0.6673
Support Vector Machine  0.5379          0.5383         0.5428           0.5431          0.9644        0.9633       0.6947    0.6946

BEST PERFORMING MODELS:
------------------------------
Test_Accuracy: Naive Bayes (0.9991)
Test_Precision: Naive Bayes (0.9994)
Test_Recall: Naive Bayes (0.9989)
Test_F1: Naive Bayes (0.9991)

OVERALL BEST MODEL: Naive Bayes (F1-Score: 0.9991)
```

## Requirements

You'll need Python 3.7+ and these packages:
- NumPy, Pandas (data handling)
- Matplotlib, Seaborn (plotting) 
- Scikit-learn, SciPy (some utilities)
- PyTorch (for neural network - optional)
- Streamlit (for dashboard - optional)

Just run `pip install -r requirements.txt` and you're good to go.

## 🚀 **Deploying on GitHub**

### **Option 1: Streamlit Cloud (Recommended)**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository
5. Set main file path to `dashboard.py`
6. Deploy! Your dashboard will be live at `https://your-app-name.streamlit.app`

### **Option 2: GitHub Pages + Static Export**
For a static version, you can export the dashboard as HTML:
```bash
python main.py --generate-reports
```
This creates HTML reports you can host on GitHub Pages.

### **Option 3: Local Development**
```bash
# Clone the repo
git clone https://github.com/yourusername/OnlineNewPopularity.git
cd OnlineNewPopularity

# Install dependencies
pip install -r requirements.txt

# Download the dataset automatically:
python download_dataset.py

# Or download manually from:
# https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
# Place OnlineNewsPopularity.csv in data/OnlineNewsPopularity/

# Run the dashboard
streamlit run dashboard.py
```

**Note**: The dataset (`OnlineNewsPopularity.csv`) is not included in this repository due to its size (~2MB). You'll need to download it separately from the UCI ML Repository and place it in the `data/OnlineNewsPopularity/` directory.

## Notes & Gotchas

- **PyTorch is optional**: If you don't have it installed, the neural network just gets skipped
- **SVM is slow**: I sample 6000 records for training to keep it reasonable
- **Naive Bayes results look too good**: Might be overfitting, but that's what the data shows
- **Everything's from scratch**: No sklearn shortcuts - coded the algorithms myself to understand them better

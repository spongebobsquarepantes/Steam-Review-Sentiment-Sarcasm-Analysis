# 🎮 Steam Game Reviews Sarcasm & Sentiment Analysis using BERT

## 📌 Business Context
As a Marketing & Strategy major, I recognize that traditional sentiment analysis often fails to detect **sarcasm** in gaming communities (e.g., leaving a "positive" review with heavily sarcastic negative text). This project leverages deep learning to help game developers and marketers uncover the *true* voice of the customers.

## ⚙️ Tech Stack & Methodology
* **Language/Frameworks:** Python, PyTorch, HuggingFace Transformers, Scikit-learn
* **Model:** Fine-tuned `bert-base-uncased` for a multi-task classification (Sentiment + Sarcasm).
* **Data Processing:** Web scraping via Selenium (Original raw data is redacted due to privacy/TOS, dummy data is provided for testing).

## 📊 Key Findings
* The BERT model outperformed traditional machine learning models (LR, SVM), achieving an overall accuracy of **73%**, with sarcasm detection accuracy reaching **95%**.
* **Business Implication:** This pipeline can be directly integrated into a company's CRM or social listening tools to flag high-priority sarcastic complaints that would otherwise be missed.

## 📁 Repository Structure
* `src/` : Python scripts for data processing and PyTorch model training.
* `data/` : `sample_data.csv` (Dummy data for code execution).
* `Project_Executive_Summary_EN.pdf` : A 2-page English presentation of the business value and model architecture.
* `Thesis_Original_Chinese.pdf` : The original academic thesis (in Chinese) for academic proof.

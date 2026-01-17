# Legal AI Summarizer ⚖️🤖

Legal AI Summarizer is a powerful tool designed to simplify complex legal documents. Using state-of-the-art Long Explanatory Document (LED) models, this application provides concise, easy-to-understand summaries of lengthy legal texts, helping users grasp key points without getting bogged down by legalese.

![Legal AI Summarizer](img.png)

## 🌟 Features

- **Automated Summarization**: Extracts the most relevant information from legal documents.
- **Easy Upload**: Support for `.txt` file uploads or direct text pasting.
- **AI-Powered**: Utilizes fine-tuned `Legal-LED` models for high-quality legal domain summarization.
- **Interactive UI**: Built with Streamlit for a smooth and responsive user experience.
- **Plain English**: Translates complex legal jargon into understandable language.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers Library

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/namanviber/Legal-AI-Summarizer.git
   cd Legal-AI-Summarizer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

To start the Streamlit web app, run:
```bash
streamlit run app.py
```

## 🧠 Model Information

The project uses the **Legal-LED-Base** model as its foundation, fine-tuned specifically for legal text summarization. It leverages the **PEFT (Parameter-Efficient Fine-Tuning)** library with LoRA adapters to achieve high performance with lower computational requirements.

- **Base Model**: `nsi319/legal-led-base-16384`
- **Architecture**: LED (Longformer Encoder-Decoder)
- **Max Input Length**: 16,384 tokens (ideal for long legal docs)

## 📊 Dataset

The dataset used in this project consists of various legal documents (e.g., UK Supreme Court cases) stored in the `dataset/` directory.

---
⚖️ Simplify Legal Understanding with AI.

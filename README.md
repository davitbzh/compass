# ⚙️ Medical Documentation Assistant for Rättsmedicinalverket

This project is an AI system for Rättsmedicinalverket staff that can provide answers about medical and administrative procedures from official documentation. It is built on Hopsworks.

- **Creates** vector embeddings for administrative protocol documents from Rättsmedicinalverket's documentation, indexing them for retrieval-augmented generation (RAG) in the Hopsworks Feature Store with Vector Indexing.
- **Based** on user input, retrieves top-ranked contexts from the vector database and generates responses using Google's `Gemini 2.5 Pro` model.
- **Provides** a UI, written in Streamlit/Python, for querying medical protocols that returns answers, citing the document IDs, pages, and paragraphs in its answer.
- **Collects** user feedback on responses to improve the system over time.

![Hopsworks Architecture for Medical Documentation Assistant](./images/llm-pdfs-architecture.gif)

## 📖 Feature Pipeline
The Feature Pipeline does the following:

- Processes administrative and medical protocol documents from Rättsmedicinalverket.
- Extracts chunks of text from the documents and creates embeddings using a sentence transformer model.
- Stores both text and embeddings in a vector-index-enabled Feature Group in Hopsworks.
- Captures user feedback in a separate Feature Group for quality monitoring.

## 🏃🏻‍♂️ Training Pipeline
This step is optional if you also want to create a fine-tuned model. Currently, we opt to use Google's `Gemini 2.5 Pro` model for generating responses.

## 🚀 Inference Pipeline
- A chatbot written in Streamlit that answers questions about medical and administrative procedures based on Rättsmedicinalverket's documentation.
- Uses both semantic search (with embeddings) and reranking to find the most relevant document sections.
- Provides a feedback mechanism for users to rate responses and submit comments.

## 🕵🏻‍♂️ Prerequisites
1. Create a free account on [Hopsworks](https://app.hopsworks.ai/app) and get an API key.
2. Get a Google Gemini API key (free tier available).
3. Clone the repository:
   ```bash
   git clone https://github.com/davitbzh/compass.git
   cd compass
   ```
4. Create `.env` file and save Hopsworks and Gemini API keys:
   ```bash
    GEMINI_KEY=your_gemini_api_key
    HOPSWORKS_API_KEY=your_hopsworks_api_key
   ```
5. Install required packages:
   ```bash
    pip install -r requirements.txt
   ```    
7. Run pipeline `feature_pipeline.ipynb`. Note that for simplicity we chose a Jupyter notebook here. However, for production environments, you can convert this to a Python script and schedule with preferred cadence.
6. For inference run:
   ```bash
    streamlit run ./app.py
   ```
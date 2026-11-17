# PDF Chat RAG – Gradio Interface

A modern, tabbed PDF question-answering frontend powered by **Gradio**, **LangChain**, **FAISS**, and **Google Gemini**.

## ✨ Features
- **Clean Tabbed Interface**:
  - **💬 Chat & Q&A**: Interactive chat interface with conversation history, citation display, and side-by-side retrieved context preview.
  - **📄 Document Indexer**: Drag-and-drop PDF upload, chunk size & overlap sliders, index status badge, and chunk preview.
  - **⚙️ Settings**: Switch between Gemini models (`gemini-3.8-flash`, `gemini-3.7-flash`, etc.), custom model input, top-k retrieval tuning, and API key management.
- **Precise Page Citations**: Answers are grounded directly in your PDF with page references.
- **Original Streamlit App Preserved**: Available as `app_streamlit.py` if needed.

## 🚀 How to Run

1. Open a terminal in this folder:
   ```bash
   cd d:\PDF_Chat_RAG_Fixed
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Gradio app:
   ```bash
   python app.py
   ```
   *The app will start at `http://127.0.0.1:7860`.*

4. Alternatively, to run the legacy Streamlit version:
   ```bash
   streamlit run app_streamlit.py
   ```

## 📝 Usage
1. Open the **📄 Document Indexer** tab, drop your PDF, and click **Process PDF & Build Search Index**.
2. Switch to the **💬 Chat & Q&A** tab and ask questions about your document.
3. Review the **📌 Retrieved Context & Sources** accordion to inspect the exact pages and passages Gemini used to answer.
"#23A81A6104_6148_rag.project"

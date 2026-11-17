import os
from functools import lru_cache
import gradio as gr
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

DEFAULT_API_KEY = "AQ.Ab8RN6Kpbsn_hklIe8vqEpayq9Ee7gX0FHM6i_NWDNp37M-AVA"

@lru_cache(maxsize=1)
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True}
        )
    except Exception:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


def process_pdf(file, chunk_size, chunk_overlap):
    if file is None:
        return (
            None,
            "⚠️ Please select a PDF file before clicking process.",
            "⚪ **No document indexed yet.**",
            ""
        )

    file_path = file if isinstance(file, str) else getattr(file, "name", str(file))
    file_name = os.path.basename(file_path)

    try:
        reader = PdfReader(file_path)
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"source": file_name, "page": page_num}
                ))

        if not pages:
            return (
                None,
                "❌ No selectable text found. This may be a scanned or image-only PDF; OCR is required.",
                "⚪ **No document indexed yet.**",
                ""
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap)
        )
        chunks = splitter.split_documents(pages)
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)

        state_data = {
            "store": vector_store,
            "filename": file_name,
            "page_count": len(pages),
            "chunk_count": len(chunks)
        }

        success_msg = f"✅ **Successfully processed `{file_name}`**\n\n- **Extracted Pages:** {len(pages)}\n- **Created Chunks:** {len(chunks)}"
        status_badge = f"🟢 **Active Document:** `{file_name}` ({len(pages)} pages, {len(chunks)} chunks)"

        preview_snippets = []
        for idx, chunk in enumerate(chunks[:3], start=1):
            preview_snippets.append(
                f"**Chunk #{idx} (Page {chunk.metadata.get('page', '?')}):**\n```text\n{chunk.page_content[:350]}...\n```"
            )
        preview_text = "\n\n".join(preview_snippets)

        return state_data, success_msg, status_badge, preview_text

    except Exception as e:
        return (
            None,
            f"❌ PDF processing failed: `{e}`",
            "⚪ **No document indexed yet.**",
            ""
        )


def reset_index():
    return (
        None,
        "🔄 Index and document cleared.",
        "⚪ **No document indexed yet. Upload a PDF to begin.**",
        "",
        [],
        ""
    )


def chat_response(user_message, history, state, api_key, model_choice, custom_model, top_k):
    if not user_message or not user_message.strip():
        return history, "", ""

    history = list(history or [])

    if state is None or state.get("store") is None:
        warning_msg = "⚠️ Please upload and process a PDF first in the **'📄 Document Indexer'** tab."
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": warning_msg})
        return history, "", "No sources retrieved (no document active)."

    effective_api_key = (api_key or "").strip()
    if not effective_api_key:
        warning_msg = "⚠️ Please provide a Google Gemini API key in the **'⚙️ Settings'** tab."
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": warning_msg})
        return history, "", "No sources retrieved (API key required)."

    if "gemini-3.5-flash-lite" in model_choice:
        selected_model = "gemini-3.5-flash-lite"
    elif model_choice == "Custom...":
        selected_model = custom_model.strip()
    else:
        selected_model = model_choice

    history.append({"role": "user", "content": user_message})

    try:
        store = state["store"]
        retrieved = store.similarity_search(user_message, k=int(top_k))

        context = "\n\n".join(
            f"[PDF page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in retrieved
        )

        system_instruction = (
            "You are a helpful teaching assistant. Answer ONLY using the "
            "provided PDF context. If the answer is absent, say: "
            "'I could not find this in the provided document.' "
            "Use clear language and cite PDF page numbers where possible. "
            "Do not invent facts or citations."
        )
        prompt = f"{system_instruction}\n\nPDF context:\n{context}\n\nUser Question:\n{user_message}"

        models_to_try = [selected_model]
        if selected_model != "gemini-3.5-flash-lite":
            models_to_try.append("gemini-3.5-flash-lite")

        client = genai.Client(api_key=effective_api_key)
        answer = None
        for m in models_to_try:
            try:
                r = client.models.generate_content(model=m, contents=prompt)
                answer = r.text
                break
            except Exception:
                continue

        if not answer:
            answer = "⚠️ The model was unable to generate an answer. Please check your API key or try again."

        page_numbers = list(dict.fromkeys(
            doc.metadata.get("page", "?") for doc in retrieved
        ))
        sources_summary = "\n".join(f"- **PDF Page {p}**" for p in page_numbers)
        sources_detail = f"### 📌 Retrieved Context Sources\n{sources_summary}\n\n---\n"
        for i, doc in enumerate(retrieved, start=1):
            sources_detail += f"**Chunk #{i} — Page {doc.metadata.get('page', '?')}**\n> {doc.page_content}\n\n"

        history.append({"role": "assistant", "content": answer})
        return history, "", sources_detail

    except Exception as e:
        error_msg = f"Error occurred: `{e}`"
        history.append({"role": "assistant", "content": error_msg})
        return history, "", error_msg


custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.app-header {
    text-align: center;
    padding: 20px;
    margin-bottom: 12px;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    border-radius: 12px;
    color: white;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: #ffffff !important;
}
.app-header p {
    font-size: 0.95rem;
    margin: 0;
    opacity: 0.9;
    color: #f1f5f9;
}
"""

with gr.Blocks(title="PDF Chat RAG Assistant") as demo:
    doc_state = gr.State(None)

    gr.HTML("""
        <div class="app-header">
            <h1>📚 PDF Chat RAG Assistant</h1>
            <p>Upload any PDF document, build instant neural search embeddings, and ask questions with precise page citations.</p>
        </div>
    """)

    status_display = gr.Markdown("⚪ **No document indexed yet. Upload a PDF in the 'Document Indexer' tab to get started.**")

    with gr.Tabs():
        with gr.Tab("💬 Chat & Q&A"):
            with gr.Row():
                with gr.Column(scale=8):
                    chatbot = gr.Chatbot(
                        label="Discussion",
                        height=500
                    )
                    with gr.Row():
                        query_input = gr.Textbox(
                            show_label=False,
                            placeholder="Type your question about the PDF and press Enter...",
                            scale=9,
                            lines=1,
                            max_lines=3,
                            autofocus=True
                        )
                        send_btn = gr.Button("Send 🚀", variant="primary", scale=1)

                    with gr.Row():
                        clear_chat_btn = gr.Button("🧹 Clear Conversation", size="sm")

                with gr.Column(scale=4):
                    with gr.Accordion("📌 Retrieved Context & Sources", open=True):
                        sources_display = gr.Markdown("Sources for the latest answer will appear here.")

        with gr.Tab("📄 Document Indexer"):
            with gr.Row():
                with gr.Column(scale=6):
                    pdf_input = gr.File(label="Upload PDF Document", file_types=[".pdf"], type="filepath")
                    with gr.Row():
                        chunk_size_slider = gr.Slider(minimum=200, maximum=2000, value=700, step=50, label="Chunk Size (chars)")
                        chunk_overlap_slider = gr.Slider(minimum=0, maximum=400, value=120, step=10, label="Chunk Overlap (chars)")

                    process_btn = gr.Button("📥 Process PDF & Build Search Index", variant="primary", size="lg")
                    clear_doc_btn = gr.Button("🗑️ Reset / Clear Document", variant="secondary")

                with gr.Column(scale=6):
                    process_output = gr.Markdown("Ready to process your PDF.")
                    with gr.Accordion("🔍 Extracted Chunks Preview", open=False):
                        chunks_preview = gr.Markdown("No chunks generated yet.")

        with gr.Tab("⚙️ Settings"):
            with gr.Row():
                with gr.Column(scale=6):
                    api_key_input = gr.Textbox(
                        label="Google Gemini API Key",
                        type="password",
                        value=os.environ.get("GEMINI_API_KEY", DEFAULT_API_KEY)
                    )
                    model_dropdown = gr.Dropdown(
                        label="Gemini Model",
                        choices=[
                            "gemini-3.5-flash-lite (Ultra Fast ⚡)",
                            "gemini-3.8-flash",
                            "gemini-3.7-flash",
                            "gemini-3.5-flash",
                            "Custom..."
                        ],
                        value="gemini-3.5-flash-lite (Ultra Fast ⚡)"
                    )
                    custom_model_input = gr.Textbox(label="Custom Model Name", value="gemini-3.5-flash-lite", visible=False)

                    def update_custom_visibility(choice):
                        return gr.update(visible=(choice == "Custom..."))

                    model_dropdown.change(fn=update_custom_visibility, inputs=[model_dropdown], outputs=[custom_model_input])

                with gr.Column(scale=6):
                    top_k_slider = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Top-K Retrieved Chunks")
                    gr.Markdown(
                        "### ℹ️ Configuration Details\n"
                        "- **Speed:** Powered by `gemini-3.5-flash-lite` for near-instant responses.\n"
                        "- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`\n"
                        "- **Vector Store:** Local in-memory FAISS indexing\n"
                        "- **Privacy:** Only retrieved chunks are forwarded to Gemini."
                    )

    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input, chunk_size_slider, chunk_overlap_slider],
        outputs=[doc_state, process_output, status_display, chunks_preview]
    )

    clear_doc_btn.click(
        fn=reset_index,
        inputs=[],
        outputs=[doc_state, process_output, status_display, chunks_preview, chatbot, sources_display]
    )

    query_input.submit(
        fn=chat_response,
        inputs=[query_input, chatbot, doc_state, api_key_input, model_dropdown, custom_model_input, top_k_slider],
        outputs=[chatbot, query_input, sources_display]
    )

    send_btn.click(
        fn=chat_response,
        inputs=[query_input, chatbot, doc_state, api_key_input, model_dropdown, custom_model_input, top_k_slider],
        outputs=[chatbot, query_input, sources_display]
    )

    clear_chat_btn.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, sources_display]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=custom_css
    )

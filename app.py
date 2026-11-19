# """College RAG Chatbot — modern chat UI.  Run:  python app.py  then open http://127.0.0.1:7860"""
# import shutil
# from pathlib import Path
# import gradio as gr
# from src.config import PATHS, NOT_FOUND_MSG
# from src.system import get_models, build_all, indexes_exist

# STATE = {"baseline": None, "advanced": None}

# CSS = """
# body { font-family: 'Segoe UI', sans-serif; }
# #chatbot { height: 520px; overflow-y: auto; border-radius: 12px; }
# #chatbot .message.user { background: #0078D4; color: white; border-radius: 18px 18px 4px 18px; }
# #chatbot .message.bot  { background: #F3F4F6; color: #111;   border-radius: 18px 18px 18px 4px; }
# #send-btn { background: #0078D4 !important; color: white !important; border-radius: 8px !important; }
# #send-btn:hover { background: #005A9E !important; }
# #sidebar { background: #F9FAFB; border-right: 1px solid #E5E7EB; padding: 16px; border-radius: 12px; }
# .status-box textarea { font-size: 0.85rem !important; color: #374151 !important; }
# footer { display: none !important; }
# """

# def _ensure_models(model_choice):
#     key = "advanced" if model_choice.startswith("Advanced") else "baseline"
#     if STATE[key] is None:
#         if not list(PATHS["raw"].glob("*.pdf")):
#             raise gr.Error("No PDFs found. Upload PDFs and click 'Build Index'.")
#         STATE["baseline"], STATE["advanced"] = get_models()
#     return STATE[key]


# def list_pdfs():
#     pdfs = sorted(PATHS["raw"].glob("*.pdf"))
#     if not pdfs:
#         return "No PDFs uploaded yet."
#     return "\n".join(f"📄 {p.name}" for p in pdfs)


# def upload_pdfs(files):
#     if not files:
#         return "⚠️ No files selected.", list_pdfs()
#     saved = []
#     for f in files:
#         src = Path(getattr(f, "name", f))
#         if src.suffix.lower() != ".pdf":
#             continue
#         shutil.copy(src, PATHS["raw"] / src.name)
#         saved.append(src.name)
#     if not saved:
#         return "⚠️ No valid PDF files found.", list_pdfs()
#     msg = f"✅ Saved {len(saved)} PDF(s):\n" + "\n".join(f"  • {n}" for n in saved) + "\n\nClick 'Build Index' to continue."
#     return msg, list_pdfs()


# def rebuild_index():
#     try:
#         yield "⏳ Building index, please wait..."
#         build_all(verbose=False)
#         STATE["baseline"] = STATE["advanced"] = None
#         n = len(list(PATHS["raw"].glob("*.pdf")))
#         yield f"✅ Index built from {n} PDF(s). Ready to answer questions!"
#     except Exception as e:
#         yield f"❌ Build failed: {e}"


# def ask(question, model_choice, history):
#     question = (question or "").strip()
#     if not question:
#         return history, ""
#     history = history + [{"role": "user", "content": question}]
#     yield history, ""

#     try:
#         model = _ensure_models(model_choice)
#         res = model.answer(question)
#         if res["found"]:
#             sources = "  \n".join(f"📄 {c}" for c in res["citations"])
#             reply = f"{res['answer']}\n\n---\n{sources}\n_({model.name} · {res['time']:.2f}s)_"
#         else:
#             reply = f"_{NOT_FOUND_MSG}_\n\n_({model.name} · {res['time']:.2f}s)_"
#     except Exception as e:
#         reply = f"❌ Error: {e}"

#     history = history + [{"role": "assistant", "content": reply}]
#     yield history, ""


# def update_history(history):
#     if not history:
#         return "No questions asked yet."
#     questions = [m["content"] for m in history if m["role"] == "user"]
#     return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))


# def clear_chat():
#     return [], "", "No questions asked yet."


# with gr.Blocks(title="College Document QA") as demo:
#     gr.Markdown(
#         "# 🎓 College Document QA\n"
#         "Ask questions about your college documents. Answers are grounded in the uploaded PDFs."
#     )

#     with gr.Row():
#         # ── Sidebar ──────────────────────────────────────────────
#         with gr.Column(scale=1, elem_id="sidebar"):
#             gr.Markdown("### ⚙️ Setup")
#             files = gr.File(
#                 label="Upload PDFs",
#                 file_count="multiple",
#                 file_types=[".pdf"],
#             )
#             up_btn = gr.Button("📤 Save PDFs")
#             build_btn = gr.Button("🔨 Build Index", variant="primary")
#             status = gr.Textbox(
#                 label="Status",
#                 lines=4,
#                 interactive=False,
#                 elem_classes=["status-box"],
#                 value="✅ Index ready." if indexes_exist() else "⚠️ No index yet. Upload PDFs and build.",
#             )
#             gr.Markdown("---")
#             gr.Markdown("### 📂 Uploaded PDFs")
#             pdf_list = gr.Textbox(
#                 label="",
#                 lines=4,
#                 interactive=False,
#                 value=list_pdfs(),
#             )
#             refresh_btn = gr.Button("🔄 Refresh list", size="sm")
#             gr.Markdown("---")
#             gr.Markdown("### 🤖 Model")
#             model_choice = gr.Radio(
#                 choices=["Advanced (MiniLM + FAISS + FLAN-T5)", "Baseline (TF-IDF)"],
#                 value="Advanced (MiniLM + FAISS + FLAN-T5)",
#                 label="",
#             )

#         # ── Chat area ─────────────────────────────────────────────
#         with gr.Column(scale=3):
#             chatbot = gr.Chatbot(
#                 elem_id="chatbot",
#                 label="",
#                 show_label=False,
#                 height=520,
#             )
#             with gr.Row():
#                 question = gr.Textbox(
#                     placeholder="Ask a question about your college documents…",
#                     show_label=False,
#                     scale=5,
#                     container=False,
#                 )
#                 send_btn = gr.Button("Send ➤", elem_id="send-btn", scale=1)
#             clear_btn = gr.Button("🗑️ Clear chat", size="sm")
#             with gr.Accordion("📜 Question History", open=False):
#                 history_box = gr.Textbox(
#                     label="",
#                     lines=6,
#                     interactive=False,
#                     value="No questions asked yet.",
#                 )

#     # ── Events ────────────────────────────────────────────────────
#     up_btn.click(upload_pdfs, inputs=files, outputs=[status, pdf_list])
#     refresh_btn.click(list_pdfs, inputs=None, outputs=pdf_list)
#     build_btn.click(rebuild_index, inputs=None, outputs=status)

#     send_btn.click(ask, inputs=[question, model_choice, chatbot], outputs=[chatbot, question]).then(
#         update_history, inputs=chatbot, outputs=history_box)
#     question.submit(ask, inputs=[question, model_choice, chatbot], outputs=[chatbot, question]).then(
#         update_history, inputs=chatbot, outputs=history_box)
#     clear_btn.click(clear_chat, outputs=[chatbot, question, history_box])

# if __name__ == "__main__":
#     demo.launch(css=CSS)


"""College RAG Chatbot - improved UI.  Run:  python app.py  then open http://127.0.0.1:7860"""
import shutil
import threading
import time
from pathlib import Path
import gradio as gr
from src.config import PATHS, NOT_FOUND_MSG
from src.system import get_models, build_all, indexes_exist

STATE = {"baseline": None, "advanced": None}
LOCK = threading.Lock()

CSS = """
.gradio-container { max-width: 1250px !important; margin: auto; font-family: 'Inter','Segoe UI',sans-serif; }
#hero { background: linear-gradient(120deg,#4f46e5,#0ea5e9); color: white; padding: 22px 28px;
        border-radius: 16px; margin-bottom: 14px; }
#hero h1 { color: white; margin: 0 0 4px 0; font-size: 1.7rem; }
#hero p  { color: #e0f2fe; margin: 0; }
#sidebar { background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 14px; }
#chatbot { border-radius: 14px; border: 1px solid #E2E8F0; }
#send-btn { background: linear-gradient(120deg,#4f46e5,#0ea5e9) !important; color: white !important;
            border-radius: 10px !important; border: none !important; }
#send-btn:hover { filter: brightness(1.1); }
details { background: rgba(99,102,241,0.07); border-radius: 8px; padding: 6px 10px; margin-top: 8px; }
summary { cursor: pointer; font-weight: 600; font-size: 0.9rem; }
blockquote { font-size: 0.82rem; opacity: 0.85; margin: 4px 0 8px 0; }
footer { display: none !important; }
"""

EXAMPLES = [
    "How many units are there in the Data Structures syllabus?",
    "What are the prerequisites for Machine Learning?",
    "What is the credit and marks distribution for Operating Systems?",
    "List the topics in Unit 3 of Database Management Systems.",
]


# ---------------- model loading (pre-warmed in background) ----------------
def _load_models():
    with LOCK:
        if STATE["advanced"] is None:
            STATE["baseline"], STATE["advanced"] = get_models()


def _warmup():
    try:
        if indexes_exist():
            _load_models()
            print("Models pre-loaded. First question will be fast.")
    except Exception as e:
        print("Warm-up skipped:", e)


threading.Thread(target=_warmup, daemon=True).start()


def _ensure_models(model_choice):
    if STATE["advanced"] is None:
        if not list(PATHS["raw"].glob("*.pdf")):
            raise gr.Error("No PDFs found. Upload PDFs and click 'Build Index'.")
        _load_models()
    return STATE["advanced" if model_choice.startswith("Advanced") else "baseline"]


# ---------------- PDF management ----------------
def pdf_names():
    return [p.name for p in sorted(PATHS["raw"].glob("*.pdf"))]


def list_pdfs():
    names = pdf_names()
    return "\n".join(f"📄 {n}" for n in names) if names else "No PDFs uploaded yet."


def _dd():
    return gr.update(choices=pdf_names())


def upload_pdfs(files):
    if not files:
        return "⚠️ No files selected.", list_pdfs(), _dd()
    saved = []
    for f in files:
        src = Path(getattr(f, "name", f))
        if src.suffix.lower() != ".pdf":
            continue
        shutil.copy(src, PATHS["raw"] / src.name)
        saved.append(src.name)
    if not saved:
        return "⚠️ No valid PDF files found.", list_pdfs(), _dd()
    msg = f"✅ Saved {len(saved)} PDF(s). Click 'Build Index' to continue."
    return msg, list_pdfs(), _dd()


def rebuild_index():
    try:
        yield "⏳ Building index, please wait..."
        build_all(verbose=False)
        with LOCK:
            STATE["baseline"] = STATE["advanced"] = None
        yield f"✅ Index built from {len(pdf_names())} PDF(s). Ready!"
    except Exception as e:
        yield f"❌ Build failed: {e}"


# ---------------- chat ----------------
def contextualize(q, history):
    """Very small follow-up handling: a short question is joined with the previous one."""
    prev = [m["content"] for m in history if m["role"] == "user"]
    if prev and len(q.split()) <= 5:
        return f"{prev[-1]} {q}"
    return q


def format_reply(res, model):
    if not res["found"]:
        return (f"_{NOT_FOUND_MSG}_\n\n💡 Try rephrasing, or select the right document in "
                f"**Search in**.\n\n<sub>{model.name} · {res['time']:.2f}s</sub>")
    cites = res["citations"]
    snippets = []
    for p in res["passages"][:3]:
        snippets.append(f"**📄 {p['doc_name']} · page {p['page_no']}** (score {p['score']:.2f})\n"
                        f"> {p['text'][:220]}…")
    src = "\n\n".join(snippets)
    return (f"{res['answer']}\n\n"
            f"<details><summary>📚 Sources ({len(cites)})</summary>\n\n{src}\n\n</details>\n\n"
            f"<sub>{model.name} · {res['time']:.2f}s</sub>")


def ask(question, model_choice, docs, history):
    q = (question or "").strip()
    if not q:
        yield history, ""
        return
    history = history + [{"role": "user", "content": q}]
    yield history + [{"role": "assistant", "content": "💭 _Searching your documents..._"}], ""

    try:
        model = _ensure_models(model_choice)
        res = model.answer(contextualize(q, history[:-1]), docs=docs or None)
        final = format_reply(res, model)
        if res["found"]:  # typing effect on the answer text only
            words = res["answer"].split()
            for i in range(3, len(words), 3):
                yield history + [{"role": "assistant", "content": " ".join(words[:i]) + " ▌"}], ""
                time.sleep(0.03)
    except Exception as e:
        final = f"❌ Error: {e}"
    yield history + [{"role": "assistant", "content": final}], ""


def update_history(history):
    qs = [m["content"] for m in (history or []) if m["role"] == "user"]
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs)) if qs else "No questions asked yet."


def clear_chat():
    return [], "", "No questions asked yet."


with gr.Blocks(title="College Document QA") as demo:
    gr.HTML('<div id="hero"><h1>🎓 College Document QA</h1>'
            '<p>Ask questions about your college documents. Every answer is grounded in your PDFs and cited.</p></div>')

    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar", min_width=290):
            gr.Markdown("### 🤖 Model")
            model_choice = gr.Radio(
                choices=["Advanced (MiniLM + FAISS + FLAN-T5)", "Baseline (TF-IDF)"],
                value="Advanced (MiniLM + FAISS + FLAN-T5)", show_label=False)
            gr.Markdown("### 🔎 Search in")
            doc_filter = gr.Dropdown(choices=pdf_names(), multiselect=True, value=None,
                                     label="Documents (empty = all)", info="Pick one syllabus for precise answers")
            with gr.Accordion("⚙️ Setup: upload & index", open=False):
                files = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
                up_btn = gr.Button("📤 Save PDFs")
                build_btn = gr.Button("🔨 Build Index", variant="primary")
                status = gr.Textbox(label="Status", lines=2, interactive=False,
                                    value="✅ Index ready." if indexes_exist() else "⚠️ No index yet.")
                pdf_list = gr.Textbox(label="Uploaded PDFs", lines=4, interactive=False, value=list_pdfs())
                refresh_btn = gr.Button("🔄 Refresh list", size="sm")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot", show_label=False, height=520,
                placeholder="### 👋 Hi! Ask me anything about your documents.\nTry one of the examples below.")
            with gr.Row():
                question = gr.Textbox(placeholder="Ask a question about your college documents…",
                                      show_label=False, scale=6, container=False, autofocus=True)
                send_btn = gr.Button("Send ➤", elem_id="send-btn", scale=1)
            gr.Examples(examples=EXAMPLES, inputs=question, label="💡 Try these")
            clear_btn = gr.Button("🗑️ Clear chat", size="sm")
            with gr.Accordion("📜 Question History", open=False):
                history_box = gr.Textbox(show_label=False, lines=5, interactive=False,
                                         value="No questions asked yet.")

    up_btn.click(upload_pdfs, files, [status, pdf_list, doc_filter])
    refresh_btn.click(lambda: (list_pdfs(), _dd()), None, [pdf_list, doc_filter])
    build_btn.click(rebuild_index, None, status).then(lambda: (list_pdfs(), _dd()), None, [pdf_list, doc_filter])

    inputs = [question, model_choice, doc_filter, chatbot]
    send_btn.click(ask, inputs, [chatbot, question]).then(update_history, chatbot, history_box)
    question.submit(ask, inputs, [chatbot, question]).then(update_history, chatbot, history_box)
    clear_btn.click(clear_chat, outputs=[chatbot, question, history_box])

if __name__ == "__main__":
    demo.queue().launch(css=CSS, theme=gr.themes.Soft(primary_hue="indigo"))
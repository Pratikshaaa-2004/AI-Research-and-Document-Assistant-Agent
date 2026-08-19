import streamlit as st
from rag.ask_document import ask_document


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
        .main {
            padding-top: 1.5rem;
        }

        .app-header {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0;
        }

        .app-subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-top: 0;
            margin-bottom: 1.5rem;
        }

        .answer-box {
            background-color: #f8f9fb;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1.2rem 1.5rem;
            margin-top: 0.5rem;
            line-height: 1.7;
            font-size: 1rem;
        }

        .source-chip {
            display: inline-block;
            background-color: #eef2ff;
            color: #4338ca;
            padding: 0.35rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin: 0.2rem 0.35rem 0.2rem 0;
            font-weight: 500;
        }

        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
        }

        .history-item {
            background-color: #fafafa;
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 0.7rem 1rem;
            margin-bottom: 0.6rem;
            font-size: 0.9rem;
        }

        .history-question {
            font-weight: 600;
            color: #111827;
        }
    </style>
""", unsafe_allow_html=True)


# ---------- SESSION STATE ----------
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0

if "history" not in st.session_state:
    st.session_state.history = []


# ---------- SIDEBAR ----------
with st.sidebar:

    st.markdown("### 🔬 Research Assistant")
    st.caption("AI-powered document Q&A")

    st.divider()

    st.markdown("**Navigation**")
    st.markdown("🏠 Home")
    st.markdown("📄 Documents")
    st.markdown("💬 Ask")
    st.markdown("📚 History")
    st.markdown("⚙️ Settings")

    st.divider()

    st.markdown("**Recent Questions**")

    if st.session_state.history:

        for item in reversed(st.session_state.history[-5:]):

            st.markdown(
                f'<div class="history-item">'
                f'<div class="history-question">'
                f'Q: {item["question"]}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    else:
        st.caption("No questions asked yet.")

    st.divider()

    st.caption("Built with Streamlit + ChromaDB + Claude")


# ---------- HEADER ----------
st.markdown(
    '<p class="app-header">🔬 AI Research Assistant</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="app-subtitle">'
    'Ask questions about your documents and get cited, sourced answers.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ---------- MAIN LAYOUT ----------
col1, col2 = st.columns([2, 1])


# ---------- LEFT COLUMN ----------
with col1:

    st.markdown("#### Ask a question")

    question = st.text_area(
        label="Your question",
        placeholder="e.g. What does the document say about CRM?",
        height=100,
        label_visibility="collapsed"
    )

    ask_col, clear_col = st.columns([1, 1])

    with ask_col:

        ask_clicked = st.button(
            "🔍 Ask",
            use_container_width=True,
            type="primary"
        )

    with clear_col:

        clear_clicked = st.button(
            "🧹 Clear",
            use_container_width=True
        )


    # ---------- CLEAR BUTTON ----------
    if clear_clicked:
        st.rerun()


    # ---------- ASK BUTTON ----------
    if ask_clicked:

        if not question.strip():

            st.warning("⚠️ Please type a question first.")

        else:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                try:

                    result = ask_document(question)

                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )

                    result = None


            # ---------- DISPLAY RESULT ----------
            if result:

                st.session_state.questions_asked += 1

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"]
                    }
                )


                # ---------- ANSWER ----------
                st.markdown("#### 🤖 Answer")

                st.markdown(
                    f'<div class="answer-box">'
                    f'{result["answer"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )


                # ---------- SOURCES ----------
                st.markdown("#### 📚 Sources")

                if result["sources"]:

                    chips_html = "".join(
                        f'<span class="source-chip">'
                        f'📄 {s["filename"]} — p.{s["page"]}'
                        f'</span>'
                        for s in result["sources"]
                    )

                    st.markdown(
                        chips_html,
                        unsafe_allow_html=True
                    )

                else:

                    st.caption(
                        "No sources available for this answer."
                    )


    # ---------- FULL HISTORY ----------
    if st.session_state.history:

        st.divider()

        with st.expander(
            f"📜 Full question history "
            f"({len(st.session_state.history)})"
        ):

            for i, item in enumerate(
                reversed(st.session_state.history),
                start=1
            ):

                st.markdown(
                    f"**Q{len(st.session_state.history) - i + 1}: "
                    f"{item['question']}**"
                )

                st.markdown(
                    f'<div class="answer-box">'
                    f'{item["answer"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )


                if item["sources"]:

                    chips_html = "".join(
                        f'<span class="source-chip">'
                        f'📄 {s["filename"]} — p.{s["page"]}'
                        f'</span>'
                        for s in item["sources"]
                    )

                    st.markdown(
                        chips_html,
                        unsafe_allow_html=True
                    )

                st.markdown("---")


# ---------- RIGHT COLUMN ----------
with col2:

    st.markdown("#### 📊 Session Info")

    st.metric(
        "Questions Asked",
        st.session_state.questions_asked
    )


    # ---------- CHROMADB STATUS ----------
    try:

        from rag.retriever import collection_status

        status = collection_status()

        doc_count = status.get(
            "document_count",
            "?"
        )

    except Exception:

        doc_count = "?"


    st.metric(
        "Chunks Indexed",
        doc_count
    )


    st.info(
        "💡 Tip: Ask specific questions for more "
        "accurate, well-cited answers."
    )


    # ---------- NO DOCUMENT WARNING ----------
    if doc_count == 0:

        st.warning(
            "⚠️ No documents indexed yet. "
            "Check your chroma_db folder."
        )
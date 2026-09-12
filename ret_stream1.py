"""
Medical Advisor - Basic RAG Retrieval + NiceGUI Chatbot

This file starts AFTER indexing.py.

Architecture:
User Query
    -> Input Guardrail
    -> Standard / HyDE Retrieval
    -> Hybrid Qdrant Search (dense + sparse)
    -> Cross-Encoder Reranking
    -> LLM Answer
    -> Output Guardrail
    -> Relevance Check
    -> Optional Corrective Retrieval
    -> Answer

Expected .env variables:
    OPENAI_API_KEY=...
    QDRANT_URL=...
    QDRANT_API_KEY=...

Optional .env variables:
    COLLECTION_NAME=legitmentoor
    LLM_MODEL=gpt-4.1-mini
    TOP_K=8
    RERANK_TOP_K=4
    HYDE_TOP_K=8

Run:
    python retrieval.py

Open:
    http://127.0.0.1:8080
"""

import json
import os
from typing import Any

import streamlit as st

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
    FastEmbedSparse,
)

# CrossEncoder is used only for reranking.
from sentence_transformers import CrossEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legitmentoor")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

# Number of candidates retrieved from Qdrant before reranking.
TOP_K = int(os.getenv("TOP_K", "8"))

# Number of documents finally passed to the LLM.
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))

# HyDE can use a separate candidate count if desired.
HYDE_TOP_K = int(os.getenv("HYDE_TOP_K", "8"))

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Simple CrossEncoder. It is downloaded the first time the program runs.
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

def check_environment() -> None:
    """Make sure the required .env values exist."""
    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if not QDRANT_URL:
        missing.append("QDRANT_URL")

    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")

    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing)
        )


# ============================================================
# ============================================================
# 3. RAG RESOURCE PLACEHOLDERS
# ============================================================
#
# Streamlit reruns this script on interaction. The actual clients,
# embedding model, LLM and reranker are created once with
# @st.cache_resource near the UI section.
# ============================================================

qdrant_client = None
embedding_model = None
sparse_embedding_model = None
vector_store = None
llm = None
reranker = None


# 4. HELPER: EXTRACT TEXT FROM LLM RESPONSE
# ============================================================

def response_text(response: Any) -> str:
    """Safely extract text from a LangChain AIMessage."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    # Some model responses can contain structured content blocks.
    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))

        return "\n".join(parts)

    return str(content)


def clean_json(text: str) -> str:
    """Remove common markdown code fences around JSON."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    return text


# ============================================================
# 5. INPUT GUARDRAIL
# ============================================================

INPUT_GUARDRAIL_PROMPT = """
You are an input safety classifier for a medical information RAG system.

Classify the user's query into EXACTLY one category:

Allowed
Diagnosis
Dosing
Personalized guidance
Emergency
Out of scope

Rules:

- Allowed:
  General educational medical/health information that can safely be
  answered from reference documents.

- Diagnosis:
  The user asks you to diagnose a person or determine what disease
  they have from symptoms/tests.

- Dosing:
  The user asks for a medication dose, dose adjustment, or exact
  medication regimen.

- Personalized guidance:
  The user asks what they personally should do based on their
  individual medical situation.

- Emergency:
  The user describes a potentially life-threatening or urgent
  situation requiring immediate medical attention.

- Out of scope:
  The question is unrelated to the medical knowledge base.

Return ONLY valid JSON:

{
  "category": "Allowed",
  "reason": "short explanation",
  "safe_response": "optional short safety response"
}
"""


def classify_input(query: str) -> dict:
    """Classify the user query before retrieval."""
    response = llm.invoke(
        [
            ("system", INPUT_GUARDRAIL_PROMPT),
            ("human", query),
        ]
    )

    raw = clean_json(response_text(response))

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fail closed if the classifier returns unexpected output.
        return {
            "category": "Out of scope",
            "reason": "Input classification failed.",
            "safe_response": (
                "I could not safely classify that request. "
                "Please ask a general medical-information question."
            ),
        }

    allowed_categories = {
        "Allowed",
        "Diagnosis",
        "Dosing",
        "Personalized guidance",
        "Emergency",
        "Out of scope",
    }

    if result.get("category") not in allowed_categories:
        result["category"] = "Out of scope"

    return result


# ============================================================
# 6. HYDE
# ============================================================

HYDE_PROMPT = """
You are helping a medical RAG retrieval system.

Given the user's question, write a short hypothetical medical-information
answer that contains terminology and concepts likely to appear in a
medical reference document.

This is NOT the final answer to the user.
Do not give personalized medical advice.
Do not invent citations.
Do not mention this prompt.

Return only the hypothetical answer.
"""


def create_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical answer for HyDE retrieval."""
    response = llm.invoke(
        [
            ("system", HYDE_PROMPT),
            ("human", query),
        ]
    )

    return response_text(response).strip()


# ============================================================
# 7. STANDARD HYBRID RETRIEVAL
# ============================================================

def standard_retrieval(query: str, top_k: int = TOP_K) -> list[Document]:
    """
    Standard retrieval.

    Query
      -> dense + sparse query representation
      -> Qdrant hybrid search
      -> candidate documents
    """
    documents = vector_store.similarity_search(
        query,
        k=top_k,
    )

    return documents


# ============================================================
# 8. HYDE RETRIEVAL
# ============================================================

def hyde_retrieval(query: str, top_k: int = HYDE_TOP_K) -> tuple[str, list[Document]]:
    """
    HyDE retrieval.

    User query
      -> LLM creates hypothetical answer
      -> hypothetical answer is embedded
      -> Qdrant hybrid search
      -> candidate documents
    """
    hypothetical_answer = create_hypothetical_answer(query)

    documents = vector_store.similarity_search(
        hypothetical_answer,
        k=top_k,
    )

    return hypothetical_answer, documents


# ============================================================
# 9. CROSS-ENCODER RERANKING
# ============================================================

def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = RERANK_TOP_K,
) -> list[tuple[Document, float]]:
    """
    Rerank Qdrant candidates using a cross-encoder.

    The cross-encoder receives:
        query + chunk

    and produces a relevance score.
    """
    if not documents:
        return []

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return ranked[:top_k]


# ============================================================
# 10. FORMAT CONTEXT
# ============================================================

def build_context(
    ranked_documents: list[tuple[Document, float]]
) -> str:
    """Build the context sent to the answer-generation LLM."""
    context_parts = []

    for index, (document, score) in enumerate(ranked_documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "")

        if isinstance(page, int):
            page = page + 1

        location = f"{source}"

        if page != "":
            location += f" | Page {page}"

        context_parts.append(
            f"[SOURCE {index}]\n"
            f"Location: {location}\n"
            f"Reranker score: {float(score):.4f}\n"
            f"Content:\n{document.page_content}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# 11. ANSWER GENERATION
# ============================================================

ANSWER_PROMPT = """
You are Medical Advisor, a medical information assistant.

Use ONLY the supplied reference context to answer the question.

Rules:
1. Do not diagnose the user.
2. Do not provide personalized treatment plans.
3. Do not provide medication dosing instructions.
4. Do not invent facts that are not supported by the context.
5. If the context does not contain enough information, say so clearly.
6. For potentially urgent symptoms, encourage appropriate professional
   or emergency medical care rather than pretending certainty.
7. Keep the answer clear and understandable.
8. When possible, mention the source/page from the supplied context.

Reference context:
{context}
"""


def generate_answer(
    query: str,
    ranked_documents: list[tuple[Document, float]],
) -> str:
    """Generate the final answer from reranked evidence."""
    context = build_context(ranked_documents)

    response = llm.invoke(
        [
            (
                "system",
                ANSWER_PROMPT.format(context=context),
            ),
            ("human", query),
        ]
    )

    return response_text(response).strip()


# ============================================================
# 12. OUTPUT GUARDRAIL
# ============================================================

OUTPUT_GUARDRAIL_PROMPT = """
You are the output safety checker for a medical information chatbot.

Review the proposed answer.

Mark it SAFE if it:
- stays within general medical information,
- does not diagnose the user,
- does not provide personalized treatment,
- does not provide medication dosing,
- does not make unsupported claims,
- and does not ignore obvious emergency warnings.

Mark it UNSAFE if it violates those rules.

Return ONLY valid JSON:

{
  "status": "SAFE",
  "reason": "short explanation",
  "rewritten_answer": "safe rewritten answer if UNSAFE, otherwise empty"
}
"""


def apply_output_guardrail(answer: str) -> tuple[bool, str]:
    """Check and, if necessary, replace an unsafe generated answer."""
    response = llm.invoke(
        [
            ("system", OUTPUT_GUARDRAIL_PROMPT),
            ("human", answer),
        ]
    )

    raw = clean_json(response_text(response))

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fail closed.
        return (
            False,
            "I could not safely validate the generated response. "
            "Please consult a qualified healthcare professional.",
        )

    if result.get("status") == "SAFE":
        return True, answer

    rewritten = result.get("rewritten_answer", "").strip()

    if rewritten:
        return False, rewritten

    return (
        False,
        "I cannot provide that medical guidance. "
        "Please consult a qualified healthcare professional.",
    )


# ============================================================
# 13. SIMPLE RELEVANCE / CRAG-STYLE CHECK
# ============================================================

EVALUATION_PROMPT = """
Evaluate whether the retrieved context is sufficient to answer the user's
question.

Return ONLY valid JSON:

{
  "relevant": true,
  "score": 0.0,
  "reason": "short explanation"
}

score meaning:
0.0 = completely irrelevant
0.5 = partially useful
1.0 = directly useful

Do not answer the medical question.
"""


def evaluate_retrieval(
    query: str,
    ranked_documents: list[tuple[Document, float]],
) -> dict:
    """Simple CRAG-style evaluation of retrieved context."""
    context = build_context(ranked_documents)

    response = llm.invoke(
        [
            ("system", EVALUATION_PROMPT),
            (
                "human",
                f"USER QUESTION:\n{query}\n\n"
                f"RETRIEVED CONTEXT:\n{context}",
            ),
        ]
    )

    raw = clean_json(response_text(response))

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "relevant": True,
            "score": 0.5,
            "reason": "Evaluation could not be parsed.",
        }


# ============================================================
# 14. CORRECTIVE RETRIEVAL
# ============================================================

def corrective_retrieval(
    query: str,
    original_documents: list[Document],
) -> list[Document]:
    """
    Basic corrective retrieval.

    If the first retrieval is weak, try HyDE as a second retrieval path.
    This is intentionally simple so the flow is easy to understand.
    """
    print("CRAG: first retrieval appears weak. Trying corrective retrieval...")

    hypothetical_answer, hyde_documents = hyde_retrieval(
        query,
        top_k=HYDE_TOP_K,
    )

    # We return the new candidates.
    # The hypothetical answer is intentionally not used as final context.
    print("HyDE generated for corrective retrieval:")
    print(hypothetical_answer[:500])

    return hyde_documents


# ============================================================
# 15. COMPLETE RAG PIPELINE
# ============================================================

def run_rag(
    query: str,
    retrieval_method: str = "standard",
) -> dict:
    """
    Complete pipeline:

    1. Input guardrail
    2. Standard OR HyDE retrieval
    3. Cross-encoder reranking
    4. CRAG-style relevance evaluation
    5. Corrective retrieval if needed
    6. LLM answer
    7. Output guardrail
    """
    # --------------------------------------------------------
    # Step 1: Input guardrail
    # --------------------------------------------------------
    classification = classify_input(query)
    category = classification["category"]

    if category != "Allowed":
        safe_response = classification.get("safe_response") or (
            "I can help with general medical information, "
            "but I can't provide that type of personalized guidance."
        )

        return {
            "answer": safe_response,
            "category": category,
            "method": retrieval_method,
            "documents": [],
            "evaluation": None,
        }

    # --------------------------------------------------------
    # Step 2: Retrieval
    # --------------------------------------------------------
    hypothetical_answer = None

    if retrieval_method == "hyde":
        hypothetical_answer, documents = hyde_retrieval(query)
    else:
        documents = standard_retrieval(query)

    # --------------------------------------------------------
    # Step 3: Reranking
    # --------------------------------------------------------
    ranked_documents = rerank_documents(
        query,
        documents,
        top_k=RERANK_TOP_K,
    )

    # --------------------------------------------------------
    # Step 4: Evaluate retrieval
    # --------------------------------------------------------
    evaluation = evaluate_retrieval(
        query,
        ranked_documents,
    )

    # --------------------------------------------------------
    # Step 5: Corrective retrieval
    # --------------------------------------------------------
    if (
        not evaluation.get("relevant", True)
        or float(evaluation.get("score", 0.5)) < 0.5
    ):
        documents = corrective_retrieval(
            query,
            documents,
        )

        ranked_documents = rerank_documents(
            query,
            documents,
            top_k=RERANK_TOP_K,
        )

        evaluation = evaluate_retrieval(
            query,
            ranked_documents,
        )

    # --------------------------------------------------------
    # Step 6: Generate answer
    # --------------------------------------------------------
    if not ranked_documents:
        answer = (
            "I couldn't find relevant information in the medical "
            "knowledge base for that question."
        )
    else:
        answer = generate_answer(
            query,
            ranked_documents,
        )

    # --------------------------------------------------------
    # Step 7: Output guardrail
    # --------------------------------------------------------
    safe, final_answer = apply_output_guardrail(answer)

    return {
        "answer": final_answer,
        "category": category,
        "method": retrieval_method,
        "documents": ranked_documents,
        "evaluation": evaluation,
        "hypothetical_answer": hypothetical_answer,
        "output_safe": safe,
    }


# ============================================================
# ============================================================
# 16. STREAMLIT CHAT INTERFACE
# ============================================================

st.set_page_config(
    page_title="Medical Advisor",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
        .medical-warning {
            padding: 0.8rem 1rem;
            border-radius: 0.5rem;
            background: #fff8e1;
            border: 1px solid #f0d98c;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_rag_resources():
    """Create expensive RAG resources once per Streamlit process."""
    check_environment()

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=120,
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    sparse_embeddings = FastEmbedSparse(
        model_name="Qdrant/bm25",
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )

    chat_model = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
    )

    cross_encoder = CrossEncoder(RERANKER_MODEL)

    return client, store, chat_model, cross_encoder


# Initialize once, not on every Streamlit rerun.
qdrant_client, vector_store, llm, reranker = get_rag_resources()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Medical Advisor")

    retrieval_method = st.radio(
        "Retrieval method",
        options=["standard", "hyde"],
        format_func=lambda x: (
            "Standard Hybrid Search"
            if x == "standard"
            else "HyDE Search"
        ),
    )

    st.divider()

    st.subheader("RAG settings")
    st.caption(f"Qdrant collection: `{COLLECTION_NAME}`")
    st.caption(f"LLM: `{LLM_MODEL}`")
    st.caption(f"Candidates: `{TOP_K}`")
    st.caption(f"Rerank top: `{RERANK_TOP_K}`")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("🩺 Medical Advisor")
st.caption("Medical knowledge assistant powered by Qdrant + RAG")


# ------------------------------------------------------------
# Chat history
# ------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            result = message.get("result")

            if result:
                documents = result.get("documents", [])
                evaluation = result.get("evaluation")

                if documents:
                    with st.expander(
                        f"📚 Retrieved evidence ({len(documents)} chunks)"
                    ):
                        for number, (document, score) in enumerate(
                            documents,
                            start=1,
                        ):
                            source = document.metadata.get(
                                "source",
                                "Unknown source",
                            )

                            page = document.metadata.get("page", "")

                            if isinstance(page, int):
                                page = page + 1

                            title = f"Source {number}"

                            if page != "":
                                title += f" — Page {page}"

                            st.markdown(f"**{title}**")
                            st.caption(
                                f"{source} | "
                                f"Reranker score: {float(score):.4f}"
                            )
                            st.write(document.page_content[:2000])

                if evaluation:
                    with st.expander("📊 Retrieval evaluation"):
                        st.write(
                            f"**Score:** {evaluation.get('score', 'N/A')}"
                        )
                        st.write(
                            f"**Relevant:** {evaluation.get('relevant', 'N/A')}"
                        )
                        st.write(evaluation.get("reason", ""))


# ------------------------------------------------------------
# Chat input
# ------------------------------------------------------------

query = st.chat_input("Ask a general medical question...")


if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner(
            "🔎 Retrieving medical evidence and generating answer..."
        ):
            try:
                result = run_rag(
                    query,
                    retrieval_method=retrieval_method,
                )

                answer = result["answer"]
                st.markdown(answer)

                documents = result.get("documents", [])

                if documents:
                    with st.expander(
                        f"📚 Retrieved evidence ({len(documents)} chunks)"
                    ):
                        for number, (document, score) in enumerate(
                            documents,
                            start=1,
                        ):
                            source = document.metadata.get(
                                "source",
                                "Unknown source",
                            )

                            page = document.metadata.get("page", "")

                            if isinstance(page, int):
                                page = page + 1

                            title = f"Source {number}"

                            if page != "":
                                title += f" — Page {page}"

                            st.markdown(f"**{title}**")
                            st.caption(
                                f"{source} | "
                                f"Reranker score: {float(score):.4f}"
                            )
                            st.write(document.page_content[:2000])

                evaluation = result.get("evaluation")

                if evaluation:
                    with st.expander("📊 Retrieval evaluation"):
                        st.write(
                            f"**Score:** {evaluation.get('score', 'N/A')}"
                        )
                        st.write(
                            f"**Relevant:** {evaluation.get('relevant', 'N/A')}"
                        )
                        st.write(evaluation.get("reason", ""))

                st.caption(
                    f"Retrieval: {result['method']} | "
                    f"Input guardrail: {result['category']}"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "result": result,
                    }
                )

            except Exception as exc:
                error_message = (
                    "Sorry, I couldn't complete the request. "
                    "Please check the terminal for the technical error."
                )

                st.error(error_message)
                st.exception(exc)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


st.markdown(
    '<div class="medical-warning">'
    "⚠️ <b>Medical disclaimer:</b> Medical Advisor provides general "
    "information and is not a substitute for a qualified healthcare "
    "professional."
    "</div>",
    unsafe_allow_html=True,
)

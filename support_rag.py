import logging, os, requests
from typing import TypedDict, List
from functools import lru_cache
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("support_rag")
FAQ = """
Order Tracking
Customers can view shipment updates in the Orders section. A tracking link appears after dispatch. If tracking has not updated for 48 hours, contact support with the order reference. The assistant cannot access live order status.
Returns and Refunds
Unused items in original packaging may be requested for return within 14 days of delivery. Open Orders, choose the item, and select Return Request. After an approved return is received, a refund is usually processed within 5 to 7 business days. The assistant cannot approve a refund.
Account Access
If you cannot sign in, use Forgot Password. Check email and spam for the reset link. Never share your password or one-time verification code.
Billing and Payments
Payment status is shown in order details. If charged twice, do not submit repeated payments. Contact billing support with the transaction date and order reference. Do not share full card numbers or security codes.
Technical Support
If the website is not loading, refresh, check your internet connection, and try an updated browser. If the issue continues, clear browser cache or try a private window. Never share passwords or verification codes.
Contact Support
For issues not covered by these FAQs, contact support through the Help section. Include a brief description and relevant order reference, but do not send passwords, one-time codes, or full payment-card details.
"""

@lru_cache(maxsize=1)
def _get_chunks():
    doc = Document(page_content=FAQ, metadata={"source": "support_faq.txt"})
    return RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60).split_documents([doc])

@lru_cache(maxsize=1)
def _get_matrix_and_vectorizer():
    chunks = _get_chunks()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([d.page_content for d in chunks])
    return matrix, vectorizer

def _get_matrix():
    return _get_matrix_and_vectorizer()[0]

# Expose vectorizer via a lightweight cached property-style helper.
class _VectorizerProxy:
    def transform(self, texts):
        return _get_matrix_and_vectorizer()[1].transform(texts)

_vectorizer = _VectorizerProxy()

class State(TypedDict, total=False):
    question: str
    intent: str
    search_query: str
    context: str
    sources: List[str]
    draft: str
    retry_count: int
    final_answer: str

def classify(s):
    q = s["question"].lower()
    if any(w in q for w in ["refund", "return", "exchange"]): intent = "returns_refunds"
    elif any(w in q for w in ["track", "tracking", "shipment", "delivery", "order status"]): intent = "order_tracking"
    elif any(w in q for w in ["password", "sign in", "login", "account", "verification"]): intent = "account_access"
    elif any(w in q for w in ["charged", "payment", "billing", "invoice", "card"]): intent = "billing_payments"
    elif any(w in q for w in ["website", "error", "loading", "browser", "technical"]): intent = "technical_support"
    else: intent = "general_support"
    return {"intent": intent, "search_query": s["question"], "retry_count": s.get("retry_count", 0)}

def retrieve(s):
    # Lightweight TF-IDF retrieval avoids loading PyTorch and a transformer model on Render's 512 MiB free instance.
    chunks = _get_chunks()
    matrix = _get_matrix()
    query_vector = _vectorizer.transform([s["search_query"]])
    scores = cosine_similarity(query_vector, matrix).ravel()
    top_indices = scores.argsort()[::-1][:3]
    docs = [chunks[i] for i in top_indices if scores[i] > 0]
    return {"context": "\n\n".join(d.page_content for d in docs),
            "sources": [d.metadata.get("source", "support_faq.txt") for d in docs]}

def generate(s):
    # Use Hugging Face Inference Providers; local Ollama is not reachable from a hosted Space.
    token = os.getenv("HF_TOKEN", "").strip()
    if not token: raise RuntimeError("Set HF_TOKEN in Space Settings → Variables and secrets.")
    url = os.getenv("HF_API_URL", "https://router.huggingface.co/v1/chat/completions")
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    prompt = f"FAQ CONTEXT:\n{s.get('context','')}\n\nCUSTOMER QUESTION:\n{s['question']}\n\nAnswer only using the FAQ. If insufficient, reply exactly INSUFFICIENT_CONTEXT. Do not invent policies or live order status."
    response = requests.post(url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": model, "messages":[
            {"role":"system","content":"You are a careful customer support assistant. Never ask for passwords, one-time codes, or full payment-card details."},
            {"role":"user","content":prompt}], "max_tokens":180, "temperature":0}, timeout=90)
    response.raise_for_status()
    return {"draft": response.json()["choices"][0]["message"]["content"].strip()}

def review(s):
    d = s.get("draft","").strip()
    return {"final_answer": "" if (not d or "INSUFFICIENT_CONTEXT" in d) else d}

def route(s):
    if s.get("final_answer"): return "finish"
    if s.get("retry_count",0) < 1: return "retry"
    return "fallback"

def rewrite(s):
    terms = {"returns_refunds":"return eligibility refund process","order_tracking":"order dispatch tracking shipment status",
             "account_access":"account sign in password reset","billing_payments":"billing payment transaction",
             "technical_support":"website technical troubleshooting","general_support":"customer support help contact"}
    return {"search_query":s["question"]+" "+terms.get(s.get("intent","general_support"),"customer support"),
            "retry_count":s.get("retry_count",0)+1}

def fallback(s):
    return {"final_answer":"I couldn't find a reliable answer in the available FAQ. Please contact the support team through the Help section. Do not share your password, one-time code, or full payment-card details."}

# Define the intent → retrieval → generation → review → retry/fallback graph.
g = StateGraph(State)
for name, fn in [("classify",classify),("retrieve",retrieve),("generate",generate),("review",review),("rewrite",rewrite),("fallback",fallback)]:
    g.add_node(name,fn)
g.add_edge(START,"classify")
g.add_edge("classify","retrieve")
g.add_edge("retrieve","generate")
g.add_edge("generate","review")
g.add_conditional_edges("review",route,{"finish":END,"retry":"rewrite","fallback":"fallback"})
g.add_edge("rewrite","retrieve")
g.add_edge("fallback",END)
graph = g.compile()

def ask_support(question):
    result = graph.invoke({"question":question,"retry_count":0})
    return {"answer":result.get("final_answer",""),"intent":result.get("intent","general_support"),
            "sources":sorted(set(result.get("sources",[]))),"retry_count":result.get("retry_count",0)}


from typing import List, TypedDict, Literal, Annotated
import json
import operator
import random
import re
import sqlite3
import time
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient

from src.config import get_settings
from src.vectorstore import get_retriever


class RAGState(TypedDict, total=False):
    user_question: str
    question: str
    memory: Annotated[List[str], operator.add]
    retrieval_query: str
    web_query: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    support_status: Literal["fully_supported", "partially_supported", "no_support", ""]
    evidence: List[str]
    usefulness: Literal["useful", "not_useful", ""]
    use_reason: str
    support_retries: int
    retrieval_rewrites: int
    web_rewrites: int
    source_mode: Literal["internal", "web", "direct", "none"]
    used_web_search: bool
    trace: List[str]


class RetrieveDecision(BaseModel):
    should_retrieve: bool

class RelevanceDecision(BaseModel):
    is_relevant: bool

class SupportDecision(BaseModel):
    status: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

class UsefulnessDecision(BaseModel):
    status: Literal["useful", "not_useful"]
    reason: str

class QueryRewrite(BaseModel):
    query: str


def _llm():
    s = get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    allowed = {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b"}
    if s.groq_model not in allowed:
        raise RuntimeError(
            f"Unsupported Groq model '{s.groq_model}'. This account exposes only: {', '.join(sorted(allowed))}"
        )
    return ChatGroq(
        api_key=s.groq_api_key,
        model=s.groq_model,
        temperature=0,
    )


def _trace(state: RAGState, item: str):
    return [*(state.get("trace") or []), item]


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in [
        "rate limit",
        "rate_limit_exceeded",
        "too many requests",
        "tokens per minute",
        "tpm",
        "429",
    ])


def _invoke_model_with_retry(model, prompt, *, schema=None):
    s = get_settings()
    max_retries = max(0, s.groq_max_retries)
    for attempt in range(max_retries + 1):
        try:
            if schema is None:
                return model.invoke(prompt)
            return model.with_structured_output(schema).invoke(prompt)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            delay = s.groq_retry_base_delay * (2 ** attempt) + random.uniform(0.2, 0.8)
            time.sleep(delay)


def _invoke_structured(model, schema, prompt):
    try:
        return _invoke_model_with_retry(model, prompt, schema=schema)
    except Exception as exc:
        message = str(exc)
        tool_error = (
            "tool" in message.lower()
            and (
                "required" in message.lower()
                or "validation" in message.lower()
                or "not in request.tools" in message.lower()
                or "tool_use_failed" in message.lower()
            )
        )
        if not tool_error:
            raise

        response = _invoke_model_with_retry(model, prompt)
        text = getattr(response, "content", str(response)).strip()
        if text.startswith("```"):
            text = text.strip("`\n ")
            if text.lower().startswith("json"):
                text = text.split("\n", 1)[1] if "\n" in text else text

        cleaned = text.strip()
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                return schema(**data)
            except Exception:
                pass

        try:
            data = json.loads(cleaned)
            return schema(**data)
        except Exception:
            pass

        lowered = cleaned.lower()
        if schema is RetrieveDecision:
            should_retrieve = any(token in lowered for token in ["true", "yes", "retrieve", "need", "incident", "runbook", "debug", "deploy", "troubleshoot", "error"])
            return RetrieveDecision(should_retrieve=should_retrieve)
        if schema is RelevanceDecision:
            is_relevant = any(token in lowered for token in ["true", "yes", "relevant", "match", "related", "supports"])
            return RelevanceDecision(is_relevant=is_relevant)
        if schema is QueryRewrite:
            quoted = re.findall(r'"(.*?)"|\'(.*?)\'', cleaned)
            if quoted:
                extracted = next((a or b for a, b in quoted if (a or b).strip()), cleaned)
                return QueryRewrite(query=extracted.strip()[:200])
            return QueryRewrite(query=cleaned.strip()[:200])
        if schema is SupportDecision:
            status = "fully_supported"
            if "partially" in lowered:
                status = "partially_supported"
            elif "not" in lowered or "unsupported" in lowered or "no support" in lowered:
                status = "no_support"
            evidence_lines = [line.strip(" -•*\n") for line in cleaned.splitlines() if line.strip()]
            evidence = evidence_lines[:5] if evidence_lines else [cleaned[:200]]
            return SupportDecision(status=status, evidence=evidence)
        if schema is UsefulnessDecision:
            status = "useful" if any(token in lowered for token in ["useful", "yes", "directly addresses", "answers the question"]) else "not_useful"
            reason = cleaned[:200] or ("Direct answer to the question." if status == "useful" else "Answer does not directly address the request.")
            return UsefulnessDecision(status=status, reason=reason)

        raise ValueError(f"Could not parse structured response from model for {schema.__name__}: {cleaned}")


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        if meta.get("source_type") == "web":
            head = f"[WEB {i}] {meta.get('title','')} | {meta.get('url','')}"
        else:
            head = f"[INTERNAL {i}] {meta.get('title') or meta.get('document_name') or meta.get('source','')}"
            if meta.get("page") is not None:
                head += f" | page {int(meta['page']) + 1}"
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)




def _memory_text(state: RAGState, limit: int = 4) -> str:
    items = state.get("memory") or []
    return "\n\n".join(items[-limit:]) if items else "No previous conversation context."


def contextualize_question(state: RAGState):
    history = _memory_text(state)
    user_question = state.get("user_question") or state.get("question", "")
    if not state.get("memory"):
        return {"question": user_question, "trace": _trace(state, "Memory: new incident session")}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the newest user message as a standalone cloud-operations question using the previous conversation only when needed. Preserve service names, symptoms, errors, and constraints. If the message already stands alone, return it unchanged. Do not answer the question."),
        ("human", "Previous conversation:\n{history}\n\nNewest message:\n{question}"),
    ])
    out = _invoke_structured(_llm(), QueryRewrite, prompt.format_messages(history=history, question=user_question))
    return {"question": out.query, "trace": _trace(state, f"Memory contextualized question: {out.query}")}


def commit_memory(state: RAGState):
    user_question = state.get("user_question") or state.get("question", "")
    answer = state.get("answer", "")
    route = state.get("source_mode", "none")
    entry = f"User: {user_question}\nAssistant ({route}): {answer}"
    return {"memory": [entry], "trace": _trace(state, "SQLite memory checkpoint updated")}


def decide_retrieval(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You decide whether the question needs retrieval. Choose true for cloud operations, production incidents, service runbooks, deployment procedures, infrastructure behavior, troubleshooting steps, specific/current technical facts, or whenever evidence is needed. Choose false only for generic technical explanations that can safely be answered from general knowledge. If unsure choose true."),
        ("human", "Question: {question}"),
    ])
    out = _invoke_structured(_llm(), RetrieveDecision, prompt.format_messages(question=state["question"]))
    return {"need_retrieval": out.should_retrieve, "trace": _trace(state, f"Retrieval decision: {out.should_retrieve}")}


def route_after_decide(state: RAGState) -> Literal["direct", "retrieve"]:
    return "retrieve" if state.get("need_retrieval", True) else "direct"


def generate_direct(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior cloud operations and software engineering expert. Answer the user question clearly, accurately, and with practical depth. Prefer concise but complete explanations, relevant examples, and actionable next steps when appropriate. Do not invent organization-specific infrastructure, credentials, runbooks, incident histories, or deployment procedures. If the answer is uncertain, say so and explain the safest assumption."),
        ("human", "{question}"),
    ])
    ans = _llm().invoke(prompt.format_messages(question=state["question"])).content
    return {"answer": ans, "source_mode": "direct", "trace": _trace(state, "Generated direct answer")}


def retrieve_internal(state: RAGState):
    q = state.get("retrieval_query") or state["question"]
    docs = get_retriever().invoke(q)
    for d in docs:
        d.metadata = {**(d.metadata or {}), "source_type": "internal"}
    return {"docs": docs, "relevant_docs": [], "source_mode": "internal", "trace": _trace(state, f"Internal retrieval: {len(docs)} chunks")}


def grade_relevance(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Judge relevance at the topic/evidence level. A document is relevant when it contains information useful for answering the user's question. Do not require the exact final answer. Be strict about unrelated content."),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ])
    grader = _llm()
    relevant = []
    for d in state.get("docs", []):
        try:
            decision = _invoke_structured(grader, RelevanceDecision, prompt.format_messages(question=state["question"], document=d.page_content[:7000]))
            if decision.is_relevant:
                relevant.append(d)
        except Exception:
            continue
    mode = state.get("source_mode", "internal")
    return {"relevant_docs": relevant, "trace": _trace(state, f"Relevance grade ({mode}): {len(relevant)}/{len(state.get('docs', []))} relevant")}


def route_after_relevance(state: RAGState) -> Literal["generate", "rewrite_internal", "rewrite_web", "no_answer"]:
    if state.get("relevant_docs"):
        return "generate"
    s = get_settings()
    if state.get("source_mode") == "web":
        if state.get("web_rewrites", 0) < s.max_web_rewrites:
            return "rewrite_web"
        return "no_answer"
    if state.get("retrieval_rewrites", 0) < s.max_retrieval_rewrites:
        return "rewrite_internal"
    return "rewrite_web"


def rewrite_internal_query(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the operations question for semantic vector retrieval over internal cloud runbooks, SOPs, postmortems, architecture notes, and troubleshooting documents. Use 6-18 words, preserve service names and error symptoms, add useful operations keywords, remove filler, and do not answer."),
        ("human", "Question: {question}\nPrevious query: {previous}"),
    ])
    out = _invoke_structured(_llm(), QueryRewrite, prompt.format_messages(question=state["question"], previous=state.get("retrieval_query", "")))
    return {"retrieval_query": out.query, "retrieval_rewrites": state.get("retrieval_rewrites", 0)+1, "docs": [], "relevant_docs": [], "trace": _trace(state, f"Rewrote internal query: {out.query}")}


def rewrite_web_query(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question into a concise internet search query of 6-14 words. Preserve important entities. Add recency wording when the question asks for latest/current/today. Do not answer."),
        ("human", "Question: {question}\nPrevious web query: {previous}"),
    ])
    out = _invoke_structured(_llm(), QueryRewrite, prompt.format_messages(question=state["question"], previous=state.get("web_query", "")))
    return {"web_query": out.query, "web_rewrites": state.get("web_rewrites", 0)+1, "docs": [], "relevant_docs": [], "trace": _trace(state, f"Prepared internet search query: {out.query}")}


def web_search(state: RAGState):
    s = get_settings()
    if not s.tavily_api_key:
        return {"docs": [], "source_mode": "web", "used_web_search": True, "trace": _trace(state, "Internet search unavailable: TAVILY_API_KEY missing")}
    client = TavilyClient(api_key=s.tavily_api_key)
    q = state.get("web_query") or state["question"]
    response = client.search(query=q, search_depth="advanced", max_results=5, include_answer=False)
    docs = []
    for r in response.get("results", []):
        content = r.get("content", "")
        docs.append(Document(
            page_content=content,
            metadata={"source_type": "web", "source": r.get("url", ""), "url": r.get("url", ""), "title": r.get("title", "")},
        ))
    return {"docs": docs, "source_mode": "web", "used_web_search": True, "trace": _trace(state, f"Internet search: {len(docs)} results")}


def generate_from_context(state: RAGState):
    context = _format_context(state.get("relevant_docs", []))
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are AI Operations assistant, a senior enterprise cloud operations and incident-response advisor. Base your answer only on the supplied evidence and cite the facts carefully. Prefer private runbooks, SOPs, architecture notes, and postmortems when present. If the evidence is from the web, clearly label it as external guidance and never present it as an internal policy or company-specific procedure. Do not invent infrastructure facts, credentials, commands, incident history, or security-sensitive actions. Produce a clear, reliable, and actionable response with concise reasoning, practical next steps, and explicit cautions when the evidence requires them."),
        ("human", "Question:\n{question}\n\nEvidence:\n{context}"),
    ])
    ans = _llm().invoke(prompt.format_messages(question=state["question"], context=context)).content
    return {"answer": ans, "context": context, "support_retries": 0, "trace": _trace(state, f"Generated answer from {state.get('source_mode','')} evidence")}


def check_support(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Verify whether every meaningful claim in the answer is supported by the supplied evidence. Return fully_supported only when all important claims are grounded; partially_supported when some claims are grounded but some are not; no_support when key claims are unsupported. Evidence excerpts should be short."),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nEvidence:\n{context}"),
    ])
    out = _invoke_structured(_llm(), SupportDecision, prompt.format_messages(question=state["question"], answer=state.get("answer", ""), context=state.get("context", "")))
    return {"support_status": out.status, "evidence": out.evidence, "trace": _trace(state, f"Support check: {out.status}")}


def route_after_support(state: RAGState) -> Literal["usefulness", "revise"]:
    if state.get("support_status") == "fully_supported":
        return "usefulness"
    if state.get("support_retries", 0) >= get_settings().max_support_retries:
        return "usefulness"
    return "revise"


def revise_answer(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the answer so every factual claim is directly supported by the provided evidence. Remove unsupported interpretation and speculation. Still answer the question naturally; do not mention this verification process."),
        ("human", "Question:\n{question}\n\nCurrent answer:\n{answer}\n\nEvidence:\n{context}"),
    ])
    ans = _llm().invoke(prompt.format_messages(question=state["question"], answer=state.get("answer", ""), context=state.get("context", ""))).content
    return {"answer": ans, "support_retries": state.get("support_retries", 0)+1, "trace": _trace(state, "Revised answer for grounding")}


def check_usefulness(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Judge only whether the answer directly addresses the user's question. Do not re-grade factual grounding. Return useful or not_useful and a one-line reason."),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ])
    out = _invoke_structured(_llm(), UsefulnessDecision, prompt.format_messages(question=state["question"], answer=state.get("answer", "")))
    return {"usefulness": out.status, "use_reason": out.reason, "trace": _trace(state, f"Usefulness check: {out.status}")}


def route_after_usefulness(state: RAGState) -> Literal["end", "rewrite_internal", "rewrite_web", "no_answer"]:
    if state.get("usefulness") == "useful":
        return "end"
    s = get_settings()
    if state.get("source_mode") == "internal":
        if state.get("retrieval_rewrites", 0) < s.max_retrieval_rewrites:
            return "rewrite_internal"
        return "rewrite_web"
    if state.get("source_mode") == "web" and state.get("web_rewrites", 0) < s.max_web_rewrites:
        return "rewrite_web"
    return "no_answer"


def no_answer(state: RAGState):
    return {"answer": "I could not find enough reliable runbook or external evidence to recommend a safe troubleshooting action.", "source_mode": "none", "trace": _trace(state, "Stopped: no reliable answer found")}


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("contextualize", contextualize_question)
    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("direct", generate_direct)
    g.add_node("retrieve", retrieve_internal)
    g.add_node("grade", grade_relevance)
    g.add_node("rewrite_internal", rewrite_internal_query)
    g.add_node("rewrite_web", rewrite_web_query)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate_from_context)
    g.add_node("support", check_support)
    g.add_node("revise", revise_answer)
    g.add_node("usefulness", check_usefulness)
    g.add_node("no_answer", no_answer)
    g.add_node("commit_memory", commit_memory)

    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "decide_retrieval")
    g.add_conditional_edges("decide_retrieval", route_after_decide, {"direct":"direct", "retrieve":"retrieve"})
    g.add_edge("direct", "commit_memory")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_relevance, {
        "generate":"generate", "rewrite_internal":"rewrite_internal", "rewrite_web":"rewrite_web", "no_answer":"no_answer"
    })
    g.add_edge("rewrite_internal", "retrieve")
    g.add_edge("rewrite_web", "web_search")
    g.add_edge("web_search", "grade")
    g.add_edge("generate", "support")
    g.add_conditional_edges("support", route_after_support, {"usefulness":"usefulness", "revise":"revise"})
    g.add_edge("revise", "support")
    g.add_conditional_edges("usefulness", route_after_usefulness, {
        "end":"commit_memory", "rewrite_internal":"rewrite_internal", "rewrite_web":"rewrite_web", "no_answer":"no_answer"
    })
    g.add_edge("no_answer", "commit_memory")
    g.add_edge("commit_memory", END)

    db_path = Path(__file__).resolve().parents[1] / "data" / "langgraph_memory.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return g.compile(checkpointer=checkpointer)


def _sources(docs: List[Document]):
    seen, out = set(), []
    for d in docs or []:
        m = d.metadata or {}
        typ = "web" if m.get("source_type") == "web" else "internal"
        key = (typ, m.get("url") or m.get("source"), m.get("page"))
        if key in seen:
            continue
        seen.add(key)
        item = {
            "type": typ,
            "title": m.get("title") or m.get("document_name") or "",
            "source": m.get("source") or "",
            "url": m.get("url") if typ == "web" else None,
            "page": (int(m["page"]) + 1) if typ == "internal" and m.get("page") is not None else None,
        }
        out.append(item)
    return out


_graph = None

def run_self_rag(question: str, thread_id: str) -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()
    initial: RAGState = {
        "user_question": question,
        "question": question,
        "memory": [],
        "retrieval_query": question,
        "web_query": "",
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "support_status": "",
        "evidence": [],
        "usefulness": "",
        "use_reason": "",
        "support_retries": 0,
        "retrieval_rewrites": 0,
        "web_rewrites": 0,
        "source_mode": "internal",
        "used_web_search": False,
        "trace": [],
    }
    result = _graph.invoke(initial, config={"configurable": {"thread_id": thread_id}, "recursion_limit": 60})
    mode = result.get("source_mode", "none")
    route = {"internal":"Private Runbooks", "web":"Internet Search", "direct":"General Knowledge", "none":"No Reliable Evidence"}.get(mode, mode)
    return {
        "answer": result.get("answer", ""),
        "route": route,
        "used_web_search": bool(result.get("used_web_search")),
        "support_status": result.get("support_status", ""),
        "usefulness": result.get("usefulness", ""),
        "sources": _sources(result.get("relevant_docs", [])),
        "trace": result.get("trace", []),
        "thread_id": thread_id,
        "memory_turns": len(result.get("memory", [])),
    }
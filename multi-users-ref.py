"""
PDF 기반 멀티유저 멀티세션 RAG 챗봇 (Supabase Auth + 사용자별 세션/벡터 RLS).
로컬: 프로젝트 루트에서 `streamlit run 7.MultiService/code/multi-users-ref.py`
Secrets: Cloud **Settings → Secrets**, 로컬 **프로젝트 루트 `.env`**, 또는 `.streamlit/secrets.toml`에
SUPABASE_URL, SUPABASE_ANON_KEY(권장) 또는 SUPABASE_SERVICE_ROLE_KEY.
`.env`는 시작 시 `load_dotenv`로 `os.environ`에 올린 뒤 **`os.getenv`** 로 읽습니다.
API 키(OpenAI 등)는 사이드바에서 입력 (Streamlit Cloud 배포용).
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pypdf import PdfReader
from supabase import Client, create_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
load_dotenv(override=False)

SYSTEM_RAG = (
    "너는 참고자료를 바탕으로 정확하게 답하는 도우미입니다. "
    "참고자료에 없는 내용은 추측하지 말고 모른다고 말하세요. "
    "답변은 한국어 존댓말로, 중학생도 이해할 수 있게 풀어서 설명하되 핵심을 빠뜨리지 마세요."
)


def hydrate_supabase_env_from_streamlit_secrets() -> None:
    """Streamlit Secrets(toml/클라우드 대시보드) 값을 os.environ에 주입해 `os.getenv`로 읽히게 함.

    - Streamlit Community Cloud는 대시보드 Secrets를 환경변수로도 노출하는 경우가 많음.
    - 로컬 `.streamlit/secrets.toml`만 쓰는 경우엔 자동 노출이 없을 수 있어, 여기서 비어 있을 때만 복사.
    """
    keys = ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY")
    try:
        sec = st.secrets
    except Exception:
        return
    for k in keys:
        if (os.getenv(k) or "").strip():
            continue
        try:
            if k not in sec:
                continue
            val = sec[k]
            if val is not None and str(val).strip():
                os.environ[k] = str(val).strip()
        except Exception:
            continue


def get_supabase_env_url() -> str:
    return (os.getenv("SUPABASE_URL") or "").strip().strip('"').strip("'")


def get_supabase_env_key() -> str:
    return (
        (os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "")
        .strip()
        .strip('"')
        .strip("'")
    )


def create_supabase_client() -> Client:
    url = get_supabase_env_url()
    key = get_supabase_env_key()
    if not url or not key:
        raise RuntimeError(
            "Supabase 연결 정보가 없습니다.\n\n"
            "• Streamlit Cloud: 앱 **Settings → Secrets**에 `SUPABASE_URL`과 "
            "`SUPABASE_ANON_KEY`(권장) 또는 `SUPABASE_SERVICE_ROLE_KEY`를 등록하세요. "
            "등록된 값은 실행 시 환경변수로 제공되며, 이 앱은 **`os.getenv`** 로만 읽습니다.\n\n"
            "• 로컬: 저장소 **루트**의 `.env`에 `SUPABASE_URL`, `SUPABASE_ANON_KEY`(또는 "
            "`SUPABASE_SERVICE_ROLE_KEY`)를 넣으세요(앱이 `load_dotenv`로 읽습니다). "
            "또는 OS 환경변수, `.streamlit/secrets.toml`에 동일 키로 설정할 수 있습니다."
        )
    return create_client(url, key)


def restore_auth_session(client: Client) -> None:
    at = st.session_state.get("_sb_access_token")
    rt = st.session_state.get("_sb_refresh_token")
    if not at or not rt:
        return
    try:
        client.auth.set_session(at, rt)
    except Exception:
        st.session_state.pop("_sb_access_token", None)
        st.session_state.pop("_sb_refresh_token", None)


def persist_auth_session(client: Client) -> None:
    try:
        sess = client.auth.get_session()
        if sess is None:
            return
        access = getattr(sess, "access_token", None)
        refresh = getattr(sess, "refresh_token", None)
        if access and refresh:
            st.session_state._sb_access_token = access
            st.session_state._sb_refresh_token = refresh
    except Exception:
        pass


def clear_auth_session() -> None:
    st.session_state.pop("_sb_access_token", None)
    st.session_state.pop("_sb_refresh_token", None)
    st.session_state.pop("_sb_email", None)
    st.session_state.db_session_id = None
    st.session_state.messages = []
    st.session_state.uploaded_file_names = []
    if "sess_sb" in st.session_state:
        del st.session_state["sess_sb"]


def current_user_id(client: Client) -> Optional[str]:
    try:
        u = client.auth.get_user()
        if u and getattr(u, "user", None):
            uid = getattr(u.user, "id", None)
            if uid:
                return str(uid)
    except Exception:
        pass
    return None


def format_pg_vector(values: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in values) + "]"


def to_lc_messages(chat_messages: List[dict]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for message in chat_messages:
        if message["role"] == "user":
            out.append(HumanMessage(content=message["content"]))
        else:
            out.append(AIMessage(content=message["content"]))
    return out


def extract_pdf_text(uploaded_files) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        reader = PdfReader(temp_path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        pairs.append((uploaded_file.name, "\n".join(pages_text)))
    return pairs


def get_llm(model_id: str, openai_key: str, anthropic_key: str, google_key: str):
    if model_id == "gpt-4o-mini":
        if not openai_key:
            raise RuntimeError("OpenAI API 키가 필요합니다.")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True, api_key=openai_key)
    if model_id == "claude-sonnet-4-5":
        if not anthropic_key:
            raise RuntimeError("Anthropic API 키가 필요합니다.")
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=0, streaming=True, api_key=anthropic_key)
    if model_id == "gemini-3-pro-preview":
        if not google_key:
            raise RuntimeError("Google(Gemini) API 키가 필요합니다.")
        return ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            temperature=0,
            streaming=True,
            google_api_key=google_key,
        )
    raise RuntimeError(f"지원하지 않는 모델입니다: {model_id}")


def get_title_llm(openai_key: str):
    if not openai_key:
        raise RuntimeError("세션 제목 생성에 OpenAI API 키가 필요합니다.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False, api_key=openai_key)


def stream_text(llm: Any, messages: List[BaseMessage]) -> Iterator[str]:
    for chunk in llm.stream(messages):
        if isinstance(chunk, AIMessageChunk):
            content = chunk.content
        elif hasattr(chunk, "content"):
            content = getattr(chunk, "content", None)
        else:
            content = None
        if not content:
            continue
        if isinstance(content, str):
            yield content
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    yield part.get("text", "") or ""


def semantic_search(
    supabase: Client,
    session_id: str,
    question: str,
    embeddings: OpenAIEmbeddings,
    k: int = 4,
) -> str:
    qvec = embeddings.embed_query(question)
    payload = {
        "query_embedding": format_pg_vector(qvec),
        "match_count": k,
        "filter_session_id": session_id,
    }
    try:
        res = supabase.rpc("match_vector_documents", payload).execute()
        rows = res.data or []
    except Exception:
        res = (
            supabase.table("vector_documents")
            .select("content")
            .eq("session_id", session_id)
            .limit(50)
            .execute()
        )
        rows = [{"content": r.get("content", "")} for r in (res.data or [])]
    parts = [r.get("content", "") for r in rows if r.get("content")]
    return "\n\n".join(parts)


def upsert_session_vectors(
    supabase: Client,
    session_id: str,
    file_chunks: List[tuple[str, str]],
    embeddings: OpenAIEmbeddings,
    batch_size: int = 10,
) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    batch: List[dict[str, Any]] = []

    for file_name, raw_text in file_chunks:
        supabase.table("vector_documents").delete().eq("session_id", session_id).eq("file_name", file_name).execute()
        docs = splitter.create_documents([raw_text], metadatas=[{"file_name": file_name}])
        texts = [d.page_content for d in docs]
        embs = embeddings.embed_documents(texts)
        for text, emb in zip(texts, embs):
            batch.append(
                {
                    "session_id": session_id,
                    "file_name": file_name,
                    "content": text,
                    "metadata": {"file_name": file_name},
                    "embedding": format_pg_vector(emb),
                }
            )
            if len(batch) >= batch_size:
                supabase.table("vector_documents").insert(batch).execute()
                batch.clear()
    if batch:
        supabase.table("vector_documents").insert(batch).execute()


def persist_messages(supabase: Client, session_id: str, messages: List[dict]) -> None:
    supabase.table("chat_messages").delete().eq("session_id", session_id).execute()
    rows = []
    for i, m in enumerate(messages):
        rows.append(
            {
                "session_id": session_id,
                "role": m["role"],
                "content": m["content"],
                "ord": i,
            }
        )
    if rows:
        supabase.table("chat_messages").insert(rows).execute()
    supabase.table("chat_sessions").update(
        {"updated_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", session_id).execute()


def ensure_db_session(supabase: Client, user_id: str) -> str:
    if st.session_state.db_session_id:
        return st.session_state.db_session_id
    row = supabase.table("chat_sessions").insert({"title": "새 세션", "user_id": user_id}).execute()
    sid = row.data[0]["id"]
    st.session_state.db_session_id = sid
    return sid


def fetch_sessions(supabase: Client) -> List[dict]:
    res = supabase.table("chat_sessions").select("id,title,updated_at").order("updated_at", desc=True).execute()
    return res.data or []


def load_session_into_ui(supabase: Client, session_id: str) -> None:
    st.session_state.db_session_id = session_id
    res = supabase.table("chat_messages").select("role,content,ord").eq("session_id", session_id).order("ord").execute()
    rows = res.data or []
    st.session_state.messages = [{"role": r["role"], "content": r["content"]} for r in rows]


def generate_session_title(first_q: str, first_a: str, openai_key: str) -> str:
    llm = get_title_llm(openai_key)
    prompt = (
        "다음은 채팅의 첫 질문과 첫 답변입니다. "
        "이 대화를 대표하는 짧은 한국어 제목을 35자 이내로 한 줄만 출력하세요. "
        "따옴표나 부가 설명 없이 제목만.\n\n"
        f"질문: {first_q}\n\n답변: {first_a[:2000]}"
    )
    msg = llm.invoke([HumanMessage(content=prompt)])
    title = (msg.content or "").strip().splitlines()[0].strip()
    return title[:120] if title else "새 세션"


def clone_session_snapshot(supabase: Client, source_session_id: str, title: str, user_id: str) -> str:
    ins = supabase.table("chat_sessions").insert({"title": title, "user_id": user_id}).execute()
    new_id = ins.data[0]["id"]

    msgs = (
        supabase.table("chat_messages")
        .select("role,content,ord")
        .eq("session_id", source_session_id)
        .order("ord")
        .execute()
    ).data or []
    if msgs:
        supabase.table("chat_messages").insert(
            [{"session_id": new_id, "role": m["role"], "content": m["content"], "ord": m["ord"]} for m in msgs]
        ).execute()

    vecs = supabase.table("vector_documents").select("file_name,content,metadata,embedding").eq("session_id", source_session_id).execute().data or []
    batch: List[dict[str, Any]] = []
    for v in vecs:
        emb = v.get("embedding")
        if isinstance(emb, list):
            emb_out = format_pg_vector([float(x) for x in emb])
        else:
            emb_out = str(emb)
        batch.append(
            {
                "session_id": new_id,
                "file_name": v["file_name"],
                "content": v["content"],
                "metadata": v.get("metadata") or {},
                "embedding": emb_out,
            }
        )
        if len(batch) >= 10:
            supabase.table("vector_documents").insert(batch).execute()
            batch.clear()
    if batch:
        supabase.table("vector_documents").insert(batch).execute()

    return new_id


def followup_questions_block(llm: Any, question: str, answer: str, context: str) -> str:
    sys = SystemMessage(
        content=(
            "사용자가 문서 기반 답변을 받았다. 앞으로 더 깊게 파고들기 좋은 후속 질문을 한국어로 정확히 3개만 번호 목록으로 출력하라. "
            "형식만 사용: 1. ...\n2. ...\n3. ..."
        )
    )
    human = HumanMessage(
        content=f"원 질문:\n{question}\n\n답변 요약:\n{answer[:4000]}\n\n참고 발췌:\n{context[:2000]}"
    )
    out = llm.invoke([sys, human])
    return (out.content or "").strip()


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
          .brand-wrap {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
          }
          .brand-logo {
            font-size: 2rem;
            line-height: 1;
          }
          .brand-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1e3a5f;
            margin: 0;
          }
          .brand-sub {
            color: #5c6b7a;
            margin-top: 0.15rem;
            font-size: 0.95rem;
          }
          div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "db_session_id" not in st.session_state:
        st.session_state.db_session_id = None
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "gpt-4o-mini"


def auth_sidebar_block(supabase: Client) -> Optional[str]:
    st.markdown("### 계정")
    st.caption("Supabase Auth는 이메일 형식의 로그인 ID를 사용합니다.")
    login_id = st.text_input("로그인 ID (이메일)", key="auth_login_id")
    password = st.text_input("비밀번호", type="password", key="auth_password")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("로그인", use_container_width=True):
            if not login_id or not password:
                st.warning("이메일과 비밀번호를 입력하세요.")
            else:
                try:
                    res = supabase.auth.sign_in_with_password({"email": login_id.strip(), "password": password})
                    sess = getattr(res, "session", None)
                    if sess and getattr(sess, "access_token", None):
                        st.session_state._sb_access_token = sess.access_token
                        st.session_state._sb_refresh_token = getattr(sess, "refresh_token", None) or ""
                        st.session_state._sb_email = login_id.strip()
                        st.success("로그인되었습니다.")
                        st.rerun()
                    else:
                        st.error("로그인에 실패했습니다. 이메일 인증이 필요할 수 있습니다.")
                except Exception as e:
                    st.error(f"로그인 오류: {e}")
    with c2:
        if st.button("회원가입", use_container_width=True):
            if not login_id or not password:
                st.warning("이메일과 비밀번호를 입력하세요.")
            else:
                try:
                    supabase.auth.sign_up({"email": login_id.strip(), "password": password})
                    st.success("가입 요청을 보냈습니다. 이메일을 확인하거나 대시보드에서 인증 설정을 확인하세요.")
                except Exception as e:
                    st.error(f"회원가입 오류: {e}")

    if st.button("로그아웃", use_container_width=True):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        clear_auth_session()
        st.rerun()

    uid = current_user_id(supabase)
    if uid:
        st.caption(f"사용자: `{st.session_state.get('_sb_email') or login_id or '…'}`")
    return uid


def main() -> None:
    st.set_page_config(page_title="멀티유저 멀티세션 RAG", page_icon="📚", layout="wide")
    apply_custom_css()
    init_state()

    col1, col2 = st.columns([0.12, 0.88])
    with col1:
        st.markdown('<div class="brand-logo">📚</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="brand-title">PDF 기반 멀티유저 멀티세션 RAG 챗봇</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="brand-sub">Supabase 계정별로 세션·벡터를 저장하고, 선택한 모델로 스트리밍 답변을 받습니다.</p>',
            unsafe_allow_html=True,
        )

    try:
        hydrate_supabase_env_from_streamlit_secrets()
        supabase = create_supabase_client()
        restore_auth_session(supabase)
        persist_auth_session(supabase)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    openai_key = anthropic_key = google_key = ""
    uid: Optional[str] = None
    uploaded_files = None

    with st.sidebar:
        st.markdown("### API 키")
        st.caption("Streamlit Cloud 배포 시 저장소에 키를 넣지 말고 여기서 입력하세요.")
        openai_key = (st.text_input("OpenAI API Key", type="password", key="sk_openai") or "").strip()
        anthropic_key = (st.text_input("Anthropic API Key", type="password", key="sk_anthropic") or "").strip()
        google_key = (st.text_input("Google (Gemini) API Key", type="password", key="sk_google") or "").strip()
        st.divider()

        uid = auth_sidebar_block(supabase)
        st.divider()

        if not uid:
            st.info("로그인 후 세션·채팅·PDF 기능을 사용할 수 있습니다.")
        else:
            st.markdown("### 모델 선택")
            st.session_state.model_choice = st.selectbox(
                "LLM",
                options=["gpt-4o-mini", "claude-sonnet-4-5", "gemini-3-pro-preview"],
                index=["gpt-4o-mini", "claude-sonnet-4-5", "gemini-3-pro-preview"].index(st.session_state.model_choice)
                if st.session_state.model_choice in ("gpt-4o-mini", "claude-sonnet-4-5", "gemini-3-pro-preview")
                else 0,
                label_visibility="collapsed",
            )

            sessions = fetch_sessions(supabase)
            id_to_label = {s["id"]: f"{s['title']} · {str(s['id'])[:8]}…" for s in sessions}

            st.markdown("### 세션 관리")
            base_ids = [s["id"] for s in sessions]
            if st.session_state.db_session_id and st.session_state.db_session_id not in base_ids:
                base_ids = [st.session_state.db_session_id] + base_ids
            options_ids: List[Optional[str]] = [None] + base_ids
            labels = ["(새 대화 · 목록만 보기)"] + [
                id_to_label.get(sid, f"진행 중 · {str(sid)[:8]}") for sid in base_ids
            ]
            st.session_state._session_id_options = options_ids
            st.session_state._supabase_client = supabase

            def on_session_pick() -> None:
                opts: List[Optional[str]] = st.session_state.get("_session_id_options") or []
                idx = int(st.session_state.get("sess_sb", 0))
                if idx <= 0 or idx >= len(opts):
                    return
                sid = opts[idx]
                if sid:
                    load_session_into_ui(st.session_state._supabase_client, sid)

            if "sess_sb" not in st.session_state:
                cur = st.session_state.db_session_id
                if cur and cur in base_ids:
                    st.session_state.sess_sb = list(base_ids).index(cur) + 1
                else:
                    st.session_state.sess_sb = 0
            if st.session_state.sess_sb >= len(options_ids):
                st.session_state.sess_sb = max(0, len(options_ids) - 1)

            st.selectbox(
                "저장된 세션",
                options=list(range(len(options_ids))),
                format_func=lambda i: labels[i],
                key="sess_sb",
                on_change=on_session_pick,
            )
            pick_idx = int(st.session_state.sess_sb)
            selected_id = options_ids[pick_idx] if 0 <= pick_idx < len(options_ids) else None

            if st.button("세션로드", use_container_width=True) and selected_id:
                load_session_into_ui(supabase, selected_id)
                st.success("세션을 불러왔습니다.")
                st.rerun()

            st.divider()

            if st.button("세션저장", use_container_width=True):
                msgs = st.session_state.messages
                u = current_user_id(supabase)
                if len(msgs) < 2:
                    st.warning("첫 질문과 답변이 있어야 세션을 저장할 수 있습니다.")
                elif not st.session_state.db_session_id:
                    st.warning("먼저 대화를 시작하거나 PDF를 올려 세션을 만든 뒤 저장하세요.")
                elif not u:
                    st.warning("로그인이 필요합니다.")
                elif not openai_key:
                    st.warning("세션 제목 생성에 OpenAI API 키가 필요합니다.")
                else:
                    first_q = next((m["content"] for m in msgs if m["role"] == "user"), "")
                    first_a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                    title = generate_session_title(first_q, first_a, openai_key)
                    clone_session_snapshot(supabase, st.session_state.db_session_id, title, u)
                    st.success(f"새 세션으로 저장했습니다: {title}")
                    st.rerun()

            if st.button("세션삭제", use_container_width=True) and selected_id:
                supabase.table("chat_sessions").delete().eq("id", selected_id).execute()
                if st.session_state.db_session_id == selected_id:
                    st.session_state.db_session_id = None
                    st.session_state.messages = []
                st.success("세션이 삭제되었습니다.")
                st.rerun()

            if st.button("화면초기화", use_container_width=True):
                st.session_state.messages = []
                st.session_state.db_session_id = None
                st.session_state.uploaded_file_names = []
                if "sess_sb" in st.session_state:
                    del st.session_state["sess_sb"]
                st.success("화면을 초기화했습니다. 새 대화를 시작할 수 있습니다.")
                st.rerun()

            if st.button("vectordb", use_container_width=True):
                sid = st.session_state.db_session_id or selected_id
                if not sid:
                    st.info("먼저 세션을 선택하거나 대화를 시작하세요.")
                else:
                    res = supabase.table("vector_documents").select("file_name").eq("session_id", sid).execute()
                    names = sorted({r.get("file_name") for r in (res.data or []) if r.get("file_name")})
                    st.write("현재 세션 벡터 DB 파일명:", names or "(없음)")

            st.markdown("### PDF 업로드")
            uploaded_files = st.file_uploader("PDF", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if not uid:
        st.warning("왼쪽 사이드바에서 로그인하거나 회원가입하세요 (`multi-users-ref.sql` 스키마 적용 및 Supabase Auth 활성화 필요).")
        st.stop()

    if not openai_key:
        st.info("PDF 임베딩 및 OpenAI 모델 사용을 위해 사이드바에 OpenAI API 키를 입력하세요.")

    embeddings: Optional[OpenAIEmbeddings] = None
    if openai_key:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

    current_names = [f.name for f in uploaded_files] if uploaded_files else []
    if uploaded_files and embeddings and current_names != st.session_state.uploaded_file_names:
        sid = ensure_db_session(supabase, uid)
        pairs = extract_pdf_text(uploaded_files)
        upsert_session_vectors(supabase, sid, pairs, embeddings)
        st.session_state.uploaded_file_names = current_names
        persist_messages(supabase, sid, st.session_state.messages)
        st.sidebar.success("문서를 분할·임베딩하여 저장했습니다.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("문서에 대해 질문해 보세요…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    sid = st.session_state.db_session_id
    if not sid:
        sid = ensure_db_session(supabase, uid)

    context = ""
    if embeddings:
        try:
            context = semantic_search(supabase, sid, prompt, embeddings, k=4)
        except Exception:
            context = ""

    llm = get_llm(st.session_state.model_choice, openai_key, anthropic_key, google_key)
    history = to_lc_messages(st.session_state.messages[:-1])
    sys = SystemMessage(content=f"{SYSTEM_RAG}\n\n참고자료:\n{context or '(없음)'}")
    msgs_for_llm: List[BaseMessage] = [sys, *history, HumanMessage(content=prompt)]

    assistant_text = ""
    with st.chat_message("assistant"):
        try:
            assistant_text = st.write_stream(stream_text(llm, msgs_for_llm))
        except Exception as e:
            assistant_text = f"(모델 호출 중 오류: {e})"
            st.markdown(assistant_text)
        if openai_key:
            try:
                extra_llm = get_title_llm(openai_key)
                extras = followup_questions_block(extra_llm, prompt, assistant_text, context)
                st.markdown("---\n\n**이어서 묻기 좋은 질문**\n\n" + extras)
                assistant_text = assistant_text + "\n\n---\n\n**이어서 묻기 좋은 질문**\n\n" + extras
            except Exception:
                pass

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    persist_messages(supabase, sid, st.session_state.messages)

    if len(st.session_state.messages) >= 2 and openai_key:
        try:
            u0 = st.session_state.messages[0]["content"]
            a0 = st.session_state.messages[1]["content"]
            title = generate_session_title(u0, a0, openai_key)
            supabase.table("chat_sessions").update(
                {"title": title, "updated_at": datetime.now(timezone.utc).isoformat()}
            ).eq("id", sid).execute()
        except Exception:
            pass


if __name__ == "__main__":
    main()

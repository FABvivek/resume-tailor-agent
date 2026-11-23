"""
Streamlit Web Interface for Resume Tailor Agent.
Provides an interactive experience for students:
- Upload resume PDF.
- Paste target job description.
- Interactive chat conversation with the agent.
- Color-coded gap analysis table (strong, partial, missing).
- Bullet rewrites with unsupported claims prominently highlighted ('unsupported: confirm or remove').
- Structured learning plan.
- Sidebar with cross-session remembered preferences and a 'Delete My Data' button.
- Expandable step trace inspector showing tool calls, latency, and tokens.
"""
import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Resume Tailor Agent",
    page_icon="📄",
    layout="wide"
)

# Initialize Session State variables
if "user_id" not in st.session_state:
    st.session_state.user_id = "student_user_1"
if "session_id" not in st.session_state:
    st.session_state.session_id = "sess_101"
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None


# ---------------------------------------------------------------------------
# Sidebar: User Settings, Cross-Session Memory & Privacy Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("👤 Candidate Profile")
    st.caption("Cross-Session Memory & Privacy")

    user_id_input = st.text_input("User ID", value=st.session_state.user_id)
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input
        st.session_state.chat_history = []
        st.session_state.last_response = None
        st.rerun()

    session_id_input = st.text_input("Session ID", value=st.session_state.session_id)
    st.session_state.session_id = session_id_input

    st.markdown("---")
    st.subheader("⚙️ Backend Health")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=2.0)
        if health_resp.status_code == 200:
            h_data = health_resp.json()
            if h_data.get("ollama_reachable") and h_data.get("model_available"):
                st.success(f"Ollama Online: `{h_data.get('active_model')}`")
            elif h_data.get("ollama_reachable"):
                st.warning(f"Ollama Online, but model `{h_data.get('active_model')}` not pulled.")
                st.info(h_data.get("message"))
            else:
                st.error("Ollama Offline (falling back to mocks/fallbacks)")
        else:
            st.error(f"API Error: HTTP {health_resp.status_code}")
    except Exception:
        st.warning("FastAPI backend not running at localhost:8000.")

    st.markdown("---")
    st.subheader("🛡️ Data Privacy")
    st.caption("Resumes contain sensitive personal info. Wiping removes all traces and uploaded files.")
    if st.button("🗑️ Delete My Data", type="secondary"):
        try:
            del_resp = requests.delete(f"{API_BASE_URL}/users/{st.session_state.user_id}", timeout=5.0)
            if del_resp.status_code == 200:
                st.success(f"All data wiped for {st.session_state.user_id}.")
                st.session_state.file_id = None
                st.session_state.filename = None
                st.session_state.chat_history = []
                st.session_state.last_response = None
                st.rerun()
            else:
                st.info("No prior data found for this user.")
        except Exception as exc:
            st.error(f"Failed to wipe data: {exc}")


# ---------------------------------------------------------------------------
# Main Screen: Header and Inputs
# ---------------------------------------------------------------------------
st.title("🎯 Resume Tailor Agent")
st.markdown(
    "Upload your resume PDF and paste a target job description. "
    "The agent analyzes semantic gaps, produces **strictly honest bullet rewrites**, "
    "and devises an actionable learning plan."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume PDF")
    uploaded_file = st.file_uploader("Choose a PDF resume", type=["pdf"])
    if uploaded_file is not None and (st.session_state.filename != uploaded_file.name):
        with st.spinner("Uploading and registering resume..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"user_id": st.session_state.user_id}
                upload_resp = requests.post(f"{API_BASE_URL}/upload", files=files, data=data, timeout=10.0)
                if upload_resp.status_code == 200:
                    res_json = upload_resp.json()
                    st.session_state.file_id = res_json.get("file_id")
                    st.session_state.filename = uploaded_file.name
                    st.success(f"Uploaded: {uploaded_file.name} (File ID: `{st.session_state.file_id}`)")
                else:
                    st.error(f"Upload failed: {upload_resp.text}")
            except Exception as exc:
                st.error(f"Could not connect to API: {exc}")
    elif st.session_state.file_id:
        st.info(f"Active Resume: **{st.session_state.filename}** (ID: `{st.session_state.file_id}`)")

with col2:
    st.subheader("2. Target Job Description")
    jd_input = st.text_area(
        "Paste the Job Description (JD)",
        height=180,
        placeholder="Paste requirements, tech stack, and responsibilities here..."
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# Chat Interface & Execution Trigger
# ---------------------------------------------------------------------------
st.subheader("3. Tailoring Chat & Instructions")

for chat_msg in st.session_state.chat_history:
    with st.chat_message(chat_msg["role"]):
        st.write(chat_msg["content"])

user_query = st.chat_input("Ask a question or type 'Tailor my resume for this JD'...")

trigger_analysis = False
if st.button("🚀 Analyze Gap & Tailor Resume", type="primary", use_container_width=True):
    user_query = "Please analyze my resume against this job description, highlight gaps, and suggest honest bullet rewrites."
    trigger_analysis = True

if user_query or trigger_analysis:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Agent analyzing requirements, computing embeddings, and formulating rewrites..."):
            payload = {
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.user_id,
                "message": user_query,
                "file_id": st.session_state.file_id,
                "jd_text": jd_input if jd_input.strip() else None
            }
            try:
                resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=90.0)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.last_response = data
                    reply_text = data.get("response", "Analysis complete.")
                    st.write(reply_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
                else:
                    err_msg = f"API Error {resp.status_code}: {resp.text}"
                    st.error(err_msg)
            except Exception as exc:
                st.error(f"Error communicating with backend: {exc}")


# ---------------------------------------------------------------------------
# Visual Output Dashboard (Gaps, Rewrites, Learning Plan, Traces)
# ---------------------------------------------------------------------------
res = st.session_state.last_response
if res:
    st.markdown("---")
    st.header("📊 Tailoring Analysis & Proposals")

    tab_gaps, tab_rewrites, tab_plan, tab_trace = st.tabs([
        "🔍 Skill Gap Matrix",
        "✍️ Grounded Bullet Rewrites",
        "📚 Learning Plan",
        "🕵️ Agent Execution Trace"
    ])

    with tab_gaps:
        gaps = res.get("gaps", [])
        if gaps:
            for idx, g in enumerate(gaps, 1):
                req = g.get("requirement", {})
                status_val = g.get("status", "missing").lower()

                if status_val == "strong":
                    color = "🟢 Strong Match"
                elif status_val == "partial":
                    color = "🟡 Partial Match"
                else:
                    color = "🔴 Missing"

                with st.expander(f"{color} — [{req.get('importance', 'must').upper()}] {req.get('text')}", expanded=True):
                    col_ev, col_adv = st.columns(2)
                    with col_ev:
                        st.markdown("**Resume Evidence:**")
                        ev = g.get("evidence")
                        if ev:
                            st.info(ev)
                        else:
                            st.caption("No matching evidence found in profile.")
                    with col_adv:
                        st.markdown("**Tailoring Advice:**")
                        st.write(g.get("advice", ""))
        else:
            st.info("No gap records returned yet. Upload a resume and paste a JD to generate.")

    with tab_rewrites:
        rewrites = res.get("rewrites", [])
        if rewrites:
            for idx, rw in enumerate(rewrites, 1):
                st.markdown(f"#### Bullet Rewrite #{idx}")
                col_orig, col_sugg = st.columns(2)
                with col_orig:
                    st.markdown("**Original:**")
                    st.text_area(f"orig_{idx}", rw.get("original", ""), height=100, disabled=True)
                with col_sugg:
                    st.markdown("**Suggested Rewrite:**")
                    st.text_area(f"sugg_{idx}", rw.get("suggested", ""), height=100)

                st.caption(f"**Reason:** {rw.get('reason', '')}")

                # Unsupported claims highlight
                unsupported = rw.get("unsupported_claims", [])
                if unsupported:
                    claims_str = ", ".join(unsupported)
                    st.error(f"⚠️ **Unsupported claims detected (confirm or remove):** {claims_str}")
                else:
                    st.success("✅ **Verified Grounded:** All tools and numbers appear in candidate profile.")
                st.markdown("---")
        else:
            st.info("No bullet rewrites generated yet.")

    with tab_plan:
        plan = res.get("learning_plan", "")
        if plan:
            st.markdown(plan)
        else:
            st.info("No learning plan items generated.")

    with tab_trace:
        st.markdown(f"**Execution Path:** `{res.get('path_used', 'tool_calling')}`")
        try:
            trace_resp = requests.get(f"{API_BASE_URL}/sessions/{st.session_state.session_id}/trace", timeout=5.0)
            if trace_resp.status_code == 200:
                traces = trace_resp.json()
                if traces:
                    for t in traces:
                        with st.expander(f"Step {t['step']}: Tool `{t['tool']}` ({t['latency_ms']} ms, {t['tokens']} tokens)"):
                            st.json(t["args"])
                            if t.get("error"):
                                st.error(f"Error: {t['error']}")
                            st.caption(f"Path: {t.get('path_used')}")
                else:
                    st.caption("No step traces recorded for this session.")
        except Exception as exc:
            st.warning(f"Could not load step trace: {exc}")

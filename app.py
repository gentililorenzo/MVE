import streamlit as st

from datetime import datetime

# Create the instance only once and reuse it.
# Use @st.cache_resource to avoid recreating it on every rerun.
@st.cache_resource
def create_system():
    try:
        from RAG.RAG import rag
    except Exception as e:
        # If the import fails, show the error in the UI instead of crashing everything.
        st.error(f"Error importing RAG.sectoral_prompt: {e}")
        return None
    return rag()

system = create_system()

# Initialize session state
if "company_profile" not in st.session_state:
    st.session_state["company_profile"] = {"num_employees": None, "activity": ""}
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts: {"question":..., "response":..., "ts":...}
if "profile_saved" not in st.session_state:
    st.session_state["profile_saved"] = False

st.set_page_config(page_title="Minimum Viable ESG", layout="wide")

# --- Sidebar: company profile ---
with st.sidebar:
    st.title("Company profile")
    num_employees = st.number_input(
        "Number of employees",
        min_value=0,
        step=1,
        value=st.session_state["company_profile"]["num_employees"] or 0
    )
    activity_desc = st.text_area(
        "Business description",
        value=st.session_state["company_profile"]["activity"],
        height="content"
    )
    save_profile = st.button("Save profile")

    if save_profile:
        if num_employees == 0:
            st.error("The number of employees must be greater than 0.")  # We could set default to 1 to force
                                                                    # the user to enter an exact number.
            st.session_state["profile_saved"] = False
        elif not activity_desc or not activity_desc.strip():
            st.error("The business description is required.")
            st.session_state["profile_saved"] = False
        else:
            st.session_state["company_profile"] = {"num_employees": int(num_employees), "activity": activity_desc}
            st.session_state["profile_saved"] = True
            st.success("Profile saved successfully ✅")

    st.markdown("---")
    if st.button("Clear history"):
        st.session_state["history"] = []
        st.info("History cleared")

    st.markdown("---")
    st.markdown("**Suggested questions:**")
    st.write("- `Which ESG practices are most important for a company like ours?`")
    st.write("- `Give me an action plan to prepare a basic environmental report.`")
    st.write("- `What sustainability plan or proposed sustainable practices can I present to the bank to obtain financing?`")

# --- Main layout ---
st.title("🟢 Sustainability assistant")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Ask a question")
    # we use a form to avoid accidental multiple submits
    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area("Your question", height=120)
        submitted = st.form_submit_button("Submit") # if profile not filled/saved block the request.
                                                    # It would be wasteful, too little context.

    if submitted:
        if not st.session_state["profile_saved"]:
            st.error("⚠️ You must complete and save the company profile before proceeding.")
        elif not question or not question.strip():
            st.warning("⚠️ Please ask a question to receive an answer.")
        elif system is None:
            st.error("The system is not properly initialized. Check the logs.")
        else:
            profile = st.session_state["company_profile"]   # show spinner while processing
            with st.spinner("Generating response..."):
                try:
                    response = system.consult(profile, question)
                except Exception as e:
                    st.error(f"Error during consultation: {e}")
                    response = f"Internal error: {e}"

            # save to history
            st.session_state["history"].append({
                "question": question,
                "response": response,
                "ts": datetime.now().strftime("%Y/%m/%d - %H:%M")
            })
            st.success("Response generated ✅")

    # Show history (newest first)
    st.markdown("### History")
    if st.session_state["history"]:
        for item in reversed(st.session_state["history"]):
            st.markdown(f"**You —** {item['question']}")
            st.markdown(f"> {item['response']}")
            st.caption(item["ts"])
            st.markdown("---")
    else:
        st.info("No questions submitted yet.")

with col2:
    st.subheader("Current profile")
    prof = st.session_state["company_profile"]
    st.write(f"- **Number of employees:** {prof.get('num_employees')}")
    st.write(f"- **Business activity:** {prof.get('activity') or '*not set*'}")
    st.markdown("---")
    # Download history as text
    if st.session_state["history"]:
        transcript = ""
        for item in st.session_state["history"]:
            transcript += f"[{item['ts']}] Question: {item['question']}\nAnswer: {item['response']}\n\n"
        st.download_button("Download history (.txt)", transcript, file_name="consultation_history.txt", mime="text/plain")

# --- footer / notes ---
st.markdown("---")
st.markdown("If the app does not update automatically when you modify project files, press `R` in the Streamlit UI or use the 'Rerun' option.")
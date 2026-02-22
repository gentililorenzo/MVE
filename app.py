import streamlit as st
from datetime import datetime
from RAG.RAG import rag
from utilities import TITLES, VSME_STEPS, generate_vsme_pdf

@st.cache_resource
def create_system():
    return rag()

system = create_system()

# ###### CREATE SESSION STATE PARAMETERS with default values ######

# Mode selection
if "mode" not in st.session_state:
    st.session_state["mode"] = "Sustainability awareness"

# Left side bar parameters
if "company_profile" not in st.session_state:
    st.session_state["company_profile"] = {"num_employees": None, "activity": ""}
if "history" not in st.session_state:
    st.session_state["history"] = [] 
if "profile_saved" not in st.session_state:
    st.session_state["profile_saved"] = False

# Guided reporting states
if "guided_reporting" not in st.session_state:
    st.session_state["guided_reporting"] = False
if "vsme_step" not in st.session_state:
    st.session_state["vsme_step"] = 0
    
# Customized response --> prompt LLM with the data of the company
if "customized_response" not in st.session_state: 
    st.session_state["customized_response"] = False

st.set_page_config(page_title="Minimum Viable ESG", layout="wide")

# --- Company profile ---
with st.sidebar:
    st.title("Company profile")
    num_employees = st.number_input("Number of employees", min_value=0, value=st.session_state["company_profile"]["num_employees"] or 0)
    activity_desc = st.text_area("Business description", value=st.session_state["company_profile"]["activity"])
            
    save_profile = st.button("Save profile")

    if save_profile:
        if num_employees == 0:
            st.error("The number of employees must be greater than 0.")
            st.session_state["profile_saved"] = False
        elif not activity_desc or not activity_desc.strip():
            st.error("The business description is required.")
            st.session_state["profile_saved"] = False
        else:
            st.session_state["company_profile"] = {"num_employees": int(num_employees), "activity": activity_desc}
            st.session_state["profile_saved"] = True
            st.success("Profile saved successfully ✅")
        
    st.markdown("---")
    # Reset consultation
    if st.button("Reset Consultation"):
        st.session_state["guided_reporting"] = False
        st.session_state["vsme_step"] = 0
        st.session_state["history"] = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("**Suggested questions:**")
    st.write("- `Which ESG practices are most important for a company like ours?`")
    st.write("- `Give me an action plan to prepare a basic environmental report.`")
    st.write("- `What sustainability plan or proposed sustainable practices can I present to the bank to obtain financing?`")


# --- Main Layout ---
st.title("🟢 Sustainability Assistant")

col1, col2 = st.columns([3, 1])

with col1:    
    st.write("**Select Mode:**")
    # Two columns to flank the buttons and an empty one to fill the remaining space.
    btn_col1, btn_col2, _ = st.columns([1, 1, 1])
    
    with btn_col1:
        if st.button("Sustainability awareness", 
                     use_container_width=True, 
                     type="primary" if st.session_state["mode"] == "Sustainability awareness" else "secondary"):
            st.session_state["mode"] = "Sustainability awareness"
            st.rerun()
            
    with btn_col2:
        if st.button("Guided reporting", 
                     use_container_width=True, 
                     type="primary" if st.session_state["mode"] == "Guided reporting" else "secondary"):
            st.session_state["mode"] = "Guided reporting"
            st.rerun()
    
    # --- Single answer, oriented through checkbox (one-shot) ---
    if st.session_state["mode"] == "Sustainability awareness":
        
        st.session_state["customized_response"] = st.toggle("Provide personalized recommendations for my company in the field of sustainability.")
        st.markdown(":small[:grey[See below left for some example questions]]")
        
        st.session_state["guided_reporting"] = False
        
        with st.form("ask_form"):
            question = st.text_area("Your question", height=100)
            submitted = st.form_submit_button("Submit")

        if submitted: # Do not allow response if the user wants personalized response but did not provide any detail
            if not st.session_state["profile_saved"] and st.session_state["customized_response"]:
                st.error("⚠️ Save company profile first.")
            else:
                with st.spinner("Generating..."):
                                  
                    response = system.consult(company_profile=st.session_state["company_profile"], user_question=question,
                                              customized_response=st.session_state["customized_response"])
                    st.session_state["history"].append({
                        "question": question,
                        "response": response,
                        "ts": datetime.now().strftime("%H:%M - %Y/%m/%d")
                    })
                    st.rerun()
    
    # --- Guided reporting through the analysis of the VSME standard ---
    else:
        st.session_state["guided_reporting"] = True

        current_step = st.session_state["vsme_step"]
        
        # Check guided reporting state
        if current_step < len(VSME_STEPS):
            st.subheader(f"{TITLES[current_step]}")
            
            # Show actual question
            question_text = VSME_STEPS[current_step]
            st.markdown(f"{question_text}")
            
            # User question
            with st.form(key=f"guided_reporting_form_{current_step}"):
                user_question = st.text_area("Any question?", height=100)
                ask_button = st.form_submit_button("Ask me")
                if ask_button:
                    if not user_question.strip():
                        st.error("Please provide a question.")
                    else:
                        # Ask LLM for clarifications TODO matchare metrica per metrica con un prompt specifico??? proviamo "generale"
                        response = system.consult(user_question=user_question, vsme_question=question_text)
                        st.session_state["history"].append({
                            "question": user_question,
                            "response": response,
                            "topic": TITLES[current_step],
                            "ts": datetime.now().strftime("%H:%M - %Y/%m/%d")
                        })
            
            # TODO controllo max e min steps (non fargli fare over/under range)
            
            prev_col, next_col, _ = st.columns([1, 1, 3])
            with prev_col:
                back_btn = st.button("Previous", use_container_width=True) 
                if back_btn:
                    st.session_state["vsme_step"] -= 1
                    st.rerun()
            with next_col:
                next_btn = st.button("Next", use_container_width=True)    
                if next_btn:
                    st.session_state["vsme_step"] += 1
                    st.rerun()

        else:
            # --- Guided Reporting end ---
            # Generating a prototype report simply with what the user asked to demonstrate how much the 
            # user is up to date with sustainability and VSME 
            st.success("Guided reporting ended! Thank you!")
            st.markdown(":small[I hope you now have a better understanding of what VSME is and how to report with it!]")
            
            # Questions review
            with st.expander("Review your questions"):
                for title in TITLES:
                    st.markdown(f"#### {title}")
                    topic_questions = [item for item in st.session_state["history"] if item.get("topic") == title]
                    
                    if topic_questions:
                        for item in topic_questions:
                            st.markdown(f"- **Question:** *{item['question']}*")
                            st.markdown(f"**Response:** {item['response']}") # response not used --> question is the information the user lacks of
                    else:
                        st.markdown("- *No question asked.*")
                        
                    st.markdown("---")
                                        
            st.markdown("---")
            st.markdown("### Download your VSME Knowledge Report")
            
            # PDF bytes generation
            pdf_bytes = generate_vsme_pdf(
                company_profile=st.session_state["company_profile"],
                history=st.session_state["history"],
                titles=TITLES
            )
            # Download report
            st.download_button(
                label="Download VSME Knowledge Report (PDF)",
                data=pdf_bytes,
                file_name="VSME_Knowledge_Report.pdf",
                mime="application/pdf"
            )

            # Home - for now do not reset everything (company profile)
            if st.button("Back to home"):
                st.session_state["vsme_step"] = 0
                st.session_state["guided_reporting"] = False
                st.session_state["mode"] = "Sustainability awareness"
                st.session_state["history"] = []
                st.rerun()

    # --- History ---
    st.markdown("### History")
    if st.session_state["history"]:
        for item in reversed(st.session_state["history"]):
            st.markdown(f"**You —** {item['question']}")
            st.markdown(f"> {item['response']}")
            st.caption(item["ts"])
            st.markdown("---")

# --- Company profile ---
with col2:
    st.subheader("Current profile")
    prof = st.session_state["company_profile"]
    st.write(f"- **Employees:** {prof.get('num_employees')}")
    st.write(f"- **Activity:** {prof.get('activity')}")
    
    if st.session_state["guided_reporting"]:
        st.markdown("Guided reporting progress:")
        st.progress(st.session_state["vsme_step"] / len(VSME_STEPS))
        
    st.markdown("---")
    # Download history as text
    if st.session_state["history"]:
        transcript = ""
        for item in st.session_state["history"]:
            transcript += f"[{item['ts']}] Question: {item['question']}\nAnswer: {item['response']}\n\n"
        st.download_button("Download history (.txt)", transcript, file_name="consultation_history.txt", mime="text/plain")
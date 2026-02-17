import streamlit as st
from datetime import datetime
from RAG.RAG4_general import rag

@st.cache_resource
def create_system():
    return rag()

system = create_system()

if "company_profile" not in st.session_state:
    st.session_state["company_profile"] = {"num_employees": None, "activity": ""}
if "history" not in st.session_state:
    st.session_state["history"] = [] 
if "profile_saved" not in st.session_state:
    st.session_state["profile_saved"] = False

# Interview states
if "interview_mode" not in st.session_state:
    st.session_state["interview_mode"] = False
if "interview_step" not in st.session_state:
    st.session_state["interview_step"] = 0
if "interview_answers" not in st.session_state:
    st.session_state["interview_answers"] = [] # Lista di tuple (Domanda, Risposta)

# Static interview questions (COULD BE DYNAMIC) --- improvable
INTERVIEW_QUESTIONS = [
    "What are the main raw materials or resources you use in your daily operations?",
    "Do you currently track your energy or water consumption? If yes, how?",
    "Do you own your facilities/vehicles or do you lease them?",
    "Are you currently asked for ESG data by banks or clients? If so, which data?"
]

PROFILES = [
    "Sustainability Reporting Advisor",
    "ESG Integration Specialist",
    "Green Finance Consultant"
]

# Put scope into prompt --- totally aligned with RAG script
def profileToScope(profile):
    if profile==PROFILES[0]:
        return "REPORTING_COMPLIANCE"
    if profile==PROFILES[1]:
        return "GENERAL_ADVICE"
    if profile==PROFILES[2]:
        return "FINANCE_ALIGNMENT"
    return

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
    # Reset interview
    if st.button("Reset Consultation"):
        st.session_state["interview_mode"] = False
        st.session_state["interview_step"] = 0
        st.session_state["interview_answers"] = []
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
    # Mode selection
    mode = st.radio("Select Mode:", ["Quick Question", "Guided Consultation"], horizontal=True)
    
    # --- Single answer, oriented through checkbox (one-shot) ---
    if mode == "Quick Question":
        st.session_state["interview_mode"] = False
        
        profile = st.radio("Select Profile:", PROFILES, horizontal=False)
        
        with st.form("ask_form"):
            question = st.text_area("Your question", height=100)
            submitted = st.form_submit_button("Submit")

        if submitted:
            if not st.session_state["profile_saved"]:
                st.error("⚠️ Save company profile first.")
            else:
                with st.spinner("Generating..."):
                                  
                    response = system.consult(company_profile=st.session_state["company_profile"], question=question, scope=profileToScope(profile))
                    st.session_state["history"].append({
                        "question": question,
                        "response": response,
                        "ts": datetime.now().strftime("%H:%M - %Y/%m/%d")
                    })
                    st.rerun()
    
    # --- Consultation with user (guided interview to enrich the prompt at the best) ---
    else:
        st.session_state["interview_mode"] = True
        
        if not st.session_state["profile_saved"]:
            st.warning("Please fill and save the Company Profile in the sidebar to start the interview.")
        else:
            current_step = st.session_state["interview_step"]
            
            # Check interview state
            if current_step < len(INTERVIEW_QUESTIONS):
                st.subheader(f"Step {current_step + 1} of {len(INTERVIEW_QUESTIONS)}")
                
                # Show actual question
                question_text = INTERVIEW_QUESTIONS[current_step]
                st.markdown(f"**{question_text}**")
                
                # Interview answer
                with st.form(key=f"interview_form_{current_step}"):
                    user_answer = st.text_area("Your answer:", height=100)
                    next_btn = st.form_submit_button("Next")
                    
                    if next_btn:
                        if not user_answer.strip():
                            st.error("Please provide an answer.")
                        else:
                            # Save response and porceed with next step
                            st.session_state["interview_answers"].append((question_text, user_answer))
                            st.session_state["interview_step"] += 1
                            st.rerun()
            else:
                # --- INTERVIEW COMPLETED: Generating action plan ---
                st.success("Interview completed! This will help me better understand your context.") 
                
                # Answers review
                with st.expander("Review your answers"):
                    for q, a in st.session_state["interview_answers"]:
                        st.write(f"**Q:** {q}")
                        st.write(f"**A:** {a}")
                        st.write("---")
                
                profile = st.radio("Select Profile:", PROFILES, horizontal=False)
                
                with st.form("ask_form"):
                    question = st.text_area("Your question", height=100)
                    submitted = st.form_submit_button("Submit")
                
                if submitted:
                    with st.spinner("Analyzing your profile and answers..."):
                        profile = st.session_state["company_profile"]
                        interview_data = st.session_state["interview_answers"]
                                                
                        response = system.consult(
                            company_profile=profile, 
                            question=question, 
                            scope=profileToScope(profile),
                            interview_history=interview_data
                        )
                        
                        st.session_state["history"].append({
                            "question": "Guided Consultation Report",
                            "response": response,
                            "ts": datetime.now().strftime("%H:%M - %Y/%m/%d")
                        })
                        
                        st.session_state["interview_step"] = 0
                        st.session_state["interview_answers"] = []
                        st.session_state["interview_mode"] = False
                        st.rerun()

    # --- History ---
    st.markdown("### History")
    if st.session_state["history"]:
        for item in reversed(st.session_state["history"]):
            st.markdown(f"**You —** {item['question']}")
            st.markdown(f"> {item['response']}")
            st.caption(item["ts"])
            st.markdown("---")

# --- Profile ---
with col2:
    st.subheader("Current profile")
    prof = st.session_state["company_profile"]
    st.write(f"- **Employees:** {prof.get('num_employees')}")
    st.write(f"- **Activity:** {prof.get('activity')}")
    
    if st.session_state["interview_answers"]:
        st.markdown("**Interview Progress:**")
        st.progress(len(st.session_state["interview_answers"]) / len(INTERVIEW_QUESTIONS))
        
    st.markdown("---")
    # Download history as text
    if st.session_state["history"]:
        transcript = ""
        for item in st.session_state["history"]:
            transcript += f"[{item['ts']}] Question: {item['question']}\nAnswer: {item['response']}\n\n"
        st.download_button("Download history (.txt)", transcript, file_name="consultation_history.txt", mime="text/plain")
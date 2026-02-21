import streamlit as st
from datetime import datetime
from RAG.RAG import rag

@st.cache_resource
def create_system():
    return rag()

system = create_system()

# ###### SESSION STATE PARAMETERS ######

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

# Interview states
if "interview_mode" not in st.session_state:
    st.session_state["interview_mode"] = False
if "interview_step" not in st.session_state:
    st.session_state["interview_step"] = 0
if "interview_answers" not in st.session_state:
    st.session_state["interview_answers"] = [] # Q -> A list
    
# Customized response --> prompt LLM with the data of the company
if "customized_response" not in st.session_state: 
    st.session_state["customized_response"] = False

# Static interview questions (COULD BE DYNAMIC) --- improvable
INTERVIEW_QUESTIONS = [
    "What are the main raw materials or resources you use in your daily operations?",
    "Do you currently track your energy or water consumption? If yes, how?",
    "Do you own your facilities/vehicles or do you lease them?",
    "Are you currently asked for ESG data by banks or clients? If so, which data?"
]

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
    
    if st.session_state["mode"] == "Sustainability awareness":
        st.session_state["customized_response"] = st.checkbox("Provide personalized recommendations for my company in the field of sustainability.")
        st.markdown(" :small[:grey[See below left for some example questions]]")

    # --- Single answer, oriented through checkbox (one-shot) ---
    if st.session_state["mode"] == "Sustainability awareness":
        st.session_state["interview_mode"] = False
        
        with st.form("ask_form"):
            question = st.text_area("Your question", height=100)
            submitted = st.form_submit_button("Submit")

        if submitted: # Do not allow response if the user wants personalized response but did not provide any detail
            if not st.session_state["profile_saved"] and st.session_state["customized_response"]:
                st.error("⚠️ Save company profile first.")
            else:
                with st.spinner("Generating..."):
                                  
                    response = system.consult(company_profile=st.session_state["company_profile"], question=question,
                                              customized_response=st.session_state["customized_response"])
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
                                
                with st.form("ask_form"):
                    question = st.text_area("Your question", height=100)
                    submitted = st.form_submit_button("Submit")
                
                if submitted:
                    with st.spinner("Analyzing your profile and answers..."):
                        company_profile = st.session_state["company_profile"]
                        interview_data = st.session_state["interview_answers"]
                                                
                        response = system.consult(
                            company_profile=company_profile, 
                            question=question,
                            interview_history=interview_data
                        )
                        
                        st.session_state["history"].append({
                            "question": "Guided reporting result",
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

# --- Company profile ---
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
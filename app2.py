import streamlit as st
from datetime import datetime
from RAG.RAG3 import rag

# --- Setup e Cache (Invariato) ---
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

# --- NUOVI STATI PER L'INTERVISTA ---
if "interview_mode" not in st.session_state:
    st.session_state["interview_mode"] = False
if "interview_step" not in st.session_state:
    st.session_state["interview_step"] = 0
if "interview_answers" not in st.session_state:
    st.session_state["interview_answers"] = [] # Lista di tuple (Domanda, Risposta)

# Domande statiche per l'intervista (possono essere rese dinamiche in futuro)
INTERVIEW_QUESTIONS = [
    "What are the main raw materials or resources you use in your daily operations?",
    "Do you currently track your energy or water consumption? If yes, how?",
    "Do you own your facilities/vehicles or do you lease them?",
    "Are you currently asked for ESG data by banks or clients? If so, which data?"
]

st.set_page_config(page_title="Minimum Viable ESG", layout="wide")

# --- Sidebar (Invariato) ---
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
    mode = st.radio("Select Mode:", ["Quick Question", "Guided Consultation)"], horizontal=True)

    if mode == "Quick Question":
        # --- Single answer, oriented through checkbox (one-shot) ---
        st.session_state["interview_mode"] = False
        
        VSME_oriented = st.checkbox('Advise me on how to create sustainability reports')
        ESG_oriented = st.checkbox('Tell me how I can be more sustainable')
        SFDR_oriented = st.checkbox('How to align my sustainability with finance')
        
        with st.form("ask_form"):
            question = st.text_area("Your question", height=100)
            submitted = st.form_submit_button("Submit")

        if submitted:
            if not st.session_state["profile_saved"]:
                st.error("⚠️ Save company profile first.")
            else:
                with st.spinner("Generating..."):
                    selected_options = []
                    if VSME_oriented: selected_options.append("VSME oriented")
                    if ESG_oriented: selected_options.append("ESG oriented")
                    if SFDR_oriented: selected_options.append("SFDR oriented")
                    
                    response = system.consult(st.session_state["company_profile"], question, selected_options)
                    # TODO mettere bottone salva history come in app vecchio
                    st.session_state["history"].append({
                        "question": question,
                        "response": response,
                        "ts": datetime.now().strftime("%H:%M - %Y/%m/%d")
                    })
                    st.rerun()

    else:
        # --- Consultation with user (guided interview to enrich the prompt at the best) ---
        st.session_state["interview_mode"] = True
        
        if not st.session_state["profile_saved"]:
            st.warning("Please fill and save the Company Profile in the sidebar to start the interview.")
        else:
            current_step = st.session_state["interview_step"]
            
            # Controllo se l'intervista è in corso o finita
            if current_step < len(INTERVIEW_QUESTIONS):
                st.subheader(f"Step {current_step + 1} of {len(INTERVIEW_QUESTIONS)}")
                
                # Mostra la domanda corrente
                question_text = INTERVIEW_QUESTIONS[current_step]
                st.markdown(f"**{question_text}**")
                
                # Form per la risposta corrente
                with st.form(key=f"interview_form_{current_step}"):
                    user_answer = st.text_area("Your answer:", height=100)
                    next_btn = st.form_submit_button("Next")
                    
                    if next_btn:
                        if not user_answer.strip():
                            st.error("Please provide an answer.")
                        else:
                            # Salva la risposta
                            st.session_state["interview_answers"].append((question_text, user_answer))
                            # Avanza step
                            st.session_state["interview_step"] += 1
                            st.rerun()
            else:
                # --- INTERVISTA COMPLETATA: GENERAZIONE REPORT ---
                st.success("Interview completed! Generating your tailored Action Plan...")
                
                # Mostra riassunto
                with st.expander("Review your answers"):
                    for q, a in st.session_state["interview_answers"]:
                        st.write(f"**Q:** {q}")
                        st.write(f"**A:** {a}")
                        st.write("---")
                
                if st.button("Generate response"):
                    with st.spinner("Analyzing your profile and answers..."):
                        profile = st.session_state["company_profile"]
                        interview_data = st.session_state["interview_answers"]
                        
                        # Definiamo una domanda "implicita" per il sistema basata sull'intervista
                        final_question = "Create a comprehensive ESG action plan and VSME report structure based on the interview details provided."
                        
                        # Chiamata al sistema con interview_history
                        response = system.consult(
                            profile, 
                            final_question, 
                            scope=["VSME oriented", "ESG oriented"], # Default scopes for full report
                            interview_history=interview_data
                        )
                        
                        st.session_state["history"].append({
                            "question": "Guided Consultation Report",
                            "response": response,
                            "ts": datetime.now().strftime("%H:%M")
                        })
                        
                        # Reset intervista opzionale
                        st.session_state["interview_step"] = 0
                        st.session_state["interview_answers"] = []
                        st.session_state["interview_mode"] = False
                        st.rerun()

    # --- Sezione History (Comune a entrambi) ---
    st.markdown("### History")
    if st.session_state["history"]:
        for item in reversed(st.session_state["history"]):
            st.markdown(f"**You —** {item['question']}")
            st.markdown(f"> {item['response']}")
            st.caption(item["ts"])
            st.markdown("---")

# --- Colonna 2 (Profilo) ---
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
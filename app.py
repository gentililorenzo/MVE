import streamlit as st
from datetime import datetime

# Crea l'istanza una sola volta e la riutilizza.
# Usa @st.cache_resource per evitare di ricrearlo ad ogni rerun.
@st.cache_resource
def create_system():
    try:
        from src.prompt_hybrid import HybridPromptSystem
    except Exception as e:
        # Se import fallisce, mostriamo l'errore in UI invece di far crashare tutto.
        st.error(f"Errore import src.prompt_hybrid: {e}")
        return None
    return HybridPromptSystem()

system = create_system()

# Inizializzazione session state
if "company_profile" not in st.session_state:
    st.session_state["company_profile"] = {"dimensione": None, "attivita": ""}
if "history" not in st.session_state:
    st.session_state["history"] = []  # lista di dict: {"question":..., "response":..., "ts":...}
if "profile_saved" not in st.session_state:
    st.session_state["profile_saved"] = False

st.set_page_config(page_title="Consulente Sostenibilità VSME", layout="wide")

# --- Sidebar: profilo azienda ---
with st.sidebar:
    st.title("Profilo azienda")
    dim = st.number_input("Numero dipendenti", min_value=0, step=1, value=st.session_state["company_profile"]["dimensione"] or 0)
    att = st.text_area("Descrizione attività", value=st.session_state["company_profile"]["attivita"], height="content")
    save_profile = st.button("Salva profilo")

    if save_profile:
        if dim == 0:
            st.error("Il numero di dipendenti deve essere maggiore di 0.")  # Si poteva magari mettere anche default a 1, cosi' "forziamo"
                                                                            # l'utente a inserire il numero preciso.
            st.session_state["profile_saved"] = False
        elif not att or not att.strip():
            st.error("La descrizione dell'attività è obbligatoria.")
            st.session_state["profile_saved"] = False
        else:
            st.session_state["company_profile"] = {"dimensione": int(dim), "attivita": att}
            st.session_state["profile_saved"] = True
            st.success("Profilo salvato correttamente ✅")

    st.markdown("---")
    if st.button("Cancella cronologia"):
        st.session_state["history"] = []
        st.info("Cronologia cancellata")

    st.markdown("---")
    st.markdown("**Suggerimenti domande:**")
    st.write("- `Quali pratiche ESG sono più importanti per una PMI di 10 dipendenti nel settore X?`")
    st.write("- `Dammi un piano d'azione per stilare un reporting ambientale basilare.`")

# --- Main layout ---
st.title("🟢 Sustainability assistant") # VSME") # il nostro MVE
# st.caption("Interfaccia web con Streamlit — basata su `HybridPromptSystem`")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Fai una domanda")
    # usiamo un form per evitare submit multipli accidentali
    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area("La tua domanda", height=120)
        submitted = st.form_submit_button("Invia domanda")  # se profilo non inserito/salvato blocco la richiesta domanda.
                                                            # Sarebbe uno spreco, troppo poco contesto. 

    if submitted:
        if not st.session_state["profile_saved"]:
            st.error("⚠️ Devi compilare e salvare il profilo aziendale (Sidebar) prima di procedere.")
        elif not question or not question.strip():
            st.warning("⚠️ Scrivi una domanda prima di inviare.")
        elif system is None:
            st.error("Il sistema non è inizializzato correttamente. Controlla i log.")
        else:
            profile = st.session_state["company_profile"]   # mostra spinner mentre elabora
            with st.spinner("Generazione risposta..."):
                try:
                    response = system.consult(profile, question)
                except Exception as e:
                    st.error(f"Errore durante la consultazione: {e}")
                    response = f"Errore interno: {e}"

            # salva nella cronologia
            st.session_state["history"].append({
                "question": question,
                "response": response,
                "ts": datetime.utcnow().isoformat()
            })
            st.success("Risposta generata ✅")

    # Mostra cronologia (nuova prima)
    st.markdown("### Cronologia")
    if st.session_state["history"]:
        for item in reversed(st.session_state["history"]):
            st.markdown(f"**Tu —** {item['question']}")
            st.markdown(f"> {item['response']}")
            st.caption(item["ts"])
            st.markdown("---")
    else:
        st.info("Nessuna domanda ancora inviata.")

with col2:
    st.subheader("Profilo attuale")
    prof = st.session_state["company_profile"]
    st.write(f"- **Numero dipendenti:** {prof.get('dimensione')}")
    st.write(f"- **Attività:** {prof.get('attivita') or '*non impostata*'}")
    st.markdown("---")
    # Download della cronologia come testo
    if st.session_state["history"]:
        transcript = ""
        for item in st.session_state["history"]:
            transcript += f"[{item['ts']}] Domanda: {item['question']}\nRisposta: {item['response']}\n\n"
        st.download_button("Scarica cronologia (.txt)", transcript, file_name="cronologia_consulenza.txt", mime="text/plain")

# --- footer / note ---
st.markdown("---")
st.markdown("Se l'app si aggiornasse automaticamente quando modifichi file nel progetto, usa `R` nella UI di Streamlit o l'opzione 'Rerun'.")


# OPTIONAL: CSS leggero per migliorare look (non obbligatorio)
st.markdown(
    """
    <style>
    .stDownloadButton > button { border-radius: 8px; padding: 8px 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

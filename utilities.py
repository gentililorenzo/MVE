# Guided reporting titles - what to expect
TITLES = [
    "Introducing the VSME standard by EFRAG",
    "VSME Metrics - B3 - Energy and greenhouse gas emissions",
    "VSME Metrics - B4 - Pollution of air, water and soil"
]

# Static interview questions (COULD BE DYNAMIC) --- improvable --> ...deliverable of the European Commission's SME Relief Package (September 2023) tasking EFRAG to develop a...
# TODO links ai paper ufficiali?
VSME_STEPS = [
    "The VSME is a simple and standardised framework for SMEs to report on ESG issues, creating better opportunities to obtain green financing and thus facilitating the transition to a sustainable economy.\n\n **I will help you better understand what is needed to create voluntary reporting based on VSME. You can ask me for more information on any question at any time. At the end, you will receive a descriptive report based on what you asked or not.**",
    """29. The undertaking shall disclose its total energy consumption in MWh, with a breakdown as per the table below, if it can obtain the necessary information to provide such a breakdown:\n
    | | Renewable | Non-renewable | Total |
    |---|---|---|---|
    | **Electricity** (as reflected in utility billings) | | | |
    | **Fuels** | | | |
    | **Total** | | | |

30. The undertaking shall disclose its estimated _**gross greenhouse gas (GHG) emissions**_ in tons of CO\u2082 equivalent (tCO\u2082eq) considering the content of the GHG Protocol Corporate Standard (version 2004), including:\n
    (a) the Scope 1 GHG emissions in tCO\u2082eq (from owned or controlled sources); and\n
    (b) the _**location-based Scope 2 emissions**_ in tCO\u2082eq (i.e. emissions from the generation of purchased energy, such as electricity, heat, steam or cooling).

31. The undertaking shall disclose its GHG intensity calculated by dividing '_**gross greenhouse gas (GHG) emissions**_' disclosed under paragraph 30 by 'turnover (in Euro)' disclosed under paragraph
24(e)(iv)^5
""", #TODO markdown non supporta esponenti quindi ^5 rimane così
    "If the undertaking is already required by law or other national regulations to report to competent authorities its emissions of pollutants, or if it voluntarily reports on them according to an Environmental Management System, it shall disclose the pollutants it emits to air, water and soil in its own operations, with the respective amount for each pollutant. If this information is already publicly available, the undertaking may alternatively refer to the document where it is reported, for example, by providing the relevant URL link or embedding a hyperlink."
]

# TODO implementare Comprehensive module 

from fpdf import FPDF

def generate_vsme_pdf(company_profile, history, titles):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    
    pdf.add_font("AlbertSans", "", "fonts/AlbertSans-Regular.ttf", uni=True)
    pdf.add_font("AlbertSans", "B", "fonts/AlbertSans-Bold.ttf", uni=True)
    pdf.add_font("AlbertSans", "I", "fonts/AlbertSans-Italic.ttf", uni=True)
    font_name = "AlbertSans"
    
    # epw (Effective Page Width) is the printable width of the sheet 
    # (e.g., 210 mm total - 40 mm margins = 170 mm).
    epw = pdf.epw 

    # --- Title ---
    pdf.set_font(font_name, "B", 16)
    pdf.set_x(20)
    pdf.multi_cell(epw, 10, "VSME Knowledge Report", align="C")
    pdf.ln(6)

    # --- Company Profile ---
    pdf.set_font(font_name, "B", 14)
    pdf.set_x(20) # Always return the cursor to the left margin.
    pdf.multi_cell(epw, 8, "Company Profile", align="L")
    
    pdf.set_font(font_name, "", 12)
    emp = company_profile.get("num_employees", "N/A")
    act = company_profile.get("activity", "N/A")

    pdf.set_x(20)
    pdf.multi_cell(epw, 7, f"Number of employees: {emp}", align="L")
    
    pdf.set_x(20)
    pdf.multi_cell(epw, 7, f"Activity: {act}", align="L")
    
    pdf.ln(6)

    # --- Q&A ---
    pdf.set_font(font_name, "B", 14)
    pdf.set_x(20)
    pdf.multi_cell(epw, 8, "Q&A about VSME (knowledge inspection)", align="L")
    pdf.ln(4)

    for title in titles:
        # Section TITLE
        pdf.set_font(font_name, "B", 12)
        pdf.set_x(20)
        pdf.multi_cell(epw, 7, title, align="L")

        topic_questions = [it for it in history if it.get("topic") == title]
        
        if topic_questions:
            for it in topic_questions:
                domanda = "- Question: " + it.get("question", "")
                risposta = "  Answer: " + it.get("response", "")

                pdf.set_font(font_name, "", 11)
                pdf.set_x(20)
                pdf.multi_cell(epw, 6, domanda, align="L")
                
                pdf.set_font(font_name, "I", 10)
                pdf.set_x(20)
                pdf.multi_cell(epw, 6, risposta, align="L")
                
                pdf.ln(2)
        else:
            pdf.set_font(font_name, "I", 11)
            pdf.set_x(20)
            pdf.multi_cell(epw, 6, "- No question asked.", align="L")
        
        pdf.ln(4)

    # download is done from app.py
    return bytes(pdf.output())
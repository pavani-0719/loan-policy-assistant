
import streamlit as st
import ollama
from pathlib import Path
from rag import search_documents

OLLAMA_API_KEY = st.secrets["OLLAMA_API_KEY"]

client = ollama.Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)
st.write("DEBUG: Ollama client initialized successfully")
st.set_page_config(
    page_title="Loan Policy Assistant",
    page_icon="🏦",
    layout="centered"
)


# -----------------------------
# Language Selection
# -----------------------------

languages = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn"
}

selected_language = st.selectbox(
    "🌐 Select your language",
    list(languages.keys())
)

target_language = languages[selected_language]
example_questions = {
    "English": [
        "What is the minimum age required for a personal loan?",
        "What documents are required for a loan?"
    ],
    "Telugu": [
        "వ్యక్తిగత రుణం తీసుకోవడానికి కనీస వయస్సు ఎంత?",
        "లోన్ కోసం ఏ పత్రాలు అవసరం?"
    ],
    "Hindi": [
        "पर्सनल लोन लेने के लिए न्यूनतम उम्र कितनी होनी चाहिए?",
        "लोन के लिए कौन से दस्तावेज़ आवश्यक हैं?"
    ],
    "Tamil": [
        "தனிநபர் கடன் பெற குறைந்தபட்ச வயது என்ன?",
        "கடனுக்கு என்ன ஆவணங்கள் தேவை?"
    ],
    "Kannada": [
        "ವೈಯಕ್ತಿಕ ಸಾಲ ಪಡೆಯಲು ಕನಿಷ್ಠ ವಯಸ್ಸು ಎಷ್ಟು?",
        "ಸಾಲಕ್ಕೆ ಯಾವ ದಾಖಲೆಗಳು ಅಗತ್ಯವಿದೆ?"
    ],
    "Malayalam": [
        "പേഴ്സണൽ ലോൺ എടുക്കാൻ കുറഞ്ഞ പ്രായം എത്രയാണ്?",
        "ലോണിന് ആവശ്യമായ രേഖകൾ എന്തൊക്കെയാണ്?"
    ],
    "Marathi": [
        "वैयक्तिक कर्ज घेण्यासाठी किमान वय किती असावे?",
        "कर्जासाठी कोणती कागदपत्रे आवश्यक आहेत?"
    ],
    "Bengali": [
        "পার্সোনাল লোন নেওয়ার জন্য ন্যূনতম বয়স কত?",
        "লোনের জন্য কী কী নথি প্রয়োজন?"
    ]
}


# -----------------------------
# Title
# -----------------------------

st.title("🏦 Loan Policy Assistant")

st.write(
    "Ask questions about loan eligibility, documents, "
    "interest rates, EMI, KYC, fees and repayment policies."
)


# -----------------------------
# Chat History
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User Input
# -----------------------------

question = st.chat_input(
    "Ask your loan-related question..."
)


if question:

    # Keep the original question
    original_question = question


    # -----------------------------
    # Translate question to English
    # -----------------------------

    if target_language != "en":

        translation_response = client.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    }
Translate the following loan-related question into English.

Return ONLY the translated question.
Do not explain anything.

Question:
{question}
"""
       
response = client.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Read the response safely
if isinstance(response, dict):
    answer = response.get("message", {}).get("content", "")
else:
    answer = response.message.content

if not answer:
    st.error("The AI service returned an empty answer. Please check the app logs.")
    st.stop()



    # -----------------------------
    # Display user's original question
    # -----------------------------

    with st.chat_message("user"):
        st.markdown(original_question)


    st.session_state.messages.append(
        {
            "role": "user",
            "content": original_question
        }
    )


    # -----------------------------
    # Search loan documents
    # -----------------------------

    results = search_documents(
        question,
        top_k=5
    )


    context = ""
    source_files = []


    for result in results:

        filename = result["filename"]
        text = result["text"]

        context += (
            "\n\nSOURCE: "
            + filename
            + "\n"
            + text
        )

        source_files.append(filename)


    # -----------------------------
    # Ask Llama
    # -----------------------------

    prompt = f"""
You are a Loan Policy Assistant.

Answer the user's question using ONLY
the information contained in the retrieved
loan policy documents.

IMPORTANT RULES:

1. Do not invent information.
2. Do not create loan policies.
3. Do not make up interest rates.
4. Do not make up eligibility criteria.
5. Do not make up fees or charges.
6. If the answer is not available in the
   documents, say:

"I could not find this information
in the available loan policy documents."

7. Give clear and simple answers.
8. Mention important conditions when
   they are available.
9. Use bullet points when useful.

Retrieved Documents:

{context}

User Question:

{question}
"""


    response = client.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

st.write("DEBUG: reached response section")
st.write("DEBUG: client exists:", "client" in globals())
    # -----------------------------
    # Get Llama's answer
    # -----------------------------

answer = response["message"]["content"]


    # -----------------------------
    # Translate answer back
    # -----------------------------

if target_language != "en":

        translation_response = client.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Translate the following loan-policy answer into {selected_language}.

Return ONLY the translated answer.
Do not add explanations.

Answer:

{answer}
"""
                }
            ]
        )

        answer = translation_response["message"]["content"]


    # -----------------------------
    # Display answer
    # -----------------------------

with st.chat_message("assistant"):

        st.markdown(answer)

        st.markdown(
            "**Sources:** "
            + ", ".join(set(source_files))
        )


st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

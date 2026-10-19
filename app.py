
import streamlit as st
import ollama
from rag import search_documents

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Policy Assistant",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------
# Ollama Client
# -----------------------------
try:
    OLLAMA_API_KEY = st.secrets["OLLAMA_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "Ollama API key is missing. "
        "Add OLLAMA_API_KEY in Streamlit app Secrets."
    )
    st.stop()

client = ollama.Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
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

# -----------------------------
# Title and Description
# -----------------------------
st.title("🏦 Loan Policy Assistant")
st.write(
    "Ask questions about loan eligibility, documents, "
    "interest rates, EMI, KYC, fees and repayment policies."
)

# -----------------------------
# Example Questions
# -----------------------------
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

with st.expander("💡 Example questions"):
    for example in example_questions[selected_language]:
        st.markdown(f"- {example}")

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Ollama Chat Helper
# -----------------------------
def ask_ollama(message):
    result = client.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": message
            }
        ]
    )

    # The Ollama Python client returns a response object.
    return result.message.content.strip()

# -----------------------------
# User Input
# -----------------------------
question = st.chat_input("Ask your loan-related question...")

if question:
    original_question = question

    # Display and save the original question
    st.session_state.messages.append({
        "role": "user",
        "content": original_question
    })

    with st.chat_message("user"):
        st.markdown(original_question)

    try:
        with st.spinner("Finding relevant loan policy information..."):

            # -----------------------------
            # Translate Question to English
            # -----------------------------
            if target_language != "en":
                question = ask_ollama(
                    f"""
Translate this loan-related question into English.
Return only the translated question.

Question:
{original_question}
"""
                )

            # -----------------------------
            # Search Loan Documents
            # -----------------------------
            results = search_documents(question, top_k=5)

            context_parts = []
            source_files = []

            for result in results:
                filename = result.get("filename", "Unknown source")
                text = result.get("text", "")

                if text.strip():
                    context_parts.append(
                        f"SOURCE: {filename}\n{text}"
                    )
                    source_files.append(filename)

            context = "\n\n".join(context_parts)

            # -----------------------------
            # Build Prompt
            # -----------------------------
            if context.strip():
                prompt = f"""
You are a Loan Policy Assistant.

Answer the user's question using only the
information in the retrieved loan policy documents.

Rules:
1. Do not invent information.
2. Do not make up interest rates, fees, or eligibility criteria.
3. If the answer is not present in the documents,
   say that you could not find the information.
4. Explain the answer in clear, simple language.
5. Mention important conditions when available.

Retrieved loan policy documents:
{context}

User question:
{question}
"""
            else:
                prompt = f"""
You are a Loan Policy Assistant.

No relevant information was found in the available
loan policy documents.

Tell the user:
"I could not find this information in the available
loan policy documents."

Do not invent an answer.

User question:
{question}
"""

            # -----------------------------
            # Generate Answer
            # -----------------------------
            answer = ask_ollama(prompt)

            if not answer:
                answer = (
                    "The AI service returned an empty answer. "
                    "Please try again."
                )

            # -----------------------------
            # Translate Answer
            # -----------------------------
            if target_language != "en":
                answer = ask_ollama(
                    f"""
Translate the following answer into {selected_language}.
Preserve the meaning and all policy conditions.
Return only the translated answer.

Answer:
{answer}
"""
                )

        # -----------------------------
        # Display Answer
        # -----------------------------
        with st.chat_message("assistant"):
            st.markdown(answer)

            if source_files:
                unique_sources = sorted(set(source_files))
                st.markdown(
                    "**Sources:** " + ", ".join(unique_sources)
                )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as error:
        # Keep the app usable and avoid displaying secrets.
        st.error(
            "Sorry, something went wrong while processing "
            "your question. Please check the app logs."
        )
        print(
            f"Chat processing error: "
            f"{type(error).__name__}: {error}"
        )

import streamlit as st
from streamlit_mic_recorder import speech_to_text
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import difflib
import re

# --- Custom CSS for Hackathon Styling ---
st.markdown("""
<style>
    /* Global Background and Font */
    .stApp {
        background-color: #f7f9fc; /* Light, clean background */
        color: #1a1a1a;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Header and Title Styling */
    h1 {
        color: #007bff; /* Primary Blue for Title */
        font-weight: 700;
        border-bottom: 3px solid #007bff;
        padding-bottom: 10px;
    }
    h3 {
        color: #198754; /* Secondary Green for sections */
        border-left: 5px solid #198754;
        padding-left: 10px;
        margin-top: 20px;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    .css-1lcbmhc, .css-1d3vo9v { /* Target sidebar elements */
        background-color: #e9f5ff; /* Light blue background for sidebar */
    }
    .st-emotion-cache-1ftc0w4 { /* Sidebar header/title */
        color: #007bff !important;
    }

    /* Container/Card Styles for Critical Areas and Tips */
    .stAlert, .stInfo, .stSuccess, .stWarning {
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-size: 16px;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-left: 5px solid #155724;
    }
    .stInfo {
        background-color: #cfe2ff;
        color: #052c65;
        border-left: 5px solid #052c65;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border-left: 5px solid #856404;
    }
    
    /* Quick Reply Buttons (WhatsApp Style) */
    .stButton > button {
        background-color: #25d366; /* WhatsApp Green */
        color: white;
        border-radius: 20px;
        border: none;
        padding: 8px 15px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover {
        background-color: #128c7e;
        color: white;
    }
    
    /* Send Button */
    .st-emotion-cache-1c9v60b button { /* Target the send button in the form */
        background-color: #007bff !important;
        color: white !important;
        border-radius: 50% !important; /* Circular button */
        width: 40px !important;
        height: 40px !important;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 18px;
        padding: 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .st-emotion-cache-1c9v60b button:hover {
        background-color: #0056b3 !important;
    }

    /* Chat Messages - Ensure the embedded HTML styles remain, but add overall consistency */
    .whatsapp-style-chat div[style*="background: #DCF8C6"] {
        /* User message styling, adjusted for clarity */
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .whatsapp-style-chat div[style*="background: #FFFFFF"] {
        /* Bot message styling, adjusted for clarity */
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Metrics Styling */
    [data-testid="stMetric"] {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #555;
    }
    
    /* Top 3 Critical Areas Boxes */
    .stMarkdown div[style*="boxcolor"] {
        border: 1px solid #ccc;
        transition: transform 0.2s ease;
    }
    .stMarkdown div[style*="boxcolor"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }

</style>
""", unsafe_allow_html=True)
# --- END Custom CSS ---


# Initialize session state FIRST
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "language" not in st.session_state:
    st.session_state.language = "hi"

load_dotenv()
# Note: Using st.secrets is correct for Streamlit Cloud, but for local testing,
# ensure your .env file is loaded correctly or use os.environ.get if needed.
# groq_api_key = st.secrets["GROQ_API_KEY"] 
# For demonstration purposes, I'll comment out the line above if it causes an error
# and assume you have st.secrets set up or will mock the LLM response.

# Mocking the LLM for self-contained example, please uncomment your actual LLM setup:
# llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")
class MockLLM:
    def invoke(self, messages):
        # Simple mock logic based on the prompt structure
        query = messages[0].content
        if "बीकानेर" in query: return HumanMessage(content="बीकानेर में जल स्तर मध्यम है। किसानों को ड्रिप सिंचाई का प्रयोग करने की सलाह दी जाती है।")
        if "Bikaner" in query: return HumanMessage(content="The water level in Bikaner is moderate. Farmers are advised to use drip irrigation.")
        if "irrigation" in query: return HumanMessage(content="The general irrigation status is challenging, with many areas facing critical groundwater levels. Focus on water-efficient methods.")
        return HumanMessage(content="यह एक महत्वपूर्ण प्रश्न है। अधिक विस्तृत सलाह के लिए, कृपया अपने जिले का नाम बताएं।")
llm = MockLLM()


st.set_page_config(page_title="FlowBot - By Team ZenFlow", layout="wide")

# --- DATA and CONFIG (Keep as is) ---
# Assuming 'cleaned_groundwater_data.csv' exists with required columns
try:
    df = pd.read_csv("cleaned_groundwater_data.csv")
except FileNotFoundError:
    # Create a dummy DataFrame for presentation if the file is missing
    df = pd.DataFrame({
        'District': ['Bikaner', 'Jaipur', 'Jodhpur', 'Pali', 'Bharatpur'],
        'State': ['Rajasthan', 'Rajasthan', 'Rajasthan', 'Rajasthan', 'Rajasthan'],
        'ExtractionVolume_ha_m': [150.0, 50.0, 95.0, 110.0, 30.0],
        'GroundWaterAvailability_ham': [180.0, 80.0, 100.0, 150.0, 100.0],
        'ExtractionStage_Percent': [83.3, 62.5, 95.0, 73.3, 30.0],
        'Rainfall_mm': [300, 550, 250, 400, 700]
    })
    
districts = sorted([str(d).strip() for d in df['District'].unique() if pd.notna(d)])

# Premium Features definition
premium_feature_dict = {
    "AI Crop Water Calculator": "Get AI-driven recommendations on optimal crop choices based on groundwater data.",
    "Recommendation Model for Water Conservation": "Receive tailored advice on water-saving techniques for your area.",
    "Predictive Analytics": "Forecast future groundwater trends with advanced analytics.",
}

# Language Configuration
LANGUAGES = {
    "hi": {"name": "हिंदी", "flag": "🇮🇳"},
    "en": {"name": "English", "flag": "🇬🇧"},
    "te": {"name": "తెలుగు", "flag": "🇮🇳"},
    "ta": {"name": "தமிழ்", "flag": "🇮🇳"},
    "bn": {"name": "বাংলা", "flag": "🇧🇩"},
    "mr": {"name": "मराठी", "flag": "🇮🇳"},
    "gu": {"name": "ગુજરાતી", "flag": "🇮🇳"}
}

# Multilingual titles and descriptions
TITLES = {
    "hi": "🌾 FlowBot - भूजल सहायक",
    "en": "🌾 FlowBot – Groundwater Assistant", 
    "te": "🌾 FlowBot - భూగర్భ జల సహాయకుడు",
    "ta": "🌾 FlowBot - நிலத்தடி நீர் உதவியாளர்",
    "bn": "🌾 FlowBot- ভূগর্ভস্থ পানি সহায়ক",
    "mr": "🌾 FlowBot - भूजल सहायक",
    "gu": "🌾 FlowBot - ભૂગર્ભજળ સહાયક"
}

DESCRIPTIONS = {
    "hi": "अपनी जमीन का पानी कैसा है? पूछिए आसान भाषा में!",
    "en": "Ask about your land's groundwater in simple language!",
    "te": "మీ భూమి యొక్క నీరు ఎలా ఉంది? సరళ భాషలో అడగండి!",
    "ta": "உங்கள் நிலத்தின் தண்ணீர் எப்படி இருக்கிறது? எளிய மொழியில் கேளுங்கள்!",
    "bn": "আপনার জমির পানি কেমন? সহজ ভাষায় জিজ্ঞাসা করুন!",
    "mr": "तुमची जमिनीचे पाणी कसे आहे? सोप्या भाषेत विचारा!",
    "gu": "તમારી જમીનનું પાણી કેવું છે? સરળ ભાષામાં પૂછો!"
}

# Multilingual Quick Replies
QUICK_REPLIES = {
    "hi": ["मेरे गांव का पानी कैसा है?", "क्या सिंचाई के लिए पर्याप्त पानी है?", "कौन सा इलाका सबसे ज्यादा जोखिम में है?"],
    "en": ["How is the water status in my village?", "Is there enough water for irrigation?", "Which area is most at risk?"],
    "te": ["నా గ్రామంలో నీటి పరిస్థితి ఎలా ఉంది?", "నీటిపారుదల కోసం తగినంత నీరు ఉందా?", "ఏ ప్రాంతం అత్యధిక ప్రమాదంలో ఉంది?"],
    "ta": ["என் கிராமத்தில் தண்ணீர் நிலைமை எப்படி?", "நீர்ப்பாசனத்திற்கு போதுமான தண்ணீர் உள்ளதா?", "எந்தப் பகுதி மிக அதிக ஆபத்தில் உள்ளது?"],
    "ur": ["میرے گاؤں میں پانی کی صورتحال کیسی ہے؟", "کیا آبپاشی کے لیے کافی پانی ہے؟", "کون سا علاقہ سب سے زیادہ خطرے میں ہے؟"],
    "bn": ["আমার গ্রামে পানির অবস্থা কেমন?", "সেচের জন্য পর্যাপ্ত পানি আছে কি?", "কোন এলাকা সবচেয়ে বেশি ঝুঁকিতে?"],
    "mr": ["माझ्या गावात पाण्याची स्थिती कशी आहे?", "पिकांच्या पाण्यासाठी पुरेसे पाणी आहे का?", "कोणता भाग सर्वात जास्त धोक्यात आहे?"],
    "gu": ["મારા ગામમાં પાણીની સ્થિતિ કેવી છે?", "સિંચાઈ માટે પૂરતું પાણી છે?", "કયો વિસ્તાર સૌથી વધુ જોખમમાં છે?"]
}

TAB_LABELS = {
    "hi": ["💬 टेक्स्ट", "🎤 आवाज़"],
    "en": ["💬 Text", "🎤 Voice"],
    "te": ["💬 వచనం", "🎤 వాయిస్"],
    "ta": ["💬 உரை", "🎤 குரல்"],
    "ur": ["💬 متن", "🎤 آواز"],
    "bn": ["💬 টেক্সট", "🎤 কণ্ঠস্বর"],
    "mr": ["💬 मजकूर", "🎤 आवाज"],
    "gu": ["💬 ટેક્સ્ટ", "🎤 અવાજ"]
}

QUICK_MESSAGES_LABEL = {
    "hi": "त्वरित संदेश", "en": "Quick Replies", "te": "త్వరిత సందేశాలు", 
    "ta": "விரைவு பதில்கள்", "ur": "فوری پیغامات", "bn": "দ্রুত উত্তর",
    "mr": "त्वरित संदेश", "gu": "ઝડપી જવાબો"
}

PLACEHOLDERS = {
    "hi": "अपना सवाल लिखें...",
    "en": "Ask about groundwater in your area...",
    "te": "మీ ప్రశ్న రాయండి...",
    "ta": "உங்கள் கேள்வியை எழுதுங்கள்...",
    "ur": "اپنا سوال لکھیں...",
    "bn": "আপনার প্রশ্ন লিখুন...",
    "mr": "तुमचा प्रश्न लिहा...",
    "gu": "તમારો પ્રશ્ન લખો..."
}
# --- END DATA and CONFIG ---

# --- Helper Functions (Keep as is) ---
def enhanced_district_detection(query):
    query_lower = query.lower().strip()
    for d in districts:
        if d.lower() in query_lower:
            return d
    words = query_lower.split()
    best_match = None
    best_ratio = 0.6
    for word in words:
        for district in districts:
            ratio = difflib.SequenceMatcher(None, word, district.lower()).ratio()
            if ratio > best_ratio:
                best_match = district
                best_ratio = ratio
    return best_match

def detect_district(query):
    return enhanced_district_detection(query)

def generate_ai_response(query):
    current_lang = st.session_state.get("language", "hi")
    
    prompts = {
        "hi": f"""आप भारतीय किसानों के लिए एक सहायक भूजल कृषि सहायक हैं।
        केवल हिंदी में संक्षिप्त उत्तर दें (2-3 वाक्य)। यदि कोई जिला/शहर का उल्लेख है, तो संक्षिप्त सलाह दें।
        प्रश्न: {query}
        महत्वपूर्ण: केवल हिंदी में उत्तर दें, अंग्रेजी शब्दों का उपयोग न करें।""",
        # ... (rest of the language prompts)
        "en": f"""You are a helpful groundwater farming assistant for Indian farmers.
        Answer only in English with brief responses (2-3 sentences). If a district/city is mentioned, provide concise advice.
        Query: {query}
        Important: Answer only in English, do not mix Hindi words."""
    }
    
    try:
        prompt = prompts.get(current_lang, prompts["en"])
        # Mocking LLM: replace with actual llm.invoke
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        error_messages = {
            "hi": "क्षमा करें, त्रुटि हुई।",
            "en": "Sorry, an error occurred.",
        }
        return error_messages.get(current_lang, error_messages["en"])
# --- END Helper Functions ---

# --- LAYOUT AND LOGIC ---

# Move Premium Features to Sidebar
with st.sidebar:
    st.markdown("### 🌟 **Premium Features**")
    st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)
    selected_feature = st.selectbox("Choose Feature:", options=list(premium_feature_dict.keys()), key="premium_dropdown")
    st.caption(f"**{premium_feature_dict[selected_feature]}**")
    st.markdown("---")
    st.markdown("### ⚙️ **Settings**")
    st.caption("FlowBot v1.0")
    st.caption("<small>SIH 2025 Submission</small>", unsafe_allow_html=True)


# Dashboard switch - Prominent Tip
st.success("💡 **Tip:** Use the sidebar to navigate to the **'Dashboard'** for detailed analytics!")

# Multi-Language Selection - NEW DROPDOWN VERSION
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 🌍 **Select Language / भाषा चुनें**")

with col2:
    # Create dropdown options
    language_options = {}
    for code, info in LANGUAGES.items():
        language_options[f"{info['flag']} {info['name']}"] = code
    
    # Get current selection for dropdown
    current_lang = st.session_state.get("language", "hi")
    current_display = next((display for display, code in language_options.items() if code == current_lang), list(language_options.keys())[0])
    
    selected_language = st.selectbox(
        "🌐",
        options=list(language_options.keys()),
        index=list(language_options.keys()).index(current_display),
        key="language_dropdown",
        label_visibility="collapsed"
    )
    
    # Update session state when selection changes
    new_lang = language_options[selected_language]
    if new_lang != st.session_state.get("language", "hi"):
        st.session_state.language = new_lang
        st.rerun()

lang = st.session_state.get("language", "hi")


# Main Title and Description
st.title(TITLES[lang])
st.write(DESCRIPTIONS[lang])

# --- Quick Replies ---
st.markdown("---")
st.markdown("### 💬 **" + QUICK_MESSAGES_LABEL[lang] + "**")

col_quick = st.columns(len(QUICK_REPLIES[lang]))
questions = QUICK_REPLIES[lang]

for i, q in enumerate(questions):
    with col_quick[i]:
        # Using custom button style from CSS
        if st.button(q, key=f"qbtn_{i}", use_container_width=True):
            st.session_state["user_query"] = q
            st.rerun()

st.markdown("---")

# --- Input Area (Text & Voice) ---
tab1, tab2 = st.tabs(TAB_LABELS[lang])

with tab1:
    # Form for cleaner submission handling
    with st.form(key='chat_form', clear_on_submit=False):
        col_input, col_send = st.columns([5, 1])
        
        with col_input:
            user_query = st.text_input(
                "Type a message...", 
                placeholder=PLACEHOLDERS[lang],
                key="chat_text_input",
                label_visibility="collapsed"
            )
        
        with col_send:
            # Custom styled send button (using the CSS class)
            send_clicked = st.form_submit_button("➤", type="primary", use_container_width=True)


with tab2:
    # Voice input with multilingual support
    voice_prompts = {
        "hi": ["🎤 बोलकर जिले का नाम बताएं", "उदाहरण: 'बीकानेर का पानी कैसा है?'"],
        "en": ["🎤 Say your district name", "Example: 'How is water in Bikaner?'"],
    }
    
    st.markdown("### " + voice_prompts.get(lang, voice_prompts["en"])[0])
    st.caption("💡 " + voice_prompts.get(lang, voice_prompts["en"])[1])
    
    voice_lang_map = {
        "hi": "hi-IN", "en": "en-US", "te": "te-IN", "ta": "ta-IN",
        "ur": "ur-PK", "bn": "bn-BD", "mr": "mr-IN", "gu": "gu-IN"
    }
    
    voice_text = speech_to_text(
        language=voice_lang_map.get(lang, "en-US"),
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True,
        key="voice_recorder"
    )
    
    if voice_text:
        said_labels = {"hi": "आपने कहा:", "en": "You said:", "te": "మీరు చెప్పారు:", "ta": "நீங்கள் சொன்னீர்கள்:"}
        detected_labels = {"hi": "जिला पहचाना गया:", "en": "District detected:", "te": "జిల్లా గుర్తించబడింది:", "ta": "மாவட்டம் கண்டறியப்பட்டது:"}
        
        st.info("🎤 " + said_labels.get(lang, said_labels["en"]) + f" '{voice_text}'")
        detected_district = detect_district(voice_text)
        if detected_district:
            st.success("📍 " + detected_labels.get(lang, detected_labels["en"]) + f" **{detected_district}**")
        
        user_query = voice_text
        send_clicked = True # Force processing

# --- Form Submission & Quick Reply Handling ---
query_to_process = None

# Handle main text/voice input
if send_clicked and user_query and user_query.strip():
    query_to_process = user_query
    
# Handle quick replies
if "user_query" in st.session_state and st.session_state["user_query"]:
    query_to_process = st.session_state["user_query"]
    st.session_state["user_query"] = "" # Clear quick reply state

if query_to_process:
    st.session_state["messages"] = []
    
    spinner_labels = {"hi": "FlowBot सोच रहा है...", "en": "FlowBot is thinking..."}
    
    with st.spinner(spinner_labels.get(lang, spinner_labels["en"])):
        try:
            ai_ans = generate_ai_response(query_to_process)
            chosen_district = detect_district(query_to_process)
            st.session_state["messages"].append({"q": query_to_process, "a": ai_ans, "district": chosen_district})
            st.rerun()
        except Exception as e:
            st.error(f"❌ ERROR: {str(e)}")


# --- Chat Display and Charts ---
if st.session_state["messages"]:
    chat_labels = {"hi": "बातचीत", "en": "Chat"}
    st.markdown("### 💬 **" + chat_labels.get(lang, chat_labels["en"]) + "**")
    
    msg = st.session_state["messages"][-1]
    
    # User message (right-aligned, green)
    st.markdown(
        f"""
        <div class='whatsapp-style-chat' style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
            <div style='background: #DCF8C6; color: #000; padding: 12px 16px; border-radius: 18px 18px 5px 18px; max-width: 70%; box-shadow: 0 1px 2px rgba(0,0,0,0.1); font-size: 14px;'>
                {msg['q']}
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Bot message (left-aligned, white)
    st.markdown(
        f"""
        <div class='whatsapp-style-chat' style='display: flex; justify-content: flex-start; margin-bottom: 15px;'>
            <div style='background: #FFFFFF; color: #000; padding: 12px 16px; border-radius: 18px 18px 18px 5px; max-width: 70%; box-shadow: 0 1px 2px rgba(0,0,0,0.1); border: 1px solid #E5E5EA; font-size: 14px;'>
                <strong>🤖 FlowBot:</strong><br>{msg['a']}
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # --- Enhanced Charts ---
    district_name = msg.get("district")
    if not district_name:
        # Fallback detection logic (kept as is)
        if msg.get('a'):
            answer_text = msg['a'].lower()
            for index, row in df.iterrows():
                if row['District'].lower() in answer_text:
                    district_name = row['District']
                    break
        if not district_name and msg.get('q'):
            query_text = msg['q'].lower()
            for index, row in df.iterrows():
                if row['District'].lower() in query_text:
                    district_name = row['District']
                    break
    
    if district_name:
        showing_data_labels = {"hi": "जिले की जानकारी:", "en": "Showing data for district:"}
        st.markdown(f"### 📊 **{showing_data_labels.get(lang, showing_data_labels['en'])} {district_name}**")
        
        # Data Filtering
        chartdf = df[df['District'].str.lower().str.contains(district_name.lower(), na=False, regex=False)]
        
        if not chartdf.empty:
            row = chartdf.iloc[0]
            extraction = row["ExtractionVolume_ha_m"]
            available = row["GroundWaterAvailability_ham"]
            stage = row["ExtractionStage_Percent"]
            rainfall = row.get("Rainfall_mm", 0)
            
            # Risk Level Determination (Multilingual)
            risk_levels = {
                "over_exploited": {"hi": "अत्याधिक दोहन", "en": "Over-Exploited", "color": "#e53935"},
                "critical": {"hi": "संकटग्रस्त", "en": "Critical", "color": "#ffb300"},
                "safe": {"hi": "सुरक्षित", "en": "Safe", "color": "#4caf50"}
            }
            
            risk_key = "safe"
            if stage > 90: risk_key = "over_exploited"
            elif stage > 70: risk_key = "critical"
                
            risk_level = risk_levels[risk_key][lang]
            risk_color = risk_levels[risk_key]["color"]
            
            # --- Charts ---
            col1, col2, col3 = st.columns(3)
            
            with col1:
                usage_titles = {"hi": f"पानी उपयोग (%)", "en": f"Water Usage (%)"}
                fig_gauge = px.pie(
                    values=[stage, 100-stage], names=["Used", "Available"],
                    title=usage_titles.get(lang, usage_titles["en"]),
                    color_discrete_sequence=[risk_color, "#e0e0e0"], hole=0.6
                )
                fig_gauge.add_annotation(text=f"<b>{stage:.1f}%</b><br>{risk_level}", x=0.5, y=0.5, font_size=16, showarrow=False)
                fig_gauge.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                volume_titles = {"hi": "पानी की मात्रा (ha.m)", "en": "Water Volume (ha.m)"}
                bar_df = pd.DataFrame({"Category": ["Available", "Extracted"], "Volume": [available, extraction]})
                fig_bar = px.bar(
                    bar_df, x="Category", y="Volume", color="Category",
                    color_discrete_map={"Available": "#4caf50", "Extracted": risk_color},
                    title=volume_titles.get(lang, volume_titles["en"]), labels={"Volume": "ha.m", "Category": ""}
                )
                fig_bar.add_hline(y=available * 0.7, line_dash="dot", annotation_text="Safe Limit (70%)", line_color="#2196f3")
                fig_bar.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col3:
                risk_titles = {"hi": "जोखिम स्तर", "en": "Risk Level"}
                risk_categories = {"hi": ["सुरक्षित", "संकटग्रस्त", "अत्याधिक दोहन"], "en": ["Safe (<70%)", "Critical (70-90%)", "Over-Exploited (>90%)"]}
                
                cat_df = pd.DataFrame({
                    "Category": risk_categories.get(lang, risk_categories["en"]),
                    "Color": ["#4caf50", "#ffb300", "#e53935"],
                    "Value": [stage if risk_key == "safe" else 0, stage if risk_key == "critical" else 0, stage if risk_key == "over_exploited" else 0]
                })

                fig_risk = px.bar(
                    cat_df, x="Category", y=[100] * 3,
                    color="Color",
                    color_discrete_map={"#4caf50": "#4caf50", "#ffb300": "#ffb300", "#e53935": "#e53935"},
                    title=risk_titles.get(lang, risk_titles["en"]),
                    labels={"y": "% Stage"}
                )
                # Highlight the current category
                current_index = 0 if risk_key == "safe" else 1 if risk_key == "critical" else 2
                fig_risk.data[current_index].marker.line = dict(width=5, color='black')
                
                fig_risk.update_layout(height=300, showlegend=False, xaxis_tickangle=0, yaxis_visible=False)
                st.plotly_chart(fig_risk, use_container_width=True)
            
            st.markdown("---")
            
            # --- Detailed Information Cards ---
            detailed_info_labels = {"hi": "विस्तृत जानकारी", "en": "Detailed Information"}
            st.markdown("### 📋 **" + detailed_info_labels.get(lang, detailed_info_labels["en"]) + "**")
            
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            
            metric_labels = {
                "available_water": {"hi": "💧 उपलब्ध पानी", "en": "💧 Available Water"},
                "extracted_water": {"hi": "🏭 निकाला गया पानी", "en": "🏭 Extracted Water"},
                "safe_limit": {"hi": "⚠️ सुरक्षित सीमा", "en": "⚠️ Safe Limit"},
                "annual_rainfall": {"hi": "🌧️ वार्षिक वर्षा", "en": "🌧️ Annual Rainfall"},
            }
            
            with info_col1:
                st.metric(label=metric_labels["available_water"].get(lang, metric_labels["available_water"]["en"]),
                          value=f"{available:.1f} ha.m",
                          delta=f"{(available - extraction):.1f} surplus" if available > extraction else f"{(extraction - available):.1f} deficit")
            
            with info_col2:
                st.metric(label=metric_labels["extracted_water"].get(lang, metric_labels["extracted_water"]["en"]),
                          value=f"{extraction:.1f} ha.m",
                          delta=f"{stage:.1f}% of available")
            
            with info_col3:
                safe_limit = available * 0.7
                st.metric(label=metric_labels["safe_limit"].get(lang, metric_labels["safe_limit"]["en"]),
                          value=f"{safe_limit:.1f} ha.m",
                          delta="Within limit" if extraction <= safe_limit else "Exceeded!")
            
            with info_col4:
                st.metric(label=metric_labels["annual_rainfall"].get(lang, metric_labels["annual_rainfall"]["en"]),
                          value=f"{rainfall:.0f} mm",
                          delta="Good" if rainfall > 600 else "Low")
            
            # --- Recommendations ---
            recommendations_labels = {"hi": "सुझाव", "en": "Recommendations"}
            st.markdown("### 💡 **" + recommendations_labels.get(lang, recommendations_labels["en"]) + "**")
            
            recommendations = {
                "over_exploited": {"hi": ["🚨 तुरंत पानी की बचत करें", "💧 ड्रिप सिंचाई अपनाएं", "🌾 कम पानी वाली फसलें उगाएं"], "en": ["🚨 Immediate water conservation needed", "💧 Switch to drip irrigation", "🌾 Grow drought-resistant crops"]},
                "critical": {"hi": ["⚠️ सावधानी बरतें", "💧 पानी का सदुपयोग करें", "🔄 फसल चक्र अपनाएं"], "en": ["⚠️ Exercise caution", "💧 Use water efficiently", "🔄 Practice crop rotation"]},
                "safe": {"hi": ["✅ स्थिति अच्छी है", "📈 टिकाऊ खेती करें", "🌱 नई तकनीक अपनाएं"], "en": ["✅ Situation is good", "📈 Practice sustainable farming", "🌱 Adopt new technologies"]}
            }
            
            selected_recommendations = recommendations[risk_key][lang]
            
            rec_cols = st.columns(len(selected_recommendations))
            for i, rec in enumerate(selected_recommendations):
                with rec_cols[i]:
                    st.success(rec)
            
        else:
            no_data_labels = {"hi": "इस जिले के लिए डेटा उपलब्ध नहीं है", "en": "No data available for this district"}
            st.warning("⚠️ " + no_data_labels.get(lang, no_data_labels["en"]) + f": **{district_name}**")

st.markdown("---")

# --- Top 3 Dashboard + Tips ---
area_status_labels = {"hi": "सबसे महत्वपूर्ण क्षेत्र🚨", "en": "Most Critical Areas 🚨"}

st.markdown("### 📊 **" + area_status_labels.get(lang, area_status_labels["en"]) + "**")
critical_df = df.sort_values("ExtractionStage_Percent", ascending=False).head(3)

col_critical = st.columns(3)
for i, (_, row) in enumerate(critical_df.iterrows()):
    with col_critical[i]:
        level = ("अत्याधिक दोहन" if row["ExtractionStage_Percent"] > 90
                 else "संकटग्रस्त" if row["ExtractionStage_Percent"] > 70
                 else "सुरक्षित")
        level_en = ("Over-Exploited" if row["ExtractionStage_Percent"] > 90
                    else "Critical" if row["ExtractionStage_Percent"] > 70
                    else "Safe")
        
        boxcolor = "#ffdddd" if row["ExtractionStage_Percent"] > 90 else "#fff4d4" if row["ExtractionStage_Percent"] > 70 else "#e0f2f1"
        border_color = "#e53935" if row["ExtractionStage_Percent"] > 90 else "#ffb300" if row["ExtractionStage_Percent"] > 70 else "#4caf50"
        
        # Enhanced box styling
        st.markdown(
            f"""
            <div style='background:{boxcolor}; border: 2px solid {border_color}; border-radius:10px; padding:15px; margin-bottom:10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h4 style='color:{border_color}; margin-top:0; font-size:18px;'>{row['District']}</h4>
                <p style='margin: 0; font-size:14px;'>{level} / {level_en}</p>
                <p style='margin: 0; font-weight:bold; font-size:20px;'>{row['ExtractionStage_Percent']:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

farmer_tips_labels = {"hi": "किसान सुझाव", "en": "Farmer Tips"}

tips_content = {
    "hi": ["💧 ड्रिप सिंचाई अपनाएँ।", "🌾 कम पानी वाली फसलें उगाएँ।", "🚰 मिट्टी की नमी जाँचें।"],
    "en": ["💧 Use drip irrigation.", "🌾 Grow crops using less water.", "🚰 Check soil moisture weekly."],
}

st.markdown("### 💡 **" + farmer_tips_labels.get(lang, farmer_tips_labels["en"]) + "**")
tip_cols = st.columns(3)
for i, tip in enumerate(tips_content.get(lang, tips_content["en"])):
    with tip_cols[i]:
        st.info(tip)

st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'><small>FlowBot – Groundwater Assistant | **INGRES Team ZenFlow** @ SIH 2025</small></div>", unsafe_allow_html=True)


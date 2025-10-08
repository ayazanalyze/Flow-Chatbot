import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import difflib
import re
from streamlit_mic_recorder import speech_to_text

# --- 1. APP CONFIGURATION & INITIALIZATION ---

# Set page configuration first
st.set_page_config(
    page_title="INGRES FlowBot - AI Groundwater Assistant",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables from .env file
load_dotenv()

# Centralized dictionary for all User Interface text and configurations
UI_CONFIG = {
    "languages": {
        "hi": {"name": "हिंदी", "flag": "🇮🇳"}, "en": {"name": "English", "flag": "🇬🇧"},
        "te": {"name": "తెలుగు", "flag": "🇮🇳"}, "ta": {"name": "தமிழ்", "flag": "🇮🇳"},
        "bn": {"name": "বাংলা", "flag": "🇧🇩"}, "mr": {"name": "मराठी", "flag": "🇮🇳"},
        "gu": {"name": "ગુજરાતી", "flag": "🇮🇳"}
    },
    "titles": {
        "hi": "🌾 FlowBot - आपका भूजल सहायक", "en": "🌾 FlowBot – Your Groundwater Assistant",
        "te": "🌾 FlowBot - మీ భూగర్భ జల సహాయకుడు", "ta": "🌾 FlowBot - உங்கள் நிலத்தடி நீர் உதவியாளர்",
        "bn": "🌾 FlowBot- আপনার ভূগর্ভস্থ পানি সহায়ক", "mr": "🌾 FlowBot - तुमचा भूजल सहायक",
        "gu": "🌾 FlowBot - તમારો ભૂગર્ભજળ સહાયક"
    },
    "descriptions": {
        "hi": "अपनी जमीन के भूजल स्तर के बारे में पूछें और स्मार्ट खेती के लिए सुझाव पाएं।",
        "en": "Ask about your land's groundwater and get insights for smart farming.",
        "te": "మీ భూమి యొక్క నీటి గురించి అడగండి మరియు స్మార్ట్ వ్యవసాయం కోసం సలహాలు పొందండి!",
        "ta": "உங்கள் நிலத்தின் நிலத்தடி நீர் பற்றி கேட்டு, έξυπனான விவசாயத்திற்கான ஆலோசனைகளைப் பெறுங்கள்.",
        "bn": "আপনার জমির ভূগর্ভস্থ পানি সম্পর্কে জিজ্ঞাসা করুন এবং স্মার্ট চাষের জন্য অন্তর্দৃষ্টি পান।",
        "mr": "तुमच्या जमिनीच्या भूजल पातळीबद्दल विचारा आणि स्मार्ट शेतीसाठी सूचना मिळवा.",
        "gu": "તમારી જમીનના ભૂગર્ભજળ વિશે પૂછો અને સ્માર્ટ ખેતી માટે આંતરદૃષ્ટિ મેળવો."
    },
    "prompts": {
        # Prompts are designed for concise, farmer-friendly responses in the selected language.
        "hi": "आप भारतीय किसानों के लिए एक सहायक भूजल कृषि सहायक हैं। केवल हिंदी में संक्षिप्त उत्तर दें (2-3 वाक्य)। यदि कोई जिला/शहर का उल्लेख है, तो संक्षिप्त सलाह दें। प्रश्न: {}",
        "en": "You are a helpful groundwater farming assistant for Indian farmers. Answer only in English with brief responses (2-3 sentences). If a district/city is mentioned, provide concise advice. Query: {}",
        "te": "మీరు భారతీయ రైతుల కోసం ఒక సహాయక భూగర్భజల వ్యవసాయ సహాయకుడు। తెలుగులో మాత్రమే సంక్షిప్త సమాధానాలు ఇవ్వండి (2-3 వాక్యాలు)। ప్రశ్న: {}",
        "ta": "நீங்கள் இந்திய விவசாயிகளுக்கான நிலத்தடி நீர் விவசாய உதவியாளர்। தமிழில் மட்டும் குறுகிய பதில்களை கொடுங்கள் (2-3 வாக்கியங்கள்)। கேள்வி: {}",
        "bn": "আপনি ভারতীয় কৃষকদের জন্য একজন সহায়ক ভূগর্ভস্থ পানি কৃষি সহায়ক। শুধুমাত্র বাংলায় সংক্ষিপ্ত উত্তর দিন (2-3 বাক্য)। প্রশ্ন: {}",
        "mr": "तुम्ही भारतीय शेतकऱ्यांसाठी भूजल शेती सहायक आहात। फक्त मराठीत थोडक्यात उत्तरे द्या (2-3 वाक्ये)। प्रश्न: {}",
        "gu": "તમે ભારતીય ખેડૂતો માટે ભૂગર્ભજળ કૃષિ સહાયક છો। માત્ર ગુજરાતીમાં ટૂંકા જવાબો આપો (2-3 વાક્યો)। પ્રશ્ન: {}"
    },
    "quick_replies": {
        "hi": ["मेरे गांव का पानी कैसा है?", "सिंचाई के लिए पानी की स्थिति क्या है?", "कौन सा इलाका सबसे ज्यादा जोखिम में है?"],
        "en": ["What's the water status in my area?", "Is there enough water for irrigation?", "Which areas are most at risk?"],
        "te": ["నా గ్రామంలో నీటి పరిస్థితి ఎలా ఉంది?", "నీటిపారుదల కోసం తగినంత నీరు ఉందా?", "ఏ ప్రాంతం అత్యధిక ప్రమాదంలో ఉంది?"],
        "ta": ["என் கிராமத்தில் தண்ணீர் நிலைமை எப்படி?", "நீர்ப்பாசனத்திற்கு போதுமான தண்ணீர் உள்ளதா?", "எந்தப் பகுதி மிக அதிக ஆபத்தில் உள்ளது?"],
        "bn": ["আমার গ্রামে পানির অবস্থা কেমন?", "সেচের জন্য পর্যাপ্ত পানি আছে কি?", "কোন এলাকা সবচেয়ে বেশি ঝুঁকিতে?"],
        "mr": ["माझ्या गावात पाण्याची स्थिती कशी आहे?", "सिंचनासाठी पुरेसे पाणी आहे का?", "कोणता भाग सर्वात जास्त धोक्यात आहे?"],
        "gu": ["મારા ગામમાં પાણીની સ્થિતિ કેવી છે?", "સિંચાઈ માટે પૂરતું પાણી છે?", "કયો વિસ્તાર સૌથી વધુ જોખમમાં છે?"]
    },
    "labels": {
        "hi": {"chat": "बातचीत", "ask": "अपना सवाल लिखें...", "send": "भेजें", "voice_prompt": "🎤 बोलकर पूछें", "thinking": "FlowBot सोच रहा है...", "no_data": "इस जिले के लिए डेटा उपलब्ध नहीं है", "error": "क्षमा करें, त्रुटि हुई।"},
        "en": {"chat": "Conversation", "ask": "Ask your question here...", "send": "Send", "voice_prompt": "🎤 Ask with your voice", "thinking": "FlowBot is thinking...", "no_data": "No data available for this district", "error": "Sorry, an error occurred."},
        "te": {"chat": "చాట్", "ask": "మీ ప్రశ్న రాయండి...", "send": "పంపు", "voice_prompt": "🎤 వాయిస్ తో అడగండి", "thinking": "FlowBot ఆలోచిస్తోంది...", "no_data": "ఈ జిల్లాకు డేటా అందుబాటులో లేదు", "error": "క్షమించండి, లోపం జరిగింది."},
        "ta": {"chat": "அரட்டை", "ask": "உங்கள் கேள்வியை எழுதுங்கள்...", "send": "அனுப்பு", "voice_prompt": "🎤 குரல் மூலம் கேளுங்கள்", "thinking": "FlowBot சிந்தித்துக்கொண்டிருக்கிறது...", "no_data": "இந்த மாவட்டத்திற்கான தரவு கிடைக்கவில்லை", "error": "மன்னிக்கவும், பிழை ஏற்பட்டது."},
        "bn": {"chat": "চ্যাট", "ask": "আপনার প্রশ্ন লিখুন...", "send": "পাঠান", "voice_prompt": "🎤 ভয়েস দিয়ে জিজ্ঞাসা করুন", "thinking": "FlowBot ভাবছে...", "no_data": "এই জেলার জন্য তথ্য উপলব্ধ নয়", "error": "দুঃখিত, একটি ত্রুটি ঘটেছে।"},
        "mr": {"chat": "संभाषण", "ask": "तुमचा प्रश्न लिहा...", "send": "पाठवा", "voice_prompt": "🎤 आवाजाने विचारा", "thinking": "FlowBot विचार करत आहे...", "no_data": "या जिल्ह्यासाठी डेटा उपलब्ध नाही", "error": "माफ करा, त्रुटी झाली."},
        "gu": {"chat": "ચેટ", "ask": "તમારો પ્રશ્ન લખો...", "send": "મોકલો", "voice_prompt": "🎤 અવાજથી પૂછો", "thinking": "FlowBot વિચારી રહ્યું છે...", "no_data": "આ જિલ્લા માટે ડેટા ઉપલબ્ધ નથી", "error": "માફ કરશો, ભૂલ થઈ."}
    },
    "premium_features": {
        "AI Crop Water Calculator": "Get AI-driven recommendations on optimal crop choices based on groundwater data.",
        "Recommendation Model for Water Conservation": "Receive tailored advice on water-saving techniques for your area.",
        "Predictive Analytics": "Forecast future groundwater trends with advanced analytics.",
    }
}

# --- 2. CORE FUNCTIONS (DATA & AI LOGIC) ---

@st.cache_resource
def initialize_services():
    """Initializes and caches the Groq LLM client."""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not api_key:
            st.error("GROQ_API_KEY not found. Please set it in your Streamlit secrets or .env file.")
            st.stop()
        return ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
    except Exception as e:
        st.error(f"Failed to initialize AI model: {e}")
        st.stop()

@st.cache_data
def load_data(filepath="cleaned_groundwater_data.csv"):
    """Loads, processes, and caches the groundwater dataset."""
    try:
        df = pd.read_csv(filepath)
        df['District'] = df['District'].str.strip()
        districts = sorted(df['District'].dropna().unique().tolist())
        return df, districts
    except FileNotFoundError:
        st.error(f"Data file not found at '{filepath}'. Please ensure it is in the correct directory.")
        st.stop()

def detect_district(query, district_list):
    """
    Detects a district from the user query using direct and fuzzy matching.
    Designed to be robust against voice recognition variations.
    """
    query_lower = query.lower().strip()
    
    # 1. Direct match (most reliable)
    for district in district_list:
        if district.lower() in query_lower:
            return district
    
    # 2. Fuzzy matching (for voice input errors)
    words = re.split(r'\s+', query_lower)
    best_match = None
    # Start with a reasonably high threshold to avoid false positives
    best_ratio = 0.75  
    
    for word in words:
        # Using difflib's fast approximation for speed
        matches = difflib.get_close_matches(word, [d.lower() for d in district_list], n=1, cutoff=best_ratio)
        if matches:
            # Find the original cased district name
            for original_district in district_list:
                if original_district.lower() == matches[0]:
                    return original_district
    
    return best_match

def generate_ai_response(llm, query, lang):
    """Generates a response from the LLM based on the user's query and language."""
    try:
        prompt_template = UI_CONFIG["prompts"].get(lang, UI_CONFIG["prompts"]["en"])
        prompt = prompt_template.format(query)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception:
        return UI_CONFIG["labels"][lang]["error"]

# --- 3. UI RENDERING FUNCTIONS ---

def render_sidebar():
    """Renders the sidebar with branding, premium features, and language selection."""
    with st.sidebar:
        st.image("https://i.imgur.com/h5l6mG8.png", use_column_width=True) # Placeholder Logo
        st.markdown("<h1 style='text-align: center; color: #004d40;'>INGRES FlowBot</h1>", unsafe_allow_html=True)
        st.markdown("---")

        # Language Selection
        st.markdown("### 🌍 Language / भाषा")
        lang_options = {f"{info['flag']} {info['name']}": code for code, info in UI_CONFIG["languages"].items()}
        
        # Find current language display name to set dropdown default
        current_lang_code = st.session_state.get("language", "en")
        current_lang_display = "🇬🇧 English" # Default fallback
        for display, code in lang_options.items():
            if code == current_lang_code:
                current_lang_display = display
                break

        selected_display = st.selectbox(
            "Choose Language", 
            options=lang_options.keys(),
            index=list(lang_options.keys()).index(current_lang_display),
            label_visibility="collapsed"
        )
        
        new_lang_code = lang_options[selected_display]
        if new_lang_code != st.session_state.language:
            st.session_state.language = new_lang_code
            st.rerun()

        # Premium Features Showcase
        st.markdown("---")
        st.markdown("### 🌟 Premium Features")
        premium_keys = list(UI_CONFIG["premium_features"].keys())
        selected_feature = st.selectbox("Explore:", options=premium_keys)
        st.info(UI_CONFIG["premium_features"][selected_feature])
        
        st.markdown("---")
        st.info("Built for SIH 2025 by **Team ZenFlow**.")


def render_input_area(lang):
    """Renders the main input area with text, voice, and quick replies."""
    st.markdown("### 💬 Ask FlowBot")
    
    query = None

    # Quick Replies
    cols = st.columns(len(UI_CONFIG['quick_replies'][lang]))
    for i, question in enumerate(UI_CONFIG['quick_replies'][lang]):
        if cols[i].button(question, use_container_width=True, key=f"qr_{i}"):
            query = question

    # Text and Voice Input Tabs
    text_tab, voice_tab = st.tabs(["✍️ " + UI_CONFIG['labels'][lang]['ask'], "🎤 " + UI_CONFIG['labels'][lang]['voice_prompt']])

    with text_tab:
        with st.form(key="text_input_form", clear_on_submit=True):
            text_query = st.text_input("text_query", placeholder=UI_CONFIG['labels'][lang]['ask'], label_visibility="collapsed")
            if st.form_submit_button(label=UI_CONFIG['labels'][lang]['send'], use_container_width=True):
                if text_query:
                    query = text_query
    
    with voice_tab:
        voice_lang_map = {"hi": "hi-IN", "en": "en-US", "te": "te-IN", "ta": "ta-IN", "bn": "bn-BD", "mr": "mr-IN", "gu": "gu-IN"}
        voice_text = speech_to_text(
            language=voice_lang_map.get(lang, "en-US"),
            start_prompt="▶️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True,
            key="voice_recorder"
        )
        if voice_text:
            query = voice_text
            st.info(f"You said: \"{voice_text}\"")
    
    return query

def render_chat_and_charts(message, lang, df, districts):
    """Renders the chat conversation and data visualizations for a detected district."""
    st.markdown(f"### {UI_CONFIG['labels'][lang]['chat']}")
    
    # Custom CSS for WhatsApp-style chat bubbles
    st.markdown("""
        <style>
            .user-msg {
                display: flex; justify-content: flex-end; margin-bottom: 10px;
            }
            .bot-msg {
                display: flex; justify-content: flex-start; margin-bottom: 10px;
            }
            .msg-content {
                padding: 10px 15px; border-radius: 18px; max-width: 75%; font-size: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
        </style>
    """, unsafe_allow_html=True)

    # User Message
    st.markdown(f"<div class='user-msg'><div class='msg-content' style='background: #E2FDD6; color: #000; border-bottom-right-radius: 5px;'>{message['q']}</div></div>", unsafe_allow_html=True)
    
    # Bot Message
    st.markdown(f"<div class='bot-msg'><div class='msg-content' style='background: #FFF; color: #000; border: 1px solid #EAEAEA; border-bottom-left-radius: 5px;'>🤖 <b>FlowBot:</b><br>{message['a']}</div></div>", unsafe_allow_html=True)

    # --- Data Visualization Section ---
    district_name = message.get("district")
    if not district_name:
        return # Exit if no district was detected

    st.info(f"📊 Showing detailed analytics for **{district_name}**")
    
    chart_df = df[df['District'].str.lower() == district_name.lower()]
    
    if chart_df.empty:
        st.warning(f"⚠️ {UI_CONFIG['labels'][lang]['no_data']}: {district_name}")
        return

    data = chart_df.iloc[0]
    stage = data.get("ExtractionStage_Percent", 0)
    
    # Determine Risk Level and Color
    if stage > 90:
        risk_level, risk_color = "Over-Exploited", "#d32f2f" # Red
    elif stage > 70:
        risk_level, risk_color = "Critical", "#ffa000" # Amber
    else:
        risk_level, risk_color = "Safe", "#388e3c" # Green

    # --- Charts ---
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge Chart for Water Usage
            fig_gauge = px.pie(
                values=[stage, max(0, 100 - stage)], names=["Used", "Available"],
                title=f"<b>Groundwater Usage Stage</b>", hole=0.6,
                color_discrete_sequence=[risk_color, "#cfd8dc"]
            )
            fig_gauge.add_annotation(text=f"<b>{stage:.1f}%</b><br>{risk_level}", x=0.5, y=0.5, font_size=20, showarrow=False)
            fig_gauge.update_layout(height=350, showlegend=False, title_x=0.5, title_font_size=18)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Bar Chart for Water Volume
            available = data.get("GroundWaterAvailability_ham", 0)
            extracted = data.get("ExtractionVolume_ha_m", 0)
            bar_df = pd.DataFrame({
                "Category": ["Available", "Extracted"],
                "Volume (ha.m)": [available, extracted]
            })
            fig_bar = px.bar(
                bar_df, x="Category", y="Volume (ha.m)",
                color="Category", color_discrete_map={"Available": "#388e3c", "Extracted": risk_color},
                title="<b>Water Availability vs. Extraction</b>"
            )
            fig_bar.update_layout(height=350, showlegend=False, title_x=0.5, title_font_size=18)
            st.plotly_chart(fig_bar, use_container_width=True)
            
    # --- Detailed Metrics ---
    st.markdown("#### Detailed Metrics")
    with st.container(border=True):
        m_col1, m_col2, m_col3 = st.columns(3)
        safe_limit = available * 0.7
        
        m_col1.metric("💧 Available Water (ha.m)", f"{available:.1f}")
        m_col2.metric("🏭 Extracted Water (ha.m)", f"{extracted:.1f}", delta=f"{stage:.1f}% of available", delta_color="off")
        m_col3.metric("⚠️ Safe Limit (70%)", f"{safe_limit:.1f}", delta="Exceeded!" if extracted > safe_limit else "Within Limit", 
                      delta_color="inverse" if extracted > safe_limit else "normal")

# --- 4. MAIN APPLICATION ---

def main():
    """Main function to run the Streamlit application."""
    
    # Initialize session state variables
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("language", "en")
    
    # Load services and data
    llm = initialize_services()
    df, districts = load_data()
    
    # Render UI components
    render_sidebar()
    lang = st.session_state.language
    st.title(UI_CONFIG["titles"].get(lang, UI_CONFIG["titles"]["en"]))
    st.markdown(UI_CONFIG["descriptions"].get(lang, UI_CONFIG["descriptions"]["en"]))
    
    # Get user input from any source (text, voice, quick reply)
    user_query = render_input_area(lang)

    # Process new input
    if user_query:
        st.session_state.messages = [] # Clear previous messages for a fresh interaction
        with st.spinner(UI_CONFIG["labels"][lang]["thinking"]):
            detected_district = detect_district(user_query, districts)
            ai_answer = generate_ai_response(llm, user_query, lang)
            
            # Store the complete message object
            st.session_state.messages.append({
                "q": user_query, 
                "a": ai_answer, 
                "district": detected_district
            })
            st.rerun()

    # Display the latest conversation and charts
    if st.session_state.messages:
        render_chat_and_charts(st.session_state.messages[-1], lang, df, districts)
        
    # --- Final Section: Critical Areas & Tips ---
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        with st.container(border=True):
            st.markdown("##### 🚨 Most Critical Areas")
            for _, row in df.sort_values("ExtractionStage_Percent", ascending=False).head(3).iterrows():
                level_color = "#FADBD8" if row["ExtractionStage_Percent"] > 90 else "#FEF9E7"
                st.markdown(f"<div style='background:{level_color}; border-left: 5px solid {('#d32f2f' if row['ExtractionStage_Percent'] > 90 else '#ffa000')}; padding: 8px; border-radius: 5px; margin-bottom: 5px;'><b>{row['District']}</b> ({row['State']}) - {row['ExtractionStage_Percent']:.1f}% Used</div>", unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown("##### 🌱 Smart Farming Tips")
            tips = {"en": ["💧 Use drip irrigation to save water.", "🌾 Grow drought-resistant crops.", "✅ Check soil moisture regularly."],
                    "hi": ["💧 ड्रिप सिंचाई का प्रयोग करें।", "🌾 सूखा-प्रतिरोधी फसलें उगाएँ।", "✅ मिट्टी की नमी नियमित रूप से जांचें।"]}
            for tip in tips.get(lang, tips["en"]):
                 st.markdown(f"<div style='background: #E8F8F5; border-left: 5px solid #16A085; padding: 8px; border-radius: 5px; margin-bottom: 5px;'>{tip}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

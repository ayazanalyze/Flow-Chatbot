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

# Initialize session state FIRST
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "language" not in st.session_state:
    st.session_state.language = "hi"

load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-20b")
st.set_page_config(page_title="FlowBot - By Team ZenFlow", layout="wide")

df = pd.read_csv("cleaned_groundwater_data.csv")
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

# Move Premium Features to Sidebar
with st.sidebar:
    st.markdown("### 🌟 Premium Features")
    selected_feature = st.selectbox("Choose Feature:", options=list(premium_feature_dict.keys()), key="premium_dropdown")
    st.caption(premium_feature_dict[selected_feature])

# Dashboard switch
st.success("💡 Tip: Use the sidebar to navigate to the 'dashboard' for detailed analytics!")

# Multi-Language Selection - NEW DROPDOWN VERSION
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 🌍 Select Language / भाषा चुनें")

with col2:
    # Create dropdown options using your existing LANGUAGES dict
    language_options = {}
    for code, info in LANGUAGES.items():
        language_options[f"{info['flag']} {info['name']}"] = code
    
    # Get current selection for dropdown (with fallback)
    current_lang = st.session_state.get("language", "hi")
    current_display = None
    for display, code in language_options.items():
        if code == current_lang:
            current_display = display
            break
    
    # If current language not found, default to Hindi
    if current_display is None:
        current_display = list(language_options.keys())[0]
        current_lang = "hi"
    
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

st.title(TITLES[lang])
st.write(DESCRIPTIONS[lang])

def enhanced_district_detection(query):
    """Enhanced district detection with fuzzy matching for voice input"""
    query_lower = query.lower().strip()
    
    # Direct match first
    for d in districts:
        if d.lower() in query_lower:
            return d
    
    # Fuzzy matching for voice input variations
    words = query_lower.split()
    best_match = None
    best_ratio = 0.6  # Minimum similarity threshold
    
    for word in words:
        for district in districts:
            # Check similarity
            ratio = difflib.SequenceMatcher(None, word, district.lower()).ratio()
            if ratio > best_ratio:
                best_match = district
                best_ratio = ratio
    
    return best_match

def detect_district(query):
    return enhanced_district_detection(query)

def generate_ai_response(query):
    current_lang = st.session_state.get("language", "hi")
    
    # Multilingual prompts
    prompts = {
        "hi": f"""आप भारतीय किसानों के लिए एक सहायक भूजल कृषि सहायक हैं।
        केवल हिंदी में संक्षिप्त उत्तर दें (2-3 वाक्य)। यदि कोई जिला/शहर का उल्लेख है, तो संक्षिप्त सलाह दें।
        प्रश्न: {query}
        महत्वपूर्ण: केवल हिंदी में उत्तर दें, अंग्रेजी शब्दों का उपयोग न करें।""",
        
        "en": f"""You are a helpful groundwater farming assistant for Indian farmers.
        Answer only in English with brief responses (2-3 sentences). If a district/city is mentioned, provide concise advice.
        Query: {query}
        Important: Answer only in English, do not mix Hindi words.""",
        
        "te": f"""మీరు భారతీయ రైతుల కోసం ఒక సహాయక భూగర్భజల వ్యవసాయ సహాయకుడు.
        తెలుగులో మాత్రమే సంక్షిప్త సమాధానాలు ఇవ్వండి (2-3 వాక్యాలు). జిల్లా/నగరం పేర్కొంటే, సంక్షిప్త సలహా ఇవ్వండి.
        ప్రశ్న: {query}
        ముఖ్యమైనది: తెలుగులో మాత్రమే సమాధానం ఇవ్వండి.""",
        
        "ta": f"""நீங்கள் இந்திய விவசாயிகளுக்கான நிலத்தடி நீர் விவசாய உதவியாளர்.
        தமிழில் மட்டும் குறுகிய பதில்களை கொடுங்கள் (2-3 வாக்கியங்கள்). மாவட்டம்/நகரம் குறிப்பிட்டால், சுருக்கமான ஆலோசனை கொடுங்கள்.
        கேள்வி: {query}
        முக்கியம்: தமிழில் மட்டுமே பதில் கொடுங்கள்.""",
        
        "ur": f"""آپ ہندوستانی کسانوں کے لیے زیر زمین پانی کے زراعت کے مددگار ہیں۔
        صرف اردو میں مختصر جوابات دیں (2-3 جملے)۔ اگر کوئی ضلع/شہر کا ذکر ہے تو مختصر مشورہ دیں۔
        سوال: {query}
        اہم: صرف اردو میں جواب دیں۔""",
        
        "bn": f"""আপনি ভারতীয় কৃষকদের জন্য একজন সহায়ক ভূগর্ভস্থ পানি কৃষি সহায়ক।
        শুধুমাত্র বাংলায় সংক্ষিপ্ত উত্তর দিন (2-3 বাক্য)। যদি কোনো জেলা/শহরের উল্লেখ থাকে, তাহলে সংক্ষিপ্ত পরামর্শ দিন।
        প্রশ্ন: {query}
        গুরুত্বপূর্ণ: শুধুমাত্র বাংলায় উত্তর দিন।""",
        
        "mr": f"""तुम्ही भारतीय शेतकऱ्यांसाठी भूजल शेती सहाyyक आहात.
        फक्त मराठीत थोडक्यात उत्तरे द्या (2-3 वाक्ये). जिल्हा/शहराचा उल्लेख असल्यास, थोडक्यात सल्ला द्या.
        प्रश्न: {query}
        महत्वाचे: फक्त मराठीत उत्तर द्या.""",
        
        "gu": f"""તમે ભારતીય ખેડૂતો માટે ભૂગર્ભજળ કૃષિ સહાયક છો.
        માત્ર ગુજરાતીમાં ટૂંકા જવાબો આપો (2-3 વાક્યો). જિલ્લો/શહેરનો ઉલ્લેખ હોય તો, ટૂંકી સલાહ આપો.
        પ્રશ્ન: {query}
        મહત્વનું: માત્ર ગુજરાતીમાં જવાબ આપો."""
    }
    
    try:
        prompt = prompts.get(current_lang, prompts["en"])
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        error_messages = {
            "hi": "क्षमा करें, त्रुटि हुई।",
            "en": "Sorry, an error occurred.",
            "te": "క్షమించండి, లోపం జరిగింది.",
            "ta": "மன்னிக்கவும், பிழை ஏற்பட்டது.",
            "ur": "معاف کریں، خرابی ہوئی۔",
            "bn": "দুঃখিত, একটি ত্রুটি ঘটেছে।",
            "mr": "माफ करा, त्रुटी झाली.",
            "gu": "માફ કરશો, ભૂલ થઈ."
        }
        return error_messages.get(current_lang, error_messages["en"])

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

# WhatsApp-style Quick Replies
st.markdown("### 💬 " + QUICK_MESSAGES_LABEL[lang])
questions = QUICK_REPLIES[lang]

for i, q in enumerate(questions):
    if st.button(q, key=f"qbtn_{i}", use_container_width=True):
        st.session_state["user_query"] = q
        st.rerun()

# WhatsApp-style input with form + Voice Input
st.markdown("---")

# Create tabs for text and voice input
tab1, tab2 = st.tabs(TAB_LABELS[lang])

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

with tab1:
    # Clean text input
    user_query = st.text_input(
        "Type a message...", 
        placeholder=PLACEHOLDERS[lang],
        key="main_text_input"
    )
    
    # Use columns for the send button
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        send_clicked = st.button("➤ Send", key="send_btn", use_container_width=True)

with tab2:
    # Voice input with multilingual support
    voice_prompts = {
        "hi": ["🎤 बोलकर जिले का नाम बताएं", "उदाहरण: 'बीकानेर का पानी कैसा है?' या 'जयपुर में सिंचाई की स्थिति'"],
        "en": ["🎤 Say your district name", "Example: 'How is water in Bikaner?' or 'Irrigation status in Jaipur'"],
        "te": ["🎤 మీ జిల్లా పేరు చెప్పండి", "ఉదాహరణ: 'బీకానేర్‌లో నీరు ఎలా ఉంది?' లేదా 'జైపూర్‌లో నీటిపారుదల పరిస్థితి'"],
        "ta": ["🎤 உங்கள் மாவட்ட பெயரைச் சொல்லுங்கள்", "உதாரணம்: 'பீகானேரில் தண்ணீர் எப்படி?' அல்லது 'ஜெய்ப்பூரில் நீர்ப்பாசன நிலை'"],
        "ur": ["🎤 اپنے ضلع کا نام بتائیں", "مثال: 'بیکانیر میں پانی کیسا ہے؟' یا 'جے پور میں آبپاشی کی صورتحال'"],
        "bn": ["🎤 আপনার জেলার নাম বলুন", "উদাহরণ: 'বীকানেরে পানি কেমন?' বা 'জয়পুরে সেচের অবস্থা'"],
        "mr": ["🎤 तुमच्या जिल्ह्याचे नाव सांगा", "उदाहरण: 'बीकानेरमध्ये पाणी कसे आहे?' किंवा 'जयपूरमध्ये पाणी पुरवठ्याची स्थिती'"],
        "gu": ["🎤 તમારા જિલ્લાનું નામ કહો", "ઉદાહરણ: 'બીકાનેરમાં પાણી કેવું છે?' અથવા 'જયપુરમાં સિંચાઈની સ્થિતિ'"]
    }
    
    st.markdown(voice_prompts[lang][0])
    st.caption("💡 " + voice_prompts[lang][1])
    
    # Voice language mapping
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
        said_labels = {
            "hi": "आपने कहा:", "en": "You said:", "te": "మీరు చెప్పారు:",
            "ta": "நீங்கள் சொன்னீர்கள்:", "ur": "آپ نے کہا:", "bn": "আপনি বলেছেন:",
            "mr": "तुम्ही म्हणालात:", "gu": "તમે કહ્યું:"
        }
        
        detected_labels = {
            "hi": "जिला पहचाना गया:", "en": "District detected:", "te": "జిల్లా గుర్తించబడింది:",
            "ta": "மாவட்டம் கண்டறியப்பட்டது:", "ur": "ضلع کی شناخت ہوگئی:", "bn": "জেলা সনাক্ত করা হয়েছে:",
            "mr": "जिल्हा ओळखला गेला:", "gu": "જિલ્લો ઓળખાયો:"
        }
        
        st.info("🎤 " + said_labels[lang] + f" '{voice_text}'")
        
        detected_district = detect_district(voice_text)
        if detected_district:
            st.success("📍 " + detected_labels[lang] + f" {detected_district}")
        
        user_query = voice_text
        send_clicked = True

# Handle form submission (text or voice)
if send_clicked and user_query and user_query.strip():
    st.session_state["messages"] = []
    detected_dist = detect_district(user_query)
    
    spinner_labels = {
        "hi": "FlowBot सोच रहा है...", "en": "FlowBot is thinking...", "te": "FlowBot ఆలోచిస్తోంది...",
        "ta": "FlowBot சிந்தித்துக்கொண்டிருக்கிறது...", "ur": "FlowBot سوچ رہا ہے...", "bn": "FlowBot ভাবছে...",
        "mr": "FlowBot विचार करत आहे...", "gu": "FlowBot વિચારી રહ્યું છે..."
    }
    
    with st.spinner(spinner_labels[lang]):
        try:
            ai_ans = generate_ai_response(user_query)
            chosen_district = detected_dist
            st.session_state["messages"].append({"q": user_query, "a": ai_ans, "district": chosen_district})
            st.rerun()
        except Exception as e:
            st.error(f"❌ ERROR in processing: {str(e)}")

# Handle quick replies
if "user_query" in st.session_state and st.session_state["user_query"]:
    user_query_from_quick = st.session_state["user_query"]
    st.session_state["user_query"] = ""
    st.session_state["messages"] = []
    
    spinner_labels = {
        "hi": "FlowBot सोच रहा है...", "en": "FlowBot is thinking...", "te": "FlowBot ఆలోచిస్తోంది...",
        "ta": "FlowBot சிந்தித்துக்கொண்டிருக்கிறது...", "ur": "FlowBot سوچ رہا ہے...", "bn": "FlowBot ভাবছে...",
        "mr": "FlowBot विचार करत आहे...", "gu": "FlowBot વિચારી રહ્યું છે..."
    }
    
    with st.spinner(spinner_labels[lang]):
        try:
            ai_ans = generate_ai_response(user_query_from_quick)
            chosen_district = detect_district(user_query_from_quick)
            st.session_state["messages"].append({"q": user_query_from_quick, "a": ai_ans, "district": chosen_district})
            st.rerun()
        except Exception as e:
            st.error(f"❌ ERROR in quick reply processing: {str(e)}")

# Chat Display and Charts (Fixed to show for ALL languages)
# Add these imports at the top of your file if not already present


# Your corrected code:
if st.session_state["messages"]:
    chat_labels = {
        "hi": "बातचीत", "en": "Chat", "te": "చాట్", "ta": "அரட்டை", 
        "ur": "بات چیت", "bn": "চ্যাট", "mr": "संभाषण", "gu": "ચેટ"
    }
    
    st.markdown("### 💬 " + chat_labels[lang])
    
    msg = st.session_state["messages"][-1]
    
    # User message (right-aligned, green)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
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
        <div style='display: flex; justify-content: flex-start; margin-bottom: 15px;'>
            <div style='background: #FFFFFF; color: #000; padding: 12px 16px; border-radius: 18px 18px 18px 5px; max-width: 70%; box-shadow: 0 1px 2px rgba(0,0,0,0.1); border: 1px solid #E5E5EA; font-size: 14px;'>
                <strong>🤖 FlowBot:</strong><br>{msg['a']}
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Enhanced Charts - Fixed to show for ALL languages
    district_name = None
    
    # Check if district exists in msg dictionary
    if "district" in msg and msg["district"]:
        district_name = msg["district"]
    else:
        # Fallback: Try to extract district name from the query or answer
        if msg.get('a'):
            answer_text = msg['a'].lower()
            for index, row in df.iterrows():
                if row['District'].lower() in answer_text:
                    district_name = row['District']
                    break
        
        # If not found in answer, try the query
        if not district_name and msg.get('q'):
            query_text = msg['q'].lower()
            for index, row in df.iterrows():
                if row['District'].lower() in query_text:
                    district_name = row['District']
                    break
    
    # Show charts if we found a district name
    if district_name:
        showing_data_labels = {
            "hi": "जिले की जानकारी दिखाई जा रही है:",
            "en": "Showing data for district:",
            "te": "జిల్లా సమాచారం చూపిస్తోంది:",
            "ta": "மாவட்ட தரவு காட்டப்படுகிறது:",
            "ur": "ضلع کی معلومات دکھائی جا رہی ہیں:",
            "bn": "জেলার তথ্য দেখানো হচ্ছে:",
            "mr": "जिल्ह्याची माहिती दाखवली जात आहे:",
            "gu": "જિલ્લાની માહિતી બતાવવામાં આવી રહી છે:"
        }
        
        st.info("📊 " + showing_data_labels[lang] + f" {district_name}")
        
        # Use flexible matching for district names
        chartdf = df[df['District'].str.lower().str.contains(district_name.lower(), na=False)]
        
        # If exact match fails, try partial matching
        if chartdf.empty:
            clean_district = re.sub(r'\b(district|जिला|జిల్లా|மாவட்டம்|ضلع|জেলা|जिल्हा|જિલ્લો)\b', '', district_name.lower()).strip()
            chartdf = df[df['District'].str.lower().str.contains(clean_district, na=False)]
        
        if not chartdf.empty:
            row = chartdf.iloc[0]
            extraction = row["ExtractionVolume_ha_m"]
            available = row["GroundWaterAvailability_ham"]
            stage = row["ExtractionStage_Percent"]
            rainfall = row.get("Rainfall_mm", 0)
            
            # Risk Level Determination (Multilingual)
            risk_levels = {
                "over_exploited": {"hi": "अत्याधिक दोहन", "en": "Over-Exploited", "te": "అధిక దోపిడీ", "ta": "அதிகம் சுரண்டல்", "ur": "حد سے زیادہ استعمال", "bn": "অতিরিক্ত শোষণ", "mr": "अधिक शोषण", "gu": "વધુ પડતું શોષણ"},
                "critical": {"hi": "संकटग्रस्त", "en": "Critical", "te": "క్లిష్టమైన", "ta": "முக்கியமான", "ur": "نازک", "bn": "গুরুতর", "mr": "गुरुत्वाकर्षण", "gu": "નિર્ણાયક"},
                "safe": {"hi": "सुरक्षित", "en": "Safe", "te": "సురక్షితమైన", "ta": "பாதுகாப்பான", "ur": "محفوظ", "bn": "নিরাপদ", "mr": "सुरक्षित", "gu": "સુરક્ષિત"}
            }
            
            if stage > 90:
                risk_level = risk_levels["over_exploited"][lang]
                risk_color = "#e53935"
            elif stage > 70:
                risk_level = risk_levels["critical"][lang]
                risk_color = "#ffb300"
            else:
                risk_level = risk_levels["safe"][lang]
                risk_color = "#4caf50"
            
            # Create three columns for different charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Water Usage Chart with multilingual title
                usage_titles = {
                    "hi": f"{district_name} का पानी उपयोग",
                    "en": f"{district_name} Water Usage",
                    "te": f"{district_name} నీటి వినియోగం",
                    "ta": f"{district_name} நீர் பயன்பாடு",
                    "ur": f"{district_name} پانی کا استعمال",
                    "bn": f"{district_name} পানির ব্যবহার",
                    "mr": f"{district_name} पाण्याचा वापर",
                    "gu": f"{district_name} પાણીનો ઉપયોગ"
                }
                
                fig_gauge = px.pie(
                    values=[stage, 100-stage],
                    names=["Used", "Available"],
                    title=usage_titles[lang],
                    color_discrete_sequence=[risk_color, "#e0e0e0"],
                    hole=0.6
                )
                fig_gauge.add_annotation(
                    text=f"<b>{stage:.1f}%</b><br>{risk_level}",
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )
                fig_gauge.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Volume Chart with multilingual title
                volume_titles = {
                    "hi": "पानी की मात्रा (ha.m)",
                    "en": "Water Volume (ha.m)",
                    "te": "నీటి పరిమాణం (ha.m)",
                    "ta": "நீர் அளவு (ha.m)",
                    "ur": "پانی کی مقدار (ha.m)",
                    "bn": "পানির পরিমাণ (ha.m)",
                    "mr": "पाण्याचे प्रमाण (ha.m)",
                    "gu": "પાણીનું પ્રમાણ (ha.m)"
                }
                
                bar_df = pd.DataFrame({
                    "Category": ["Available", "Extracted", "Safe Limit"],
                    "Volume": [available, extraction, available * 0.7],
                    "Color": ["#4caf50", risk_color, "#2196f3"]
                })
                fig_bar = px.bar(
                    bar_df, x="Category", y="Volume",
                    color="Category",
                    color_discrete_map={
                        "Available": "#4caf50",
                        "Extracted": risk_color,
                        "Safe Limit": "#2196f3"
                    },
                    title=volume_titles[lang],
                    labels={"Volume": "ha.m", "Category": ""}
                )
                fig_bar.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col3:
                # Risk Level Chart with multilingual categories
                risk_titles = {
                    "hi": "जोखिम स्तर",
                    "en": "Risk Level",
                    "te": "రిస్క్ స్థాయి",
                    "ta": "ஆபத்து நிலை",
                    "ur": "خطرے کی سطح",
                    "bn": "ঝুঁকির মাত্রা",
                    "mr": "धोक्याची पातळी",
                    "gu": "જોખમનું સ્તર"
                }
                
                risk_categories = {
                    "hi": ["बहुत कम", "कम", "मध्यम", "उच्च", "बहुत उच्च"],
                    "en": ["Very Low", "Low", "Medium", "High", "Very High"],
                    "te": ["చాలా తక్కువ", "తక్కువ", "మధ్యస్థ", "అధిక", "చాలా అధిక"],
                    "ta": ["மிகக் குறைந்த", "குறைந்த", "நடுத்தர", "உயர்", "மிக உயர்"],
                    "ur": ["بہت کم", "کم", "درمیانہ", "زیادہ", "بہت زیادہ"],
                    "bn": ["অতি নিম্ন", "নিম্ন", "মধ্যম", "উচ্চ", "অতি উচ্চ"],
                    "mr": ["अगदी कमी", "कमी", "मध्यम", "जास्त", "खूप जास्त"],
                    "gu": ["ખૂબ ઓછું", "ઓછું", "મધ્યમ", "વધુ", "ખૂબ વધુ"]
                }
                
                categories = risk_categories[lang]
                risk_scores = [10, 30, 50, 70, 90]
                colors = ["#4caf50", "#8bc34a", "#ffeb3b", "#ff9800", "#f44336"]
                
                current_pos = 0
                for i, score in enumerate(risk_scores):
                    if stage <= score:
                        current_pos = i
                        break
                else:
                    current_pos = len(risk_scores) - 1
                
                fig_risk = px.bar(
                    x=categories, y=[100] * len(categories),
                    color=colors,
                    title=risk_titles[lang]
                )
                
                fig_risk.data[current_pos].update(
                    marker_color='red',
                    marker_line=dict(width=3, color='black')
                )
                
                fig_risk.update_layout(
                    height=300, 
                    showlegend=False,
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            
            # Comprehensive Information Cards with multilingual labels
            detailed_info_labels = {
                "hi": "विस्तृत जानकारी",
                "en": "Detailed Information",
                "te": "వివరణాత్మక సమాచారం",
                "ta": "விரிவான தகவல்",
                "ur": "تفصیلی معلومات",
                "bn": "বিস্তারিত তথ্য",
                "mr": "तपशीलवार माहिती",
                "gu": "વિગતવાર માહિતી"
            }
            
            st.markdown("### 📋 " + detailed_info_labels[lang])
            
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            
            metric_labels = {
                "available_water": {"hi": "💧 उपलब्ध पानी", "en": "💧 Available Water", "te": "💧 అందుబాటులో ఉన్న నీరు", "ta": "💧 கிடைக்கும் நீர்", "ur": "💧 دستیاب پانی", "bn": "💧 উপলব্ধ পানি", "mr": "💧 उपलब्ध पाणी", "gu": "💧 ઉપલબ્ધ પાણી"},
                "extracted_water": {"hi": "🏭 निकाला गया पानी", "en": "🏭 Extracted Water", "te": "🏭 వెలికితీసిన నీరు", "ta": "🏭 எடுக்கப்பட்ட நீர்", "ur": "🏭 نکالا گیا پانی", "bn": "🏭 উত্তোলিত পানি", "mr": "🏭 काढलेले पाणी", "gu": "🏭 કાઢેલું પાણી"},
                "safe_limit": {"hi": "⚠️ सुरक्षित सीमा", "en": "⚠️ Safe Limit", "te": "⚠️ సురక్షిత పరిమితి", "ta": "⚠️ பாதுகாப்பு வரம்பு", "ur": "⚠️ محفوظ حد", "bn": "⚠️ নিরাপদ সীমা", "mr": "⚠️ सुरक्षित मर्यादा", "gu": "⚠️ સુરક્ષિત મર્યાદા"},
                "annual_rainfall": {"hi": "🌧️ वार्षिक वर्षा", "en": "🌧️ Annual Rainfall", "te": "🌧️ వార్షిక వర్షపాతం", "ta": "🌧️ ஆண்டு மழைப்பொழிவு", "ur": "🌧️ سالانہ بارش", "bn": "🌧️ বার্ষিক বৃষ্টিপাত", "mr": "🌧️ वार्षिक पाऊस", "gu": "🌧️ વાર્ષિક વરસાદ"},
                "risk_level_metric": {"hi": "📊 जोखिम स्तर", "en": "📊 Risk Level", "te": "📊 రిస్క్ స్థాయి", "ta": "📊 ஆபத்து நிலை", "ur": "📊 خطرے کی سطح", "bn": "📊 ঝুঁকির মাত্রা", "mr": "📊 धोक्याची पातळी", "gu": "📊 જોખમનું સ્તર"}
            }
            
            with info_col1:
                st.metric(
                    label=metric_labels["available_water"][lang],
                    value=f"{available:.1f} ha.m",
                    delta=f"{available - extraction:.1f} surplus" if available > extraction else f"{extraction - available:.1f} deficit"
                )
            
            with info_col2:
                st.metric(
                    label=metric_labels["extracted_water"][lang],
                    value=f"{extraction:.1f} ha.m",
                    delta=f"{stage:.1f}% of available"
                )
            
            with info_col3:
                safe_limit = available * 0.7
                st.metric(
                    label=metric_labels["safe_limit"][lang],
                    value=f"{safe_limit:.1f} ha.m",
                    delta="Within limit" if extraction <= safe_limit else "Exceeded!"
                )
            
            with info_col4:
                if rainfall > 0:
                    st.metric(
                        label=metric_labels["annual_rainfall"][lang],
                        value=f"{rainfall:.0f} mm",
                        delta="Good" if rainfall > 600 else "Low"
                    )
                else:
                    st.metric(
                        label=metric_labels["risk_level_metric"][lang],
                        value=risk_level,
                        delta=f"{stage:.1f}% usage"
                    )
        else:
            # If no district data found, show a helpful message
            no_data_labels = {
                "hi": "इस जिले के लिए डेटा उपलब्ध नहीं है",
                "en": "No data available for this district",
                "te": "ఈ జిల్లాకు డేటా అందుబాటులో లేదు",
                "ta": "இந்த மாவட்டத்திற்கான தரவு கிடைக்கவில்லை",
                "ur": "اس ضلع کے لیے ڈیٹا دستیاب نہیں",
                "bn": "এই জেলার জন্য তথ্য উপলব্ধ নয়",
                "mr": "या जिल्ह्यासाठी डेटा उपलब्ध नाही",
                "gu": "આ જિલ્લા માટે ડેટા ઉપલબ્ધ નથી"
            }
            st.warning("⚠️ " + no_data_labels[lang] + f": {district_name}")

            
            # Recommendations based on risk level (Multilingual)
            recommendations_labels = {
                "hi": "सुझाव",
                "en": "Recommendations",
                "te": "సిఫార్సులు",
                "ta": "பரிந்துரைகள்",
                "ur": "تجاویز",
                "bn": "সুপারিশসমূহ",
                "mr": "शिफारसी",
                "gu": "ભલામણો"
            }
            
            st.markdown("### 💡 " + recommendations_labels[lang])
            
            recommendations = {
                "over_exploited": {
                    "hi": ["🚨 तुरंत पानी की बचत करें", "💧 ड्रिप सिंचाई अपनाएं", "🌾 कम पानी वाली फसलें उगाएं"],
                    "en": ["🚨 Immediate water conservation needed", "💧 Switch to drip irrigation", "🌾 Grow drought-resistant crops"],
                    "te": ["🚨 తక్షణ నీటి సంరక్షణ అవసరం", "💧 డ్రిప్ ఇరిగేషన్ అవలంబించండి", "🌾 కరువు-నిరోధక పంటలు పండించండి"],
                    "ta": ["🚨 உடனடியாக நீர் சேமிப்பு தேவை", "💧 துளி நீர்ப்பாசனத்தை பயன்படுத்துங்கள்", "🌾 வறட்சி-எதிர்ப்பு பயிர்களை வளர்க்கவும்"],
                    "ur": ["🚨 فوری طور پر پانی کی بچت کی ضرورت", "💧 ڈرپ ایریگیشن اپنائیں", "🌾 خشک سالی مزاحم فصلیں اگائیں"],
                    "bn": ["🚨 তাৎক্ষণিক পানি সংরক্ষণের প্রয়োজন", "💧 ড্রিপ সেচ গ্রহণ করুন", "🌾 খরা-প্রতিরোধী ফসল জন্মান"],
                    "mr": ["🚨 त्वरित पाणी संवर्धन आवश्यक", "💧 ठिबक सिंचन अवलंबा", "🌾 दुष्काळ-प्रतिरोधी पिके पिका"],
                    "gu": ["🚨 તાત્કાલિક પાણી સંરક્ષણ જરૂરી", "💧 ડ્રિપ ઇરિગેશન અપનાવો", "🌾 દુષ્કાળ-પ્રતિરોધી પાકો ઉગાડો"]
                },
                "critical": {
                    "hi": ["⚠️ सावधानी बरतें", "💧 पानी का सदुपयोग करें", "🔄 फसल चक्र अपनाएं"],
                    "en": ["⚠️ Exercise caution", "💧 Use water efficiently", "🔄 Practice crop rotation"],
                    "te": ["⚠️ జాగ్రత్త వహించండి", "💧 నీటిని సమర్థవంతంగా ఉపయోగించండి", "🔄 పంట మార్చుకోవడం అభ్యసించండి"],
                    "ta": ["⚠️ எச்சரிக்கையுடன் இருங்கள்", "💧 நீரை திறமையாக பயன்படுத்துங்கள்", "🔄 பயிர் சுழற்சியை கடைப்பிடியுங்கள்"],
                    "ur": ["⚠️ احتیاط برتیں", "💧 پانی کو موثر طریقے سے استعمال کریں", "🔄 فصلی گردش کریں"],
                    "bn": ["⚠️ সাবধানতা অবলম্বন করুন", "💧 পানি কার্যকরভাবে ব্যবহার করুন", "🔄 ফসল আবর্তন অনুশীলন করুন"],
                    "mr": ["⚠️ सावधगिरी बाळगा", "💧 पाणी कार्यक्षमतेने वापरा", "🔄 पिकांच्या आवर्तनाचा अभ्यास करा"],
                    "gu": ["⚠️ સાવધાની રાખો", "💧 પાણીનો કાર્યક્ષમ ઉપયોગ કરો", "🔄 પાક પરિવર્તન કરો"]
                },
                "safe": {
                    "hi": ["✅ स्थिति अच्छी है", "📈 टिकाऊ खेती करें", "🌱 नई तकनीक अपनाएं"],
                    "en": ["✅ Situation is good", "📈 Practice sustainable farming", "🌱 Adopt new technologies"],
                    "te": ["✅ పరిస్థితి మంచిది", "📈 స్థిరమైన వ్యవసాయం చేయండి", "🌱 కొత్త సాంకేతికతలను అవలంబించండి"],
                    "ta": ["✅ நிலைமை நல்லது", "📈 நிலையான விவசாயம் செய்யுங்கள்", "🌱 புதிய தொழில்நுட்பங்களை அறிமுகப்படுத்துங்கள்"],
                    "ur": ["✅ صورتحال اچھی ہے", "📈 پائیدار کاشتکاری کریں", "🌱 نئی ٹیکنالوجیز اپنائیں"],
                    "bn": ["✅ পরিস্থিতি ভাল", "📈 টেকসই চাষাবাদ করুন", "🌱 নতুন প্রযুক্তি গ্রহণ করুন"],
                    "mr": ["✅ परिस्थिती चांगली आहे", "📈 शाश्वत शेती करा", "🌱 नवीन तंत्रज्ञान स्वीकारा"],
                    "gu": ["✅ સ્થિતિ સારી છે", "📈 ટકાઉ ખેતી કરો", "🌱 નવી તકનીકો અપનાવો"]
                }
            }
            
            if stage > 90:
                selected_recommendations = recommendations["over_exploited"][lang]
            elif stage > 70:
                selected_recommendations = recommendations["critical"][lang]
            else:
                selected_recommendations = recommendations["safe"][lang]
            
            for rec in selected_recommendations:
                st.success(rec)
       
# Top 3 Dashboard + Tips (Multilingual)
area_status_labels = {
    "hi": "सबसे महत्वपूर्ण क्षेत्र🚨",
    "en": "Most Critical Areas 🚨",
    "te": "అత్యంత కీలక ప్రాంతాలు🚨",
    "ta": "மிக முக்கிய பகுதிகள்🚨",
    "ur": "علاقائی صورتحال",
    "bn": "সবচেয়ে গুরুত্বপূর্ণ এলাকা🚨",
    "mr": " सर्वात महत्त्वाच्या क्षेत्रां🚨",
    "gu": "સૌથી મહત્વના વિસ્તારો🚨"
}

st.markdown("### 📊 " + area_status_labels[lang])
for _, row in df.sort_values("ExtractionStage_Percent", ascending=False).head(3).iterrows():
    level = ("अत्याधिक दोहन / Over-Exploited" if row["ExtractionStage_Percent"] > 90
             else "संकटग्रस्त / Critical" if row["ExtractionStage_Percent"] > 70
             else "सुरक्षित / Safe")
    boxcolor = "#ffbcbc" if row["ExtractionStage_Percent"] > 90 else "#ffe066" if row["ExtractionStage_Percent"] > 70 else "#c8f7c5"
    st.markdown(
        f"<div style='background:{boxcolor};border-radius:10px;padding:10px;margin-bottom:8px;'><b>{row['District']} ({row['State']})</b> | {level} - {row['ExtractionStage_Percent']:.1f}%</div>",
        unsafe_allow_html=True
    )

farmer_tips_labels = {
    "hi": "किसान सुझाव",
    "en": "Farmer Tips",
    "te": "రైతు సూచనలు",
    "ta": "விவசாயி குறிப्पुகள்",
    "ur": "کسان تجاویز",
    "bn": "কৃষক পরামর্শ",
    "mr": "शेतकरी सूचना",
    "gu": "ખેડૂત સૂચનાઓ"
}

tips_content = {
    "hi": ["💧 ड्रिप सिंचाई अपनाएँ।", "🌾 कम पानी वाली फसलें उगाएँ।", "🚰 मिट्टी की नमी जाँचें।"],
    "en": ["💧 Use drip irrigation.", "🌾 Grow crops using less water.", "🚰 Check soil moisture weekly."],
    "te": ["💧 డ్రిప్ ఇరిగేషన్ ఉపయోగించండి.", "🌾 తక్కువ నీటితో పంటలు పండించండి.", "🚰 వారానికి మట్టి తేమను తనిఖీ చేయండి."],
    "ta": ["💧 துளி நீர்ப்பாசனத்தைப் பயன்படுத்துங்கள்.", "🌾 குறைந்த நீர் பயன்படுத்தும் பயிர்களை வளர்க்கவும்.", "🚰 வாரந்தோறும் மண் ஈரப்பதத்தை சரிபார்க்கவும்."],
    "ur": ["💧 ڈرپ ایریگیشن استعمال کریں۔", "🌾 کم پانی استعمال کرنے والی فصلیں اگائیں۔", "🚰 ہفتہ وار مٹی کی نمی چیک کریں۔"],
    "bn": ["💧 ড্রিপ সেচ ব্যবহার করুন।", "🌾 কম পানি ব্যবহারকারী ফসল জন্মান।", "🚰 সাপ্তাহিক মাটির আর্দ্রতা পরীক্ষা করুন।"],
    "mr": ["💧 ठिबक सिंचन वापरा।", "🌾 कमी पाणी वापरणारी पिके पिका।", "🚰 साप्ताहिक मातीची ओलावा तपासा।"],
    "gu": ["💧 ડ્રિપ ઇરિગેશનનો ઉપયોગ કરો.", "🌾 ઓછા પાણીના પાકો ઉગાડો.", "🚰 સાપ્તાહિક માટીની ભેજ તપાસો."]
}

st.markdown("### 💡 " + farmer_tips_labels[lang])
for tip in tips_content[lang]:
    st.info(tip)

st.markdown("<small>INGRES Groundwater Assistant @ SIH 2025</small>", unsafe_allow_html=True)

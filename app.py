import streamlit as st
from streamlit_mic_recorder import speech_to_text
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import difflib

# Initialize session state FIRST
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "language" not in st.session_state:
    st.session_state.language = "hi"

load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]

# Add debug for API key
st.sidebar.write(f"🔑 API Key loaded: {bool(groq_api_key)}")
if groq_api_key:
    st.sidebar.write(f"🔑 Key preview: {groq_api_key[:10]}...")

llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")
st.set_page_config(page_title="FlowBot - By Team ZenFlow", layout="wide")

df = pd.read_csv("cleaned_groundwater_data.csv")
districts = sorted([str(d).strip() for d in df['District'].unique() if pd.notna(d)])

# Premium Features definition
premium_feature_dict = {
    "AI Crop Water Calculator": "Get AI-driven recommendations on optimal crop choices based on groundwater data.",
    "Recommendation Model for Water Conservation": "Receive tailored advice on water-saving techniques for your area.",
    "Predictive Analytics": "Forecast future groundwater trends with advanced analytics.",
}

# Move Premium Features to Sidebar
with st.sidebar:
    st.markdown("### 🌟 Premium Features")
    selected_feature = st.selectbox("Choose Feature:", options=list(premium_feature_dict.keys()), key="premium_dropdown")
    st.caption(premium_feature_dict[selected_feature])
    
    # Debug section in sidebar
    st.markdown("### 🔧 Debug Info")
    st.write(f"Messages count: {len(st.session_state.messages)}")
    st.write(f"Current language: {st.session_state.language}")

# Dashboard switch
st.success("💡 Tip: Use the sidebar to navigate to the 'dashboard' for detailed analytics!")

# Language switch
col1, col2 = st.columns(2)
with col1:    
    st.button("🇮🇳 हिंदी", on_click=lambda: st.session_state.update(language="hi"))
with col2:    
    st.button("🇬🇧 English", on_click=lambda: st.session_state.update(language="en"))

lang = st.session_state.get("language", "hi")

st.title("🌾 " + ("किसान मित्र - भूजल सहायक" if lang == "hi" else "Farmer's Friend – Groundwater Assistant"))
st.write("अपनी जमीन का पानी कैसा है? पूछिए आसान भाषा में!" if lang == "hi" else "Ask about your land's groundwater in simple language!")

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
    
    if current_lang == "hi":
        prompt = f"""
        आप भारतीय किसानों के लिए एक सहायक भूजल कृषि सहायक हैं।
        केवल हिंदी में संक्षिप्त उत्तर दें (2-3 वाक्य)। यदि कोई जिला/शहर का उल्लेख है, तो संक्षिप्त सलाह दें।
        प्रश्न: {query}
        
        महत्वपूर्ण: केवल हिंदी में उत्तर दें, अंग्रेजी शब्दों का उपयोग न करें।
        """
    else:
        prompt = f"""
        You are a helpful groundwater farming assistant for Indian farmers.
        Answer only in English with brief responses (2-3 sentences). If a district/city is mentioned, provide concise advice.
        Query: {query}
        
        Important: Answer only in English, do not mix Hindi words.
        """
    
    try:
        st.write("🔄 DEBUG: About to call Groq API...")  # Debug line
        response = llm.invoke([HumanMessage(content=prompt)])
        st.write(f"✅ DEBUG: Response received: {response.content[:50]}...")  # Debug line
        return response.content
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")  # Show error
        error_msg = "क्षमा करें, त्रुटि हुई।" if current_lang == "hi" else "Sorry, an error occurred."
        return f"{error_msg} Error: {str(e)}"

# WhatsApp-style Quick Replies
st.markdown("### 💬 " + ("त्वरित संदेश" if lang == "hi" else "Quick Replies"))
quick_hindi = ["मेरे गांव का पानी कैसा है?", "क्या सिंचाई के लिए पर्याप्त पानी है?", "कौन सा इलाका सबसे ज्यादा जोखिम में है?"]
quick_english = ["How is the water status in my village?", "Is there enough water for irrigation?", "Which area is most at risk?"]

questions = quick_hindi if lang == "hi" else quick_english

for i, q in enumerate(questions):
    if st.button(q, key=f"qbtn_{i}", use_container_width=True):
        st.session_state["user_query"] = q
        st.rerun()

# WhatsApp-style input with form + Voice Input
st.markdown("---")

# Create tabs for text and voice input
tab1, tab2 = st.tabs(["💬 " + ("टेक्स्ट" if lang == "hi" else "Text"), "🎤 " + ("आवाज़" if lang == "hi" else "Voice")])

with tab1:
    # Text input form - FIXED VERSION
    with st.form("chat_form", clear_on_submit=False):  # Changed to False
        col_input, col_send = st.columns([4, 1])
        
        with col_input:
            user_query = st.text_input(
                "Type a message...", 
                placeholder="अपना सवाल लिखें..." if lang == "hi" else "Ask about groundwater in your area...",
                label_visibility="collapsed",
                key="text_input"
            )
        
        with col_send:
            send_clicked = st.form_submit_button("➤", help="Send message")
    
    # DEBUG SECTION - Add this after the form
    st.markdown("### 🔧 Debug Information")
    st.write(f"**Send clicked:** {send_clicked}")
    st.write(f"**User query:** '{user_query}'")
    st.write(f"**Query length:** {len(user_query) if user_query else 0}")
    st.write(f"**Query stripped:** '{user_query.strip() if user_query else ''}'")
    st.write(f"**Condition check:** {bool(send_clicked and user_query and user_query.strip())}")

with tab2:
    # Voice input
    st.markdown("🎤 " + ("बोलकर जिले का नाम बताएं" if lang == "hi" else "Say your district name"))
    st.caption("💡 " + ("उदाहरण: 'बीकानेर का पानी कैसा है?' या 'जयपुर में सिंचाई की स्थिति'" if lang == "hi" else "Example: 'How is water in Bikaner?' or 'Irrigation status in Jaipur'"))
    
    # Voice recorder component
    voice_text = speech_to_text(
        language='hi-IN' if lang == "hi" else 'en-US',
        start_prompt="🎤 " + ("बोलना शुरू करें" if lang == "hi" else "Start Recording"),
        stop_prompt="⏹️ " + ("रुकें" if lang == "hi" else "Stop Recording"),
        just_once=True,
        use_container_width=True,
        key="voice_recorder"
    )
    
    # Process voice input automatically
    if voice_text:
        st.info("🎤 " + ("आपने कहा:" if lang == "hi" else "You said:") + f" '{voice_text}'")
        
        # Detect district from voice
        detected_district = detect_district(voice_text)
        if detected_district:
            st.success("📍 " + ("जिला पहचाना गया:" if lang == "hi" else "District detected:") + f" {detected_district}")
        
        # Set voice text as user query and process
        user_query = voice_text
        send_clicked = True

# Handle form submission (text or voice) - IMPROVED VERSION
if send_clicked and user_query and user_query.strip():
    st.write("🚀 DEBUG: Processing user input...")  # Debug line
    
    # Clear previous messages for fresh conversation
    st.session_state["messages"] = []
    
    # Show which district was detected (if any)
    detected_dist = detect_district(user_query)
    st.write(f"📍 DEBUG: Detected district: {detected_dist}")  # Debug line
    
    # Process the message with loading spinner
    with st.spinner("FlowBot is thinking..." if lang == "en" else "FlowBot सोच रहा है..."):
        try:
            ai_ans = generate_ai_response(user_query)
            chosen_district = detected_dist
            
            st.session_state["messages"].append({"q": user_query, "a": ai_ans, "district": chosen_district})
            st.success("✅ DEBUG: Message added to session state")  # Debug line
            st.rerun()
        except Exception as e:
            st.error(f"❌ ERROR in processing: {str(e)}")

# Handle input from quick replies - IMPROVED VERSION
if "user_query" in st.session_state and st.session_state["user_query"]:
    user_query = st.session_state["user_query"]
    st.write(f"🔄 DEBUG: Processing quick reply: '{user_query}'")  # Debug line
    st.session_state["user_query"] = ""  # Clear immediately to prevent loops
    
    # Clear previous messages for fresh conversation
    st.session_state["messages"] = []
    
    # Process the message with loading spinner
    with st.spinner("FlowBot is thinking..." if lang == "en" else "FlowBot सोच रहा है..."):
        try:
            ai_ans = generate_ai_response(user_query)
            chosen_district = detect_district(user_query)
            
            st.session_state["messages"].append({"q": user_query, "a": ai_ans, "district": chosen_district})
            st.rerun()
        except Exception as e:
            st.error(f"❌ ERROR in quick reply processing: {str(e)}")

# Simple Test Section - ADD THIS FOR DEBUGGING
st.markdown("### 🔧 Simple Test")
test_input = st.text_input("Simple test input:", key="test_input")
if st.button("Test Submit"):
    st.write(f"✅ Test successful! You entered: '{test_input}'")
    st.write(f"✅ API Key working: {bool(groq_api_key)}")
    if groq_api_key:
        try:
            test_response = llm.invoke([HumanMessage(content="Hello, just testing")])
            st.write(f"✅ LLM Test Response: {test_response.content[:100]}...")
        except Exception as e:
            st.error(f"❌ LLM Test Failed: {str(e)}")

# WhatsApp-style Chat Display
if st.session_state["messages"]:
    st.markdown("### 💬 " + ("बातचीत" if lang == "hi" else "Chat"))
    
    # Show only the latest message
    msg = st.session_state["messages"][-1]  # Get the latest message
    
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
    
    # Enhanced Charts (if district detected)
    if msg["district"]:
        st.info("📊 " + ("जिले की जानकारी दिखाई जा रही है:" if lang == "hi" else "Showing data for district:") + f" {msg['district']}")
        
        chartdf = df[df['District'].str.lower().str.contains(msg["district"].lower())]
        if not chartdf.empty:
            # Your existing chart code here...
            st.success(f"✅ Found data for {msg['district']}")
        else:
            st.warning("❌ " + ("इस जिले के लिए डेटा उपलब्ध नहीं है:" if lang == "hi" else "No data available for district:") + f" {msg['district']}")

# Rest of your code (tips, dashboard, etc.) remains the same...
st.markdown("### 💡 " + ("किसान सुझाव" if lang == "hi" else "Farmer Tips"))
tips = [
    "💧 ड्रिप सिंचाई अपनाएँ।" if lang == "hi" else "💧 Use drip irrigation.",
    "🌾 कम पानी वाली फसलें उगाएँ।" if lang == "hi" else "🌾 Grow crops using less water.",
    "🚰 मिट्टी की नमी जाँचें।" if lang == "hi" else "🚰 Check soil moisture weekly.",
]

for tip in tips:
    st.info(tip)

st.markdown("<small>INGRES Groundwater Assistant @ SIH 2025</small>", unsafe_allow_html=True)

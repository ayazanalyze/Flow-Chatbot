import streamlit as st
from streamlit_mic_recorder import speech_to_text
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import difflib

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
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
        response = llm([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        error_msg = "क्षमा करें, त्रुटि हुई।" if current_lang == "hi" else "Sorry, an error occurred."
        return f"{error_msg}"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

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
    # Text input form
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([4, 1])
        
        with col_input:
            user_query = st.text_input(
                "Type a message...", 
                placeholder="अपना सवाल लिखें..." if lang == "hi" else "Ask about groundwater in your area...",
                label_visibility="collapsed"
            )
        
        with col_send:
            send_clicked = st.form_submit_button("➤", help="Send message")

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

# Handle form submission (text or voice)
if send_clicked and user_query and user_query.strip():
    # Clear previous messages for fresh conversation
    st.session_state["messages"] = []
    
    # Show which district was detected (if any)
    detected_dist = detect_district(user_query)
    
    # Process the message with loading spinner
    with st.spinner("FlowBot is thinking..." if lang == "en" else "FlowBot सोच रहा है..."):
        ai_ans = generate_ai_response(user_query)
        chosen_district = detected_dist
    
    st.session_state["messages"].append({"q": user_query, "a": ai_ans, "district": chosen_district})
    st.rerun()

# Handle input from quick replies
if "user_query" in st.session_state and st.session_state["user_query"]:
    user_query = st.session_state["user_query"]
    st.session_state["user_query"] = ""  # Clear immediately to prevent loops
    
    # Clear previous messages for fresh conversation
    st.session_state["messages"] = []
    
    # Process the message with loading spinner
    with st.spinner("FlowBot is thinking..." if lang == "en" else "FlowBot सोच रहा है..."):
        ai_ans = generate_ai_response(user_query)
        chosen_district = detect_district(user_query)
    
    st.session_state["messages"].append({"q": user_query, "a": ai_ans, "district": chosen_district})
    st.rerun()

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
            row = chartdf.iloc[0]
            extraction = row["ExtractionVolume_ha_m"]
            available = row["GroundWaterAvailability_ham"]
            stage = row["ExtractionStage_Percent"]
            rainfall = row.get("Rainfall_mm", 0)
            
            # Risk Level Determination
            if stage > 90:
                risk_level = "अत्याधिक दोहन" if lang == "hi" else "Over-Exploited"
                risk_color = "#e53935"
            elif stage > 70:
                risk_level = "संकटग्रस्त" if lang == "hi" else "Critical"
                risk_color = "#ffb300"
            else:
                risk_level = "सुरक्षित" if lang == "hi" else "Safe"
                risk_color = "#4caf50"
            
            # Create three columns for different charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 1. Water Balance Gauge Chart
                fig_gauge = px.pie(
                    values=[stage, 100-stage],
                    names=["Used", "Available"],
                    title=f"{msg['district']}" + (" का पानी उपयोग" if lang == "hi" else " Water Usage"),
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
                # 2. Water Available vs Extracted Bar Chart
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
                    title="पानी की मात्रा (ha.m)" if lang == "hi" else "Water Volume (ha.m)",
                    labels={"Volume": "ha.m", "Category": ""}
                )
                fig_bar.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col3:
                # 3. Risk Assessment Indicator
                categories = ["बहुत कम" if lang == "hi" else "Very Low",
                             "कम" if lang == "hi" else "Low", 
                             "मध्यम" if lang == "hi" else "Medium",
                             "उच्च" if lang == "hi" else "High",
                             "बहुत उच्च" if lang == "hi" else "Very High"]
                
                risk_scores = [10, 30, 50, 70, 90]
                colors = ["#4caf50", "#8bc34a", "#ffeb3b", "#ff9800", "#f44336"]
                
                # Find current risk position
                current_pos = 0
                for i, score in enumerate(risk_scores):
                    if stage <= score:
                        current_pos = i
                        break
                else:
                    current_pos = len(risk_scores) - 1
                
                # Create risk indicator
                fig_risk = px.bar(
                    x=categories, y=[100] * len(categories),
                    color=colors,
                    title="जोखिम स्तर" if lang == "hi" else "Risk Level"
                )
                
                # Highlight current risk
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
            
            # 4. Comprehensive Information Cards
            st.markdown("### 📋 " + ("विस्तृत जानकारी" if lang == "hi" else "Detailed Information"))
            
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            
            with info_col1:
                st.metric(
                    label="💧 " + ("उपलब्ध पानी" if lang == "hi" else "Available Water"),
                    value=f"{available:.1f} ha.m",
                    delta=f"{available - extraction:.1f} surplus" if available > extraction else f"{extraction - available:.1f} deficit"
                )
            
            with info_col2:
                st.metric(
                    label="🏭 " + ("निकाला गया पानी" if lang == "hi" else "Extracted Water"),
                    value=f"{extraction:.1f} ha.m",
                    delta=f"{stage:.1f}% of available"
                )
            
            with info_col3:
                safe_limit = available * 0.7
                st.metric(
                    label="⚠️ " + ("सुरक्षित सीमा" if lang == "hi" else "Safe Limit"),
                    value=f"{safe_limit:.1f} ha.m",
                    delta="Within limit" if extraction <= safe_limit else "Exceeded!"
                )
            
            with info_col4:
                if rainfall > 0:
                    st.metric(
                        label="🌧️ " + ("वार्षिक वर्षा" if lang == "hi" else "Annual Rainfall"),
                        value=f"{rainfall:.0f} mm",
                        delta="Good" if rainfall > 600 else "Low"
                    )
                else:
                    st.metric(
                        label="📊 " + ("जोखिम स्तर" if lang == "hi" else "Risk Level"),
                        value=risk_level,
                        delta=f"{stage:.1f}% usage"
                    )
            
            # 5. Recommendations based on risk level
            st.markdown("### 💡 " + ("सुझाव" if lang == "hi" else "Recommendations"))
            
            if stage > 90:
                recommendations = [
                    "🚨 तुरंत पानी की बचत करें" if lang == "hi" else "🚨 Immediate water conservation needed",
                    "💧 ड्रिप सिंचाई अपनाएं" if lang == "hi" else "💧 Switch to drip irrigation",
                    "🌾 कम पानी वाली फसलें उगाएं" if lang == "hi" else "🌾 Grow drought-resistant crops"
                ]
            elif stage > 70:
                recommendations = [
                    "⚠️ सावधानी बरतें" if lang == "hi" else "⚠️ Exercise caution",
                    "💧 पानी का सदुपयोग करें" if lang == "hi" else "💧 Use water efficiently",
                    "🔄 फसल चक्र अपनाएं" if lang == "hi" else "🔄 Practice crop rotation"
                ]
            else:
                recommendations = [
                    "✅ स्थिति अच्छी है" if lang == "hi" else "✅ Situation is good",
                    "📈 टिकाऊ खेती करें" if lang == "hi" else "📈 Practice sustainable farming",
                    "🌱 नई तकनीक अपनाएं" if lang == "hi" else "🌱 Adopt new technologies"
                ]
            
            for rec in recommendations:
                st.success(rec)
        else:
            st.warning("❌ " + ("इस जिले के लिए डेटा उपलब्ध नहीं है:" if lang == "hi" else "No data available for district:") + f" {msg['district']}")

# Top 3 Dashboard + Tips
st.markdown("### 📊 " + ("क्षेत्रीय स्थिति" if lang == "hi" else "Area Status"))
for _, row in df.sort_values("ExtractionStage_Percent", ascending=False).head(3).iterrows():
    level = ("अत्याधिक दोहन / Over-Exploited" if row["ExtractionStage_Percent"] > 90
             else "संकटग्रस्त / Critical" if row["ExtractionStage_Percent"] > 70
             else "सुरक्षित / Safe")
    boxcolor = "#ffbcbc" if row["ExtractionStage_Percent"] > 90 else "#ffe066" if row["ExtractionStage_Percent"] > 70 else "#c8f7c5"
    st.markdown(
        f"<div style='background:{boxcolor};border-radius:10px;padding:10px;margin-bottom:8px;'><b>{row['District']} ({row['State']})</b> | {level} - {row['ExtractionStage_Percent']:.1f}%</div>",
        unsafe_allow_html=True
    )

st.markdown("### 💡 " + ("किसान सुझाव" if lang == "hi" else "Farmer Tips"))
tips = [
    "💧 ड्रिप सिंचाई अपनाएँ।" if lang == "hi" else "💧 Use drip irrigation.",
    "🌾 कम पानी वाली फसलें उगाएँ।" if lang == "hi" else "🌾 Grow crops using less water.",
    "🚰 मिट्टी की नमी जाँचें।" if lang == "hi" else "🚰 Check soil moisture weekly.",
]

for tip in tips:
    st.info(tip)

st.markdown("<small>INGRES Groundwater Assistant @ SIH 2025</small>", unsafe_allow_html=True)

# Premium feature handling
if selected_feature == "AI Crop Water Calculator":
    st.warning("🚧 This feature is under development. Stay tuned!")

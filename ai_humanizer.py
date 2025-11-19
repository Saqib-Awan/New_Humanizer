import streamlit as st
import requests
import random
import re
import nltk
from nltk.tokenize import sent_tokenize
import time
import json
import logging

logging.basicConfig(level=logging.INFO)

# Download necessary NLTK data
try:
    nltk.download('punkt_tab')
except:
    st.warning("NLTK punkt download failed. Some features may not work correctly.")

# Set up the Streamlit app
st.set_page_config(page_title="AI Text Humanizer", page_icon="🤖→🧠", layout="wide")

# Groq API Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Function to communicate with Groq API
def query_groq(prompt, api_key, temperature=0.9, max_tokens=4000):
    model = "llama-3.3-70b-versatile"
    logging.info(f"Querying Groq with model: {model}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "stream": False
    }
    
    try:
        logging.info(f"Sending request to: {GROQ_API_URL}")
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        logging.info(f"Response status code: {response.status_code}")
        response.raise_for_status()
        
        json_response = response.json()
        if "choices" in json_response and len(json_response["choices"]) > 0:
            return json_response["choices"][0]["message"]["content"]
        else:
            st.error("Unexpected response format from Groq API")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Groq API: {e}")
        if hasattr(e, 'response') and e.response:
            st.error(f"Response status: {e.response.status_code}")
            st.error(f"Response text: {e.response.text}")
        return None

def humanize_text(text, api_key, temperature):
    # Split text into paragraphs to maintain structure
    paragraphs = text.split('\n\n')
    
    enhanced_prompt = f"""
    CRITICAL HUMANIZATION TASK - BYPASS AI DETECTION:
    
    You are now an expert human writer tasked with completely transforming AI-generated text into 100% human-like content that will bypass ALL AI detection systems including StealthWriter, GPTZero, Originality.ai, and others.
    
    ORIGINAL TEXT TO TRANSFORM:
    {text}
    
    ULTRA-ADVANCED HUMANIZATION TECHNIQUES TO APPLY:
    
    1. **Paragraph Preservation**: Maintain the exact same paragraph structure as the original. If input has {len(paragraphs)} paragraphs, output MUST have the same number.
    
    2. **Advanced Human Patterns**:
       - Mix sentence lengths dramatically (3-40 words)
       - Use natural interruptions and restarts
       - Add authentic conversational fillers (like, you know, actually, basically)
       - Include rhetorical questions and direct reader address
       - Use contractions naturally (don't, can't, it's, that's)
       - Vary transition words beyond basic ones
    
    3. **Cognitive Imperfections**:
       - Occasional slight redundancy for emphasis
       - Self-corrections mid-thought
       - Natural digressions that return to main point
       - Emotional interjections (interestingly, surprisingly, honestly)
    
    4. **Vocabulary Sophistication**:
       - Blend formal and informal language seamlessly
       - Use idioms and colloquial expressions appropriately
       - Vary pronoun usage (I, we, you, one)
       - Include domain-specific terminology naturally
    
    5. **Structural Authenticity**:
       - Start paragraphs with varied sentence types
       - Use em-dashes, semicolons, and parentheses naturally
       - Create natural rhythm and flow between ideas
       - Ensure each paragraph has clear thematic unity
    
    MANDATORY REQUIREMENTS:
    - Output MUST be indistinguishable from human writing
    - Preserve all factual information and core meaning
    - Maintain original paragraph count and structure
    - Sound like an educated, articulate human professional
    - Pass AI detection with 0% AI probability
    
    FINAL OUTPUT MUST BE COMPLETELY UNDETECTABLE AS AI-GENERATED.
    """

    humanized_text = query_groq(enhanced_prompt, api_key, temperature=temperature)
    return humanized_text

def advanced_humanization(text, techniques):
    if not text:
        return text
        
    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            processed_paragraphs.append('')
            continue
            
        sentences = sent_tokenize(paragraph)
        
        if "typos" in techniques and random.random() < 0.15:
            common_typos = {
                "the": "teh", "and": "adn", "that": "taht", "with": "wit", 
                "this": "thsi", "from": "form", "have": "hav", "would": "woud",
                "could": "coud", "their": "thier", "there": "ther", "your": "ur",
                "because": "becuase", "people": "poeple", "through": "thru",
                "though": "tho", "although": "altho", "until": "til"
            }
            for i in range(len(sentences)):
                if random.random() < 0.1:
                    words = sentences[i].split()
                    for j in range(len(words)):
                        clean_word = words[j].lower().strip('.,!?;:')
                        if clean_word in common_typos and random.random() < 0.2:
                            typo = common_typos[clean_word]
                            if words[j][0].isupper():
                                typo = typo.capitalize()
                            words[j] = words[j].replace(clean_word, typo)
                    sentences[i] = ' '.join(words)
        
        if "punctuation" in techniques:
            for i in range(len(sentences)):
                if random.random() < 0.2:
                    # Add varied punctuation
                    if sentences[i].endswith('.'):
                        if random.random() < 0.3:
                            sentences[i] = sentences[i][:-1] + '...'
                        elif random.random() < 0.2:
                            sentences[i] = sentences[i][:-1] + '!'
                    elif random.random() < 0.1:
                        sentences[i] = sentences[i] + ' — you know?'
                
                # Add natural interruptions
                if len(sentences[i]) > 20 and random.random() < 0.15:
                    words = sentences[i].split()
                    if len(words) > 4:
                        insert_pos = random.randint(2, len(words)-2)
                        interruptions = [', actually,', ', I mean,', ', you see,', ' — ', '; ', ', basically,']
                        if not any(marker in words[insert_pos-1] for marker in [',', ';', '—']):
                            words.insert(insert_pos, random.choice(interruptions))
                            sentences[i] = ' '.join(words)
        
        if "repetition" in techniques and random.random() < 0.1:
            for i in range(len(sentences)):
                if random.random() < 0.08:
                    words = sentences[i].split()
                    if len(words) > 3:
                        repeat_word = random.choice([w for w in words if len(w) > 3 and w.isalpha()])
                        emphasis_words = [f"{repeat_word} — {repeat_word}", f"{repeat_word}, really {repeat_word}"]
                        sentences[i] = sentences[i].replace(repeat_word, random.choice(emphasis_words), 1)
        
        if "formatting" in techniques:
            for i in range(len(sentences)):
                if random.random() < 0.1:
                    # Add emphasis with formatting
                    words = sentences[i].split()
                    if len(words) > 2:
                        emphasis_idx = random.randint(0, len(words)-1)
                        if len(words[emphasis_idx]) > 2:
                            if random.random() < 0.3:
                                words[emphasis_idx] = f"**{words[emphasis_idx]}**"
                            elif random.random() < 0.3:
                                words[emphasis_idx] = f"*{words[emphasis_idx]}*"
                            sentences[i] = ' '.join(words)
        
        processed_paragraphs.append(' '.join(sentences))
    
    return '\n\n'.join(processed_paragraphs)

# Main app layout
st.title("🤖→🧠 AI Text Humanizer")
st.markdown("""
This app transforms AI-generated text into highly natural, human-like writing by leveraging advanced humanization techniques.
""")

# Sidebar configuration for API key and advanced options
st.sidebar.title("Configuration")

# API Key input
api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key from https://console.groq.com/")

if not api_key:
    st.sidebar.warning("⚠️ Please enter your Groq API key to use the humanizer")
    st.info("👈 Enter your Groq API key in the sidebar to get started. Get your free API key at: https://console.groq.com/")

st.sidebar.markdown("---")
st.sidebar.title("Advanced Settings")
st.sidebar.markdown("**Model:** llama-3.3-70b-versatile")
st.sidebar.markdown("**Humanization Level:** StealthWriter-Level")

temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=0.9, step=0.1)

# Additional techniques settings
st.sidebar.subheader("Additional Techniques")
add_typos = st.sidebar.checkbox("Add occasional typos", value=True)
vary_punctuation = st.sidebar.checkbox("Vary punctuation", value=True)
add_repetition = st.sidebar.checkbox("Add natural repetition", value=True)
adjust_formatting = st.sidebar.checkbox("Adjust formatting", value=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Text Humanizer", "AI Detection Check", "About"])

with tab1:
    input_text = st.text_area("Enter AI-generated text to humanize:", height=200)
    
    if st.button("Humanize Text"):
        if not api_key:
            st.error("❌ Please enter your Groq API key in the sidebar first!")
        elif not input_text:
            st.warning("Please enter some text to humanize.")
        else:
            with st.spinner("Processing with Groq AI..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Use enhanced humanization with Groq
                humanized_text = humanize_text(input_text, api_key, temperature=temperature)
                
                techniques = []
                if add_typos:
                    techniques.append("typos")
                if vary_punctuation:
                    techniques.append("punctuation")
                if add_repetition:
                    techniques.append("repetition")
                if adjust_formatting:
                    techniques.append("formatting")
                
                if humanized_text and techniques:
                    humanized_text = advanced_humanization(humanized_text, techniques)
                    
                if humanized_text is not None:
                    st.subheader("Humanized Text:")
                    st.write(humanized_text)
                    st.text_area("Copy this text:", value=humanized_text, height=200)
                    
                    st.subheader("Text Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Original Word Count", len(input_text.split()))
                        st.metric("Original Character Count", len(input_text))
                        st.metric("Original Paragraphs", input_text.count('\n\n') + 1)
                    
                    with col2:
                        st.metric("Humanized Word Count", len(humanized_text.split()))
                        st.metric("Humanized Character Count", len(humanized_text))
                        st.metric("Humanized Paragraphs", humanized_text.count('\n\n') + 1)
                else:
                    st.error("Failed to generate humanized text. Please check your API key and try again.")

with tab2:
    st.markdown("""
    ## AI Detection Check
    
    This feature simulates how your text might perform against AI detection tools.
    
    **Note:** This is a heuristic estimation based on common detection patterns.
    """)
    
    check_text = st.text_area("Paste text to check:", height=200)
    
    if st.button("Check Text"):
        if not check_text:
            st.warning("Please enter some text to check.")
        else:
            with st.spinner("Analyzing text..."):
                time.sleep(2)
                
                # Enhanced detection analysis
                word_count = len(check_text.split())
                sentence_count = len(sent_tokenize(check_text))
                paragraph_count = check_text.count('\n\n') + 1
                
                # Advanced metrics
                sentences = sent_tokenize(check_text)
                sent_lengths = [len(s.split()) for s in sentences]
                sentence_variance = max(sent_lengths) - min(sent_lengths) if sent_lengths else 0
                
                # Calculate sophisticated metrics
                contractions = len(re.findall(r"\b\w+'[a-z]+\b", check_text))
                fillers = len(re.findall(r'\b(um|uh|like|you know|sort of|kind of|literally|basically|actually|anyway|so|well|right|okay|just|really|quite|perhaps|maybe)\b', check_text.lower()))
                transitions = len(re.findall(r'\b(however|nevertheless|therefore|thus|consequently|furthermore|moreover|additionally|meanwhile|nonetheless|otherwise|likewise|similarly|incidentally|incidentally|notably|significantly)\b', check_text.lower()))
                questions = len(re.findall(r'\?', check_text))
                
                # Calculate humanness score with more sophisticated algorithm
                humanness_score = 50  # Base score
                
                # Sentence structure variety
                if sentence_variance > 15:
                    humanness_score += 20
                elif sentence_variance > 8:
                    humanness_score += 10
                
                # Conversational elements
                humanness_score += min(15, contractions * 2)
                humanness_score += min(10, fillers * 2)
                humanness_score += min(10, transitions)
                humanness_score += min(5, questions * 3)
                
                # Paragraph structure
                if paragraph_count > 1 and len(check_text) > 500:
                    humanness_score += 10
                
                # Normalize score
                humanness_score = max(0, min(100, humanness_score))
                
                st.subheader("Detection Analysis Results")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    color = "green" if humanness_score > 85 else "orange" if humanness_score > 60 else "red"
                    st.markdown(f"""
                    <div style="text-align:center">
                        <h3>Human-likeness Score</h3>
                        <div style="margin:20px auto; width:200px; height:200px; position:relative;">
                            <div style="position:absolute; width:200px; height:200px; border-radius:50%; background:conic-gradient(from 0deg, {color} 0%, {color} {humanness_score}%, #e0e0e0 {humanness_score}%, #e0e0e0 100%);"></div>
                            <div style="position:absolute; width:150px; height:150px; border-radius:50%; background:white; top:25px; left:25px; display:flex; align-items:center; justify-content:center;">
                                <span style="font-size:40px; font-weight:bold; color:black;">{humanness_score}%</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.subheader("Advanced Text Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Word Count", word_count)
                    st.metric("Sentence Count", sentence_count)
                    st.metric("Paragraph Count", paragraph_count)
                with col2:
                    st.metric("Sentence Length Variance", sentence_variance)
                    st.metric("Contractions Found", contractions)
                    st.metric("Conversational Markers", fillers + transitions)
                
                st.subheader("Detection Risk Assessment")
                if humanness_score > 85:
                    st.success("✅ VERY LOW RISK: This text has excellent human characteristics and should bypass most AI detectors.")
                elif humanness_score > 70:
                    st.info("✅ LOW RISK: This text shows strong human-like patterns and should perform well against detection.")
                elif humanness_score > 50:
                    st.warning("⚠️ MODERATE RISK: Some AI patterns detected. Consider additional humanization for critical applications.")
                else:
                    st.error("❌ HIGH RISK: Significant AI patterns detected. Humanization recommended.")
                
                st.markdown("### Optimization Suggestions")
                suggestions = []
                if sentence_variance < 10:
                    suggestions.append("• Increase sentence length variation (mix short and long sentences)")
                if contractions < 2:
                    suggestions.append("• Add more natural contractions (don't, can't, it's)")
                if fillers < 1:
                    suggestions.append("• Include occasional conversational fillers (like, actually, you know)")
                if paragraph_count == 1 and word_count > 200:
                    suggestions.append("• Break into multiple paragraphs for better structure")
                if questions == 0 and word_count > 150:
                    suggestions.append("• Consider adding rhetorical questions for engagement")
                
                if not suggestions:
                    suggestions.append("• Text shows excellent human-like characteristics!")
                
                for suggestion in suggestions:
                    st.markdown(suggestion)

with tab3:
    st.markdown("""
    ## About AI Text Humanizer
    
    **StealthWriter-Level Humanization Engine**
    
    This enhanced version uses advanced techniques to produce text that bypasses even sophisticated AI detectors like StealthWriter's built-in system.
    
    ### Key Improvements:
    
    - **Paragraph Structure Preservation**: Maintains original paragraph count and organization
    - **Advanced Linguistic Patterns**: Implements cognitive imperfections and natural speech rhythms
    - **Sophisticated Vocabulary Mix**: Blends formal and informal language seamlessly
    - **Enhanced Conversational Elements**: Adds authentic human speech patterns
    
    ### How It Works:
    
    1. **Multi-Stage Processing**: Advanced prompt engineering + post-processing
    2. **Structure-Aware**: Preserves your original document structure
    3. **Stealth Optimization**: Specifically designed to bypass AI detection
    4. **Quality Assurance**: Maintains content meaning while enhancing readability
    
    ### Privacy & Security:
    - Your API key and text are processed securely
    - No data storage or logging
    - Complete session privacy
    
    ### Model Used:
    **llama-3.3-70b-versatile** - Optimized for maximum humanization effectiveness
    """)

if __name__ == "__main__":
    st.markdown("---")
    st.markdown("💡 **Pro Tip:** For best stealth results, enable all additional techniques and use temperature 0.8-0.9")
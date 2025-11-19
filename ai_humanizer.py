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
def query_groq(prompt, api_key, temperature=0.8, max_tokens=2000):
    model = "llama-3.1-405b-reasoning"  # Upgraded to more advanced model for better humanization
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
        "top_p": 1,
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

# Enhanced function to humanize text using Groq with multi-pass refinement
def humanize_text(text, api_key, temperature):
    # Initial advanced prompt for deep humanization
    initial_prompt = f"""
    Rewrite this text to make it indistinguishable from human writing, evading AI detectors completely. 
    Achieve 'Ultimate' humanization level by:
    - Preserving all original meaning, facts, and intent.
    - Using highly varied sentence lengths, structures, and rhythms to mimic natural thought flow.
    - Incorporating subtle filler words, hesitations, idioms, colloquialisms, and contractions organically.
    - Adding minor redundancies, asides, personal touches, or slight tangents that feel authentic without altering core content.
    - Mixing vocabulary: common words with occasional sophisticated ones, avoiding repetitive patterns.
    - Introducing natural imperfections like varied pacing, subtle opinions, or transitional phrases.
    - Ensuring the text flows conversationally with warmth, personality, and genuine expression.
    - Avoid perfect symmetry, lists, or robotic phrasing; make it read like casual human drafting.

    Original text:
    {text}

    Output only the rewritten text, nothing else.
    """
    
    # First pass: Initial humanization
    first_pass = query_groq(initial_prompt, api_key, temperature=temperature)
    if not first_pass:
        return None
    
    # Second pass: Refinement for evasion
    refinement_prompt = f"""
    Refine this rewritten text to maximize human-likeness and bypass AI detection tools entirely. 
    Focus on:
    - Enhancing variability in phrasing and word choice to eliminate any detectable patterns.
    - Adding more nuanced human elements: slight digressions, rhetorical questions, or emphatic expressions.
    - Ensuring irregular punctuation use, mixed formality, and authentic voice.
    - Making it feel like a human edited their own draft multiple times for natural flow.
    - Keep meaning identical, but amplify imperfections and personality.

    Text to refine:
    {first_pass}

    Output only the final refined text.
    """
    
    # Second pass: Refine for better evasion
    refined_text = query_groq(refinement_prompt, api_key, temperature=temperature + 0.1)  # Slightly higher temp for variability
    return refined_text

# Enhanced function for additional humanization techniques with more variations
def additional_humanization(text, techniques):
    if not text:
        return text
        
    sentences = sent_tokenize(text)
    
    # Expanded typos with more common errors
    if "typos" in techniques and random.random() < 0.5:
        common_typos = {
            "the": ["teh", "hte", "th"], "and": ["adn", "nad", "an"], "that": ["taht", "tht", "tha"], 
            "with": ["wtih", "wiht", "wit"], "this": ["tihs", "thsi", "thi"], "from": ["form", "fro", "frm"],
            "have": ["ahve", "hvae", "hav"], "would": ["woudl", "wuold", "wuld"], "could": ["cuold", "coudl", "coud"],
            "their": ["thier", "theri", "thir"], "there": ["tehre", "ther", "thre"], "your": ["yoru", "yuor", "yur"],
            "because": ["becuase", "becasue", "becuz"], "people": ["peopl", "pople"], "about": ["abotu", "abot"],
            "which": ["wich", "whcih"], "they": ["tehy", "thy"], "what": ["waht", "wht"]
        }
        for i in range(len(sentences)):
            if random.random() < 0.25:
                words = sentences[i].split()
                for j in range(len(words)):
                    word_lower = words[j].lower().rstrip('.,!?;')
                    if word_lower in common_typos and random.random() < 0.35:
                        typo = random.choice(common_typos[word_lower])
                        if words[j][0].isupper():
                            typo = typo.capitalize()
                        words[j] = typo + words[j][len(word_lower):]
                sentences[i] = ' '.join(words)
    
    # Enhanced punctuation variation
    if "punctuation" in techniques:
        for i in range(len(sentences)):
            if random.random() < 0.2:
                end_punct = random.choice(['..', '...', '!!', '??', '!?', '?!']) if sentences[i][-1] in '.!?' else ''
                sentences[i] = sentences[i].rstrip('.!?') + end_punct
            if len(sentences[i]) > 40 and random.random() < 0.6:
                words = sentences[i].split()
                if len(words) > 8:
                    splice_point = random.randint(4, len(words) - 4)
                    if not words[splice_point-1].endswith((',', ';', ':')):
                        punct = random.choice([',', ';', ':', ' -', ' --'])
                        words[splice_point-1] += punct
                    sentences[i] = ' '.join(words)
    
    # Enhanced repetition with phrases
    if "repetition" in techniques:
        for i in range(len(sentences)):
            if random.random() < 0.15:
                words = sentences[i].split()
                if len(words) > 5:
                    repeat_index = random.randint(0, len(words) - 2)
                    phrase = ' '.join(words[repeat_index:repeat_index+2])
                    if len(phrase) > 5 and not phrase.endswith((',', '.')):
                        words.insert(repeat_index + 2, phrase)
                    sentences[i] = ' '.join(words)
    
    # Enhanced formatting with more options
    if "formatting" in techniques:
        for i in range(len(sentences)):
            if random.random() < 0.1:
                words = sentences[i].split()
                if len(words) > 4:
                    emp_index = random.randint(0, len(words) - 1)
                    if len(words[emp_index]) > 4 and not re.search(r'[.,:;!?]', words[emp_index]):
                        style = random.choice(['upper', 'italic', 'bold'])
                        if style == 'upper':
                            words[emp_index] = words[emp_index].upper()
                        elif style == 'italic':
                            words[emp_index] = f"*{words[emp_index]}*"
                        else:
                            words[emp_index] = f"**{words[emp_index]}**"
                    sentences[i] = ' '.join(words)
    
    # New: Add random idioms or fillers
    fillers = ["you know", "like", "sort of", "kind of", "actually", "basically", "I mean", "well", "so", "right"]
    if random.random() < 0.3:
        for i in range(len(sentences)):
            if random.random() < 0.2 and len(sentences[i].split()) > 5:
                insert_point = random.randint(1, len(sentences[i].split()) - 1)
                words = sentences[i].split()
                words.insert(insert_point, random.choice(fillers))
                sentences[i] = ' '.join(words)
    
    return ' '.join(sentences)

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
st.sidebar.markdown("**Model:** llama-3.1-405b-reasoning")
st.sidebar.markdown("**Humanization Level:** Ultimate (fixed)")

# Temperature setting remains adjustable
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

# Additional techniques settings
st.sidebar.subheader("Additional Techniques")
add_typos = st.sidebar.checkbox("Add occasional typos", value=False)
vary_punctuation = st.sidebar.checkbox("Vary punctuation", value=True)
add_repetition = st.sidebar.checkbox("Add natural repetition", value=False)
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
                    humanized_text = additional_humanization(humanized_text, techniques)
                    
                if humanized_text is not None:
                    st.subheader("Humanized Text:")
                    st.write(humanized_text)
                    st.text_area("Copy this text:", value=humanized_text, height=200)
                    
                    st.subheader("Text Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Original Word Count", len(input_text.split()))
                        st.metric("Original Character Count", len(input_text))
                    
                    with col2:
                        st.metric("Humanized Word Count", len(humanized_text.split()))
                        st.metric("Humanized Character Count", len(humanized_text))
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
                word_count = len(check_text.split())
                avg_word_length = sum(len(word) for word in check_text.split()) / word_count if word_count > 0 else 0
                sentence_count = len(sent_tokenize(check_text))
                avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
                punctuation_count = len(re.findall(r'[.,;:!?]', check_text))
                words = check_text.lower().split()
                repeated_phrases = 0
                for i in range(len(words) - 3):
                    phrase = ' '.join(words[i:i+3])
                    if ' '.join(words[i+3:]).count(phrase) > 0:
                        repeated_phrases += 1
                humanness_score = 0
                sent_lengths = [len(s.split()) for s in sent_tokenize(check_text)]
                if sent_lengths:
                    avg_sentence_length = sum(sent_lengths) / len(sent_lengths)
                    sentence_length_variance = sum((x - avg_sentence_length) ** 2 for x in sent_lengths) / len(sent_lengths)
                    if sentence_length_variance > 10:
                        humanness_score += 20
                    elif sentence_length_variance > 5:
                        humanness_score += 10
                contractions = len(re.findall(r"\b\w+'[a-z]+\b", check_text))
                if contractions > 0:
                    humanness_score += min(15, contractions * 3)
                transitions = len(re.findall(r'\b(however|nevertheless|therefore|thus|consequently|furthermore|moreover|in addition|in fact|actually|basically|arguably|indeed|instead|meanwhile|nonetheless|otherwise|likewise|similarly|in other words|for example|for instance|in particular|specifically|especially|notably|chiefly|mainly|mostly)\b', check_text.lower()))
                humanness_score += min(15, transitions * 3)
                fillers = len(re.findall(r'\b(um|uh|like|you know|sort of|kind of|literally|basically|actually|anyway|so|well|right|okay|just)\b', check_text.lower()))
                humanness_score += min(10, fillers * 2)
                if sentence_count > 5 and abs(max(sent_lengths) - min(sent_lengths)) < 3:
                    humanness_score -= 20
                if repeated_phrases > 3:
                    humanness_score -= min(20, repeated_phrases * 2)
                humanness_score = max(0, min(100, humanness_score + 50))
                
                st.subheader("Detection Analysis Results")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="text-align:center">
                        <h3>Human-likeness Score</h3>
                        <div style="margin:20px auto; width:200px; height:200px; position:relative;">
                            <div style="position:absolute; width:200px; height:200px; border-radius:50%; background:conic-gradient(from 0deg, {'green' if humanness_score > 70 else 'orange' if humanness_score > 40 else 'red'} 0%, {'green' if humanness_score > 70 else 'orange' if humanness_score > 40 else 'red'} {humanness_score}%, #e0e0e0 {humanness_score}%, #e0e0e0 100%);"></div>
                            <div style="position:absolute; width:150px; height:150px; border-radius:50%; background:white; top:25px; left:25px; display:flex; align-items:center; justify-content:center;">
                                <span style="font-size:40px; font-weight:bold; color:black;">{humanness_score}%</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.subheader("Text Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Word Count", word_count)
                    st.metric("Average Word Length", f"{avg_word_length:.2f} characters")
                    st.metric("Sentence Count", sentence_count)
                with col2:
                    st.metric("Average Sentence Length", f"{avg_sentence_length:.2f} words")
                    st.metric("Punctuation Count", punctuation_count)
                    st.metric("Repeated Phrases", repeated_phrases)
                
                st.subheader("Detection Risk")
                if humanness_score > 70:
                    st.success("✅ LOW RISK: This text has a good chance of passing AI detection tools.")
                elif humanness_score > 40:
                    st.warning("⚠️ MODERATE RISK: This text may trigger some AI detection tools. Consider further humanization.")
                else:
                    st.error("❌ HIGH RISK: This text is likely to be flagged by AI detection tools. Significant humanization recommended.")
                
                st.markdown("### Improvement Suggestions")
                suggestions = []
                if avg_sentence_length > 20:
                    suggestions.append("• Try using shorter sentences in some places")
                if avg_sentence_length < 10:
                    suggestions.append("• Try using longer, more complex sentences occasionally")
                if contractions < 3 and word_count > 200:
                    suggestions.append("• Add more contractions (e.g., don't, can't, it's)")
                if fillers < 2 and word_count > 200:
                    suggestions.append("• Add a few natural filler words (like, actually, just)")
                if transitions < 3 and word_count > 200:
                    suggestions.append("• Add more transitional phrases (however, additionally, etc.)")
                if repeated_phrases > 3:
                    suggestions.append("• Reduce repetitive phrases and patterns")
                if not suggestions:
                    suggestions.append("• Text appears natural, no specific improvements needed")
                for suggestion in suggestions:
                    st.markdown(suggestion)

with tab3:
    st.markdown("""
    ## About AI Text Humanizer
    
    This tool uses Groq's powerful **llama-3.1-405b-reasoning** model to transform AI-generated text into highly natural, human-like writing.
    It applies advanced humanization techniques including:
    
    - Restructuring sentences with varied structures
    - Incorporating conversational fillers and transitions
    - Introducing natural redundancies and minor imperfections
    - Balancing formal and informal language usage
    
    ### How it works
    
    1. Your AI-generated text is sent to Groq's cloud API with your API key.
    2. The llama-3.1-405b-reasoning model rewrites the text using advanced humanization prompts.
    3. Additional post-processing techniques further refine the natural flow.
    4. The output maintains the original meaning but reads with genuine human expression.
    
    ### Privacy & Security
    
    - Your API key is never stored and only used for the current session
    - Text is processed through Groq's secure API
    - No conversation history is saved
    
    ### Getting Your API Key
    
    1. Visit [Groq Console](https://console.groq.com/)
    2. Sign up or log in to your account
    3. Navigate to API Keys section
    4. Create a new API key
    5. Copy and paste it into the sidebar
    
    ### Requirements
    
    - Python 3.7+
    - Groq API key (free tier available)
    - NLTK library
    - Streamlit
    - Internet connection
    
    ### Model Information
    
    **llama-3.1-405b-reasoning** is one of the most advanced open models available, offering:
    - 405 billion parameters for nuanced understanding
    - Excellent at maintaining context and meaning
    - Strong performance on text rewriting tasks
    - Fast inference times through Groq's infrastructure
    """)

if __name__ == "__main__":
    st.markdown("---")
    st.markdown("💡 **Tip:** For best results, enable 'Vary punctuation' and 'Adjust formatting' in the sidebar settings.")
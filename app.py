import streamlit as st
import requests
import pandas as pd
import re
import json
import time
import base64
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
import openai
from docx2python import docx2python
import docx2txt

# Initialize session state for storing processing state
if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = []

# Configure page settings
st.set_page_config(
    page_title="YouTube Downloader",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# YouTube Downloader\nDownload comments and transcripts from multiple YouTube videos."
    }
)

# Load API keys from st.secrets if available
@st.cache_resource
def get_api_keys():
    config = {"youtube_api_key": None, "openai_api_key": None}
    try:
        if "YOUTUBE_API_KEY" in st.secrets:
            config["youtube_api_key"] = st.secrets["YOUTUBE_API_KEY"]
        if "OPENAI_API_KEY" in st.secrets:
            config["openai_api_key"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return config

api_keys = get_api_keys()
YOUTUBE_API_KEY = api_keys["youtube_api_key"]
OPENAI_API_KEY = api_keys["openai_api_key"]

# Load angle_bot_prompt.txt
def load_analysis_prompt():
    try:
        with open("angle_bot_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading analysis prompt file: {str(e)}")
        return "Analyze the transcript and comments from this YouTube video. Identify key themes, interesting insights, and what seems to resonate with viewers."

ANALYSIS_PROMPT = load_analysis_prompt()

# Create downloads directory if it doesn't exist
os.makedirs("downloads", exist_ok=True)

# Apply custom CSS for dark mode and modern UI
st.markdown("""
<style>
    /* Dark mode theme */
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    /* Card-like containers */
    .card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Header styling */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    /* Accent colors for highlights */
    .highlight {
        color: #ff5252;
    }
    /* Button hover effects */
    .stButton>button:hover {
        background-color: #ff5252;
        border-color: #ff5252;
    }
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #ff5252;
    }
    /* Custom divider */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #ff5252, transparent);
        margin: 20px 0;
        border-radius: 10px;
    }
    /* Download button styling */
    .download-button {
        display: inline-block;
        background-color: #ff5252;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        text-decoration: none;
        font-weight: bold;
        margin: 10px 0;
        transition: background-color 0.3s;
    }
    .download-button:hover {
        background-color: #ff3333;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to remove invalid filename characters
def clean_filename(filename: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return cleaned.strip('. ')[:100]

# Helper function: extract video ID from a YouTube URL
def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Helper function: get binary file download link
def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate a link to download a binary file."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}" class="download-button">{file_label}</a>'

# Helper function: fetch comments for a video using the YouTube Data API v3
def get_comments(video_id: str, api_key: str, page_token: Optional[str] = None) -> Dict:
    api_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet,replies",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": api_key,
        "order": "relevance"
    }
    if page_token:
        params["pageToken"] = page_token
    try:
        response = requests.get(api_url, params=params, timeout=10)
        if response.status_code != 200:
            error_message = f"HTTP error {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data and "message" in error_data["error"]:
                    error_message = error_data["error"]["message"]
            except Exception:
                pass
            return {"success": False, "error": error_message}
        return {"success": True, "data": response.json()}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. YouTube API might be slow or experiencing issues."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection error. Please check your internet connection."}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

# Helper function: process comment data from the API response
def process_comment_data(data: Dict) -> Tuple[List[Dict], Optional[str]]:
    comments = []
    for item in data.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        comments.append({
            "author": snippet.get("authorDisplayName", "Unknown"),
            "text": snippet.get("textDisplay", ""),
            "likeCount": snippet.get("likeCount", 0),
            "publishedAt": snippet.get("publishedAt", ""),
            "isReply": False
        })
        if "replies" in item:
            for reply in item["replies"]["comments"]:
                reply_snippet = reply["snippet"]
                comments.append({
                    "author": reply_snippet.get("authorDisplayName", "Unknown"),
                    "text": reply_snippet.get("textDisplay", ""),
                    "likeCount": reply_snippet.get("likeCount", 0),
                    "publishedAt": reply_snippet.get("publishedAt", ""),
                    "isReply": True
                })
    next_page_token = data.get("nextPageToken")
    return comments, next_page_token

# Helper function: format API timestamps to a readable date
def format_timestamp(timestamp: str) -> str:
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%b %d, %Y")
    except Exception:
        return timestamp

# Helper function: get basic video information
def get_video_info(video_id: str, api_key: str) -> Dict:
    api_url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": api_key
    }
    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        return {"success": False, "error": f"HTTP error {response.status_code}: {response.text}"}
    data = response.json()
    if not data.get("items"):
        return {"success": False, "error": "Video not found"}
    video_data = data["items"][0]
    snippet = video_data["snippet"]
    statistics = video_data["statistics"]
    return {
        "success": True,
        "title": snippet.get("title", "Unknown Title"),
        "channel": snippet.get("channelTitle", "Unknown Channel"),
        "published": snippet.get("publishedAt", "Unknown Date"),
        "comment_count": statistics.get("commentCount", "0"),
        "view_count": statistics.get("viewCount", "0"),
        "like_count": statistics.get("likeCount", "0"),
        "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", "")
    }

# Helper function: fetch all comments with pagination
def get_all_comments_with_callback(video_id: str, api_key: str, status_text=None) -> List[Dict]:
    all_comments = []
    next_page_token = None
    page_num = 1
    max_pages = 500
    error_count = 0
    max_errors = 3
    while page_num <= max_pages:
        if status_text:
            status_text.text(f"Fetching page {page_num} of comments...")
        result = get_comments(video_id, api_key, next_page_token)
        if not result["success"]:
            error_count += 1
            if error_count >= max_errors:
                if status_text:
                    status_text.error("Too many errors. Stopping comment retrieval.")
                break
            time.sleep(2)
            continue
        error_count = 0
        comments, next_page_token = process_comment_data(result["data"])
        all_comments.extend(comments)
        if not next_page_token:
            break
        time.sleep(0.5)
        page_num += 1
    return all_comments

# Helper function: create a download link for a DataFrame as CSV
def create_download_link(df: pd.DataFrame, filename: str, link_text: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{link_text}</a>'
    return href

# Function to download audio from a YouTube video
def download_youtube_audio(youtube_url, output_path="./downloads", format="mp3", quality="192"):
    """Download audio from a YouTube video using yt-dlp."""
    try:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            return {"success": False, "error": "Could not extract video ID from URL"}
            
        video_info = get_video_info(video_id, YOUTUBE_API_KEY)
        if not video_info["success"]:
            return video_info
            
        title = video_info["title"]
        base_filename = clean_filename(title)
        output_file = f"{output_path}/{base_filename}.{format}"
        
        command = [
            "yt-dlp", "-f", "bestaudio",
            "--extract-audio", "--audio-format", format,
            "--audio-quality", quality,
            "-o", output_file,
            "--no-playlist", "--quiet", "--progress",
            youtube_url
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Downloading audio...")
        
        process = subprocess.run(command, capture_output=True, text=True)
        
        if process.returncode != 0:
            return {"success": False, "error": process.stderr}
            
        progress_bar.progress(100)
        status_text.empty()
        
        if not os.path.exists(output_file):
            potential_files = list(Path(output_path).glob(f"{base_filename}.*"))
            if potential_files:
                output_file = str(potential_files[0])
            else:
                return {"success": False, "error": "File not found after download"}
                
        return {
            "success": True,
            "file_path": output_file,
            "title": title,
            "file_name": os.path.basename(output_file)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Function to transcribe audio using OpenAI's Whisper
def transcribe_with_whisper(audio_file_path, api_key):
    """Convert audio to text using OpenAI's Whisper API."""
    try:
        import openai
        openai.api_key = api_key
        
        with open(audio_file_path, "rb") as audio_file:
            transcript_response = openai.Audio.transcribe("whisper-1", audio_file)
            
        transcript_text = transcript_response.get("text", "")
        return {"success": True, "transcript": transcript_text}
    except Exception as e:
        return {"success": False, "error": f"Error transcribing audio with OpenAI: {str(e)}"}
        
# Function to extract text content from DOCX file
def extract_text_from_docx(docx_path):
    """Extract all text content from a DOCX file."""
    try:
        return docx2txt.process(docx_path)
    except Exception as e:
        st.error(f"Error extracting text from document: {str(e)}")
        return None
        
# Function to analyze document with OpenAI's ChatGPT
def analyze_document_with_chatgpt(document_text, prompt_text, api_key):
    """Analyze document content using ChatGPT."""
    try:
        openai.api_key = api_key
        
        # Combine the prompt and document text
        full_prompt = f"{prompt_text}\n\nHere is the document containing the transcript and comments:\n\n{document_text}"
        
        # Make API call to ChatGPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",  # Using a model with larger context window
            messages=[
                {"role": "system", "content": "You are an expert analyst of YouTube video content and audience engagement."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        
        # Extract and return the analysis
        analysis = response.choices[0].message.content
        return {"success": True, "analysis": analysis}
    except Exception as e:
        return {"success": False, "error": f"Error analyzing document with ChatGPT: {str(e)}"}

# Function to create a Word document with transcript and comments
def create_word_doc(video_info, transcript, comments, output_path="./downloads"):
    """Create a Word document with the video transcript and comments."""
    try:
        # Create a new Document
        doc = Document()
        
        # Add document title
        title = doc.add_heading(f"{video_info['title']}", level=1)
        title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        
        # Add video metadata
        doc.add_paragraph(f"Channel: {video_info['channel']}")
        doc.add_paragraph(f"Published: {video_info['published']}")
        
        # Add transcript section
        doc.add_heading("Transcript", level=2)
        doc.add_paragraph(transcript)
        
        # Add comments section
        doc.add_heading(f"Comments ({len(comments)})", level=2)
        
        # Add table for comments
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Set table header
        header_cells = table.rows[0].cells
        header_cells[0].text = "Author"
        header_cells[1].text = "Comment"
        header_cells[2].text = "Likes"
        
        # Add comments to the table
        for comment in comments:
            row_cells = table.add_row().cells
            row_cells[0].text = comment["author"]
            row_cells[1].text = comment["text"]
            row_cells[2].text = str(comment["likeCount"])
        
        # Save the document
        safe_title = clean_filename(video_info["title"])
        file_path = f"{output_path}/{safe_title}_transcript_comments.docx"
        doc.save(file_path)
        
        return {"success": True, "file_path": file_path}
    except Exception as e:
        return {"success": False, "error": f"Error creating Word document: {str(e)}"}

# Main Application UI
st.markdown('<h1 class="highlight">YouTube Comments & Transcript Downloader</h1>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# API Key Input Section
with st.expander("API Keys (Required)", expanded=not (YOUTUBE_API_KEY and OPENAI_API_KEY)):
    col1, col2 = st.columns(2)
    
    with col1:
        youtube_api_key_input = st.text_input(
            "YouTube API Key", 
            value=YOUTUBE_API_KEY if YOUTUBE_API_KEY else "",
            type="password", 
            help="Required for fetching video info and comments"
        )
    
    with col2:
        openai_api_key_input = st.text_input(
            "OpenAI API Key", 
            value=OPENAI_API_KEY if OPENAI_API_KEY else "",
            type="password", 
            help="Required for audio transcription with Whisper"
        )
    
    if not youtube_api_key_input:
        st.warning("⚠️ YouTube API key is required.")
    
    if not openai_api_key_input:
        st.warning("⚠️ OpenAI API key is required for transcription.")

# Input area for multiple YouTube video URLs
st.markdown("### Enter up to 5 YouTube video URLs (one per line):")
video_urls_text = st.text_area("YouTube Video URLs", placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...")

if st.button("Process Videos (Download Audio, Generate Transcript & Comments)"):
    # Validate API keys
    if not youtube_api_key_input:
        st.error("YouTube API key is required. Please enter your API key in the section above.")
        st.stop()
    
    if not openai_api_key_input:
        st.error("OpenAI API key is required for transcription. Please enter your API key in the section above.")
        st.stop()
    
    # Split the input text into lines and take up to 5 non-empty URLs
    video_urls = [url.strip() for url in video_urls_text.splitlines() if url.strip()]
    
    if not video_urls:
        st.error("Please enter at least one YouTube URL.")
    elif len(video_urls) > 5:
        st.error("Please enter no more than 5 URLs. Processing multiple videos consumes significant API resources.")
    else:
        # Clear previously processed videos
        st.session_state.processed_videos = []
        
        for url in video_urls:
            st.markdown(f"### Processing: {url}")
            with st.spinner(f"Processing {url} ..."):
                video_id = extract_video_id(url)
                if not video_id:
                    st.error(f"Could not extract video ID from the URL: {url}")
                    continue
                
                # Step 1: Fetch video info
                video_info = get_video_info(video_id, youtube_api_key_input)
                if not video_info.get("success"):
                    st.error(f"Error fetching video info: {video_info.get('error', 'Unknown error')}")
                    continue
                
                # Display video info
                st.subheader(f"📺 {video_info['title']}")
                st.write(f"Channel: {video_info['channel']}")
                st.write(f"View Count: {int(video_info['view_count']):,}")
                
                # Step 2: Download audio
                audio_status = st.empty()
                audio_status.info("Downloading audio...")
                audio_result = download_youtube_audio(url)
                
                if not audio_result.get("success"):
                    st.error(f"Error downloading audio: {audio_result.get('error', 'Unknown error')}")
                    continue
                
                audio_status.success("✅ Audio downloaded successfully")
                
                # Step 3: Generate transcript with Whisper
                transcript_status = st.empty()
                transcript_status.info("Generating transcript with OpenAI Whisper...")
                
                transcript_result = transcribe_with_whisper(audio_result["file_path"], openai_api_key_input)
                
                if not transcript_result.get("success"):
                    st.error(f"Error generating transcript: {transcript_result.get('error', 'Unknown error')}")
                    continue
                
                transcript_status.success("✅ Transcript generated successfully")
                
                # Step 4: Fetch all comments
                comment_status = st.empty()
                comment_status.info("Fetching comments...")
                
                comments = get_all_comments_with_callback(video_id, youtube_api_key_input, status_text=comment_status)
                
                if not comments:
                    comment_status.warning("No comments retrieved or comments are disabled for this video.")
                    # Continue processing even if no comments are found
                else:
                    comment_status.success(f"✅ Fetched {len(comments):,} comments")
                
                # Step 5: Create Word document with transcript and comments
                doc_status = st.empty()
                doc_status.info("Creating Word document...")
                
                doc_result = create_word_doc(
                    video_info, 
                    transcript_result["transcript"], 
                    comments
                )
                
                if not doc_result.get("success"):
                    st.error(f"Error creating document: {doc_result.get('error', 'Unknown error')}")
                    continue
                
                doc_status.success("✅ Word document created successfully")
                
                # Create download link for the Word document
                safe_title = clean_filename(video_info["title"])
                doc_link = get_binary_file_downloader_html(
                    doc_result["file_path"], 
                    f"{safe_title}_transcript_comments.docx"
                )
                
                # Store processed video data in session state
                st.session_state.processed_videos.append({
                    "video_id": video_id,
                    "title": video_info["title"],
                    "safe_title": safe_title,
                    "doc_path": doc_result["file_path"],
                    "url": url
                })
                
                st.markdown(doc_link, unsafe_allow_html=True)
                
                # Initialize session state for this video if not exists
                video_state_key = f"video_state_{video_id}"
                if video_state_key not in st.session_state:
                    st.session_state[video_state_key] = {
                        "analyzed": False,
                        "analysis_result": None,
                        "show_analysis": False
                    }
                
                # Create a button for analysis instead of a checkbox
                analyze_button_key = f"analyze_btn_{video_id}"
                analyze_col, status_col = st.columns([1, 3])
                
                with analyze_col:
                    if not st.session_state[video_state_key]["analyzed"]:
                        if st.button("Analyze with ChatGPT", key=analyze_button_key):
                            with status_col:
                                analysis_status = st.empty()
                                analysis_status.info("Analyzing video content with ChatGPT...")
                                
                                # Extract text from document
                                doc_text = extract_text_from_docx(doc_result["file_path"])
                                
                                if doc_text:
                                    # Analyze with ChatGPT
                                    analysis_result = analyze_document_with_chatgpt(
                                        doc_text, 
                                        ANALYSIS_PROMPT, 
                                        openai_api_key_input
                                    )
                                    
                                    # Store in session state
                                    st.session_state[video_state_key]["analyzed"] = True
                                    st.session_state[video_state_key]["analysis_result"] = analysis_result
                                    st.session_state[video_state_key]["show_analysis"] = True
                                    
                                    # Rerun to show results
                                    st.experimental_rerun()
                                else:
                                    analysis_status.error("Could not extract text from document for analysis.")
                    else:
                        # Already analyzed, show toggle button
                        if st.button("Show/Hide Analysis", key=f"toggle_{video_id}"):
                            st.session_state[video_state_key]["show_analysis"] = not st.session_state[video_state_key]["show_analysis"]
                            st.experimental_rerun()
                
                # Display analysis if available and show_analysis is True
                if st.session_state[video_state_key]["analyzed"] and st.session_state[video_state_key]["show_analysis"]:
                    analysis_result = st.session_state[video_state_key]["analysis_result"]
                    
                    if analysis_result["success"]:
                        status_col.success("✅ Analysis completed successfully")
                        
                        # Display the analysis in a nice format
                        st.subheader("ChatGPT Analysis")
                        
                        with st.expander("View Analysis", expanded=True):
                            st.markdown(analysis_result["analysis"])
                            
                        # Create a download link for the analysis
                        analysis_text = analysis_result["analysis"]
                        analysis_filename = f"{safe_title}_analysis.txt"
                        
                        # Save analysis to file
                        analysis_path = f"downloads/{analysis_filename}"
                        with open(analysis_path, "w", encoding="utf-8") as f:
                            f.write(analysis_text)
                            
                        # Create download link
                        analysis_link = get_binary_file_downloader_html(
                            analysis_path,
                            analysis_filename
                        )
                        
                        st.markdown(analysis_link, unsafe_allow_html=True)
                    else:
                        st.error(f"Error analyzing content: {analysis_result['error']}")
                
                # Add a divider between videos
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

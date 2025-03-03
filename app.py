import streamlit as st
import requests
import pandas as pd
import re
import json
import time
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Configure page settings
st.set_page_config(
    page_title="YouTube Comments Downloader",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# YouTube Comments Downloader\nDownload all comments from multiple YouTube videos."
    }
)

# Load API keys from st.secrets if available
@st.cache_resource
def get_api_keys():
    config = {"youtube_api_key": None}
    try:
        if "YOUTUBE_API_KEY" in st.secrets:
            config["youtube_api_key"] = st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        pass
    return config

api_keys = get_api_keys()
YOUTUBE_API_KEY = api_keys["youtube_api_key"]

# Verify that API key exists
if not YOUTUBE_API_KEY:
    st.error("⚠️ YouTube API key not found in secrets. Please add it to your .streamlit/secrets.toml file.")
    st.stop()

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
    return re.sub(r'[\\/*?:"<>|]', "", filename)

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

# Main Application UI
st.markdown('<h1 class="highlight">YouTube Comments Downloader</h1>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Input area for multiple YouTube video URLs
st.markdown("### Enter up to 10 YouTube video URLs (one per line):")
video_urls_text = st.text_area("YouTube Video URLs", placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...")

if st.button("Download Comments for Videos"):
    # Split the input text into lines and take up to 10 non-empty URLs
    video_urls = [url.strip() for url in video_urls_text.splitlines() if url.strip()]
    if not video_urls:
        st.error("Please enter at least one YouTube URL.")
    elif len(video_urls) > 10:
        st.error("Please enter no more than 10 URLs.")
    else:
        for url in video_urls:
            with st.spinner(f"Processing {url} ..."):
                video_id = extract_video_id(url)
                if not video_id:
                    st.error(f"Could not extract video ID from the URL: {url}")
                    continue
                # Fetch video info
                video_info = get_video_info(video_id, YOUTUBE_API_KEY)
                if not video_info.get("success"):
                    st.error(f"Error fetching video info for {url}: {video_info.get('error', 'Unknown error')}")
                    continue
                title = video_info["title"]
                safe_title = clean_filename(title).replace(" ", "_")
                # Fetch all comments
                status_text = st.empty()
                status_text.info(f"Fetching comments for '{title}'...")
                comments = get_all_comments_with_callback(video_id, YOUTUBE_API_KEY, status_text=status_text)
                if not comments:
                    st.warning(f"No comments retrieved for '{title}'.")
                    continue
                # Create a DataFrame and a download link
                df_comments = pd.DataFrame(comments)
                if "publishedAt" in df_comments.columns:
                    df_comments["publishedAt"] = df_comments["publishedAt"].apply(format_timestamp)
                filename = f"{safe_title}_comments.csv"
                download_link = create_download_link(df_comments, filename, f"Download '{title}' Comments")
                st.success(f"Fetched {len(comments):,} comments for '{title}'.")
                st.markdown(download_link, unsafe_allow_html=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

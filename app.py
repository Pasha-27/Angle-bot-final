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
        'About': "# YouTube Comments Downloader\nDownload all comments from any YouTube video sorted by like count."
    }
)

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
    
    /* Improved spacing */
    .spacer {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #ff5252;
    }

    /* Improved table styling */
    .dataframe {
        font-size: 14px;
    }
    
    /* Custom divider */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #ff5252, transparent);
        margin: 20px 0;
        border-radius: 10px;
    }

    /* Comment styling */
    .comment-card {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 3px solid #ff5252;
        word-wrap: break-word;
    }
    
    .comment-card.reply-comment {
        margin-left: 25px;
        border-left: 3px solid #4d7aff;
        background-color: #252525;
    }
    
    .author-name {
        color: #ff5252;
        font-weight: bold;
    }
    
    .reply-comment .author-name {
        color: #4d7aff;
    }
    
    .comment-text {
        margin-top: 8px;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    
    .like-count {
        color: #aaaaaa;
        font-size: 0.8rem;
    }
    
    .timestamp {
        color: #888888;
        font-size: 0.8rem;
        display: inline-block;
        margin-left: 10px;
    }
    
    /* Pagination styling */
    .pagination {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .pagination-btn {
        margin: 0 5px;
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
    
    /* Error and warning messages */
    .warning-box {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 4px 4px 0;
    }
    
    .error-box {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #f44336;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def extract_video_id(url: str) -> Optional[str]:
    """Extract the video ID from a YouTube URL."""
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

def get_comments(video_id: str, api_key: str, page_token: Optional[str] = None) -> Dict:
    """Fetch comments for a video using YouTube Data API v3."""
    api_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet,replies",
        "videoId": video_id,
        "maxResults": 100,  # Maximum allowed by the API
        "textFormat": "plainText",
        "key": api_key,
        "order": "relevance"  # This helps get more popular comments first
    }
    
    if page_token:
        params["pageToken"] = page_token
    
    try:
        response = requests.get(api_url, params=params, timeout=10)  # Add timeout for better error handling
        
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

def process_comment_data(data: Dict) -> Tuple[List[Dict], Optional[str]]:
    """Extract comment information from API response."""
    comments = []
    
    for item in data.get("items", []):
        # Process top-level comment
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        author = snippet.get("authorDisplayName", "Unknown")
        text = snippet.get("textDisplay", "")
        like_count = snippet.get("likeCount", 0)
        published_at = snippet.get("publishedAt", "")
        
        comments.append({
            "author": author,
            "text": text,
            "likeCount": like_count,
            "publishedAt": published_at,
            "isReply": False
        })
        
        # Process replies if any
        if "replies" in item:
            for reply in item["replies"]["comments"]:
                reply_snippet = reply["snippet"]
                reply_author = reply_snippet.get("authorDisplayName", "Unknown")
                reply_text = reply_snippet.get("textDisplay", "")
                reply_like_count = reply_snippet.get("likeCount", 0)
                reply_published_at = reply_snippet.get("publishedAt", "")
                
                comments.append({
                    "author": reply_author,
                    "text": reply_text,
                    "likeCount": reply_like_count,
                    "publishedAt": reply_published_at,
                    "isReply": True
                })
    
    # Check if there are more pages of comments
    next_page_token = data.get("nextPageToken")
    
    return comments, next_page_token

def format_timestamp(timestamp: str) -> str:
    """Format API timestamp to a readable date."""
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%b %d, %Y")
    except Exception:
        return timestamp

def get_video_info(video_id: str, api_key: str) -> Dict:
    """Get basic information about the video."""
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

def get_all_comments(video_id: str, api_key: str, progress_bar=None, status_text=None) -> List[Dict]:
    """Fetch all comments by paging through results."""
    all_comments = []
    next_page_token = None
    page_num = 1
    max_pages = 500  # Safety limit for extremely popular videos
    error_count = 0
    max_errors = 3  # Maximum consecutive errors before giving up
    
    while True and page_num <= max_pages:
        # Update progress status
        if status_text:
            status_text.text(f"Fetching page {page_num} of comments...")
        
        try:
            # Fetch comments for current page
            result = get_comments(video_id, api_key, next_page_token)
            
            if not result["success"]:
                error_count += 1
                error_message = result.get('error', 'Unknown error')
                
                if error_count >= max_errors:
                    if status_text:
                        status_text.error(f"Too many errors: {error_message}. Stopping comment retrieval.")
                    break
                
                if "quota" in error_message.lower():
                    if status_text:
                        status_text.error(f"API quota exceeded: {error_message}")
                    break
                
                if status_text:
                    status_text.warning(f"Error on page {page_num}: {error_message}. Retrying in 2 seconds...")
                
                # Wait longer before retry for rate limiting issues
                time.sleep(2)
                continue
            
            # Reset error counter on success
            error_count = 0
            
            # Process comments from current page
            comments, next_page_token = process_comment_data(result["data"])
            all_comments.extend(comments)
            
            # Update progress
            if progress_bar and "commentCount" in st.session_state:
                try:
                    total_comments = int(st.session_state["commentCount"])
                    if total_comments > 0:
                        progress = min(1.0, len(all_comments) / total_comments)
                        progress_bar.progress(progress)
                except (ValueError, TypeError):
                    # If commentCount is invalid, show progress based on pages
                    progress_bar.progress(min(1.0, page_num / 50))
            
            # Display warning for large comment retrieval
            if page_num % 10 == 0 and status_text:
                status_text.info(f"Downloaded {len(all_comments)} comments so far...")
            
            # Check if we've reached the last page
            if not next_page_token:
                break
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            page_num += 1
            
            # Safety check for extremely large comment sets
            if page_num == max_pages and status_text:
                status_text.warning(f"Reached maximum page limit ({max_pages}). Some comments may not be retrieved.")
        
        except Exception as e:
            error_count += 1
            if status_text:
                status_text.warning(f"Unexpected error on page {page_num}: {str(e)}. Retrying...")
            
            if error_count >= max_errors:
                if status_text:
                    status_text.error(f"Too many consecutive errors. Stopping comment retrieval.")
                break
            
            time.sleep(2)  # Wait before retry
    
    return all_comments

def create_download_link(df: pd.DataFrame, filename: str, link_text: str) -> str:
    """Generate a download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{link_text}</a>'
    return href

# Main application layout
st.markdown('<h1 class="highlight">YouTube Comments Downloader</h1>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Sidebar for API key and options
with st.sidebar:
    st.markdown('<h3>⚙️ Settings</h3>', unsafe_allow_html=True)
    
    api_key = st.text_input(
        "YouTube API Key",
        type="password",
        help="Get your API key from the Google Developer Console"
    )
    
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    
    # Sorting options
    st.markdown('<h3>🔄 Sort Options</h3>', unsafe_allow_html=True)
    sort_by = st.radio(
        "Sort comments by",
        options=["Most Likes", "Newest First", "Oldest First"],
        index=0
    )
    
    # Filtering options
    st.markdown('<h3>🔍 Filter Options</h3>', unsafe_allow_html=True)
    
    min_likes = st.number_input(
        "Minimum likes",
        min_value=0,
        value=0,
        step=1
    )
    
    include_replies = st.checkbox("Include replies", value=True)
    
    # Footer info
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.8rem; color: #888888;">
        This app uses the YouTube Data API v3.<br>
        API quota limit: 10,000 units per day.<br>
        Each comment fetch uses ~1 unit.
    </div>
    """, unsafe_allow_html=True)

# Main content area
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    youtube_url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        process_button = st.button("Get Comments", type="primary")
    with col2:
        if st.button("Clear Results"):
            # Reset session state to clear results
            for key in ['comments', 'video_info', 'commentCount']:
                if key in st.session_state:
                    del st.session_state[key]
            st.experimental_rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

    # Process video when the button is clicked
    if process_button and youtube_url and api_key:
        # Extract video ID
        video_id = extract_video_id(youtube_url)
        
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
        else:
            # Get video information first
            with st.spinner("Fetching video information..."):
                video_info = get_video_info(video_id, api_key)
                
                if not video_info["success"]:
                    st.error(f"Error: {video_info.get('error', 'Unable to fetch video information')}")
                else:
                    st.session_state["video_info"] = video_info
                    st.session_state["commentCount"] = video_info["comment_count"]
                    
                    # Display video information
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.image(video_info["thumbnail"], use_column_width=True)
                        
                    with col2:
                        st.markdown(f"### {video_info['title']}")
                        st.markdown(f"**Channel:** {video_info['channel']}")
                        st.markdown(f"**Published:** {format_timestamp(video_info['published'])}")
                        st.markdown(f"**Views:** {int(video_info['view_count']):,}")
                        st.markdown(f"**Likes:** {int(video_info['like_count']):,}")
                        st.markdown(f"**Comments:** {int(video_info['comment_count']):,}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Fetch all comments with progress tracking
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.markdown("### Downloading Comments")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text(f"Fetching comments... This might take a while for videos with many comments.")
                    
                    # Get all comments
                    all_comments = get_all_comments(video_id, api_key, progress_bar, status_text)
                    
                    # Store in session state
                    st.session_state["comments"] = all_comments
                    
                    # Update status
                    progress_bar.progress(1.0)
                    
                    if len(all_comments) == 0:
                        status_text.error("No comments were retrieved. The video might have comments disabled.")
                    elif len(all_comments) < int(video_info["comment_count"]) * 0.5 and int(video_info["comment_count"]) > 10:
                        # If we retrieved less than 50% of the expected comments
                        status_text.warning(f"Downloaded {len(all_comments)} comments, but the video has approximately {video_info['comment_count']} comments. Some comments may be missing due to API limitations.")
                    else:
                        status_text.success(f"Downloaded {len(all_comments)} comments successfully!")
                        
                    # Add explanation if needed
                    if int(video_info["comment_count"]) > 10000:
                        st.info("⚠️ Note: For videos with a very large number of comments, YouTube's API may not return all historical comments.")
    
    # Display fetched comments if available
    if "comments" in st.session_state and "video_info" in st.session_state:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Comments")
        
        # Apply filters and sorting
        filtered_comments = st.session_state["comments"].copy()
        
        # Filter by minimum likes
        if min_likes > 0:
            filtered_comments = [c for c in filtered_comments if c["likeCount"] >= min_likes]
        
        # Filter replies if needed
        if not include_replies:
            filtered_comments = [c for c in filtered_comments if not c["isReply"]]
        
        # Apply sorting
        if sort_by == "Most Likes":
            filtered_comments = sorted(filtered_comments, key=lambda x: x["likeCount"], reverse=True)
        elif sort_by == "Newest First":
            filtered_comments = sorted(filtered_comments, key=lambda x: x["publishedAt"], reverse=True)
        elif sort_by == "Oldest First":
            filtered_comments = sorted(filtered_comments, key=lambda x: x["publishedAt"])
        
        # Show comment count
        st.write(f"Showing {len(filtered_comments)} comments out of {len(st.session_state['comments'])} total")
        
        # Create DataFrame for download
        df_comments = pd.DataFrame(filtered_comments)
        if not df_comments.empty:
            # Format timestamps
            if "publishedAt" in df_comments.columns:
                df_comments["publishedAt"] = df_comments["publishedAt"].apply(format_timestamp)
            
            # Generate download buttons
            video_title = st.session_state["video_info"]["title"]
            safe_title = re.sub(r'[^\w\s-]', '', video_title).strip().replace(' ', '_')
            filename = f"{safe_title}_comments.csv"
            
            st.markdown(
                create_download_link(df_comments, filename, "Download as CSV"),
                unsafe_allow_html=True
            )
            
            # Display comments - showing all comments in the UI
            if filtered_comments:
                # Add pagination for better performance
                items_per_page = 50
                total_pages = (len(filtered_comments) + items_per_page - 1) // items_per_page
                
                if total_pages > 1:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"Total comments: {len(filtered_comments)}")
                    with col2:
                        page_num = st.selectbox("Page", range(1, total_pages + 1), index=0)
                else:
                    page_num = 1
                    st.write(f"Total comments: {len(filtered_comments)}")
                
                start_idx = (page_num - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(filtered_comments))
                
                page_comments = filtered_comments[start_idx:end_idx]
                
                # Display warning for large comment sets
                if len(filtered_comments) > 1000:
                    st.warning("⚠️ This video has a large number of comments. The UI might be slower to respond.")
                
                # Display comments for this page
                for i, comment in enumerate(page_comments):
                    # Determine if it's a reply
                    indent = "" if not comment["isReply"] else "↪️ "
                    reply_class = "comment-card" if not comment["isReply"] else "comment-card reply-comment"
                    
                    # Format the comment
                    st.markdown(f"""
                    <div class="{reply_class}">
                        <div class="author-name">{indent}{comment["author"]}</div>
                        <div class="timestamp">{format_timestamp(comment["publishedAt"])}</div>
                        <div class="comment-text">{comment["text"]}</div>
                        <div class="like-count">👍 {comment["likeCount"]} likes</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add pagination controls at the bottom for convenience
                if total_pages > 1:
                    st.write(f"Showing page {page_num} of {total_pages}")
                    
                    cols = st.columns([1, 1, 1, 1])
                    with cols[0]:
                        if page_num > 1:
                            if st.button("⏮️ First"):
                                st.session_state["page_num"] = 1
                                st.experimental_rerun()
                    with cols[1]:
                        if page_num > 1:
                            if st.button("◀️ Previous"):
                                st.session_state["page_num"] = page_num - 1
                                st.experimental_rerun()
                    with cols[2]:
                        if page_num < total_pages:
                            if st.button("Next ▶️"):
                                st.session_state["page_num"] = page_num + 1
                                st.experimental_rerun()
                    with cols[3]:
                        if page_num < total_pages:
                            if st.button("Last ⏭️"):
                                st.session_state["page_num"] = total_pages
                                st.experimental_rerun()
            else:
                st.info("No comments match your current filter settings.")
    
    # Display help when no URL is provided
    if not youtube_url:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📝 How to use this app
        
        1. Enter your YouTube API key in the sidebar
        2. Paste a YouTube video URL in the text box
        3. Click "Get Comments" to download comments
        4. Use the sidebar options to sort and filter comments
        5. Download all comments as a CSV file
        
        ### 🔑 Getting a YouTube API Key
        
        1. Go to [Google Developer Console](https://console.developers.google.com/)
        2. Create a new project
        3. Enable the YouTube Data API v3
        4. Create an API key and copy it
        5. Paste the API key in the sidebar
        """)
        st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import json
import sqlite3
import datetime
import os
from scipy.stats import median_abs_deviation
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load API Keys from Streamlit Secrets
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

# Function to fetch top 50 comments for a given video
def get_video_comments(video_id, max_results=50):
    youtube = get_youtube_service()
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        )
        response = request.execute()
        items = response.get("items", [])
        for item in items:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
    except HttpError as e:
        st.error(f"API Error when fetching comments for video {video_id}: {e}")
    return comments

# Initialize YouTube API
def get_youtube_service():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Improved Database Schema
def initialize_db():
    db_exists = os.path.exists("youtube_data.db")
    conn = sqlite3.connect("youtube_data.db")
    cursor = conn.cursor()
    
    if not db_exists:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT UNIQUE,
                channel_id TEXT,
                channel_name TEXT,
                title TEXT,
                description TEXT,
                thumbnail TEXT,
                published_date TEXT,
                fetch_date TEXT,
                views INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                comments INTEGER DEFAULT 0,
                outlier_score REAL DEFAULT 0
            )
        """)
        # Add indices for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON search_results (video_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_channel_id ON search_results (channel_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_outlier_score ON search_results (outlier_score)")
    else:
        cursor.execute("PRAGMA table_info(search_results)")
        existing_columns = [column[1] for column in cursor.fetchall()]
        expected_columns = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "video_id": "TEXT UNIQUE",
            "channel_id": "TEXT",
            "channel_name": "TEXT",
            "title": "TEXT",
            "description": "TEXT",
            "thumbnail": "TEXT",
            "published_date": "TEXT",
            "fetch_date": "TEXT",
            "views": "INTEGER DEFAULT 0",
            "likes": "INTEGER DEFAULT 0",
            "comments": "INTEGER DEFAULT 0",
            "outlier_score": "REAL DEFAULT 0"
        }
        for column, column_type in expected_columns.items():
            if column not in existing_columns and column != "id":
                try:
                    cursor.execute(f"ALTER TABLE search_results ADD COLUMN {column} {column_type}")
                    st.info(f"Added missing column: {column}")
                except sqlite3.Error as e:
                    st.error(f"Error adding column {column}: {e}")
    
    conn.commit()
    conn.close()

# Clear Cache (Delete all records from the database)
def clear_cache():
    conn = sqlite3.connect("youtube_data.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM search_results")
    conn.commit()
    conn.close()

# Improved database search function with better keyword matching
def search_db_results(niche=None, keyword=None, min_outlier_score=None, sort_by="views", 
                      niche_data=None, excluded_channels=None, video_type=None):
    conn = sqlite3.connect("youtube_data.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Include the duration column if needed by video_type filtering
    query_parts = ["""
        SELECT video_id, channel_id, channel_name, title, description, thumbnail, 
               published_date, fetch_date, views, likes, comments, outlier_score, duration
        FROM search_results
    """]
    
    params = []
    where_conditions = []
    
    if niche and niche_data and niche in niche_data:
        channel_ids = [channel["channel_id"] for channel in niche_data[niche] 
                       if not excluded_channels or channel["channel_id"] not in excluded_channels]
        if channel_ids:
            channel_placeholders = ",".join(["?"] * len(channel_ids))
            where_conditions.append(f"channel_id IN ({channel_placeholders})")
            params.extend(channel_ids)
        else:
            # If all channels are excluded, return empty result
            return []
    
    # Modified keyword search logic: Each search term must be present
    if keyword and keyword.strip():
        search_terms = keyword.strip().lower().split()
        term_conditions = []
        for term in search_terms:
            term_conditions.append("(LOWER(title) LIKE ? OR LOWER(description) LIKE ?)")
            params.extend([f"%{term}%", f"%{term}%"])
        # Join the individual term conditions with AND so all terms must be present
        where_conditions.append(" AND ".join(term_conditions))
    
    if min_outlier_score is not None:
        where_conditions.append("outlier_score >= ?")
        params.append(min_outlier_score)
    
    # Add video duration filtering
    if video_type == "short":
        # Short videos are <= 3 minutes (180 seconds)
        where_conditions.append("duration <= 180")
    elif video_type == "long":
        # Long videos are > 3 minutes
        where_conditions.append("duration > 180")
    
    if where_conditions:
        query_parts.append("WHERE " + " AND ".join(where_conditions))
    
    if sort_by:
        query_parts.append(f"ORDER BY {sort_by} DESC")
    
    final_query = " ".join(query_parts)
    
    try:
        cursor.execute(final_query, params)
        results = cursor.fetchall()
        result_list = [dict(row) for row in results]
        # Remove duplicates based on video_id
        unique_results = list({video["video_id"]: video for video in result_list}.values())
        conn.close()
        return unique_results
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        conn.close()
        return []

# Custom keyword search function for memory filtering
def keyword_match(text, keyword):
    if not keyword or not text:
        return True
    
    text = text.lower()
    search_terms = keyword.lower().strip().split()
    
    # Return True if all terms are present in the text
    for term in search_terms:
        if term not in text:
            return False
    
    return True

# Improved save to db function
def save_to_db(video_data):
    if not video_data:
        return
    conn = sqlite3.connect("youtube_data.db")
    cursor = conn.cursor()
    current_date = datetime.datetime.now().isoformat()
    for video in video_data:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO search_results (
                    video_id, channel_id, channel_name, title, description, thumbnail, 
                    published_date, fetch_date, views, likes, comments, outlier_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video["video_id"],
                video.get("channel_id", ""),
                video.get("channel_name", ""),
                video["title"],
                video.get("description", ""),
                video.get("thumbnail", ""),
                video.get("published_date", ""),
                current_date,
                video.get("views", 0),
                video.get("likes", 0),
                video.get("comments", 0),
                video.get("outlier_score", 0)
            ))
        except sqlite3.Error as e:
            st.error(f"Error saving video {video.get('title', 'unknown')}: {e}")
    conn.commit()
    conn.close()

# Load Niche Channels with error handling
def load_niche_channels():
    try:
        with open("channels.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("channels.json file not found!")
        return {}
    except json.JSONDecodeError:
        st.error("Error parsing channels.json - invalid format!")
        return {}

# Compute outlier scores using Modified Z-Score with numpy for better performance
def compute_outlier_scores(videos, metric="views"):
    if not videos:
        return videos
    video_ids = [video["video_id"] for video in videos]
    values = np.array([video.get(metric, 0) for video in videos])
    if len(values) < 2:
        for video in videos:
            video["outlier_score"] = 0
        return videos
    median_value = np.median(values)
    mad = median_abs_deviation(values)
    if mad == 0:
        for video in videos:
            video["outlier_score"] = 0
        return videos
    scores = 0.6745 * (values - median_value) / mad
    for i, video in enumerate(videos):
        video["outlier_score"] = round(float(scores[i]), 2)
    return videos

# Fetch all videos from a channel with pagination
def get_channel_videos(channel_id, channel_name, max_results=100):
    youtube = get_youtube_service()
    videos = []
    next_page_token = None
    total_results = 0
    try:
        while total_results < max_results:
            batch_size = min(50, max_results - total_results)
            request_params = {
                "part": "id,snippet",
                "channelId": channel_id,
                "maxResults": batch_size,
                "type": "video",
                "order": "date"
            }
            if next_page_token:
                request_params["pageToken"] = next_page_token
            request = youtube.search().list(**request_params)
            response = request.execute()
            items = response.get("items", [])
            if not items:
                break
            for item in items:
                if "videoId" in item["id"]:
                    video_id = item["id"]["videoId"]
                    title = item["snippet"]["title"]
                    description = item["snippet"].get("description", "")
                    thumbnail = item["snippet"]["thumbnails"]["high"]["url"]
                    published_date = item["snippet"]["publishedAt"]
                    videos.append({
                        "video_id": video_id,
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "title": title,
                        "description": description,
                        "thumbnail": thumbnail,
                        "published_date": published_date
                    })
            total_results += len(items)
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        return videos
    except HttpError as e:
        st.error(f"API Error for channel {channel_name}: {e}")
        return []

# Fetch video statistics in batches
def get_video_statistics(videos):
    youtube = get_youtube_service()
    if not videos:
        return videos
    video_ids = [video["video_id"] for video in videos]
    video_dict = {video["video_id"]: video for video in videos}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        try:
            request = youtube.videos().list(
                part="statistics",
                id=",".join(chunk)
            )
            response = request.execute()
            for item in response.get("items", []):
                video_id = item["id"]
                if video_id in video_dict:
                    stats = item.get("statistics", {})
                    video_dict[video_id]["views"] = int(stats.get("viewCount", 0))
                    video_dict[video_id]["likes"] = int(stats.get("likeCount", 0))
                    video_dict[video_id]["comments"] = int(stats.get("commentCount", 0))
        except HttpError as e:
            st.error(f"API Error when fetching statistics: {e}")
    return list(video_dict.values())

# Check if data needs refreshing
def needs_refresh(data, max_age_days=7):
    if not data:
        return True
    current_date = datetime.datetime.now()
    for item in data:
        fetch_date_str = item.get("fetch_date")
        if not fetch_date_str:
            return True
        try:
            fetch_date = datetime.datetime.fromisoformat(fetch_date_str)
            age = (current_date - fetch_date).days
            if age > max_age_days:
                return True
        except ValueError:
            return True
    return False

# Recreate database if needed
def recreate_database():
    try:
        if os.path.exists("youtube_data.db"):
            os.remove("youtube_data.db")
            st.success("Database file removed successfully.")
        initialize_db()
        st.success("Database has been recreated with the new schema.")
    except Exception as e:
        st.error(f"Error recreating database: {e}")

# Initialize Database
initialize_db()

# Apply Styling with additional CSS for metrics and channel pills
st.set_page_config(layout="wide", page_title="YouTube Outlier Detector")
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .video-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .video-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .video-stats {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }
    /* Reduce the font size for metric values and labels */
    div[data-testid="stMetricValue"] {
        font-size: 14px !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12px !important;
    }
    /* Channel pill styling */
    .channel-pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0;
    }
    .channel-pill {
        background-color: #f0f2f6;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .channel-pill-active {
        background-color: #e0f7fa;
        border: 1px solid #26c6da;
    }
    .channel-pill-excluded {
        background-color: #ffebee;
        border: 1px solid #ef5350;
        text-decoration: line-through;
    }
    .channel-pill-button {
        cursor: pointer;
        margin-left: 5px;
        color: #616161;
        font-weight: bold;
    }
    .channel-pill-button:hover {
        color: #ef5350;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎥 YouTube Outlier Video Detector")
st.markdown("Find standout videos across channels based on performance metrics")

# Initialize session state for excluded channels if it doesn't exist
if 'excluded_channels' not in st.session_state:
    st.session_state.excluded_channels = set()

# Sidebar for controls
with st.sidebar:
    st.header("🔍 Filter Options")
    niche_data = load_niche_channels()
    niches = list(niche_data.keys())
    
    if not niches:
        st.error("No niches found in channels.json. Please check the file.")
        selected_niche = None
    else:
        selected_niche = st.selectbox("Select a Niche", niches)
    
    # Channel selection section
    if selected_niche:
        st.subheader("Channel Selection")
        st.markdown("Click on a channel to exclude/include it in the search:")
        
        # Create a container for channel pills
        channel_pills_html = '<div class="channel-pill-container">'
        
        for channel in niche_data.get(selected_niche, []):
            channel_id = channel["channel_id"]
            channel_name = channel["channel_name"]
            
            is_excluded = channel_id in st.session_state.excluded_channels
            pill_class = "channel-pill channel-pill-excluded" if is_excluded else "channel-pill channel-pill-active"
            
            channel_pills_html += f"""
            <div class="{pill_class}">
                {channel_name}
                <span class="channel-pill-button" onclick="
                    const data = {{
                        channel_id: '{channel_id}',
                        action: '{'include' if is_excluded else 'exclude'}'
                    }};
                    window.parent.postMessage({{
                        type: 'streamlit:toggleChannel',
                        data: data
                    }}, '*');
                ">{"+" if is_excluded else "×"}</span>
            </div>
            """
        
        channel_pills_html += '</div>'
        
        # Display the channel pills
        st.markdown(channel_pills_html, unsafe_allow_html=True)
        
        # Handle channel toggle from JavaScript via a hidden button click
        toggle_channel_placeholder = st.empty()
        toggle_channel = toggle_channel_placeholder.button("Toggle Channel", key="toggle_channel_btn", help="This button is programmatically clicked by JavaScript")
        
        if toggle_channel:
            # Extract channel ID and action from query parameters
            query_params = st.experimental_get_query_params()
            channel_id = query_params.get("channel_id", [""])[0]
            action = query_params.get("action", [""])[0]
            
            if channel_id and action:
                if action == "exclude" and channel_id not in st.session_state.excluded_channels:
                    st.session_state.excluded_channels.add(channel_id)
                elif action == "include" and channel_id in st.session_state.excluded_channels:
                    st.session_state.excluded_channels.remove(channel_id)
                
                # Clear query parameters
                st.experimental_set_query_params()
                # Force a rerun to update the UI
                st.experimental_rerun()
        
        # Alternative channel selection method using checkboxes (no JavaScript needed)
        st.markdown("### Alternative Channel Selection")
        excluded_channels_temp = set(st.session_state.excluded_channels)
        
        for channel in niche_data.get(selected_niche, []):
            channel_id = channel["channel_id"]
            channel_name = channel["channel_name"]
            
            is_included = st.checkbox(
                f"Include {channel_name}", 
                value=channel_id not in st.session_state.excluded_channels,
                key=f"channel_{channel_id}"
            )
            
            if not is_included:
                excluded_channels_temp.add(channel_id)
            elif channel_id in excluded_channels_temp:
                excluded_channels_temp.remove(channel_id)
        
        st.session_state.excluded_channels = excluded_channels_temp
        
        # Button to reset channel selection
        if st.button("Reset Channel Selection"):
            st.session_state.excluded_channels = set()
            st.experimental_rerun()
    
    keyword = st.text_input("🔎 Enter keyword to search within videos")
    
    st.subheader("Advanced Options")
    outlier_threshold = st.slider("Minimum Outlier Score", min_value=0, max_value=20, value=5, step=1)
    
    max_results_per_channel = st.slider("Max Videos per Channel", min_value=10, max_value=200, value=50, step=10)
    
    refresh_days = st.slider("Refresh data older than (days)", min_value=1, max_value=30, value=7, step=1)
    force_refresh = st.checkbox("Force refresh all data")
    
    sort_options = {
        "View Count": "views",
        "Outlier Score": "outlier_score",
        "Likes": "likes",
        "Comments": "comments" 
    }
    
    sort_option = st.selectbox("Sort results by", list(sort_options.keys()))
    
    fetch_button = st.button("🔍 Find Outliers")
    
    st.header("🛠️ Database Maintenance")
    
    if st.button("🗑️ Clear Cache"):
        clear_cache()
        st.success("Cache cleared! All data has been removed from the database.")
    
    if st.button("🔄 Recreate Database"):
        recreate_database()
        st.success("Database has been recreated with the updated schema.")

# Add JavaScript to handle channel toggling
st.markdown("""
<script>
// Listen for messages from the channel pills
window.addEventListener('message', function(event) {
    if (event.data.type === 'streamlit:toggleChannel') {
        const channelData = event.data.data;
        // Set query parameters to communicate with Streamlit
        const searchParams = new URLSearchParams(window.location.search);
        searchParams.set('channel_id', channelData.channel_id);
        searchParams.set('action', channelData.action);
        // Update URL (doesn't reload the page)
        window.history.pushState({}, '', '?' + searchParams.toString());
        // Click the hidden button to trigger the Streamlit event
        setTimeout(() => {
            document.querySelector('[data-testid="stButton"] button').click();
        }, 100);
    }
});
</script>
""", unsafe_allow_html=True)

# Display currently excluded channels info
if st.session_state.excluded_channels:
    excluded_channel_names = []
    for channel in niche_data.get(selected_niche, []):
        if channel["channel_id"] in st.session_state.excluded_channels:
            excluded_channel_names.append(channel["channel_name"])
    
    st.warning(f"Currently excluding {len(excluded_channel_names)} channels: {', '.join(excluded_channel_names)}")

# Main content area
if fetch_button and selected_niche:
    with st.spinner("Searching for videos..."):
        try:
            db_results = search_db_results(
                niche=selected_niche,
                keyword=keyword, 
                min_outlier_score=outlier_threshold,
                sort_by=sort_options[sort_option],
                niche_data=niche_data,
                excluded_channels=st.session_state.excluded_channels
            )
            
            refresh_needed = needs_refresh(db_results, max_age_days=refresh_days) or force_refresh
            
            if db_results and not refresh_needed:
                st.success(f"✅ Found {len(db_results)} videos in the database")
                video_data = db_results
            else:
                st.info("Fetching fresh data from YouTube...")
                niche_channels = [
                    channel for channel in niche_data.get(selected_niche, []) 
                    if channel["channel_id"] not in st.session_state.excluded_channels
                ]
                if not niche_channels:
                    st.warning(f"No channels selected for the niche: {selected_niche}")
                    video_data = []
                else:
                    all_videos = []
                    progress_bar = st.progress(0)
                    for i, channel in enumerate(niche_channels):
                        channel_videos = get_channel_videos(
                            channel["channel_id"], 
                            channel["channel_name"],
                            max_results=max_results_per_channel
                        )
                        if channel_videos:
                            progress_bar.progress((i + 0.5) / len(niche_channels))
                            st.text(f"Fetching statistics for {len(channel_videos)} videos...")
                            channel_videos = get_video_statistics(channel_videos)
                            all_videos.extend(channel_videos)
                        progress_bar.progress((i + 1) / len(niche_channels))
                    if all_videos:
                        st.text("Calculating outlier scores...")
                        all_videos = compute_outlier_scores(all_videos, metric="views")
                        save_to_db(all_videos)
                        video_data = [
                            video for video in all_videos
                            if (not keyword or 
                                (keyword.lower() in video["title"].lower() or 
                                 keyword.lower() in video.get("description", "").lower())) and
                            video.get("outlier_score", 0) >= outlier_threshold
                        ]
                        # Remove duplicates if any
                        video_data = list({video["video_id"]: video for video in video_data}.values())
                        video_data = sorted(
                            video_data, 
                            key=lambda x: x.get(sort_options[sort_option], 0), 
                            reverse=True
                        )
                        st.success(f"✅ Found {len(video_data)} videos matching your criteria")
                    else:
                        st.warning("No videos found for the selected niche.")
                        video_data = []
            
            # After obtaining search results, fetch top 50 comments for each video
            if video_data:
                with st.spinner("Fetching top comments for each video..."):
                    for video in video_data:
                        video["top_comments"] = get_video_comments(video["video_id"], max_results=50)
            
            # Display the results
            if video_data:
                df = pd.DataFrame(video_data)
                if 'channel_name' in df.columns and 'title' in df.columns:
                    display_cols = ['channel_name', 'title', 'views', 'likes', 'comments', 'outlier_score']
                    display_names = ['Channel', 'Title', 'Views', 'Likes', 'Comments', 'Outlier Score']
                    for col in display_cols:
                        if col not in df.columns:
                            df[col] = "N/A"
                    df_display = df[display_cols].copy()
                    df_display.columns = display_names
                    st.subheader("Results Overview")
                    st.dataframe(df_display, height=300)
                # Add pagination for results
                items_per_page = 10
                total_pages = len(video_data) // items_per_page + (1 if len(video_data) % items_per_page > 0 else 0)
                if total_pages > 1:
                    page = st.slider("Page", 1, max(1, total_pages), 1)
                    start_idx = (page - 1) * items_per_page
                    end_idx = min(start_idx + items_per_page, len(video_data))
                    display_videos = video_data[start_idx:end_idx]
                    st.info(f"Showing results {start_idx+1}-{end_idx} of {len(video_data)}")
                else:
                    display_videos = video_data[:10]
                
                st.subheader("Top Videos")
                # Create two columns for displaying videos side by side
                cols = st.columns(2)
                for idx, video in enumerate(display_videos):
                    with cols[idx % 2]:
                        st.image(video.get("thumbnail"), use_container_width=True)
                        st.markdown(f"### {video.get('title', 'No Title')}")
                        st.markdown(f"**Channel:** {video.get('channel_name', 'Unknown')}")
                        stat_cols = st.columns(4)
                        with stat_cols[0]:
                            st.metric("Views", f"{video.get('views', 0):,}")
                        with stat_cols[1]:
                            st.metric("Likes", f"{video.get('likes', 0):,}")
                        with stat_cols[2]:
                            st.metric("Comments", f"{video.get('comments', 0):,}")
                        with stat_cols[3]:
                            st.metric("Outlier Score", f"{video.get('outlier_score', 0):.2f}")
                        video_url = f"https://www.youtube.com/watch?v={video.get('video_id', '')}"
                        st.markdown(f"[Watch Video]({video_url})")
                        with st.expander("Show Top Comments"):
                            comments = video.get("top_comments", [])
                            if comments:
                                for comment in comments:
                                    st.markdown(f"- {comment}")
                            else:
                                st.markdown("No comments available.")
                        st.markdown("---")
            else:
                st.warning("No videos found matching your criteria. Try adjusting your filters.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
elif fetch_button and not selected_niche:
    st.warning("Please select a niche first!")
else:
    st.info("👈 Select filters and click 'Find Outliers' to search for videos")
    st.markdown("""
    ## How to use this app
    
    1. **Select a niche** from the dropdown menu
    2. **Select/deselect channels** you want to include in your search
    3. Optionally **enter a keyword** to search within video titles and descriptions
    4. Adjust the **minimum outlier score** to filter for truly exceptional videos
    5. Choose how to **sort** your results
    6. Click the **Find Outliers** button to see the results
    
    ## What is an outlier score?
    
    The outlier score measures how much a video deviates from the typical performance of videos in the same niche:
    
    - **Score > 5**: The video is performing better than most others
    - **Score > 10**: The video is a significant outlier (very successful)
    - **Score > 15**: The video is an extreme outlier (viral potential)
    
    The score is calculated using a statistical method called Modified Z-Score based on median absolute deviation.
    """)

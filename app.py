import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np
import pandas as pd
import time
import urllib.parse
import platform

# --- STAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Olympus AI", page_icon="🏆")

# --- PERSISTENT SESSION STATE MEMORY LOCKS ---
if 'history_log' not in st.session_state:
    st.session_state.history_log = []
if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = {}  

# --- CORE MATH UTILITIES ---
def calculate_angle(a, b, c):
    """Calculates internal joint angle between three 2D coordinates."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    angle = np.arctan2(np.linalg.det([ba, bc]), np.dot(ba, bc))
    angle = np.abs(angle * 180 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

@st.cache_resource
def load_models():
    """Loads Intel-optimized OpenVINO models if present, otherwise switches to universal weights."""
    try:
        det = YOLO('yolov8n_openvino_model')
        pose = YOLO('yolov8n-pose_openvino_model')
    except Exception:
        det = YOLO('yolov8n.pt')
        pose = YOLO('yolov8n-pose.pt')
        
    return det, pose

det_model, pose_model = load_models()

# --- DYNAMIC HARDWARE DETECTOR ---
current_os = platform.system()
current_processor = platform.processor()

st.sidebar.image("https://img.icons8.com/fluent/96/000000/sports.png", width=80)
st.sidebar.title("Olympus AI Engine")

if "openvino" in str(type(det_model.model)):
    st.sidebar.success("🚀 Optimization Engine: Hardware Acceleration Active!")
    st.sidebar.markdown(f"**Platform Profile:** Intel Optimized Engine")
else:
    st.sidebar.info("⚡ Optimization Engine: Mobile Responsive Core Active")
    st.sidebar.markdown(f"**Host Device OS:** {current_os}")

st.sidebar.divider()

app_mode = st.sidebar.radio("Navigate Workspace:", ["🏠 Home & Interactive Guide", "🚀 Real-Time AI Analyzer", "📊 Analytics Vault & Log"])

# ==========================================
# TAB 1: INTERACTIVE HOME & INITIALIZATION
# ==========================================
if app_mode == "🏠 Home & Interactive Guide":
    st.markdown("# Welcome to Olympus AI 🤖🏆")
    st.markdown("### The Universal Autonomous Sports Science & Biomechanics Interface")
    
    st.info("💡 **Cross-Device Ready:** This app automatically adapts to iPhones, Androids, iPads, and computers. Our dual-layer Neural Networks automatically identify the sport and track your body mechanics instantly.")
    st.divider()
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### How the Automation Operates")
        with st.expander("👁️ Computer Vision Asset Tracking"):
            st.write("The app uses standard Convolutional Neural Networks to scan for items like sports balls, rackets, bats, and goalposts to dynamically deduce what sport is being played in real-time.")
        with st.expander("🦴 17-Point Skeleton Spatial Math"):
            st.write("Using trigonometry tracking loops, we calculate the exact degree profiles of your joints 30 times per second to catch form breaks invisible to the naked eye.")
            
    with col_right:
        st.markdown("### Interactive Features Checklist")
        st.checkbox("Autonomous Sport Profiling", value=True, disabled=True)
        st.checkbox("Batch Multi-Video Queue Processing", value=True, disabled=True)
        st.checkbox("Fully Labeled Kinetic Waveform Charts", value=True, disabled=True)
        st.checkbox("Actionable Physical Corrective Drills", value=True, disabled=True)

    st.divider()
    st.success("👉 Head over to the **Real-Time AI Analyzer** on the sidebar menu to upload your first batch of clips!")

# ==========================================
# TAB 2: AI BATCH PROCESSING ENGINE
# ==========================================
elif app_mode == "🚀 Real-Time AI Analyzer":
    st.markdown("# Autonomous Batch AI Analysis Terminal")
    st.caption("Upload one or multiple sports clips. The pipeline will automatically loop through and process each file sequentially.")
    
    uploaded_files = st.file_uploader("Drop one or multiple sports clips here (.mp4, .mov, .avi)...", type=["mp4", "mov", "avi"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"📋 Video Queue Configured: **{len(uploaded_files)} files** detected.")
        
        if st.button("⚡ Execute Batch Biomechanical Scan", use_container_width=True):
            for idx, single_file in enumerate(uploaded_files):
                st.subheader(f"🎬 Processing Video {idx+1} of {len(uploaded_files)}: `{single_file.name}`")
                
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(single_file.read())
                
                video_cap = cv2.VideoCapture(tfile.name)
                
                st_video_placeholder = st.empty()
                
                col1, col2 = st.columns(2)
                with col1:
                    sport_card = st.empty()
                    tips_placeholder = st.empty()
                    drills_placeholder = st.empty() 
                with col2:
                    st.markdown("#### Real-Time KPI Performance Cards")
                    kpi1_placeholder = st.empty()
                    kpi2_placeholder = st.empty()
                    
                chart_title_placeholder = st.empty()
                chart_placeholder = st.empty()
                youtube_placeholder = st.empty()
                
                frame_count = 0
                angle_history = []
                max_angle = 0
                min_angle = 180
                detected_sport = "Scanning..."
                coaching_tip = "Isolating athlete profiles..."
                actionable_drill = ""
                recommended_drill = "" 
                weakness_tag = ""  
                
                frame_skip_rate = 4  
                
                while video_cap.isOpened():
                    flag = video_cap.grab()
                    if not flag:
                        break
                    
                    frame_count += 1
                    if frame_count % frame_skip_rate != 0:
                        continue
                    
                    ret, frame = video_cap.retrieve()
                    if not ret:
                        break
                        
                    frame = cv2.resize(frame, (480, 270))
                    
                    det_results = det_model(frame, verbose=False)
                    found_objects = [det_model.names[int(box.cls[0])] for box in det_results[0].boxes]
                    pose_results = pose_model(frame, verbose=False)
                    annotated_frame = pose_results[0].plot()
                    
                    try:
                        if pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy) > 0:
                            keypoints = pose_results[0].keypoints.xy[0].cpu().numpy()
                            
                            if len(keypoints) >= 17:
                                wrist, elbow, shoulder = keypoints[10], keypoints[8], keypoints[6]
                                hip, knee, ankle = keypoints[12], keypoints[14], keypoints[16]
                                
                                # A. Basketball Engine
                                if "sports ball" in found_objects and wrist[1] < shoulder[1] and wrist[1] > 0:
                                    detected_sport = "Basketball"
                                    current_angle = calculate_angle(shoulder, elbow, wrist)
                                    angle_history.append(current_angle)
                                    max_angle = max(max_angle, current_angle)
                                    min_angle = min(min_angle, current_angle)
                                    
                                    if max_angle > 150:
                                        coaching_tip = f"🎯 **Form Insights:** Release extension peaked at **{int(max_angle)}°**. Solid lockout form!"
                                        actionable_drill = "💡 **Maintenance Drill:** Keep doing 20 reps of fluid 'Form Shooting' from 3 feet away to lock this muscle memory into your kinetic chain."
                                        recommended_drill = "Basketball shooting rhythm and follow through drills"
                                        weakness_tag = "None (Good Release)"
                                    else:
                                        coaching_tip = f"🎯 **Form Insights:** Release extension peaked at **{int(max_angle)}°**. Warning: Your shooting elbow is short-arming."
                                        actionable_drill = "🛠️ **How to fix it:** Stand 2 feet from the basket and shoot one-handed. Force your elbow to finish completely straight, pointing above the rim until the ball drops."
                                        recommended_drill = "How to fix a short arm basketball shot follow through"
                                        weakness_tag = "Incomplete Shooting Elbow Lockout"
                                
                                # B. Soccer Engine
                                elif "sports ball" in found_objects and ankle[1] > hip[1]:
                                    detected_sport = "Soccer"
                                    current_angle = calculate_angle(hip, knee, ankle)
                                    angle_history.append(current_angle)
                                    max_angle = max(max_angle, current_angle)
                                    min_angle = min(min_angle, current_angle)
                                    
                                    if min_angle < 120:
                                        coaching_tip = f"⚽ **Form Insights:** Knee torque flex reached **{int(min_angle)}°**. Excellent high-velocity snap!"
                                        actionable_drill = "💡 **Maintenance Drill:** Practice standard wall-kicks ensuring you maintain a locked ankle position on impact to maximize force output."
                                        recommended_drill = "Advanced soccer shooting strike power accuracy"
                                        weakness_tag = "None (Strong Kicking Flex)"
                                    else:
                                        coaching_tip = f"⚽ **Form Insights:** Knee flex reached a rigid **{int(min_angle)}°**. Stiffness Warning: Deepen your knee drop on your approach step."
                                        actionable_drill = "🛠️ **How to fix it:** Perform 'Step-Plant-Drop' slow-motion repetitions. Focus entirely on lowering your center of mass and flexing your striking leg deep before driving through."
                                        recommended_drill = "How to bend knee properly when kicking a soccer ball and fix power"
                                        weakness_tag = "Stiff Kicking Knee Load"
                                
                                # C. Racket / Badminton Engine
                                elif "tennis racket" in found_objects or (wrist[1] < keypoints[0][1] and wrist[1] > 0):
                                    detected_sport = "Racket Sports / Badminton"
                                    current_angle = calculate_angle(hip, shoulder, wrist)
                                    angle_history.append(current_angle)
                                    max_angle = max(max_angle, current_angle)
                                    min_angle = min(min_angle, current_angle)
                                    
                                    if current_angle > 140:
                                        coaching_tip = f"🏸 **Form Insights:** Reach mechanical angle timed at **{int(current_angle)}°**. Perfect apex contact point for an explosive overhead smash!"
                                        actionable_drill = "💡 **Maintenance Drill:** Focus on your drop-step footwork pattern to ensure you are always balanced directly behind the shuttle path."
                                        recommended_drill = "Advanced badminton overhead jump smash power tutorials"
                                        weakness_tag = "None (Perfect Apex Contact)"
                                    else:
                                        coaching_tip = f"🏸 **Form Insights:** Reach mechanical angle timed at **{int(current_angle)}°**. Form Break: Your overhead smash contact point is too low, crushing your angular power."
                                        actionable_drill = "🛠️ **How to fix it:** Toss a shuttle up high, extend your arm fully straight into the air, and catch it at the highest possible point. Do 30 reps to rebuild your high contact instinct."
                                        recommended_drill = "How to hit badminton smash at highest point overhead reach mechanics"
                                        weakness_tag = "Low Overhead Strike Contact Point"
                                
                                # D. Fallback Core Engine
                                else:
                                    detected_sport = "General Mobility"
                                    current_angle = calculate_angle(shoulder, hip, knee)
                                    angle_history.append(current_angle)
                                    max_angle = max(max_angle, current_angle)
                                    min_angle = min(min_angle, current_angle)
                                    
                                    if current_angle < 140:
                                        coaching_tip = f"🏃 **Form Insights:** Hip hinge is registering at **{int(current_angle)}°**. Great athletic stance balance."
                                        actionable_drill = "💡 **Maintenance Drill:** Practice lateral box-jumps while maintaining this low, loaded balance point upon landing."
                                        recommended_drill = "Athletic stance agility footwork movement tutorials"
                                        weakness_tag = "None (Balanced Base)"
                                    else:
                                        coaching_tip = f"🏃 **Form Insights:** Hip hinge is registering at **{int(current_angle)}°**. Warning: Your posture is too erect."
                                        actionable_drill = "🛠️ **How to fix it:** Perform Kettlebell or bodyweight hip hinges against a wall. Push your glutes backward until they touch the wall while keeping your spine straight."
                                        recommended_drill = "How to improve hip hinge mechanics athletic lower body load"
                                        weakness_tag = "Erect Posture / Lack of Hip Hinge"

                    except (IndexError, TypeError, ValueError):
                        pass

                    # Stream Live Data to active frame widgets
                    sport_card.markdown(f"#### Identified Event Cluster:\n## {detected_sport}")
                    tips_placeholder.info(coaching_tip)
                    if actionable_drill:
                        drills_placeholder.success(actionable_drill)
                        
                    kpi1_placeholder.metric("Peak Joint Angle", f"{int(max_angle)}°")
                    kpi2_placeholder.metric("Minimum Joint Flexion", f"{int(min_angle)}°")
                    
                    # --- REAL-TIME EXPLICITLY LABELED GRAPH ---
                    if angle_history:
                        chart_title_placeholder.markdown("##### 📈 Live Biomechanical Waveform Tracker")
                        df_chart = pd.DataFrame({
                            "Video Frame Progress Over Time": range(len(angle_history)),
                            "Joint Flexion (Degrees)": angle_history
                        })
                        # Plot chart using native explicit column declarations for X and Y
                        chart_placeholder.line_chart(df_chart, x="Video Frame Progress Over Time", y="Joint Flexion (Degrees)")

                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st_video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                    
                video_cap.release()
                
                youtube_search_url = ""
                if recommended_drill:
                    encoded_query = urllib.parse.quote(recommended_drill)
                    youtube_search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
                    with youtube_placeholder.container():
                        st.markdown(f"🔗 **Targeted YouTube Training Links Generated for `{single_file.name}`:**")
                        st.link_button(f"📺 Click to open recommended drills for: {recommended_drill}", youtube_search_url, use_container_width=True)
                
                # SAVE COMPREHENSIVE RECORDS TO STATE DICTIONARY
                st.session_state.processed_videos[single_file.name] = {
                    "sport": detected_sport,
                    "tip": coaching_tip,
                    "drill": actionable_drill,
                    "max": f"{int(max_angle)}°",
                    "min": f"{int(min_angle)}°",
                    "chart_data": angle_history,
                    "yt_link": youtube_search_url,
                    "drill_name": recommended_drill
                }
                
                st.session_state.history_log.append({
                    "Video Source File": single_file.name,
                    "Sport Identified": detected_sport,
                    "Peak Extension": f"{int(max_angle)}°",
                    "Max Flexion": f"{int(min_angle)}°",
                    "Primary Form Fault Detected": weakness_tag
                })
                
            st.balloons()
        
        # --- MEMORY RENDERING LOOP (LABELED CHART PERSISTENCE IN HISTORY EXPANDERS) ---
        if st.session_state.processed_videos:
            st.divider()
            st.markdown("## 💾 Loaded Video Cache Readings (Stored System Memory)")
            
            for name, data in st.session_state.processed_videos.items():
                with st.expander(f"📁 Verified Metrics Profile: {name}", expanded=True):
                    vcol1, vcol2 = st.columns(2)
                    with vcol1:
                        st.markdown(f"### Sport Context: {data['sport']}")
                        st.info(data['tip'])
                        if data['drill']:
                            st.success(data['drill'])
                        if data['yt_link']:
                            st.link_button(f"📺 Open Target Video Drills on YouTube", data['yt_link'], use_container_width=True)
                    with vcol2:
                        st.metric("Peak Joint Extension", data['max'])
                        st.metric("Maximum Joint Flexion", data['min'])
                    
                    if data['chart_data']:
                        st.markdown("##### 📈 Biomechanical Waveform Chart")
                        st.caption("ℹ️ **What this graph means:** This wave tracks your joint angle degree shifts frame-by-frame. Drops represent deep joint flexion (loading power), and peaks represent high mechanical extensions (the follow-through or strike apex).")
                        
                        # --- EXPLICIT CACHED GRAPH LABELS ---
                        df_cached_chart = pd.DataFrame({
                            "Video Frame Progress Over Time": range(len(data['chart_data'])),
                            "Joint Flexion (Degrees)": data['chart_data']
                        })
                        st.line_chart(df_cached_chart, x="Video Frame Progress Over Time", y="Joint Flexion (Degrees)")

# ==========================================
# TAB 3: PERFORMANCE METRICS RECORD LOG
# ==========================================
elif app_mode == "📊 Analytics Vault & Log":
    st.markdown("# Performance Analytics Vault")
    st.caption("Review separate metrics compiled for each individual video alongside your comprehensive cross-sport performance diagnostic report.")
    
    if st.session_state.history_log:
        df_logs = pd.DataFrame(st.session_state.history_log)
        
        st.markdown("### 📋 File-by-File Processing Breakdown")
        st.dataframe(df_logs, use_container_width=True)
        
        st.divider()
        
        st.markdown("### 🧠 Master Athlete Diagnostic Report")
        
        raw_weaknesses = [w for w in df_logs["Primary Form Fault Detected"].tolist() if "None" not in w]
        
        if raw_weaknesses:
            primary_overall_weakness = max(set(raw_weaknesses), key=raw_weaknesses.count)
            occurrence_count = raw_weaknesses.count(primary_overall_weakness)
            
            st.error(f"⚠️ **Primary Overarching Technical Weakness Isolated:** **{primary_overall_weakness}**")
            st.write(f"This specific mechanical flaw was flagged across **{occurrence_count} separate movement cycles** during this testing session.")
            st.info("💡 **Sports Science Strategic Advice:** Your kinetic chain shows consistent compression breakdowns during high-impact moments. Prioritize joint flexibility and muscle memory drills matching this target fault area to unlock efficient power dispersion across all multi-sport events.")
        else:
            st.success("🏆 **Master Athlete Status Verified:** No consistent form faults or critical mechanical bottlenecks were flagged during this processing batch! Keep maintaining this balance profile.")
            
        st.divider()
        st.download_button("📥 Export Performance Stance Logs as CSV", data=df_logs.to_csv(index=False), file_name="olympus_session_metrics.csv", use_container_width=True)
    else:
        st.warning("The Performance Vault is currently empty. Run the Autonomous AI Analyzer first to compile tracking logs.")
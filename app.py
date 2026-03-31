import streamlit as st
import cv2
import numpy as np
import pickle
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Student Voting System", layout="wide")
st.title("🗳️ Student Online Voting System - Face Recognition")

# ====================== ADMIN PASSWORD ======================
ADMIN_PASSWORD = "admin2026"   # ← CHANGE THIS TO A STRONG PASSWORD!

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

admin_pass = st.sidebar.text_input("🔑 Admin Password", type="password")
if admin_pass == ADMIN_PASSWORD:
    st.session_state.admin_logged_in = True
elif admin_pass != "":
    st.sidebar.error("❌ Wrong Admin Password")

if st.session_state.admin_logged_in and st.sidebar.button("Logout Admin"):
    st.session_state.admin_logged_in = False
    st.rerun()

# ====================== LOAD DATA ======================
DATA_FILE = "voting_data.pkl"

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
else:
    data = {"students": {}, "candidates": [], "votes": {}, "recognizer": None, "label_map": {}}

students = data.get("students", {})
candidates = data.get("candidates", [])
votes = data.get("votes", {})
recognizer = data.get("recognizer")
label_map = data.get("label_map", {})

def save_data():
    with open(DATA_FILE, "wb") as f:
        pickle.dump({
            "students": students,
            "candidates": candidates,
            "votes": votes,
            "recognizer": recognizer,
            "label_map": label_map
        }, f)

# Train LBPH recognizer using only registered face data
def train_recognizer():
    global recognizer, label_map
    faces = []
    labels = []
    label_map = {}
    current_label = 0
    for reg, info in students.items():
        label_map[current_label] = reg
        for face in info.get("faces", []):
            faces.append(face)
            labels.append(current_label)
        current_label += 1
    if len(faces) == 0:
        recognizer = None
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(np.array(faces), np.array(labels))

if recognizer is None and students:
    train_recognizer()

def is_good_lighting(gray):
    return 70 <= np.mean(gray) <= 220

# ====================== MENU ======================
if st.session_state.admin_logged_in:
    page = st.sidebar.radio("Menu", ["Vote (Public)", "Register Student", "Manage Candidates", "View Results"])
else:
    page = "Vote (Public)"

# ====================== VOTE PAGE ======================
if page == "Vote (Public)":
    st.header("🗳️ Cast Your Vote")
    st.write("**Only registered students can vote.** Look straight at the camera.")

    class VoteProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')\
                .detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            recognized_reg = None
            min_conf = 999

            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                if not is_good_lighting(face_gray):
                    continue

                resized = cv2.resize(face_gray, (100, 100))
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

                if recognizer:
                    label, conf = recognizer.predict(resized)
                    if label in label_map and conf < min_conf and conf < 75:   # Stricter threshold
                        min_conf = conf
                        recognized_reg = label_map[label]

            if recognized_reg:
                student = students[recognized_reg]
                cv2.putText(img, f"Identified: {student['name']}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                st.session_state.recognized_reg = recognized_reg
            else:
                cv2.putText(img, "Face not recognized - Only registered students allowed", (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="vote",
        video_processor_factory=VoteProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if "recognized_reg" in st.session_state:
        reg = st.session_state.recognized_reg
        student = students.get(reg)

        if student:
            st.success(f"✅ Recognized: **{student['name']}** (Reg: {reg})")

            if student.get("voted"):
                st.error("❌ You have already voted!")
            elif not candidates:
                st.warning("No candidates available yet.")
            else:
                st.subheader("Choose your candidate")
                cols = st.columns(len(candidates))
                for idx, cand in enumerate(candidates):
                    with cols[idx]:
                        if cand.get("photo"):
                            st.image(cand["photo"], use_column_width=True)
                        st.write(f"**{cand['name']}**")
                        st.write(f"Dept: {cand.get('department', '')}")
                        if st.button(f"Vote for {cand['name']}", key=f"vote_{idx}"):
                            student["voted"] = True
                            votes[cand["name"]] = votes.get(cand["name"], 0) + 1
                            save_data()
                            st.success(f"🎉 Vote recorded for **{cand['name']}**!")
                            st.balloons()
                            if "recognized_reg" in st.session_state:
                                del st.session_state.recognized_reg
                            st.rerun()

    if st.button("Clear Recognition & Try Again"):
        st.session_state.pop("recognized_reg", None)
        st.rerun()

# ====================== ADMIN PAGES ======================
elif st.session_state.admin_logged_in:
    if page == "Register Student":
        st.header("📝 Register New Student")
        reg_number = st.text_input("Registration Number", placeholder="REG12345").strip().upper()
        name = st.text_input("Full Name", placeholder="John Doe").strip()

        if st.button("Start Face Registration"):
            if reg_number and name:
                if reg_number in students:
                    st.error("❌ Registration number already exists!")
                else:
                    st.session_state.reg_number = reg_number
                    st.session_state.name = name
                    st.session_state.faces = []
                    st.rerun()

        if "reg_number" in st.session_state:
            st.success(f"Recording face data for **{st.session_state.name}**")

            class RegisterProcessor(VideoProcessorBase):
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')\
                        .detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

                    for (x, y, w, h) in faces:
                        face_gray = gray[y:y+h, x:x+w]
                        if is_good_lighting(face_gray):
                            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            resized = cv2.resize(face_gray, (100, 100))
                            if len(st.session_state.faces) < 250 and len(st.session_state.faces) % 3 == 0:
                                st.session_state.faces.append(resized)
                        else:
                            cv2.putText(img, "Bad Lighting - Improve light", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.putText(img, f"Collected: {len(st.session_state.faces)}/250", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(
                key="register",
                video_processor_factory=RegisterProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )

            if len(st.session_state.get("faces", [])) >= 250:
                students[st.session_state.reg_number] = {
                    "name": st.session_state.name,
                    "faces": st.session_state.faces,
                    "voted": False
                }
                train_recognizer()
                save_data()
                st.success("✅ Face data recorded and student registered!")
                st.balloons()
                for key in ["reg_number", "name", "faces"]:
                    st.session_state.pop(key, None)
                st.rerun()

    elif page == "Manage Candidates":
        st.header("Manage Candidates")
        st.subheader("Add New Candidate")
        c_name = st.text_input("Candidate Name")
        c_dept = st.text_input("Department")
        c_photo = st.file_uploader("Candidate Photo (optional)", type=["jpg", "png", "jpeg"])

        if st.button("Add Candidate"):
            if c_name and c_dept:
                photo_bytes = c_photo.read() if c_photo else None
                candidates.append({"name": c_name.strip(), "department": c_dept.strip(), "photo": photo_bytes})
                save_data()
                st.success("Candidate added!")
                st.rerun()

        st.subheader("Current Candidates")
        for cand in candidates:
            st.write(f"**{cand['name']}** - {cand.get('department', '')}")
            if cand.get("photo"):
                st.image(cand["photo"], width=200)

    elif page == "View Results":
        st.header("📊 Election Results")
        if votes:
            st.bar_chart(votes)
            for cand, count in sorted(votes.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{cand}**: {count} votes")
        else:
            st.info("No votes yet.")

st.caption("Only students whose faces were recorded during registration can vote. Data is saved automatically.")
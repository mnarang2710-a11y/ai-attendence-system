"""
╔══════════════════════════════════════════════════════════════════╗
║      AI ATTENDANCE SYSTEM  —  FINAL VERSION (ALL BUGS FIXED)    ║
╠══════════════════════════════════════════════════════════════════╣
║  Install (CMD only, NOT PowerShell):                             ║
║    py -3.11 -m pip install opencv-contrib-python numpy           ║
║                            pandas requests                       ║
║                                                                  ║
║  Run:                                                            ║
║    py -3.11 attendance_final.py                                  ║
║                                                                  ║
║  Folder layout:                                                  ║
║    project/                                                      ║
║    ├── attendance_final.py                                       ║
║    ├── known_faces/                                              ║
║    │   ├── Manish.jpg   ← filename = student name (title case)   ║
║    │   └── Priya.jpg                                             ║
║    └── attendance.csv   ← auto-created, delete to reset          ║
║                                                                  ║
║  Camera controls:                                                ║
║    Q → quit                                                      ║
║    E → end-of-day (marks unseen students as Absent + sends n8n)  ║
╠══════════════════════════════════════════════════════════════════╣
║  BUGS FIXED IN THIS VERSION:                                     ║
║  1. Name case mismatch  (manish vs Manish) — fixed everywhere    ║
║  2. Absent loop indentation — now correctly loops all students   ║
║  3. Duplicate n8n calls removed — mark_attendance() is sole      ║
║     authority for CSV + n8n                                      ║
║  4. n8n retries (3x) with proper timeout                         ║
║  5. send_to_n8n signature consistent — confidence always passed  ║
║  6. Absent students correctly excluded from Present count        ║
║  7. Status field sent matches n8n Switch node cases exactly:     ║
║     "Present" / "Late" / "Absent" / "Duplicate"                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
#   CONFIGURATION  — only edit this section
# ══════════════════════════════════════════════════════════════════════════════

N8N_WEBHOOK_URL  = "http://localhost:5678/webhook-test/attendence"
CLASS_START_TIME = "09:15"       # students after this time → "Late"
CSV_FILE         = "attendance.csv"
KNOWN_FACES_DIR  = "known_faces"
CONFIDENCE_LIMIT = 80            # 0=strictest, 100=loosest. Keep 70–90.

# Photo filename (without extension) MUST match the key below exactly.
# e.g.  known_faces/Manish.jpg  →  key = "Manish"
STUDENT_EMAILS = {
    "Manish" : "mnarang2710@gmail.com",
    "Priya"  : "mnarang2710@gmail.com",
    # add more:  "StudentName" : "student@email.com",
}

# ══════════════════════════════════════════════════════════════════════════════
#   HELPERS  — case-insensitive name/email tools
# ══════════════════════════════════════════════════════════════════════════════

def normalize(name: str) -> str:
    """Always returns title-case: 'manish' → 'Manish', 'PRIYA' → 'Priya'."""
    return name.strip().title()


def get_email(name: str) -> str:
    """Case-insensitive email lookup so 'manish' finds key 'Manish'."""
    name_lower = name.lower().strip()
    for key, val in STUDENT_EMAILS.items():
        if key.lower() == name_lower:
            return val
    return "unknown@gmail.com"

# ══════════════════════════════════════════════════════════════════════════════
#   STEP 1 — OpenCV face detector  (no extra install needed)
# ══════════════════════════════════════════════════════════════════════════════

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ══════════════════════════════════════════════════════════════════════════════
#   STEP 2 — Train LBPH recognizer from known_faces/ folder
# ══════════════════════════════════════════════════════════════════════════════

recognizer    = cv2.face.LBPHFaceRecognizer_create()
label_to_name : dict = {}   # {0: "Manish", 1: "Priya", …}


def load_known_faces() -> bool:
    faces_pixels : list = []
    labels       : list = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"\n  Folder '{KNOWN_FACES_DIR}/' created.")
        print("  Add one photo per student (filename = name) then restart.\n")
        return False

    label_id = 0
    for filename in sorted(os.listdir(KNOWN_FACES_DIR)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path  = os.path.join(KNOWN_FACES_DIR, filename)
        raw   = os.path.splitext(filename)[0]
        name  = normalize(raw)           # always title-case
        image = cv2.imread(path)

        if image is None:
            print(f"  ⚠  Cannot read {filename} – skipping")
            continue

        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            print(f"  ⚠  No face detected in {filename} – skipping")
            continue

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # largest face
        face_roi   = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

        faces_pixels.append(face_roi)
        labels.append(label_id)
        label_to_name[label_id] = name
        label_id += 1
        print(f"  ✅  Enrolled: {name}")

    if not faces_pixels:
        print("\n  ❌  No valid faces loaded. Add photos to known_faces/ and restart.\n")
        return False

    recognizer.train(faces_pixels, np.array(labels))
    print(f"\n  Recognizer ready — {len(faces_pixels)} student(s) enrolled.\n")
    return True

# ══════════════════════════════════════════════════════════════════════════════
#   STEP 3 — CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def init_csv():
    """Create CSV with header if it does not exist yet."""
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(
            columns=["Name", "Email", "Date", "Time", "Status"]
        ).to_csv(CSV_FILE, index=False)
        print(f"  Created {CSV_FILE}\n")


def already_in_csv(name: str) -> bool:
    """
    Returns True if this student already has ANY record for today.
    Comparison is case-insensitive to fix 'manish' vs 'Manish' bug.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    df    = pd.read_csv(CSV_FILE)
    match = (
        (df["Name"].str.lower() == name.lower()) &
        (df["Date"] == today)
    )
    return not df[match].empty


def write_to_csv(name: str, email: str, status: str):
    """Writes one row. Name is always stored in title-case."""
    row = pd.DataFrame([[
        normalize(name),                          # title-case always
        email,
        datetime.now().strftime("%Y-%m-%d"),
        datetime.now().strftime("%H:%M:%S"),
        status
    ]], columns=["Name", "Email", "Date", "Time", "Status"])
    row.to_csv(CSV_FILE, mode="a", header=False, index=False)
    print(f"  💾  CSV  →  {normalize(name)} | {status}")

# ══════════════════════════════════════════════════════════════════════════════
#   STEP 4 — Send payload to n8n webhook
#            Matches the field names your n8n Switch/Filter nodes expect.
#            Retries 3 times on connection failure.
# ══════════════════════════════════════════════════════════════════════════════

def send_to_n8n(name: str, status: str, email: str, confidence: float = 0):
    """
    Sends attendance event to n8n.
    status must be one of: "Present" | "Late" | "Absent" | "Duplicate"
    These match the Switch node cases in your n8n workflow.
    """
    payload = {
        "student_name" : normalize(name),
        "status"       : status,              # exactly matches n8n Switch cases
        "parent_email" : email,
        "date"         : datetime.now().strftime("%Y-%m-%d"),
        "time"         : datetime.now().strftime("%H:%M:%S"),
        "confidence"   : int(confidence),
    }

    for attempt in range(1, 4):
        try:
            r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
            print(f"  📡  n8n {r.status_code}  |  {normalize(name)}  |  {status}")
            return
        except requests.exceptions.ConnectionError:
            print(f"  ⚠   n8n unreachable (attempt {attempt}/3) — is n8n running?")
        except requests.exceptions.Timeout:
            print(f"  ⚠   n8n timeout (attempt {attempt}/3)")
        except Exception as exc:
            print(f"  ❌  n8n error: {exc}")
            return

    print(f"  ❌  n8n failed after 3 attempts for {name}. Saved to CSV only.")

# ══════════════════════════════════════════════════════════════════════════════
#   STEP 5 — Core attendance logic
#            Single source of truth: writes CSV + sends n8n in one place only.
#            Camera loop calls ONLY this function — no direct send_to_n8n there.
# ══════════════════════════════════════════════════════════════════════════════

already_marked : set = set()   # in-memory guard; resets each run


def mark_attendance(name: str, confidence: float = 0):
    name  = normalize(name)
    email = get_email(name)

    # ── Guard 1: already processed in this camera session ──────────────────
    if name in already_marked:
        return   # silent — screen already shows the label

    # ── Guard 2: already in CSV for today ──────────────────────────────────
    if already_in_csv(name):
        print(f"  ⚠   {name} already marked today → duplicate notice sent")
        already_marked.add(name)
        send_to_n8n(name, "Duplicate", email, confidence)
        return

    # ── Determine correct status ────────────────────────────────────────────
    now_hhmm = datetime.now().strftime("%H:%M")
    status   = "Late" if now_hhmm > CLASS_START_TIME else "Present"

    # ── Save + notify (ONE call each, correct status) ───────────────────────
    write_to_csv(name, email, status)
    send_to_n8n(name, status, email, confidence)

    already_marked.add(name)
    print(f"  ✅  {name}  →  {status}  at  {datetime.now().strftime('%H:%M:%S')}\n")

# ══════════════════════════════════════════════════════════════════════════════
#   STEP 6 — End-of-day absent marking  (press E in camera window)
#            Loops EVERY enrolled student, marks those not seen today as Absent.
#            BUG FIX: all lines now correctly inside the for loop.
#            BUG FIX: uses lowercase comparison so "manish" == "Manish".
# ══════════════════════════════════════════════════════════════════════════════

def mark_end_of_day_absents():
    print("\n  ⏰  End-of-day check running …\n")
    today = datetime.now().strftime("%Y-%m-%d")
    df    = pd.read_csv(CSV_FILE)

    # build seen-set in lowercase — fixes the case-mismatch bug
    seen = set(n.lower() for n in df[df["Date"] == today]["Name"].tolist())

    absent_count  = 0
    present_count = len(seen)

    for name in label_to_name.values():                    # ← for loop
        email = get_email(name)                            # ← INSIDE loop ✅

        if name.lower() not in seen:                       # ← INSIDE loop ✅
            write_to_csv(name, email, "Absent")            # ← INSIDE loop ✅
            send_to_n8n(name, "Absent", email, 0)          # ← INSIDE loop ✅
            print(f"  📋  {name}  →  Absent")              # ← INSIDE loop ✅
            absent_count += 1                              # ← INSIDE loop ✅

    print(f"\n  ── End-of-day Summary ──")
    print(f"  Present / Late : {present_count}")
    print(f"  Absent         : {absent_count}")
    print(f"  Total enrolled : {len(label_to_name)}\n")

# ══════════════════════════════════════════════════════════════════════════════
#   STEP 7 — Camera loop
# ══════════════════════════════════════════════════════════════════════════════

def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ⚠  Camera 0 not found, trying Camera 1 …")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("  ❌  No camera found. Check connections.")
        return

    print("─" * 58)
    print("  📷  Camera ON")
    print("  Q  →  quit")
    print("  E  →  end-of-day (marks unseen students as Absent)")
    print("─" * 58 + "\n")

    process_this_frame = True   # skip alternate frames → faster on low-end PCs

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("  ❌  Camera read failed.")
            break

        if process_this_frame:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                face_roi    = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                label, conf = recognizer.predict(face_roi)

                if conf < CONFIDENCE_LIMIT:
                    # ── Known student ────────────────────────────────────
                    name  = label_to_name.get(label, "Unknown")
                    color = (0, 220, 0)                      # green
                    tag   = f"{normalize(name)}  ({int(conf)})"

                    # ✅ ONLY call — handles CSV + n8n inside
                    mark_attendance(name, conf)

                else:
                    # ── Unknown face ─────────────────────────────────────
                    name  = "Unknown"
                    color = (0, 0, 220)                      # red
                    tag   = f"Unknown  ({int(conf)})"

                # draw bounding box + name label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-32), (x+w, y), color, cv2.FILLED)
                cv2.putText(frame, tag, (x+6, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2)

        process_this_frame = not process_this_frame

        # ── Top status bar ───────────────────────────────────────────────
        bar = (f"  Marked: {len(already_marked)}/{len(label_to_name)}"
               f"  |  {datetime.now().strftime('%H:%M:%S')}"
               f"  |  Q=quit  E=absent")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30),
                      (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, bar, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 255, 200), 1)

        cv2.imshow("AI Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n  Closing camera …")
            break
        elif key == ord("e"):
            mark_end_of_day_absents()

    cap.release()
    cv2.destroyAllWindows()

# ══════════════════════════════════════════════════════════════════════════════
#   ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 58)
    print("    AI ATTENDANCE SYSTEM  —  FINAL VERSION")
    print("═" * 58 + "\n")

    print("  Loading known faces …")
    if not load_known_faces():
        input("\n  Press Enter to exit.")
        raise SystemExit

    init_csv()
    run_camera()

    # ── Print today's full attendance summary after camera closes ────────
    print("\n  ── Today's Full Attendance Summary ──")
    df    = pd.read_csv(CSV_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    summary = df[df["Date"] == today].copy()

    if summary.empty:
        print("  No records found for today.")
    else:
        # show counts
        for status in ["Present", "Late", "Absent"]:
            count = len(summary[summary["Status"] == status])
            
            print(f"  {status:<10} : {count}")
        print()
        print(summary.to_string(index=False))

    print()
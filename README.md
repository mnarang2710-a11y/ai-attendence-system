# AI Attendance System

An AI-powered attendance system using face recognition to automatically mark attendance and send alerts.

## 🚀 Features
- Face recognition using OpenCV
- Automated attendance marking
- Late/absence detection
- Email notifications (via n8n)
- Basic attendance tracking

## 🛠️ Tech Stack
- Python
- OpenCV
- n8n (automation)
- CSV (data storage)

## 🔗 n8n Automation

This project integrates with n8n for workflow automation:

- Webhook receives attendance data
- IF node checks status (Late/Present)
- Email notifications are sent automatically
-  E-mail reply sent automatically 

### 📂 Workflow
See: n8n-workflows:-attendance_alert.json

## 📂 Project Structure
- recognizer.py → main face recognition logic
- known_faces/ → dataset (not required in production)
- n8n-workflows:-attendance_alert.json

## ▶️ How to Run
1. Install dependencies:
2. Run:
3. ## 📌 Future Improvements
- Add database integration
- Improve accuracy
- Dashboard for analytics

---


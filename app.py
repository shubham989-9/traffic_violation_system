from flask import Flask, render_template, Response, request, redirect, url_for, send_file, flash, jsonify
from ultralytics import YOLO
import easyocr
import cv2
import os
import csv
import time
from datetime import datetime
import pandas as pd

app = Flask(__name__)
app.secret_key = "traffic-secret-123"

# ========= Paths =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE_DIR, "uploads")
IMG_DIR = os.path.join(UPLOADS, "images")
IN_VID_DIR = os.path.join(UPLOADS, "input_videos")
OUT_VID_DIR = os.path.join(UPLOADS, "output_videos")
CROPS_DIR = os.path.join(UPLOADS, "plate_crops")

for d in [UPLOADS, IMG_DIR, IN_VID_DIR, OUT_VID_DIR, CROPS_DIR]:
    os.makedirs(d, exist_ok=True)

CSV_PATH = os.path.join(BASE_DIR, "violations.csv")
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "vehicle", "violation", "plate", "media_path"])

# ========= Models =========
helmet_model = YOLO("helmet.pt")      # your helmet model
plate_model = YOLO("plate.pt")        # your plate model
obj_model = YOLO("yolov8n.pt")        # general model

# ========= OCR =========
reader = easyocr.Reader(['en'])

# ========= Camera =========
camera = cv2.VideoCapture(0)

# ========= Utils =========
def ocr_text(img):
    if img is None or img.size == 0: return ""
    try:
        r = reader.readtext(img, detail=0)
        if not r: return ""
        return "".join(ch for ch in r[0] if ch.isalnum()).upper()
    except:
        return ""

def save_violation(vehicle, violation, plate, media):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), vehicle, violation, plate, media])

def annotate(frame, tag="live"):
    ann = frame.copy()
    obj = obj_model(frame, verbose=False)
    ann = obj[0].plot()

    counts = {"person":0,"motorcycle":0,"car":0}
    for b in obj[0].boxes:
        cls = obj[0].names[int(b.cls[0])]
        if cls in counts: counts[cls] += 1

    persons = counts["person"]
    bikes = counts["motorcycle"]
    vehicle = "Bike" if bikes else "Car"

    # Helmet detection
    helmet = helmet_model(frame, verbose=False)
    ann = helmet[0].plot()
    no_helmet = any(
        "without" in str(helmet[0].names[int(b.cls[0])]).lower()
        for b in helmet[0].boxes
    )

    # Triple seat
    triple = persons >= 3 and bikes >= 1

    # Plate + OCR
    plate = ""
    p = plate_model(frame, verbose=False)
    ann = p[0].plot()
    for i, b in enumerate(p[0].boxes):
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            txt = ocr_text(crop)
            if txt: plate = txt

    violations = []
    if no_helmet: violations.append("No Helmet")
    if triple: violations.append("Triple Seating")

    if violations:
        save_violation(vehicle, ",".join(violations), plate, tag)

    status = " | ".join(violations) if violations else "Normal"
    cv2.putText(ann, status, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if violations else (0,255,0), 2)
    return ann

def gen_frames():
    while True:
        ret, frame = camera.read()
        if not ret: break
        ann = annotate(frame, "camera")
        _, buffer = cv2.imencode(".jpg", ann)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# ✅ ROUTES

@app.route("/")
def home():
    try:
        rows = pd.read_csv(CSV_PATH).tail(20).values.tolist()
        rows.reverse()
    except:
        rows = []
    return render_template("index.html", rows=rows)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/upload_image", methods=["POST"])
def upload_image():
    f = request.files["image_file"]
    path = os.path.join(IMG_DIR, f.filename)
    f.save(path)
    img = cv2.imread(path)
    ann = annotate(img, path)
    cv2.imwrite(path.replace(".","_OUT."), ann)
    flash("Image Processed ✅")
    return redirect("/")

@app.route("/upload_video", methods=["POST"])
def upload_video():
    f = request.files["video_file"]
    path = os.path.join(IN_VID_DIR, f.filename)
    f.save(path)

    cap = cv2.VideoCapture(path)
    out = cv2.VideoWriter(os.path.join(OUT_VID_DIR, "output_"+f.filename),
                          cv2.VideoWriter_fourcc(*"mp4v"), 25,
                          (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(annotate(frame, path))
    cap.release(); out.release()

    flash("Video Processed ✅")
    return redirect("/")

@app.route("/download_csv")
def download_csv():
    return send_file(CSV_PATH, as_attachment=True)

@app.route("/stats")
def stats():
    df = pd.read_csv(CSV_PATH)
    v = df.violation.value_counts().to_dict()
    return jsonify(v)

if __name__ == "__main__":
    app.run(debug=True)

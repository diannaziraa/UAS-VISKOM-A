import cv2
import math
from collections import OrderedDict
import numpy as np
import time

# --- KELAS TRACKER SEDERHANA (IMPLEMENTASI TRACKING) ---
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # Hitung jarak antar centroid
            D = []
            for i in range(len(objectCentroids)):
                row = []
                for j in range(len(inputCentroids)):
                    dist = math.hypot(objectCentroids[i][0] - inputCentroids[j][0], objectCentroids[i][1] - inputCentroids[j][1])
                    row.append(dist)
                D.append(row)
            D = np.array(D)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                # REVISI: Threshold jarak dinaikkan jadi 100 piksel
                # Agar kalau kepala gerak agak cepat, ID tidak ganti-ganti
                if D[row][col] < 100: 
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.disappeared[objectID] = 0
                    usedRows.add(row)
                    usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('ERROR: Gagal mengakses kamera.')

# Pemuatan Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi Tracker
tracker = CentroidTracker(maxDisappeared=20)

print("Sistem dimulai... Tekan 'q' pada jendela video untuk keluar.")
print("-" * 50)
print(f"{'WAKTU':<20} | {'TOTAL ORANG':<15} | {'LIST ID'}")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. PRE-PROCESSING
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. IMPLEMENTASI CANNY EDGE DETECTION
    edges = cv2.Canny(gray, 50, 150)
    cv2.imshow('Fitur Tepi (Canny Edge)', edges) 

    # 3. DETEKSI WAJAH (HAAR CASCADE) - REVISI PARAMETER
    # scaleFactor=1.2 (naik sedikit biar lebih cepat)
    # minNeighbors=8 (Dinaikkan dari 5 -> 8 agar lebih ketat, mengurangi hantu)
    # minSize=(80, 80) (Dinaikkan dari 30 -> 80 agar noise kecil tidak dianggap wajah)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=8, 
        minSize=(80, 80)
    )

    # Siapkan list kotak untuk tracker
    rects = []
    for (x, y, w, h) in faces:
        rects.append((x, y, x+w, y+h)) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 4. IMPLEMENTASI TRACKING & SIMULASI RECOGNITION
    objects = tracker.update(rects)

    for (objectID, centroid) in objects.items():
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)

    # --- OUTPUT KE TERMINAL VS CODE ---
    if len(objects) > 0:
        waktu_sekarang = time.strftime("%H:%M:%S")
        list_id = list(objects.keys())
        print(f"{waktu_sekarang:<20} | {len(objects):<15} | {list_id}")

    # Tampilkan Info Jumlah di Layar
    cv2.putText(frame, f'Total Orang Terdeteksi: {len(objects)}', (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.putText(frame, 'Tekan Q untuk keluar', (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow('Face Tracking & Recognition System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("-" * 50)
        print("Sistem dihentikan pengguna.")
        break

cap.release()
cv2.destroyAllWindows()
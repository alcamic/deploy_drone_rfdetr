import cv2
import numpy as np
import mss
import pygetwindow as gw
import torch
import time
from collections import deque

import supervision as sv
from ultralytics import YOLO #

def run_yolo_detector():
    print("=== SISTEM DETEKSI DRONE MEMULAI (VERSI YOLO) ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO] GPU Terdeteksi: {gpu_name}")
        print("[INFO] Model akan otomatis berjalan dengan akselerasi CUDA.")
    else:
        print("[WARN] GPU tidak terdeteksi! PyTorch akan menggunakan CPU.")
        print("[WARN] FPS mungkin akan sangat rendah/patah-patah.")

    print("\n[INFO] Memuat bobot model YOLO 'best.pt'...")
    # [UBAH] Load model YOLO. Ultralytics otomatis menggunakan GPU jika tersedia.
    model = YOLO("yolo12m_finetune.pt") 

    sct = mss.MSS()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    WINDOW_TITLE = "Pocophone F1"
    print(f"\n[INFO] Mencari window dengan nama '{WINDOW_TITLE}'...")
    print("[INFO] Pastikan HP tidak dalam keadaan terkunci (layar mati/redup).")
    print("[INFO] Tekan tombol 'q' pada keyboard di jendela video untuk keluar.\n")

    prev_frame_time = 0
    
    # Inisialisasi penyimpan riwayat (history) untuk 60 frame terakhir
    history_length = 60
    fps_history = deque(maxlen=history_length)
    latency_history = deque(maxlen=history_length)

    while True:
        windows = gw.getWindowsWithTitle(WINDOW_TITLE)
        
        if not windows:
            print(f"[WAIT] Menunggu... Window '{WINDOW_TITLE}' tidak ditemukan.")
            cv2.waitKey(2000)
            continue
            
        win = windows[0]
        
        if win.isMinimized:
            print("[WAIT] Window sedang di-minimize. Layar tidak bisa ditangkap. Silakan buka kembali.")
            cv2.waitKey(1000)
            continue
        
        monitor = {
            "top": win.top + 30,  
            "left": win.left + 8, 
            "width": win.width - 16, 
            "height": win.height - 38 
        }
        
        sct_img = sct.grab(monitor)
        
        frame_bgr = np.array(sct_img)
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
        # YOLO v8/v9/v11 Ultralytics bisa menerima BGR langsung, tapi menggunakan RGB lebih aman secara standar
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # --- MULAI HITUNG LATENCY ---
        start_inference_time = time.time()
        
        # [UBAH] Inferensi menggunakan YOLO. verbose=False agar terminal tidak penuh dengan log
        results = model(frame_rgb, conf=0.5, verbose=False)[0]
        
        end_inference_time = time.time()
        
        # Simpan data Latency ke dalam history
        latency_ms = (end_inference_time - start_inference_time) * 1000
        latency_history.append(latency_ms)
        # --- SELESAI HITUNG LATENCY ---
        
        # [UBAH] Konversi hasil Ultralytics ke format Supervision
        detections = sv.Detections.from_ultralytics(results)
        
        # [UBAH] Generate Label. YOLO menyimpan nama class di dalam model.names
        labels = [
            f"{model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        
        annotated_frame = box_annotator.annotate(scene=frame_bgr.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # --- MULAI HITUNG FPS ---
        new_frame_time = time.time()
        time_diff = new_frame_time - prev_frame_time if (new_frame_time - prev_frame_time) > 0 else 0.001
        fps = 1 / time_diff
        prev_frame_time = new_frame_time
        
        # Simpan data FPS ke dalam history
        fps_history.append(fps)
        # --- SELESAI HITUNG FPS ---

        # Hitung Nilai Rata-rata (Avg), Minimum (Min), dan Maksimum (Max)
        avg_fps = sum(fps_history) / len(fps_history)
        min_fps = min(fps_history)
        max_fps = max(fps_history)

        avg_lat = sum(latency_history) / len(latency_history)
        min_lat = min(latency_history)
        max_lat = max(latency_history)
        
        # Buat format teks untuk ditampilkan
        text_fps = f"FPS -> Avg: {int(avg_fps)} | Min: {int(min_fps)} | Max: {int(max_fps)}"
        text_lat = f"Lat -> Avg: {avg_lat:.1f}ms | Min: {min_lat:.1f}ms | Max: {max_lat:.1f}ms"

        # Tampilkan teks FPS (Hijau)
        cv2.putText(annotated_frame, text_fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Tampilkan teks Latency (Merah)
        cv2.putText(annotated_frame, text_lat, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("YOLO (GPU Mode) - DJI Air 3S", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Menutup program...")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_yolo_detector()
import cv2
import numpy as np
import mss
import pygetwindow as gw
import torch

import supervision as sv
from rfdetr import RFDETRMedium

def run_custom_rfdetr():
    print("=== SISTEM DETEKSI DRONE MEMULAI ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO]GPU Terdeteksi: {gpu_name}")
        print("[INFO] Model akan otomatis berjalan dengan akselerasi CUDA.")
    else:
        print("[WARN] GPU tidak terdeteksi! PyTorch akan menggunakan CPU.")
        print("[WARN] FPS mungkin akan sangat rendah/patah-patah.")

    CUSTOM_CLASSES = ["person"]

    print("\n[INFO] Memuat custom weights 'checkpoint_best_total.pth'...")
    model = RFDETRMedium(
        pretrain_weights="checkpoint_best_total.pth",
        num_classes=len(CUSTOM_CLASSES) 
    )
    
    # Optimasi model agar lebih efisien di memori VRAM GPU untuk video real-time
    print("[INFO] Mengoptimasi model untuk inference real-time...")
    model.optimize_for_inference()

    sct = mss.MSS()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Sesuaikan dengan nama jendela Scrcpy Anda yang sebenarnya
    WINDOW_TITLE = "Pocophone F1"
    print(f"\n[INFO] Mencari window dengan nama '{WINDOW_TITLE}'...")
    print("[INFO] Pastikan HP tidak dalam keadaan terkunci (layar mati/redup).")
    print("[INFO] Tekan tombol 'q' pada keyboard di jendela video untuk keluar.\n")

    while True:
        # Cari window berdasarkan judulnya
        windows = gw.getWindowsWithTitle(WINDOW_TITLE)
        
        # Jika window belum terbuka atau salah nama
        if not windows:
            print(f"[WAIT] Menunggu... Window '{WINDOW_TITLE}' tidak ditemukan.")
            cv2.waitKey(2000) # Tunggu 2 detik sebelum mencari lagi
            continue
            
        win = windows[0]
        
        # Cegah error jika window sedang di-minimize ke taskbar
        if win.isMinimized:
            print("[WAIT] Window sedang di-minimize. Layar tidak bisa ditangkap. Silakan buka kembali.")
            cv2.waitKey(1000)
            continue
        
        # Ambil koordinat terbaru window (otomatis mengikuti jika Anda menggeser jendelanya)
        # Offset: +30 atas, +8 kiri, untuk memotong Title Bar dan Bingkai Windows
        monitor = {
            "top": win.top + 30,  
            "left": win.left + 8, 
            "width": win.width - 16, 
            "height": win.height - 38 
        }
        
        # Tangkap area layar tersebut
        sct_img = sct.grab(monitor)
        
        # Konversi format gambar mss (BGRA) menjadi format standar OpenCV (BGR)
        frame_bgr = np.array(sct_img)
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
        
        # RF-DETR mewajibkan input gambar dalam format RGB murni
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Jalankan Prediksi RF-DETR (dikirim otomatis ke GPU oleh library)
        detections = model.predict(frame_rgb, threshold=0.5)
        
        # Generate Label dari hasil deteksi
        labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            # Proteksi agar tidak index out of range jika model mengeluarkan ID aneh
            if class_id < len(CUSTOM_CLASSES):
                class_name = CUSTOM_CLASSES[class_id]
            else:
                class_name = f"Unknown_{class_id}"
                
            labels.append(f"{class_name} {confidence:.2f}")
        
        # Gambar Bounding Box dan Teks ke atas frame (menggunakan frame BGR untuk ditampilkan OpenCV)
        annotated_frame = box_annotator.annotate(scene=frame_bgr.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Tampilkan hasil akhir
        cv2.imshow("RF-DETR (GPU Mode) - DJI Air 3S", annotated_frame)
        
        # Deteksi tombol 'q' untuk keluar dengan aman
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Menutup program...")
            break

    # Bersihkan memori RAM/VRAM
    cv2.destroyAllWindows()

# Blok pelindung ini sangat penting di OS Windows untuk mencegah error saat model dimuat
if __name__ == '__main__':
    run_custom_rfdetr()
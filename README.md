**Identitas Pengembang**
Nama  : Dian Nazira
NPM   : 2308107010011

**Sistem Pemantauan Kehadiran Otomatis (Face Counting)**
  Proyek ini adalah sistem berbasis Computer Vision yang dirancang untuk mendeteksi, melacak, dan menghitung jumlah orang di dalam ruangan secara real-time menggunakan kamera laptop.

**Fitur Utama**
**1. Deteksi Wajah Akurat:** Menggunakan algoritma Haar Cascade dengan parameter yang diperketat untuk mengurangi kesalahan deteksi.
**2. Pelacakan Objek (Object Tracking):** Menggunakan Centroid Tracker untuk memberikan ID unik pada setiap wajah dan mencegah penghitungan ganda.
**3. Analisis Visual:** Menampilkan visualisasi tepi (Canny Edge) di jendela terpisah untuk melihat struktur wajah.
**4. Logging Real-Time:** Mencetak data waktu, jumlah orang, dan ID ke terminal secara langsung.

**Prasyarat (Requirements)**
**Pastikan Anda telah menginstal Python dan pustaka berikut:**
- pip install opencv-python numpy

**Struktur File**
**- Main.py:** Kode program utama (Python).
**- haarcascade_frontalface_default.xml:** Model deteksi wajah (biasanya sudah otomatis ada di library OpenCV, tapi pastikan aksesnya benar).
**- README.md:** Dokumen ini.

**Cara Menjalankan**
1. Buka terminal atau VS Code.
2. Jalankan perintah berikut:
    python Main.py
   
3. Akan muncul dua jendela:
**- Face Tracking & Recognition:** Tampilan utama dengan kotak hijau dan ID.
**- Fitur Tepi (Canny Edge):** Visualisasi garis kontur.

4. Lihat terminal untuk log data kehadiran.
5. Tekan tombol 'q' pada jendela video untuk keluar.

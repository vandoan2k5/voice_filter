# Luồng Công việc (Work Flow)

Luồng công việc tổng thể của dự án VoiceFilter từ khâu chuẩn bị dữ liệu đến khi ra kết quả cuối cùng được chia thành 5 bước chính:

## 1. Thu thập và Chuẩn bị Dữ liệu (Data Acquisition)
- **Nguồn**: Tải bộ dữ liệu giọng nói (ví dụ: LibriSpeech hoặc VoxCeleb2).
- **Mục tiêu**: Có được các tệp âm thanh sạch của nhiều người nói khác nhau.

## 2. Tiền xử lý Âm thanh (Audio Preprocessing)
- **Chuẩn hóa**: Sử dụng `utils/normalize-resample.sh` để đưa tất cả tệp wav về cùng tốc độ lấy mẫu (16kHz) và mức âm lượng chuẩn.
- **Tạo dữ liệu hỗn hợp**: Sử dụng `generator.py` để:
  - Trộn ngẫu nhiên giọng người nói mục tiêu với một giọng nói khác làm nhiễu.
  - Tính toán phổ Magnitude Spectrogram bằng STFT.
  - Lưu trữ các cặp (Hỗn hợp, Mục tiêu) và tệp tham chiếu cho d-vector.

## 3. Trích xuất Đặc trưng Người nói (Speaker Embedding)
- **Tham chiếu**: Lấy một đoạn âm thanh ngắn của người nói mục tiêu.
- **Embedder**: Đưa qua mô hình `SpeechEmbedder` để tạo ra một **d-vector**. Vector này đóng vai trò là "chìa khóa" để mô hình biết cần phải lọc lấy giọng nói của ai.

## 4. Huấn luyện Mô hình (Model Training)
- **Đầu vào**: Phổ hỗn hợp + d-vector.
- **Quá trình**:
  - CNN trích xuất đặc trưng phổ.
  - LSTM kết hợp đặc trưng phổ và d-vector để hiểu ngữ cảnh.
  - Mô hình dự đoán một mặt nạ (Mask).
- **Tối ưu hóa**: So sánh phổ sau khi áp mặt nạ với phổ mục tiêu sạch, cập nhật trọng số thông qua hàm Loss (MSE).

## 5. Suy luận và Đánh giá (Inference & Evaluation)
- **Suy luận**: Sử dụng `inference.py` với một tệp âm thanh hỗn hợp và một tệp tham chiếu của người nói mục tiêu.
- **Khôi phục**: Áp dụng mặt nạ dự đoán lên phổ hỗn hợp $\rightarrow$ Chuyển đổi phổ ngược lại thành âm thanh (iSTFT) $\rightarrow$ Lưu tệp `.wav` kết quả.
- **Đánh giá**: Tính toán chỉ số **SDR (Signal-to-Distortion Ratio)** để đo lường chất lượng âm thanh được tách ra.

---
**Sơ đồ tóm tắt:**
`Dữ liệu thô` $\rightarrow$ `Chuẩn hóa` $\rightarrow$ `Tạo Spectrogram` $\rightarrow$ `Huấn luyện` $\rightarrow$ `Tách giọng nói` $\rightarrow$ `Đánh giá SDR`

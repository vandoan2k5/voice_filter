# Chuẩn bị Bộ dữ liệu (Dataset Preparation)

Quá trình chuẩn bị dữ liệu cho VoiceFilter bao gồm việc tải dữ liệu, chuẩn hóa âm thanh và tạo các cặp phổ (spectrogram) hỗn hợp và mục tiêu.

## 1. Tải Bộ dữ liệu
Để tái lập kết quả như trong bài báo, bạn nên sử dụng bộ dữ liệu **LibriSpeech** tại [http://www.openslr.org/12/](http://www.openslr.org/12/).
- `train-clear-100.tar.gz` (6.3G): chứa giọng nói của 252 người nói.
- `train-clear-360.tar.gz` (23G): chứa giọng nói của 922 người nói.
- Càng nhiều người nói trong bộ dữ liệu, mô hình VoiceFilter sẽ hoạt động càng hiệu quả.

## 2. Thay đổi Tốc độ lấy mẫu và Chuẩn hóa (Resample & Normalize)
Các tệp âm thanh cần được đưa về cùng một tốc độ lấy mẫu và mức âm lượng chuẩn.

1. Giải nén tệp dữ liệu:
   ```bash
   tar -xvzf train-clear-360.tar.gz
   ```
2. Sao chép tệp `utils/normalize-resample.sh` vào thư mục gốc của dữ liệu đã giải nén.
3. Cấu hình số nhân CPU (`N`) trong tệp `.sh` và chạy:
   ```bash
   chmod a+x normalize-resample.sh
   ./normalize-resample.sh
   ```

## 3. Tiền xử lý dữ liệu với `generator.py`
Để tăng tốc độ huấn luyện, thay vì tính toán STFT trong lúc train, chúng ta sẽ tính trước phổ Magnitude Spectrogram cho toàn bộ dữ liệu.

### Cách hoạt động của `generator.py`:
Hàm `mix` trong `generator.py` thực hiện các bước sau:
1. **Lựa chọn**: Chọn ngẫu nhiên hai người nói ($S1$ và $S2$).
2. **Tạo cặp**:
   - Lấy một đoạn âm thanh của $S1$ làm **âm thanh tham chiếu** (để trích xuất d-vector).
   - Lấy một đoạn âm thanh khác của $S1$ làm **âm thanh mục tiêu** (target).
   - Lấy một đoạn âm thanh của $S2$ làm **âm thanh nhiễu/nền**.
3. **Trộn (Mixing)**: Cộng tín hiệu của âm thanh mục tiêu và âm thanh nhiễu: $\text{mixed} = w_1 + w_2$.
4. **Chuẩn hóa**: Chia cho giá trị cực đại để tránh clipping.
5. **Lưu trữ**:
   - Lưu tệp wav của âm thanh mục tiêu và âm thanh hỗn hợp.
   - Chuyển đổi thành phổ Magnitude Spectrogram bằng STFT và lưu dưới dạng tệp `.pt` (PyTorch tensor).
   - Lưu đường dẫn của tệp tham chiếu để tính d-vector sau này.

### Lệnh chạy:
```bash
python generator.py -c [config yaml] -d [thư mục dữ liệu] -o [thư mục đầu ra] -p [số tiến trình chạy song song]
```
Lệnh này sẽ tạo ra khoảng 100,000 mẫu cho tập huấn luyện (train) và 1,000 mẫu cho tập kiểm tra (test).

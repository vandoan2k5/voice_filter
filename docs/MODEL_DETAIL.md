# Chi tiết Mô hình (Model Detail)

Mô hình VoiceFilter được thiết kế để thực hiện tách giọng nói mục tiêu bằng cách dự đoán một mặt nạ phổ (spectrogram mask) dựa trên điều kiện là đặc trưng của người nói.

## 1. Kiến trúc Tổng thể
Mô hình bao gồm ba thành phần chính: **CNN**, **LSTM**, và các **Lớp Tuyến tính (FC layers)**.

### A. Mạng CNN (Trích xuất Đặc trưng Cục bộ)
Đầu vào là phổ Magnitude Spectrogram của âm thanh hỗn hợp (Mixed Spectrogram).
- **Cấu trúc**: Gồm 8 lớp Convolution 2D.
- **Dilated Convolutions**: Các lớp CNN sau sử dụng *dilation* (giãn cách) tăng dần (1, 2, 4, 8, 16). Điều này cho phép mô hình mở rộng trường thụ cảm (receptive field) để bao quát nhiều thông tin thời gian và tần số hơn mà không làm tăng quá nhiều tham số.
- **Đầu ra**: Kết quả của CNN là một đặc trưng nén có kích thước `[Batch, 8, Time, Freq]`.

### B. Lớp LSTM (Mô hình hóa Chuỗi Thời gian)
Để hiểu ngữ cảnh thời gian của tín hiệu âm thanh:
- **Đầu vào**: Đặc trưng từ CNN được làm phẳng và kết hợp (concatenate) với **d-vector** (đặc trưng người nói).
  - $\text{Input} = [\text{CNN\_features} ; \text{d-vector}]$
- **Cấu trúc**: Sử dụng LSTM hai chiều (Bidirectional LSTM) để học thông tin từ cả hai phía (quá khứ và tương lai) của chuỗi âm thanh.
- **Đầu ra**: Một chuỗi các vector đặc trưng temporal.

### C. Lớp Tuyến tính và Đầu ra (Mask Prediction)
- **FC Layers**: Hai lớp Fully Connected với hàm kích hoạt ReLU.
- **Sigmoid**: Lớp cuối cùng sử dụng hàm Sigmoid để đảm bảo giá trị của mặt nạ nằm trong khoảng $[0, 1]$.
- **Kết quả**: Một mặt nạ phổ $\text{Mask}$ có cùng kích thước với phổ đầu vào.

$$\text{Estimated Spectrogram} = \text{Mixed Spectrogram} \times \text{Mask}$$

## 2. Thành phần SpeechEmbedder
`SpeechEmbedder` (trong `model/embedder.py`) chịu trách nhiệm tạo ra d-vector từ âm thanh tham chiếu:
- **LSTM**: Học đặc trưng từ phổ Mel-spectrogram của người nói.
- **Projection**: Một lớp tuyến tính để đưa đặc trưng về không gian embedding.
- **Normalization**: Vector kết quả được chuẩn hóa $L_2$ và lấy trung bình theo thời gian (average pooling) để tạo ra một vector đặc trưng duy nhất cho mỗi người nói.

## 3. Tóm tắt Luồng Xử lý
$\text{Mixed Spectrogram} \xrightarrow{\text{CNN}} \text{Local Features} \xrightarrow{+\text{d-vector}} \text{LSTM} \xrightarrow{\text{FC}} \text{Mask} \xrightarrow{\times \text{Mixed}} \text{Target Spectrogram}$

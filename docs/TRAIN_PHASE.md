# Giai đoạn Huấn luyện (Train Phase)

Quá trình huấn luyện VoiceFilter tập trung vào việc học cách tạo ra một mặt nạ (mask) để lọc lấy giọng nói mục tiêu từ một tín hiệu hỗn hợp, dựa trên đặc trưng của người nói (d-vector).

## 1. Mô hình Trích xuất Đặc trưng (Speaker Embedder)
VoiceFilter sử dụng hệ thống nhận dạng người nói để tạo ra các **d-vector embeddings**. Đây là các vector đặc trưng đại diện cho danh tính của người nói.
- Bạn cần một mô hình embedder đã được huấn luyện trước (pretrained) trên bộ dữ liệu như VoxCeleb2.
- Mô hình này sẽ chuyển đổi âm thanh tham chiếu của người nói thành một vector có kích thước cố định.

## 2. Cấu hình Huấn luyện
Trước khi chạy, hãy chỉnh sửa tệp `config.yaml` (ví dụ: `config/default.yaml`) để thiết lập:
- `train_dir`: Đường dẫn đến dữ liệu đã được tiền xử lý bởi `generator.py`.
- `test_dir`: Đường dẫn đến dữ liệu kiểm tra.
- Các siêu tham số (hyperparameters) như `learning_rate`, `batch_size`, `lstm_dim`, v.v.

## 3. Chạy Huấn luyện
Sử dụng tệp `trainer.py` để bắt đầu quá trình huấn luyện:

```bash
python trainer.py -c [config yaml] -e [đường dẫn file embedder pt] -m [tên mô hình]
```

**Giải thích các tham số:**
- `-c`: Tệp cấu hình YAML.
- `-e`: Đường dẫn đến tệp `.pt` của mô hình embedder tiền huấn luyện.
- `-m`: Tên định danh cho mô hình này (dùng để lưu checkpoint và log).

## 4. Quá trình Huấn luyện chi tiết
- **DataLoader**: `datasets/dataloader.py` sẽ tải các phổ Spectrogram hỗn hợp và mục tiêu, đồng thời sử dụng embedder để trích xuất d-vector từ âm thanh tham chiếu.
- **Hàm mất mát (Loss Function)**: Mô hình được tối ưu hóa để giảm thiểu sai số bình phương trung bình (MSE) giữa phổ mục tiêu và phổ sau khi áp dụng mặt nạ: $\text{Loss} = \text{MSE}(\text{Target}, \text{Mixed} \times \text{Mask})$.
- **Tối ưu hóa**: Sử dụng thuật toán AdaBound (trong `utils/adabound.py`) để điều chỉnh tốc độ học một cách linh hoạt.

## 5. Theo dõi và Khôi phục
- **Tensorboard**: Bạn có thể theo dõi giá trị Loss và SDR thông qua Tensorboard:
  ```bash
  tensorboard --logdir ./logs
  ```
- **Khôi phục (Resume)**: Nếu quá trình huấn luyện bị gián đoạn, bạn có thể tiếp tục từ một checkpoint:
  ```bash
  python trainer.py -c [config yaml] --checkpoint_path [đường dẫn file chkpt.pt] -e [đường dẫn file embedder pt] -m [tên mô hình]
  ```

# VoiceFilter

Một bản triển khai không chính thức bằng PyTorch cho mô hình của Google AI: [VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking](https://arxiv.org/abs/1810.04826).

![](./assets/voicefilter.png)

## Kết quả

- Thời gian huấn luyện mất khoảng 20 giờ trên AWS p3.2xlarge (NVIDIA V100).

### Mẫu âm thanh
- Nghe các mẫu âm thanh tại: http://swpark.me/voicefilter/

### Chỉ số đo lường (Metric)

| Median SDR             | Paper | Ours |
| ---------------------- | ----- | ---- |
| Trước khi dùng VoiceFilter |  2.5  |  1.9 |
| Sau khi dùng VoiceFilter  | 12.6  | 10.2 |

![](./assets/sdr-result.png)

- Chỉ số SDR hội tụ ở mức 10, thấp hơn một chút so với trong bài báo.

## Yêu cầu cài đặt

### 1. Python và các gói thư viện
Mã nguồn này được thử nghiệm trên Python 3.6 với PyTorch 1.0.1. Các gói thư viện khác có thể được cài đặt bằng lệnh:

```bash
pip install -r requirements.txt
```

### 2. Công cụ khác
[ffmpeg-normalize](https://github.com/slhck/ffmpeg-normalize) được sử dụng để thay đổi tốc độ lấy mẫu (resampling) và chuẩn hóa các tệp wav. Xem README.md của `ffmpeg-normalize` để biết cách cài đặt.

## Chuẩn bị dữ liệu
1. Tải bộ dữ liệu LibriSpeech tại http://www.openslr.org/12/.
2. Thay đổi tốc độ lấy mẫu và chuẩn hóa tệp wav bằng `utils/normalize-resample.sh`.
3. Cấu hình `config.yaml`.
4. Tiền xử lý tệp wav bằng `generator.py` để tạo spectrogram, giúp tăng tốc độ huấn luyện.

## Huấn luyện VoiceFilter
1. Tải mô hình tiền huấn luyện (pretrained model) cho hệ thống nhận dạng người nói để lấy d-vector embeddings.
2. Chạy lệnh huấn luyện:
   ```bash
   python trainer.py -c [config yaml] -e [đường dẫn file embedder pt] -m [tên mô hình]
   ```
3. Theo dõi quá trình qua Tensorboard:
   ```bash
   tensorboard --logdir ./logs
   ```

## Đánh giá (Evaluate)
Sử dụng lệnh sau để thực hiện suy luận:
```bash
python inference.py -c [config yaml] -e [đường dẫn file embedder pt] --checkpoint_path [đường dẫn file chkpt pt] -m [đường dẫn file wav hỗn hợp] -r [đường dẫn file wav tham chiếu] -o [thư mục đầu ra]
```

## Tác giả
[Seungwon Park](http://swpark.me) tại MINDsLab (yyyyy@snu.ac.kr, swpark@mindslab.ai)

## Giấy phép
Apache License 2.0

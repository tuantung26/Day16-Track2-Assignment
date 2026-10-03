1. Tôi dùng AWS Cloud, us-east-1, t3.medium.
2. Dataset có 284807 dòng, chia train/validation/test với test_size=0.2, seed 42.
3. Load dữ liệu mất 2.46 giây; training mất 3.69 giây.
4. AUC 0.8061, Accuracy 0.9985, F1 0.5849, Precision 0.5439, Recall 0.6327 trên tập test.
5. Latency 1 dòng 1.83 ms; throughput batch 1.000 dòng 193741.24 dòng/giây.
6. CPU/RAM/Network tôi quan sát lúc benchmark chạy hiển thị trong ảnh chụp terminal đính kèm.
7. Billing tại thời điểm làm lab ghi nhận Estimated $0.00 (chưa cập nhật).
8. Tôi đã tải kết quả và xóa tài nguyên bằng `terraform destroy` thành công.

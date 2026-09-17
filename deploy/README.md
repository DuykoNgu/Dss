# Chạy một máy Linux

1. Đặt repo tại `/srv/dss`, tạo user `dss` có quyền ghi `backend/data`,
   `backend/models` và `backend/reports`.
2. Dùng Python 3.12 tạo `/srv/dss/.venv`, cài
   `pip install -r /srv/dss/backend/requirements.txt`.
3. Trong `frontend/`, chạy `npm ci && npm run build` bằng Node 22.
4. Chạy `./run.sh test`, `cd frontend && npm test && npm run lint && npm run build`.
5. Chép `deploy/dss.service` vào `/etc/systemd/system/`, rồi chạy
   `systemctl daemon-reload && systemctl enable --now dss`.
6. Kiểm tra `http://127.0.0.1:8765/health/ready`; log ở `journalctl -u dss -f`.

API chỉ nghe trên loopback. Đặt HTTPS reverse proxy trước cổng 8765, giới hạn
tần suất gọi `/api/` tại proxy và chỉ cho proxy truy cập API. Nếu cần hai worker,
chạy `./run.sh web-balanced` dưới process supervisor thay cho service một tiến
trình; producer và worker dùng chung filesystem. Chưa dùng nhiều máy vì snapshot
đang là file local.

Trước khi mở công khai, xác nhận điều khoản sử dụng và quota của nguồn dữ liệu,
kiểm tra mức trễ `quote_as_of` trong phiên, tỉ lệ `quote_status=degraded` và
`sync_status=error`. Điểm ML hiện chỉ dùng để nghiên cứu/xếp hạng, không tạo
lệnh giao dịch tự động.

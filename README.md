# AI Code Model — Train Từ Số 0 (Local, Miễn Phí)

**Đã sửa lại hướng đi**: bản trước dùng model có sẵn (Qwen3-Coder qua Ollama) — không đúng ý ban đầu. Bản này train **1 mạng nơ-ron hoàn toàn mới, trọng số random**, không dùng model có sẵn nào.

## Thực tế cần biết trước khi bắt đầu

- **"Từ số 0" kiểu không dùng dữ liệu gì cả** (tự học 100% qua self-discovery) đã thử và ước tính mất **hàng trăm năm** — không khả thi, bỏ qua hướng này.
- **Hướng đang làm**: train từ đầu (random init) nhưng cho học từ **dữ liệu code thật, có sẵn, miễn phí** — khả thi trong vài giờ đến vài ngày tuỳ ngân sách GPU thuê.
- Đổi lại: model sẽ **yếu hơn Qwen3-Coder hoặc GPT-6 Astra rất nhiều** — vì những model đó tốn hàng triệu đô và hàng ngàn GPU để train. Với ngân sách của bạn, kết quả ban đầu chỉ là model nhỏ, biết hoàn thành các đoạn code đơn giản/quen thuộc — **chưa thể tự lên kế hoạch, tự test, tự sửa lỗi như agent bạn mô tả**. Đây là bước nền tảng (learning + nghiên cứu), không phải sản phẩm hoàn chỉnh ngay.

## Công cụ dùng (100% free/local)

- **nanoGPT** (github.com/karpathy/nanoGPT) — code train GPT từ đầu, ngắn gọn, dễ chỉnh, chạy được cả CPU (rất chậm) lẫn GPU thuê.
- **bigcode/the-stack-smol** — bộ dữ liệu code Python thật, ~10.000 file, giấy phép mở, tải free qua thư viện `datasets` của Hugging Face.
- Không gọi API nào, không tốn phí ngoài tiền thuê GPU bạn đã có sẵn ngân sách.

## Cài đặt — dán và chạy (chuẩn bị dữ liệu + model)

```bash
#!/usr/bin/env bash
set -e

echo "== 1. Cai Python + thu vien =="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip git
pip3 install --break-system-packages torch numpy transformers datasets tiktoken tqdm

echo "== 2. Tai nanoGPT =="
mkdir -p ~/ai-agent
git clone https://github.com/karpathy/nanoGPT.git ~/ai-agent/nanoGPT
cd ~/ai-agent/nanoGPT

echo "== 3. Chuan bi du lieu code that (mien phi, hop phap) =="
mkdir -p data/code_python
cat > data/code_python/prepare.py << 'EOF'
import os
import numpy as np
import tiktoken
from datasets import load_dataset

ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python")["train"]
enc = tiktoken.get_encoding("gpt2")

text = "\n\n".join(x["content"] for x in ds)
ids = enc.encode_ordinary(text)
ids = np.array(ids, dtype=np.uint16)

n = len(ids)
train_ids = ids[: int(n * 0.9)]
val_ids = ids[int(n * 0.9):]

train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
print(f"Da chuan bi {n} token tu {len(ds)} file Python that.")
EOF
python3 data/code_python/prepare.py

echo "== 4. Tao config model nho (phu hop GPU thue re) =="
cat > config/train_code_small.py << 'EOF'
out_dir = 'out-code-small'
dataset = 'code_python'
eval_interval = 250
eval_iters = 50
log_interval = 20

batch_size = 12
block_size = 512
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 3e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 3e-5
warmup_iters = 200

compile = False
EOF

echo ""
echo "== XONG PHAN CHUAN BI =="
echo "Train (can GPU thue moi du nhanh):"
echo "  cd ~/ai-agent/nanoGPT && python3 train.py config/train_code_small.py"
echo ""
echo "Chi co CPU thi them: python3 train.py config/train_code_small.py --device=cpu --compile=False"
echo "(CPU se rat cham, chi nen thu voi max_iters nho, vi du sua thanh 200, de kiem tra chay duoc)"
```

## Sau khi train xong

Thử model tự hoàn thành code:
```bash
cd ~/ai-agent/nanoGPT
python3 sample.py --out_dir=out-code-small --start="def "
```

## Ngân sách GPU thuê

Dùng đúng bảng giá bạn có: tier **16GB VRAM (~6.000đ/giờ)** là đủ cho model nhỏ này. `max_iters = 5000` chỉ là điểm khởi đầu — chạy thử, xem tốc độ + hết bao nhiêu tiền trong 1 giờ, rồi tăng/giảm `max_iters` cho vừa ngân sách còn lại.

## Đường đi tiếp theo

1. Model đầu tiên sẽ yếu — cải thiện bằng cách tăng dữ liệu (dùng thêm ngôn ngữ khác trong the-stack-smol, hoặc bộ lớn hơn nếu ngân sách cho phép), tăng `n_layer`/`n_embd`, train nhiều iter hơn.
2. Khi model đủ tốt để sinh code hợp lệ, mới nên nối nó với 1 vòng lặp tự động (giao việc → sinh code → test → sửa) giống hướng đi trước — nhưng lúc đó "bộ não" đã là model do bạn tự train, không phải Qwen3-Coder có sẵn.
3. Việc train từ đầu là 1 quá trình lặp đi lặp lại (thử, xem kết quả, chỉnh, thử lại) — không có 1 lần chạy là xong.

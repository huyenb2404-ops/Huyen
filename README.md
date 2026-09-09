# AI Code Model — Train Từ Số 0 (Local, Miễn Phí)

📋 Xem **[TIEP_THEO.md](TIEP_THEO.md)** để biết việc cần làm, ai làm gì, cần công cụ/tài liệu gì.

## Việc cần làm trước khi có thiết bị

**Bạn — chỉ 1 việc**: thuê VPS/máy tính (CPU cũng được để thử trước, có GPU thì train nhanh hơn). Có rồi thì quay lại chat, nhắn Claude 1 câu là đã có thiết bị.

**Sau khi có thiết bị, cách ít việc nhất cho bạn**: cài Claude Code ngay trên thiết bị đó (đăng nhập bằng tài khoản claude.ai hiện có — **không** khai báo `ANTHROPIC_API_KEY` để khỏi tốn phí API riêng), rồi chỉ cần bảo nó "làm theo README trong repo github.com/huyenb2404-ops/Huyen". Nó tự tải, tự cài, tự train, tự sửa nếu lỗi, không cần bạn copy dán từng lệnh. Không muốn cài Claude Code thì quay lại đoạn chat này, Claude dán từng lệnh để bạn tự chạy (mất công hơn).

**AI (Claude) đã làm / sẽ làm — không cần bạn động tay**:
- [x] Viết + kiểm tra cú pháp script cài đặt, chuẩn bị dữ liệu, config train
- [x] Chọn nguồn dữ liệu không cần đăng ký tài khoản nào thêm (chỉ dùng GitHub bạn đã có sẵn)
- [ ] Khi có thiết bị: cài môi trường, tải dữ liệu, chạy train, theo dõi log, tự sửa lỗi, báo kết quả

**Cần công cụ/tài khoản gì**: chỉ VPS/GPU thuê + tài khoản claude.ai đang dùng + tài khoản GitHub đã có. Không cần đăng ký thêm bất kỳ dịch vụ nào khác.

**Đã sửa lại hướng đi**: bản trước dùng model có sẵn (Qwen3-Coder qua Ollama) — không đúng ý ban đầu. Bản này train **1 mạng nơ-ron hoàn toàn mới, trọng số random**, không dùng model có sẵn nào.

## 2 bước (giống cách các AI như Claude/GPT được tạo ra)

- **Bước 1 — Pretrain (đang làm, file này)**: train model đoán token/code tiếp theo từ dữ liệu thật. Chưa biết "nghe lời" hay trả lời như trợ lý.
- **Bước 2 — Fine-tune/Alignment (chưa làm, để sau)**: dạy model biết nghe lời, trả lời hữu ích. Chỉ làm khi Bước 1 đã ra được model đoán code đủ tốt.

## Thực tế cần biết trước khi bắt đầu

- **"Từ số 0" kiểu không dùng dữ liệu gì cả** (tự học 100% qua self-discovery) đã thử và ước tính mất **hàng trăm năm** — không khả thi, bỏ qua hướng này.
- **Hướng đang làm**: train từ đầu (random init) nhưng cho học từ **dữ liệu code thật, có sẵn, miễn phí** — khả thi trong vài giờ đến vài ngày tuỳ ngân sách GPU thuê.
- Đổi lại: model sẽ **yếu hơn Qwen3-Coder hoặc GPT-6 Astra rất nhiều** — vì những model đó tốn hàng triệu đô và hàng ngàn GPU để train. Với ngân sách của bạn, kết quả ban đầu chỉ là model nhỏ, biết hoàn thành các đoạn code đơn giản/quen thuộc — **chưa thể tự lên kế hoạch, tự test, tự sửa lỗi như agent bạn mô tả**. Đây là bước nền tảng (learning + nghiên cứu), không phải sản phẩm hoàn chỉnh ngay.

## Công cụ dùng (100% free/local)

- **nanoGPT** (github.com/karpathy/nanoGPT) — code train GPT từ đầu, ngắn gọn, dễ chỉnh, chạy được cả CPU (rất chậm) lẫn GPU thuê.
- **3 loại dữ liệu** (không chỉ code — xem lý do bên dưới):
  - **Code Python thật**: clone trực tiếp vài repo mã nguồn mở nổi tiếng (Flask, Requests, pytest, tqdm, httpx) — không cần tài khoản thêm.
  - **TinyStories** (dataset free trên Hugging Face) — truyện ngắn, từ vựng đơn giản, được thiết kế riêng để dạy model NHỎ nói mạch lạc/đúng ngữ pháp — thiếu phần này model sẽ chỉ lặp cú pháp code mà không "hiểu" ngôn ngữ tự nhiên xung quanh.
  - **GSM8K** (dataset free trên Hugging Face) — toán đố học sinh cấp 2, có giải thích từng bước — cho model thấy dạng "suy luận nhiều bước" thay vì chỉ ra đáp số, đúng ý "tư duy tính toán" bạn nói.
- Không gọi API nào, không tốn phí ngoài tiền thuê GPU bạn đã có sẵn ngân sách.

**Vì sao cần trộn 3 loại**: model học bằng cách đoán chữ/token tiếp theo. Nếu chỉ cho học code, nó chỉ giỏi đoán cú pháp code, không có nền tảng ngôn ngữ/suy luận để hiểu yêu cầu hay giải thích. Đây là cách các model thật (Qwen3-Coder, GPT...) cũng làm — trộn code + văn bản + dữ liệu suy luận khi train, chỉ khác là họ trộn ở quy mô lớn hơn hàng triệu lần.

**Giới hạn thực tế**: ở quy mô nhỏ (model nhỏ, ít giờ train) như dự án này, thêm 2 loại dữ liệu trên giúp model bớt "ngu" hơn so với chỉ có code, nhưng KHÔNG biến nó thành model biết tính toán/suy luận thật — khả năng đó chỉ xuất hiện rõ ở quy mô lớn hơn rất nhiều. Coi đây là bước cải thiện nền tảng, không phải lời giải cho việc "thông minh".

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

echo "== 3. Chuan bi du lieu: code that + van ban + toan (mien phi) =="
mkdir -p data/code_python
cat > data/code_python/prepare.py << 'EOF'
import os
import subprocess
import random
import numpy as np
import tiktoken
from datasets import load_dataset

REPOS = [
    "https://github.com/pallets/flask.git",
    "https://github.com/psf/requests.git",
    "https://github.com/pytest-dev/pytest.git",
    "https://github.com/tqdm/tqdm.git",
    "https://github.com/encode/httpx.git",
]

work_dir = os.path.join(os.path.dirname(__file__), "_src")
os.makedirs(work_dir, exist_ok=True)

# 1) Code that (Python) - trong tam chinh cua model
code_texts = []
for url in REPOS:
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = os.path.join(work_dir, name)
    if not os.path.exists(dest):
        subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)
    for root, _, files in os.walk(dest):
        for fn in files:
            if fn.endswith(".py"):
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        code_texts.append(f.read())
                except Exception:
                    pass

# 2) Van ban tu nhien don gian - de model hoc ngu phap/mach lac co ban
story_ds = load_dataset("roneneldan/TinyStories", split="train[:20000]")
story_texts = [x["text"] for x in story_ds]

# 3) Tu duy tinh toan - toan giai thich tung buoc, khong chi dap so
math_ds = load_dataset("openai/gsm8k", "main", split="train")
math_texts = [f"Question: {x['question']}\nAnswer: {x['answer']}" for x in math_ds]

all_texts = code_texts + story_texts + math_texts
random.seed(42)
random.shuffle(all_texts)

text = "\n\n".join(all_texts)
enc = tiktoken.get_encoding("gpt2")
ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)

n = len(ids)
train_ids = ids[: int(n * 0.9)]
val_ids = ids[int(n * 0.9):]

train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
print(f"Da chuan bi {n} token — {len(code_texts)} file code, {len(story_texts)} truyen, {len(math_texts)} bai toan.")
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

1. Model đầu tiên sẽ yếu — cải thiện bằng cách tăng dữ liệu (thêm repo GitHub khác, thêm truyện/bài toán), tăng `n_layer`/`n_embd`, train nhiều iter hơn.
2. Khi model đủ tốt để sinh code hợp lệ, mới nên nối nó với 1 vòng lặp tự động (giao việc → sinh code → test → sửa) giống hướng đi trước — nhưng lúc đó "bộ não" đã là model do bạn tự train, không phải Qwen3-Coder có sẵn.
3. Việc train từ đầu là 1 quá trình lặp đi lặp lại (thử, xem kết quả, chỉnh, thử lại) — không có 1 lần chạy là xong.

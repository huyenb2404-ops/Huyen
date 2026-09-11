# AI Code Model — Train Từ Số 0 (Local, Miễn Phí)

📋 Xem **[TIEP_THEO.md](TIEP_THEO.md)** để biết việc cần làm, ai làm gì, cần công cụ/tài liệu gì.

## Việc cần làm trước khi có thiết bị

**Bạn — chỉ 1 việc**: thuê VPS/máy tính (CPU cũng được để thử trước, có GPU thì train nhanh hơn). Có rồi thì quay lại chat, nhắn Claude 1 câu là đã có thiết bị.

**Sau khi có thiết bị, cách ít việc nhất cho bạn**: cài Claude Code ngay trên thiết bị đó (đăng nhập bằng tài khoản claude.ai hiện có — **không** khai báo `ANTHROPIC_API_KEY` để khỏi tốn phí API riêng), rồi chỉ cần bảo nó "làm theo README trong repo github.com/huyenb2404-ops/Huyen". Nó tự tải, tự cài, tự train, tự sửa nếu lỗi, không cần bạn copy dán từng lệnh.

**Cần công cụ/tài khoản gì**: VPS/GPU thuê (cần nhiều RAM CPU + ổ NVMe/SSD nhanh — xem mục "Phần cứng" bên dưới) + tài khoản claude.ai đang dùng + tài khoản GitHub đã có. Không cần đăng ký thêm dịch vụ nào khác.

## 2 bước (giống cách các AI như Claude/GPT được tạo ra)

- **Bước 1 — Pretrain (đang làm, file này)**: train model đoán token/code tiếp theo từ dữ liệu thật. Chưa biết "nghe lời" hay trả lời như trợ lý.
- **Bước 2 — Fine-tune/Alignment (chưa làm, để sau)**: dạy model biết nghe lời, trả lời hữu ích. Chỉ làm khi Bước 1 đã ra được model đoán code đủ tốt.

## Thực tế cần biết trước khi bắt đầu

- **"Từ số 0" kiểu không dùng dữ liệu gì cả** (tự học 100% qua self-discovery) đã thử và ước tính mất **hàng trăm năm** — không khả thi, đã bỏ qua hướng này.
- **Hướng đang làm**: train từ đầu (random init, ~3B tham số) trên dữ liệu thật quy mô lớn, dùng kỹ thuật DeepSpeed ZeRO-Infinity để chứa model to trong phần cứng đi thuê.
- **Rủi ro đã biết trước, chấp nhận đi tiếp**: 3B tham số lý tưởng cần ~60 tỷ token mới học tử tế; bản đầu ở đây nhắm ~3-5 tỷ token (thực tế tải/xử lý được trong 1 lần chạy) — vẫn thiếu so với lý tưởng, có thể chạy lại nhiều lần tăng dần lượng data sau. Phần DeepSpeed **chưa test được bằng máy thật** (AI không có GPU) — khả năng cần sửa lỗi khi chạy lần đầu là có thật, đã có bước chạy thử ngắn ở dưới để giảm rủi ro tốn tiền.

## Công cụ dùng (100% free/local)

- **nanoGPT** (github.com/karpathy/nanoGPT) — code định nghĩa model GPT, ngắn gọn dễ chỉnh.
- **DeepSpeed ZeRO-Infinity** — cho phép train model 3B trên 1 GPU thuê bằng cách đẩy phần optimizer/tham số chưa cần dùng ngay sang RAM CPU và ổ NVMe.
- **Dữ liệu** (quy mô lớn hơn hẳn bản thử nghiệm trước):
  - **Code Python thật**: clone ~30 repo mã nguồn mở nổi tiếng (Flask, Django, NumPy, Pandas, FastAPI, scikit-learn...).
  - **FineWeb-Edu** (free, Hugging Face, mẫu có sẵn 10 tỷ token) — văn bản chất lượng cao quy mô lớn, không chỉ vài ngàn bài như trước.
  - **Wikipedia tiếng Anh + tiếng Việt** (free, Hugging Face).
  - **TinyStories, GSM8K, ARC-Easy** (free, Hugging Face) — ngôn ngữ mạch lạc + suy luận toán/khoa học.
- Không gọi API AI nào, không tốn phí ngoài tiền thuê GPU/VPS.

## Phần cứng cần

- **VRAM GPU**: từ 16GB trở lên là chạy được (ZeRO-Infinity đẩy bớt gánh nặng sang CPU/NVMe).
- **RAM CPU**: càng nhiều càng an toàn. Kiểm tra bằng `free -h`, báo AI biết số cụ thể để tính lại chính xác nếu cần.
- **Ổ NVMe/SSD tốc độ cao**: quyết định tốc độ train trực tiếp — ổ mạng/chia sẻ sẽ train chậm hơn nhiều, không phải lỗi cấu hình.
- **Dung lượng đĩa trống**: cần vài trăm GB cho dữ liệu — script bên dưới tự kiểm tra, báo lỗi rõ nếu thiếu thay vì tải nửa chừng rồi hỏng.

## Bước 1 — Chuẩn bị dữ liệu (~3-5 tỷ token)

```bash
#!/usr/bin/env bash
set -e

echo "== Cai Python + thu vien =="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip git
pip3 install --break-system-packages torch numpy transformers datasets tiktoken tqdm deepspeed

echo "== Tai nanoGPT =="
mkdir -p ~/ai-agent
git clone https://github.com/karpathy/nanoGPT.git ~/ai-agent/nanoGPT
cd ~/ai-agent/nanoGPT
mkdir -p data/code_3b

cat > data/code_3b/prepare.py << 'EOF'
import os
import subprocess
import shutil
import numpy as np
import tiktoken
from datasets import load_dataset

TARGET_TOKENS = 3_000_000_000  # tang so nay len de lay them data khi chay lai sau nay
BYTES_PER_TOKEN_EST = 4.5
need_gb = (TARGET_TOKENS * BYTES_PER_TOKEN_EST) / 1e9
free_gb = shutil.disk_usage(os.path.dirname(__file__)).free / 1e9
print(f"Uoc tinh can ~{need_gb:.0f}GB dia trong (con {free_gb:.0f}GB trong).")
if free_gb < need_gb * 1.3:
    raise SystemExit(
        f"KHONG DU DIA: can khoang {need_gb:.0f}GB (co du phong), chi con {free_gb:.0f}GB. "
        f"Giam TARGET_TOKENS trong file nay xuong roi chay lai, hoac them dia cho VPS."
    )

REPOS = [
    "flask:pallets/flask", "requests:psf/requests", "click:pallets/click",
    "rich:Textualize/rich", "httpx:encode/httpx", "tqdm:tqdm/tqdm",
    "pytest:pytest-dev/pytest", "django:django/django", "numpy:numpy/numpy",
    "pandas:pandas-dev/pandas", "scikit-learn:scikit-learn/scikit-learn",
    "matplotlib:matplotlib/matplotlib", "sqlalchemy:sqlalchemy/sqlalchemy",
    "pydantic:pydantic/pydantic", "fastapi:tiangolo/fastapi",
    "starlette:encode/starlette", "aiohttp:aio-libs/aiohttp",
    "celery:celery/celery", "scrapy:scrapy/scrapy", "pillow:python-pillow/Pillow",
    "black:psf/black", "mypy:python/mypy", "sympy:sympy/sympy",
    "networkx:networkx/networkx", "streamlit:streamlit/streamlit",
    "typer:tiangolo/typer", "pyyaml:yaml/pyyaml", "cryptography:pyca/cryptography",
    "paramiko:paramiko/paramiko", "gunicorn:benoitc/gunicorn",
]

enc = tiktoken.get_encoding("gpt2")
out_path = os.path.join(os.path.dirname(__file__), "train_val.txt")
total_tokens = 0
work_dir = os.path.join(os.path.dirname(__file__), "_src")
os.makedirs(work_dir, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as out:
    # 1) Code that - clone ~30 repo GitHub mo
    for entry in REPOS:
        name, gh = entry.split(":")
        dest = os.path.join(work_dir, name)
        if not os.path.exists(dest):
            try:
                subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{gh}.git", dest],
                                check=True, timeout=600)
            except Exception as e:
                print(f"Bo qua {gh}: {e}")
                continue
        for root, _, files in os.walk(dest):
            for fn in files:
                if fn.endswith(".py"):
                    try:
                        with open(os.path.join(root, fn), "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        out.write(text + "\n\n")
                        total_tokens += len(enc.encode_ordinary(text))
                    except Exception:
                        pass
        print(f"Sau {name}: ~{total_tokens:,} token")

    # 2) Van ban chat luong cao quy mo lon (mau co san 10 ty token, lay 1 phan)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for x in fw:
        if total_tokens >= TARGET_TOKENS * 0.7:
            break
        out.write(x["text"] + "\n\n")
        total_tokens += len(enc.encode_ordinary(x["text"]))

    # 3) Wikipedia Anh + Viet, TinyStories
    extra_sources = [
        ("wikimedia/wikipedia", "20231101.en", 200_000),
        ("wikimedia/wikipedia", "20231101.vi", 200_000),
        ("roneneldan/TinyStories", None, 100_000),
    ]
    for ds_name, cfg, take_n in extra_sources:
        if total_tokens >= TARGET_TOKENS:
            break
        stream = load_dataset(ds_name, cfg, split="train", streaming=True) if cfg else \
                 load_dataset(ds_name, split="train", streaming=True)
        for x in stream.take(take_n):
            out.write(x["text"] + "\n\n")
            total_tokens += len(enc.encode_ordinary(x["text"]))

    # 4) Tu duy tinh toan + suy luan khoa hoc
    math_ds = load_dataset("openai/gsm8k", "main", split="train")
    for x in math_ds:
        out.write(f"Question: {x['question']}\nAnswer: {x['answer']}\n\n")
        total_tokens += len(enc.encode_ordinary(x["answer"]))

    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    for x in arc_ds:
        pairs = list(zip(x["choices"]["label"], x["choices"]["text"]))
        choices_str = "\n".join(f"{lbl}) {txt}" for lbl, txt in pairs)
        correct = dict(pairs).get(x["answerKey"], "")
        out.write(f"Question: {x['question']}\nChoices:\n{choices_str}\nAnswer: {x['answerKey']}) {correct}\n\n")

print(f"TONG: ~{total_tokens:,} token thu duoc (muc tieu: {TARGET_TOKENS:,}).")

with open(out_path, "r", encoding="utf-8") as f:
    text = f.read()
ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)
n = len(ids)
ids[: int(n * 0.98)].tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
ids[int(n * 0.98):].tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
os.remove(out_path)
print(f"Da ghi train.bin/val.bin: {n:,} token thuc te sau tokenize.")
EOF
python3 data/code_3b/prepare.py
```

⚠️ Bước này có thể chạy **nhiều giờ** (tải + xử lý hàng tỷ token) — nên chạy trên VPS thường (CPU, rẻ hơn) trước, xong data mới thuê GPU để train, đỡ tốn tiền GPU trong lúc chờ tải.

## Bước 2 — Cài DeepSpeed + train model ~3B

```bash
cd ~/ai-agent/nanoGPT

cat > ds_config.json << 'EOF'
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 32,
  "fp16": { "enabled": true },
  "optimizer": { "type": "AdamW", "params": { "lr": 3e-4, "betas": [0.9, 0.95] } },
  "zero_optimization": {
    "stage": 3,
    "offload_param": { "device": "nvme", "nvme_path": "/root/ai-agent/nvme_offload", "pin_memory": true },
    "offload_optimizer": { "device": "nvme", "nvme_path": "/root/ai-agent/nvme_offload", "pin_memory": true },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e8,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5
  }
}
EOF
mkdir -p /root/ai-agent/nvme_offload

cat > train_deepspeed.py << 'EOF'
import os
import numpy as np
import torch
import deepspeed
from model import GPTConfig, GPT

DATA_DIR = "data/code_3b"
BLOCK_SIZE = 512
MICRO_BATCH = 1
MAX_ITERS = 3000
LOG_EVERY = 10
SAVE_EVERY = 200

def get_batch(split):
    data = np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK_SIZE, (MICRO_BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x, y

config = GPTConfig(block_size=BLOCK_SIZE, vocab_size=50304, n_layer=32,
                    n_head=20, n_embd=2560, dropout=0.1, bias=False)
model = GPT(config)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, model_parameters=model.parameters(), config="ds_config.json"
)

for it in range(MAX_ITERS):
    x, y = get_batch("train")
    x, y = x.to(model_engine.device), y.to(model_engine.device)
    _, loss = model_engine(x, y)
    model_engine.backward(loss)
    model_engine.step()
    if it % LOG_EVERY == 0:
        print(f"iter {it}: loss {loss.item():.4f}")
    if it % SAVE_EVERY == 0 and it > 0:
        model_engine.save_checkpoint("out-code-3b")

model_engine.save_checkpoint("out-code-3b")
print("Xong.")
EOF
```

### BẮT BUỘC chạy thử ngắn trước khi chạy thật

Sửa tạm `MAX_ITERS = 3000` thành `MAX_ITERS = 5` trong `train_deepspeed.py`, chạy:
```bash
deepspeed train_deepspeed.py
```
Không lỗi + thấy `loss` in ra → sửa lại `MAX_ITERS = 3000` (hoặc cao hơn nếu ngân sách cho phép) rồi chạy thật. Lỗi ngay ở bước này thì copy nguyên lỗi gửi AI ở đoạn chat mới — đỡ tốn tiền GPU cho 1 lỗi cấu hình.

## Sau khi train xong

DeepSpeed lưu checkpoint dạng riêng, cần đổi về dạng thường trước khi thử model:
```bash
cd ~/ai-agent/nanoGPT
python3 out-code-3b/zero_to_fp32.py out-code-3b model_fp32.pt

cat > sample_3b.py << 'EOF'
import torch
import tiktoken
from model import GPTConfig, GPT

config = GPTConfig(block_size=512, vocab_size=50304, n_layer=32, n_head=20, n_embd=2560, dropout=0.0, bias=False)
model = GPT(config)
state_dict = torch.load("model_fp32.pt", map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

enc = tiktoken.get_encoding("gpt2")
ids = torch.tensor([enc.encode_ordinary("def ")], dtype=torch.long)
out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=50)
print(enc.decode(out[0].tolist()))
EOF
python3 sample_3b.py
```

## Đường đi tiếp theo

1. Model đầu tiên vẫn sẽ thiếu data so với mức lý tưởng (~60 tỷ token) — muốn tốt hơn thì tăng `TARGET_TOKENS` trong `prepare.py` và chạy lại (cần thêm đĩa + thời gian tải tương ứng).
2. Khi model đủ tốt để sinh code hợp lệ, mới nên nối nó với 1 vòng lặp tự động (giao việc → sinh code → test → sửa).
3. Train từ đầu là quá trình lặp đi lặp lại (thử → xem kết quả → chỉnh → thử lại), không phải 1 lần chạy là xong.

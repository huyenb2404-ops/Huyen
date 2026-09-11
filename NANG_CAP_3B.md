# Nâng Cấp Lên ~3B (ZeRO-Infinity) — Đọc Kỹ Trước Khi Chạy

**Trạng thái**: đây là phần **rủi ro cao hơn hẳn** mọi thứ trong `README.md`. Phần đó (355M, 8-bit Adam) đã test được 1 phần thật (patch optimizer chạy đúng trên code nanoGPT thật). Phần 3B này dùng framework mới (DeepSpeed) mà AI không có GPU để test — khả năng cần sửa lỗi thật khi chạy lần đầu là có thật. Nếu lỗi, copy nguyên lỗi gửi AI ở đoạn chat mới, sẽ sửa tiếp.

**Vẫn nên biết**: dù làm đúng hết, model 3B train trên lượng data này (xem bên dưới) vẫn sẽ thiếu data so với mức lý tưởng (~60 tỷ token cho 3B) — bản này nhắm tới ~3-5 tỷ token cho lần chạy đầu (nhiều hơn bản 355M cũ khoảng 50-100 lần, chưa tới mức lý tưởng, nhưng là mức thực tế tải/xử lý được trong 1 lần chạy). Có thể chạy lại nhiều lần, tăng `TARGET_TOKENS` mỗi lần, để tiến dần lên 60 tỷ nếu muốn.

## 1. Chuẩn bị dữ liệu (~3-5 tỷ token)

```bash
#!/usr/bin/env bash
set -e
mkdir -p ~/ai-agent/nanoGPT/data/code_3b
cd ~/ai-agent/nanoGPT

cat > data/code_3b/prepare.py << 'EOF'
import os
import subprocess
import random
import shutil
import numpy as np
import tiktoken
from datasets import load_dataset

TARGET_TOKENS = 3_000_000_000  # tang so nay len de lay them data khi chay lai sau nay
BYTES_PER_TOKEN_EST = 4.5      # uoc luong tho de kiem tra dung luong dia
need_gb = (TARGET_TOKENS * BYTES_PER_TOKEN_EST) / 1e9
free_gb = shutil.disk_usage(os.path.dirname(__file__)).free / 1e9
print(f"Uoc tinh can ~{need_gb:.0f}GB dia trong (con {free_gb:.0f}GB trong). ")
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
    # 1) Code that - clone nhieu repo GitHub mo (~30 repo)
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

    # 3) Wikipedia Anh + Viet, TinyStories, GSM8K, ARC-Easy (nhu ban 355M, lay nhieu hon)
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

# Tokenize toan bo va chia train/val
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

⚠️ Bước này có thể chạy **nhiều giờ** (tải + xử lý hàng tỷ token) — chạy trên VPS thường (CPU, rẻ hơn) trước, xong data mới thuê GPU để train, đỡ tốn tiền GPU trong lúc chờ tải.

## 2. Cài DeepSpeed + patch model dùng được với ZeRO-3

```bash
pip3 install --break-system-packages deepspeed
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

## 3. BẮT BUỘC chạy thử ngắn trước

Trước khi để chạy hàng giờ (tốn tiền GPU thật), sửa `MAX_ITERS = 3000` thành `MAX_ITERS = 5` trong `train_deepspeed.py`, chạy:

```bash
deepspeed train_deepspeed.py
```

Nếu chạy hết 5 vòng không lỗi và thấy số `loss` in ra — cấu hình đúng, sửa lại `MAX_ITERS = 3000` (hoặc cao hơn nếu ngân sách cho phép) rồi chạy thật. Nếu lỗi ngay ở bước này, copy nguyên lỗi gửi AI — đỡ mất tiền GPU cho 1 lỗi cấu hình.

## 4. Phần cứng cần cho bước này

- **RAM CPU**: càng nhiều càng an toàn cho ZeRO-3, nhưng vì offload sang **NVMe** (không phải chỉ RAM) nên yêu cầu RAM thấp hơn hẳn ZeRO-Offload thường — kiểm tra RAM có bằng lệnh `free -h`, gửi AI biết nếu muốn tính chính xác hơn.
- **Ổ đĩa NVMe/SSD tốc độ cao**: tốc độ ổ đĩa quyết định tốc độ train trực tiếp — ổ càng chậm (ổ mạng, ổ chia sẻ) thì train càng chậm, không phải lỗi cấu hình.
- **Dung lượng đĩa trống**: script ở Bước 1 tự kiểm tra và báo lỗi rõ nếu thiếu, không cần đoán trước.

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

TARGET_TOKENS = 500_000_000  # thuc te hon 3 ty (xem README: MAX_ITERS moi tinh theo toc do do thuc, khong con doan mo)
BYTES_PER_TOKEN_EST = 4.5
need_gb = (TARGET_TOKENS * BYTES_PER_TOKEN_EST) / 1e9
free_gb = shutil.disk_usage(os.path.dirname(__file__)).free / 1e9
print(f"Uoc tinh can ~{need_gb:.1f}GB dia trong (con {free_gb:.0f}GB trong).")
if free_gb < need_gb * 1.3:
    raise SystemExit(
        f"KHONG DU DIA: can khoang {need_gb:.1f}GB (co du phong), chi con {free_gb:.0f}GB. "
        f"Giam TARGET_TOKENS trong file nay xuong roi chay lai, hoac them dia cho VPS."
    )

enc = tiktoken.get_encoding("gpt2")
assert enc.n_vocab <= 50304, "Tokenizer co vocab lon hon vocab_size cua model (50304) - bao AI biet"

total_tokens = 0
raw_bin = os.path.join(os.path.dirname(__file__), "all_tokens.bin")

def write_chunk(fh, text):
    # Tokenize tung doan nho va ghi thang ra dia - khong gom het van ban
    # vao 1 chuoi khong lo roi tokenize 1 lan (de no RAM voi hang ty token).
    global total_tokens
    if not text:
        return
    ids = enc.encode_ordinary(text)
    np.array(ids, dtype=np.uint16).tofile(fh)
    total_tokens += len(ids)

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
work_dir = os.path.join(os.path.dirname(__file__), "_src")
os.makedirs(work_dir, exist_ok=True)

with open(raw_bin, "wb") as fh:
    # 1) Code that - clone ~30 repo GitHub mo
    for entry in REPOS:
        if total_tokens >= TARGET_TOKENS:
            break
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
                            write_chunk(fh, f.read())
                    except Exception:
                        pass
        print(f"Sau {name}: ~{total_tokens:,} token")

    # 2) Van ban chat luong cao quy mo lon (mau co san 10 ty token, lay 1 phan)
    if total_tokens < TARGET_TOKENS * 0.8:
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        for x in fw:
            if total_tokens >= TARGET_TOKENS * 0.8:
                break
            write_chunk(fh, x.get("text", ""))

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
            if total_tokens >= TARGET_TOKENS:
                break
            write_chunk(fh, x.get("text", ""))

    # 4) Tu duy tinh toan + suy luan khoa hoc
    math_ds = load_dataset("openai/gsm8k", "main", split="train")
    for x in math_ds:
        write_chunk(fh, f"Question: {x['question']}\nAnswer: {x['answer']}\n")

    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    for x in arc_ds:
        pairs = list(zip(x["choices"]["label"], x["choices"]["text"]))
        choices_str = "\n".join(f"{lbl}) {txt}" for lbl, txt in pairs)
        correct = dict(pairs).get(x["answerKey"], "")
        write_chunk(fh, f"Question: {x['question']}\nChoices:\n{choices_str}\nAnswer: {x['answerKey']}) {correct}\n")

print(f"TONG: ~{total_tokens:,} token thu duoc (muc tieu: {TARGET_TOKENS:,}).")

# Chia train/val bang memmap - doc/ghi truc tiep tren dia, khong load het vao RAM
all_ids = np.memmap(raw_bin, dtype=np.uint16, mode="r")
n = len(all_ids)
split = int(n * 0.98)
train_arr = np.memmap(os.path.join(os.path.dirname(__file__), "train.bin"), dtype=np.uint16, mode="w+", shape=(split,))
train_arr[:] = all_ids[:split]
train_arr.flush()
val_arr = np.memmap(os.path.join(os.path.dirname(__file__), "val.bin"), dtype=np.uint16, mode="w+", shape=(n - split,))
val_arr[:] = all_ids[split:]
val_arr.flush()
del all_ids, train_arr, val_arr
os.remove(raw_bin)
print(f"Da ghi train.bin ({split:,} token) + val.bin ({n - split:,} token).")
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
import time
import numpy as np
import torch
import deepspeed
from model import GPTConfig, GPT

DATA_DIR = "data/code_3b"
BLOCK_SIZE = 512
MICRO_BATCH = 1
GRAD_ACCUM = 32          # phai khop voi gradient_accumulation_steps trong ds_config.json
TARGET_HOURS = 3.0       # SUA SO NAY theo so gio ban dinh thue GPU - train se tu dung dung gio
LOG_EVERY = 10
SAVE_EVERY_SEC = 900     # luu checkpoint moi 15 phut, mat dien/dut ket noi khong mat het

def get_batch(split):
    data = np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK_SIZE, (MICRO_BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x, y

train_size = os.path.getsize(os.path.join(DATA_DIR, "train.bin")) // 2  # uint16 = 2 byte/token
tokens_per_step = MICRO_BATCH * BLOCK_SIZE * GRAD_ACCUM
print(f"Du lieu train: {train_size:,} token. Moi step xu ly {tokens_per_step:,} token.")
print(f"De train het 1 luot du lieu can khoang {train_size / tokens_per_step:,.0f} step "
      f"(chua tinh toc do that cua may - se do o duoi).")

config = GPTConfig(block_size=BLOCK_SIZE, vocab_size=50304, n_layer=32,
                    n_head=20, n_embd=2560, dropout=0.1, bias=False)
model = GPT(config)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, model_parameters=model.parameters(), config="ds_config.json"
)

start = time.time()
last_save = start
it = 0
tokens_seen = 0
calibrated = False

while True:
    elapsed_hr = (time.time() - start) / 3600
    if elapsed_hr >= TARGET_HOURS:
        print(f"Da du {TARGET_HOURS} gio dat muc tieu, dung lai.")
        break

    x, y = get_batch("train")
    x, y = x.to(model_engine.device), y.to(model_engine.device)
    _, loss = model_engine(x, y)
    model_engine.backward(loss)
    model_engine.step()
    it += 1
    tokens_seen += tokens_per_step

    if it == 10 and not calibrated:
        sec_per_step = (time.time() - start) / 10
        total_steps_est = int(TARGET_HOURS * 3600 / sec_per_step)
        coverage = min(100, 100 * (total_steps_est * tokens_per_step) / train_size)
        print(f"\n[DO TOC DO] ~{sec_per_step:.1f} giay/step -> uoc tinh {total_steps_est:,} step "
              f"trong {TARGET_HOURS} gio -> se train qua ~{coverage:.1f}% du lieu da chuan bi.\n")
        calibrated = True

    if it % LOG_EVERY == 0:
        print(f"iter {it} ({elapsed_hr:.2f}h): loss {loss.item():.4f}, {tokens_seen:,} token da qua")

    if time.time() - last_save >= SAVE_EVERY_SEC:
        model_engine.save_checkpoint("out-code-3b")
        last_save = time.time()

model_engine.save_checkpoint("out-code-3b")
print(f"Xong. Tong {it:,} step, {tokens_seen:,} token da train qua.")
EOF

```

### BẮT BUỘC chạy thử ngắn trước khi chạy thật

Sửa tạm `TARGET_HOURS = 3.0` thành `TARGET_HOURS = 0.03` (~2 phút) trong `train_deepspeed.py`, chạy:
```bash
deepspeed train_deepspeed.py
```
Không lỗi + thấy dòng `[DO TOC DO]` và `loss` in ra → sửa lại `TARGET_HOURS` thành số giờ thật bạn định thuê GPU rồi chạy thật (script tự dừng đúng giờ, tự tính đang train qua bao nhiêu % dữ liệu đã chuẩn bị — không cần đoán số iteration). Lỗi ngay ở bước này thì copy nguyên lỗi gửi AI ở đoạn chat mới — đỡ tốn tiền GPU cho 1 lỗi cấu hình.

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

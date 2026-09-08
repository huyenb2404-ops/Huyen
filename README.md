# AI Agent Tự Viết Code (Local, Miễn Phí)

Mục tiêu: một AI agent chạy trên máy/VPS của bạn — tự nghĩ việc cần làm, tự viết code, tự test, tự sửa lỗi, tự đề xuất việc tiếp theo, và chỉ báo lại người khi gặp việc vượt khả năng. 100% dùng công cụ free/local, không tốn phí API.

## Các mảnh ghép (kiến trúc)

- **Ollama** — phần mềm chạy model AI ngay trên máy, không cần gọi ra internet mỗi lần hỏi (tải model 1 lần, dùng mãi, không tính phí theo lượt).
- **Qwen3-Coder** (chạy qua Ollama) — "bộ não" viết code.
- **Aider** — công cụ dòng lệnh: đọc code, sửa file, tự chạy test, tự lưu (commit) vào git. Đây là "tay chân" thực thi việc.
- **loop.sh** (script tự tạo) — đóng vai "quản lý": lấy việc từ danh sách, giao cho Aider làm, kiểm tra xong chưa, nếu bí thì ghi lại để báo bạn, hết việc thì tự nghĩ việc tiếp theo.

## Chọn model theo phần cứng

- **Máy/VPS chỉ có CPU** (ví dụ RAM 24GB như VPS cũ của bạn): bản Qwen3-Coder nhỏ nhất hiện có chính thức là **30B** (~19GB) — khá sát trần RAM nhưng chạy được, script bên dưới đã giảm context xuống 8192 để đỡ tốn RAM hơn. Nếu máy đơ/quá chậm, mở `setup.sh` đổi 1 dòng `MODEL_TAG` sang `qwen2.5-coder:14b` (nhẹ hơn nhiều, vẫn họ Qwen, ổn định hơn trên CPU).
- **Khi thuê thêm VPS có GPU**: tier **24GB VRAM (~9.000đ/giờ)** là vừa đủ để chạy bản 30B đầy đủ, nhanh hơn CPU rất nhiều — bật lên lúc cần xử lý việc nặng rồi tắt ngay để tiết kiệm.

## Cài đặt — dán và chạy 1 lần

Lưu đoạn dưới thành file `setup.sh` trên VPS/máy tính (Ubuntu), rồi chạy `bash setup.sh`:

```bash
#!/usr/bin/env bash
set -e

echo "== 1. Cai Ollama =="
curl -fsSL https://ollama.com/install.sh | sh

echo "== 2. Tang context window mac dinh cua Ollama =="
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_CONTEXT_LENGTH=8192"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 3

echo "== 3. Tai model =="
MODEL_TAG="qwen3-coder:30b"
# May yeu / hay bi treo thi doi dong tren thanh: MODEL_TAG="qwen2.5-coder:14b"
ollama pull "$MODEL_TAG"

echo "== 4. Cai Python + Aider =="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip git
pip3 install -U aider-chat --break-system-packages

echo "== 5. Tao thu muc lam viec =="
mkdir -p ~/ai-agent/workspace
cd ~/ai-agent
[ -d workspace/.git ] || git init workspace
echo "$MODEL_TAG" > model.txt
touch NEEDS_HUMAN.md
[ -f TASKS.md ] || echo "- [ ] Doc code trong workspace/, viet lai README.md mo ta du an" > TASKS.md

cat > workspace/.aider.model.settings.yml << EOF
- name: ollama_chat/$MODEL_TAG
  edit_format: whole
  num_ctx: 8192
EOF

echo "== 6. Tao vong lap tu dong =="
cat > loop.sh << 'SCRIPT'
#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export OLLAMA_API_BASE=http://127.0.0.1:11434
MODEL="ollama_chat/$(cat "$DIR/model.txt")"
cd "$DIR/workspace"

while true; do
  TASK=$(grep -m1 '^- \[ \]' "$DIR/TASKS.md" || true)

  if [ -z "$TASK" ]; then
    echo "[$(date)] Het viec - nho AI de xuat viec moi"
    aider --model "$MODEL" --yes-always --no-show-release-notes --message "Doc code trong thu muc nay. De xuat DUNG 1 viec cu the nen lam tiep de cai thien du an, ghi vao file $DIR/TASKS.md dang: - [ ] mo ta. Chi ghi vao TASKS.md, khong sua file code."
    sleep 30
    continue
  fi

  echo "[$(date)] Dang lam: $TASK"
  aider --model "$MODEL" --yes-always --auto-test --no-show-release-notes --message "Lam viec nay: ${TASK#- [ ] }. Sua code, dam bao chay duoc va test qua. That bai thi tu sua va thu lai ngay. Khi CHAC CHAN xong va test qua, xoa dong task nay khoi file $DIR/TASKS.md."

  STILL_THERE=$(grep -F -m1 "$TASK" "$DIR/TASKS.md" || true)
  if [ -n "$STILL_THERE" ]; then
    N=$(( $(cat "$DIR/.failcount" 2>/dev/null || echo 0) + 1 ))
    echo "$N" > "$DIR/.failcount"
    if [ "$N" -ge 3 ]; then
      echo "- $(date '+%Y-%m-%d %H:%M') KHONG LAM DUOC sau $N lan: ${TASK#- [ ] }" >> "$DIR/NEEDS_HUMAN.md"
      grep -vF "$TASK" "$DIR/TASKS.md" > "$DIR/TASKS.md.tmp" && mv "$DIR/TASKS.md.tmp" "$DIR/TASKS.md"
      rm -f "$DIR/.failcount"
    fi
  else
    rm -f "$DIR/.failcount"
  fi
  sleep 30
done
SCRIPT
chmod +x loop.sh

echo ""
echo "== XONG =="
echo "Chay nen:  cd ~/ai-agent && nohup ./loop.sh > agent.log 2>&1 &"
echo "Xem log:   tail -f ~/ai-agent/agent.log"
echo "Dung lai:  pkill -f loop.sh"
```

## Sau khi cài xong

- Chạy nền: `cd ~/ai-agent && nohup ./loop.sh > agent.log 2>&1 &`
- Theo dõi: `tail -f agent.log`
- Xem việc đang làm: `cat TASKS.md`
- Xem việc bị kẹt cần bạn xử lý: `cat NEEDS_HUMAN.md`
- Dừng agent: `pkill -f loop.sh`
- Code thật nằm trong thư mục `workspace/` (đây cũng là git repo riêng của agent).

## Giới hạn thực tế cần biết

- Đây là bản v1 "tự động ở mức hợp lý" — không phải AI có ý thức thật. Nó chạy vòng lặp: nghĩ việc → làm → test → sửa → commit, dựa trên model local nên đôi khi chậm/sai hơn model cloud lớn.
- Chạy trên CPU sẽ chậm (có thể vài phút/bước) — nên để chạy qua đêm rồi xem log, đừng kỳ vọng thấy kết quả ngay lập tức.
- Repo hiện đang trống nên chưa có bộ test nào — vài việc đầu tiên nên là tự viết test cơ bản, để `--auto-test` có cái mà chạy.
- Vài lần đầu nên đọc `agent.log` để chỉnh lại câu lệnh trong `loop.sh` cho sát với dự án cụ thể hơn.

## Nâng cấp thêm sau này (self-upgrade thật)

Bản này để agent tự sửa code trong `workspace/`, nhưng CHƯA cho nó tự sửa chính `loop.sh` của nó (tránh nó tự làm hỏng vòng lặp đang chạy nó). Khi thấy chạy ổn định, có thể thêm bước: định kỳ nhờ Aider đọc `loop.sh`, đề xuất bản cải tiến ra file riêng (`loop_v2.sh`) để duyệt trước khi thay.

## Đưa file này lên GitHub (từ iPhone)

Mở repo trên Safari, đổi `github.com` thành `github.dev` trên URL → mở trình soạn thảo code đầy đủ → dán đè nội dung này vào `README.md` → Commit. (Hoặc dùng app GitHub, chọn README.md → Edit → dán → Commit.)

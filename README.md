# 🧗 BetaFinder CNX

> Find your beta on the wall. — สำหรับยิมปีนผาเชียงใหม่

ระบบค้นหา beta (คลิปคนปีนเส้นเดียวกัน) จากรูปผนังที่ถ่าย
โดยใช้ CLIP image embeddings + FAISS similarity search

**Sources ที่รองรับ:**

| Type | Source | หมายเหตุ |
|------|--------|---------|
| Official | @the_alpine_outpost | Alpine Outpost |
| Official | @mainwallcnx | Main Wall |
| Official | @progressionvertical | Progression Vertical |
| Community | `data/contributors.json` | นักปีนที่โพสต์ beta บ่อย — เพิ่มได้เรื่อยๆ |

> Official accounts โพสต์รูปน้อยและ curated — beta จริงๆ ส่วนใหญ่อยู่ใน account ส่วนตัวของนักปีน
> ยิ่ง contributor list ยาว ยิ่ง match ได้ดี

---

## ⚙️ Setup

```bash
# 1. Clone / ดาวน์โหลดโฟลเดอร์นี้
cd betafinder-cnx

# 2. สร้าง virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. ติดตั้ง dependencies
pip install -r requirements.txt
```

---

## 🚀 การใช้งาน (3 ขั้นตอน)

### Step 1 — Scrape รูปจาก Instagram

#### Official gyms

```bash
# scrape ทุก gym + contributors (default)
python scrape.py

# scrape เฉพาะ gym เดียว
python scrape.py --gym alpine

# เพิ่ม limit และ delay
python scrape.py --limit 200 --delay 3.0
```

#### Community contributors

```bash
# เพิ่มนักปีนที่โพสต์ beta บ่อย (scrape ทันที)
python scrape.py --add-contributor somchai_climbs
python scrape.py --add-contributor malee.boulders --note "alpine beta เยอะมาก" --gyms-tag alpine

# ดู contributor list ปัจจุบัน
python scrape.py --list-contributors

# เพิ่มก่อน ยังไม่ต้อง scrape
python scrape.py --add-contributor new_user --no-scrape

# scrape เฉพาะ contributors (official ไม่เปลี่ยน)
python scrape.py --contributors-only

# scrape contributor คนเดียว
python scrape.py --contributor somchai_climbs
```

**ตัวอย่าง contributors list:**
```
────────────────────────────────────────────────────────────
  USERNAME                  GYMS                 NOTE
────────────────────────────────────────────────────────────
  @somchai_climbs           alpine, mainwall     
  @malee.boulders           alpine               alpine beta เยอะมาก
────────────────────────────────────────────────────────────
  Total: 2 active contributors
```

> ⚠️ **หมายเหตุ:** Instagram อาจ rate-limit ถ้า scrape เร็วเกินไป
> แนะนำ `--delay 3.0` และ scrape ช่วง off-peak

รูปจะถูกบันทึกใน `data/images/{official|contributor}/{key}/`  
Metadata (shortcode, caption, URL, source_type) → `data/gym_index.json`  
Contributor list → `data/contributors.json`

---

### Step 2 — สร้าง Embeddings & Index

```bash
python embed.py
```

ครั้งแรกจะดาวน์โหลด CLIP model (~350MB สำหรับ ViT-B-32)
จะใช้เวลาประมาณ 2-5 นาทีต่อ 100 รูป บน CPU

```bash
# ใช้ model ใหญ่ขึ้น (แม่นขึ้น แต่ช้ากว่า)
python embed.py --model ViT-L-14

# Rebuild ทั้งหมด
python embed.py --rebuild
```

---

### Step 3 — ค้นหา Beta

```bash
# ถ่ายรูปผนัง แล้ว search
python search.py my_wall_photo.jpg

# ระบุยิม
python search.py my_wall_photo.jpg --gym mainwall

# ดู top 10 + เปิดเบราว์เซอร์ไปยัง IG post
python search.py my_wall_photo.jpg --top 10 --open

# Output เป็น JSON (สำหรับ integrate กับ app อื่น)
python search.py my_wall_photo.jpg --json
```

**ตัวอย่าง output:**
```
============================================================
  🧗 BetaFinder CNX — Top 5 Results
============================================================

  #1  [MAINWALL]  score=0.9234
      📅 2025-02-14
      💬 New yellow route on overhang section...
      🔗 https://www.instagram.com/p/XXXXXXX/

  #2  [ALPINE]  score=0.8891
      ...
```

---

## 🏗️ Architecture

```
รูปผนัง (query)
      ↓
  CLIP Encoder (ViT-B-32)
      ↓
  Query Vector (512-dim, L2 normalized)
      ↓
  FAISS IndexFlatIP (cosine similarity)
      ↓
  Top-K matching vectors
      ↓
  Map → Instagram URLs (via gym_index.json)
```

---

## 📁 โครงสร้างไฟล์

```
betafinder-cnx/
├── scrape.py          # Step 1: ดึงรูปจาก Instagram
├── embed.py           # Step 2: สร้าง CLIP embeddings
├── search.py          # Step 3: ค้นหา
├── requirements.txt
├── README.md
└── data/
    ├── images/
    │   ├── official/
    │   │   ├── alpine/        # รูปจาก @the_alpine_outpost
    │   │   ├── mainwall/      # รูปจาก @mainwallcnx
    │   │   └── progression/   # รูปจาก @progressionvertical
    │   └── contributor/
    │       ├── username_a/    # รูปจาก community contributors
    │       └── username_b/
    ├── gym_index.json         # metadata ทุกรูป (รวม source_type)
    ├── contributors.json      # contributor list
    ├── embeddings.pkl         # CLIP vectors cache
    ├── faiss.index            # FAISS binary index
    └── faiss.index.paths.json # ordered path list
```

---

## 🔧 Tips & Troubleshooting

**Instagram scraping:**
- ถ้าได้รูปน้อย: ลอง login ด้วย `L.login("username", "password")` ใน `scrape.py`
- ถ้าถูก block: เพิ่ม `--delay 5.0` หรือรอ 1 ชั่วโมง
- รูปเก่าอาจไม่ match เพราะ gym เปลี่ยน holds บ่อย → scrape เฉพาะโพสต์ล่าสุด

**Search quality:**
- รูป query ควรถ่ายให้เห็นผนังชัด (ไม่มีคนบัง)
- ยิ่งเห็น holds เยอะ ยิ่ง match ได้ดี
- ถ้า results ไม่ดี ลอง `--model ViT-L-14`

**Hardware:**
- CPU: ทำได้, embed ~10s/รูป
- GPU (CUDA): เร็วกว่า ~10x, ติดตั้ง `faiss-gpu` แทน `faiss-cpu`

---

## 🗺️ Next Steps

- [ ] Streamlit web UI
- [ ] Auto-update cron job (scrape ทุก 6-12 ชั่วโมง)
- [ ] Web form ให้ community submit contributor account เอง
- [ ] Filter รูปที่มีคนปีน vs รูปผนังเปล่า (classifier)
- [ ] Weight search score ต่างกันระหว่าง official vs contributor
- [ ] Grade detection จาก hold color
- [ ] Line Notify / Telegram bot แจ้งเตือน beta ใหม่

---

*Built for Chiang Mai climbing community by @patipan_poty @climb.with.poom🇹🇭*
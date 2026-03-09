<img width="1905" height="905" alt="image" src="https://github.com/user-attachments/assets/dae5a5d3-d688-4bd3-9dd5-27a7a614569a" /><img width="1905" height="905" alt="image" src="https://github.com/user-attachments/assets/dae5a5d3-d688-4bd3-9dd5-27a7a614569a" /># KMITL Artificial Intelligence: Thai Sentiment Analysis

โปรเจกต์นี้คือระบบวิเคราะห์อารมณ์ข้อความภาษาไทย (Thai Sentiment Analysis) ที่พัฒนาขึ้นโดยใช้เทคนิค Machine Learning แบบ **Soft Voting Ensemble** (LightGBM + Logistic Regression + LinearSVC) ผสานกับ **Hybrid TF-IDF** และ **Smart Tokenization (Negation Binding)** เพื่อให้ความแม่นยำสูงสุด

🚀 **ทดลองใช้งานจริง (Live Demo):** [https://kmitl-artificial-intelligence-thai.onrender.com/](https://kmitl-artificial-intelligence-thai.onrender.com/)

<img width="1905" height="905" alt="image" src="https://github.com/user-attachments/assets/d34f9d3e-2922-485b-8b1e-1c60d365fc85" />
<img width="1905" height="905" alt="image" src="https://github.com/user-attachments/assets/d34f9d3e-2922-485b-8b1e-1c60d365fc85" />

* สามารถอัปโหลดได้ทั้ง Text และ JSON

---

## 👥 ผู้จัดทำ (Contributors)
**วิชา 01076582 Artificial Intelligence** | KMITL
* 66010204 ณกุล เฉลิมชัยโกศล
* 66010794 ศศิญากร จันทร์ศิริ
* 66011377 ธนกร ฟูคูฮารา
* 66011464 ราธา โรจน์รุจิพงศ์
* 66011476 วัฒน์นันท์ ธีรธนาพงษ์ 

---

## 📂 โครงสร้าง Directory และ Logic การทำงาน (Project Structure)

โปรเจกต์ถูกแบ่งออกเป็น 3 ส่วนหลัก: `Dataset` (ข้อมูล), `training` (การทดลองโมเดล), และ `web` (การนำไปใช้งานจริง)

### 1. `Dataset/` (การจัดการข้อมูล)
ส่วนนี้เก็บไฟล์ข้อมูลดิบและสคริปต์ที่ใช้เตรียมข้อมูลก่อนนำไปเทรนโมเดล
* **`checker.py` & `random_sampler.py`**: สคริปต์สำหรับสุ่มตัวอย่างและตรวจสอบความถูกต้องของข้อมูล (Data Validation & Sampling)
* **`train_sentiment.json`**: ไฟล์ข้อมูลหลักที่ใช้ในการเทรนโมเดล (มี Label ครบถ้วน)
* **`sample_test_sentiment.json`**: ข้อมูลชุดทดสอบที่แบ่งแยกไว้
* **`random_with_sentiment.json` / `random_without_sentiment.json` / `sentiment_result.json`**: ไฟล์ข้อมูลที่ถูกสุ่มขึ้นมาเพื่อใช้ทดสอบระบบ หรือเก็บผลลัพธ์จากการทำนาย

### 2. `training/` (การพัฒนาและเปรียบเทียบโมเดล)
โฟลเดอร์นี้คือหัวใจหลักของการทำ R&D (Research & Development) โดยแบ่งการทดลองออกเป็นโฟลเดอร์ตาม Algorithm อย่างเป็นระเบียบ
* **`gradient boosting/`**, **`linear svm/`**, **`logistic regression/`**:
    * แต่ละโฟลเดอร์จะบรรจุไฟล์ `.ipynb` ที่ใช้ทดลองเทรนโมเดลเดี่ยวๆ (Base Models)
    * มีการส่งออกไฟล์ Vectorizer (`tfidf_word.pkl`, `tfidf_char.pkl`), Label Encoder, และไฟล์โมเดล (`.pkl`) ของแต่ละตัวเพื่อเก็บไว้เปรียบเทียบประสิทธิภาพ
* **`combined_with_split/`**:
    * `train_with_split.py`: โค้ดสำหรับทำ **Ensemble Model** โดยมีการแบ่ง Train/Test Split เพื่อวัดผล Accuracy และดึง Confusion Matrix ออกมาดูประสิทธิภาพ
* **`combined_final/`**:
    * `train_without_split.py`: โค้ดสำหรับเทรน Ensemble Model ตัวสมบูรณ์แบบโดยใช้ข้อมูลทั้งหมด (100% Training Data) เพื่อประสิทธิภาพสูงสุดก่อนนำไปขึ้นเว็บ
    * โฟลเดอร์นี้จะเก็บโมเดลเวอร์ชันที่ดีที่สุด (`ensemble_model.pkl`)

### 3. `web/` (การใช้งานบนเว็บไซต์ - Deployment)
ส่วนแอปพลิเคชันที่พัฒนาด้วย Python (Flask/FastAPI) เพื่อนำโมเดลไปให้บริการผ่าน Web UI
* **`model/`**: แหล่งรวบรวมไฟล์ `.pkl` (Model, Word TF-IDF, Char TF-IDF, Label Encoder) ที่ดีที่สุดจาก `combined_final/`
* **`tokenizer.py`**: ไฟล์สำคัญที่เก็บ Logic **"Smart Tokenization"** และ **"Negation Binding"** (การดักจับคำว่า "ไม่" ผูกติดกับคำถัดไป) เพื่อให้ฝั่ง Web ทำ Preprocessing ได้เหมือนฝั่ง Training แบบ 100%
* **`app.py`**: สคริปต์หลักของ Backend สำหรับรับ Request, ทำความสะอาดข้อความ, ตัดคำ, และคืนค่า Prediction
* **`template/index.html` & `static/style.css`**: หน้าตา UI ของเว็บไซต์ (Frontend)
* **`requirements.txt`**: รายชื่อ Library ทั้งหมดที่จำเป็นต้องใช้ในการรันโปรเจกต์นี้

---

## 💻 วิธีการติดตั้งและรันโปรเจกต์ในเครื่อง (Local Setup Guide)

หากต้องการนำโค้ดไปรันบนเครื่องของตัวเอง สามารถทำตามขั้นตอนด้านล่างนี้ได้เลย:

**Step 1: โคลน Repository**
```bash
git clone https://github.com/Ikkyuuuu/KMITL-Artificial-Intelligence-Thai-Sentiment-Analysis.git
cd KMITL-Artificial-Intelligence-Thai-Sentiment-Analysis-main
```

**Step 2: สร้างและเปิดใช้งาน Virtual Environment (แนะนำ)**
```bash
# สำหรับ Windows
python -m venv venv
venv\Scripts\activate

# สำหรับ Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 3: ติดตั้ง Dependencies)**
```bash
pip install -r web/requirements.txt
```

**Step 4: รัน Web Application**
```bash
cd web
python app.py
```

**Step 5: ใช้งานระบบ**
* เปิด Web Browser แล้วเข้าไปที่: http://127.0.0.1:5000 (หรือพอร์ตที่ระบุไว้ใน app.py)
* คุณสามารถพิมพ์ข้อความภาษาไทยเพื่อทดสอบการทำนายอารมณ์ได้ทันที

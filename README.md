# Driver Drowsiness Detection System

Graduation Project II – El-Sewedy University of Technology

## Project Overview
This project uses Artificial Intelligence and Computer Vision to detect driver drowsiness in real time.

The system monitors:
- Eye closure (EAR)
- Yawning detection (MAR)
- Head movement
- Facial landmarks using MediaPipe

When drowsiness is detected, an alarm is triggered to alert the driver.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- MobileNetV2
- YOLOv8
- Arduino Uno

## Project Structure

```text
data/
models/
utils/
real_time.py
train_eye.py
train_yawn.py
evaluate_visual.py
requirements.txt

## Installation
```bash
pip install -r requirements.txt
```

Driver Drowsiness Project II FINAL(1).pdf
PDF
تزبيط GIT HIP
Thought for 5s

أيوه كده الرفع نجح من Windsurf ✅
ظاهر عندك الملفات كلها واسم الـ commit: First upload from Windsurf.

بس عشان تزبط GitHub صح اعمل الآتي:

افتح ملف README واكتب شرح المشروع.
يفضل تمسح الملفات الثقيلة زي:
results.rar
أي موديلات كبيرة لو مش ضرورية
تأكد إن requirements.txt موجود عشان أي حد يشغل المشروع.
خلي الملفات الأساسية واضحة:
real_time.py
train_eye.py
train_yawn.py
final_report.py

لو عايز README جاهز للمشروع ده أكتبهولك دلوقتي.

تمام مسحته في WIND كيف امسمحهم من على قيت

لو مسحت الملفات من Windsurf ولسه عايزها تتمسح من GitHub، لازم تعمل Commit وبعدها Push.

في Windsurf:

افتح Source Control (علامة Git).
هتلاقي الملفات المحذوفة مكتوب جنبها D (Deleted).

اكتب رسالة مثل:

Remove old files
اضغط Commit.
بعد الـ Commit اضغط Sync Changes أو Push.

أو من التيرمنال داخل Windsurf:

git add .
git commit -m "Remove old files"
git push origin main

بعدها اعمل Refresh لصفحة GitHub وستجد الملفات اختفت.

لو ابعتلي صورة من Source Control في Windsurf بعد الحذف، أقولك بالضبط تضغط إيه.

تم 

ممتاز 👍

دلوقتي ادخل على GitHub واعمل:

Refresh (F5) للصفحة.
تأكد أن الملفات اللي مسحتها اختفت.
اضغط على Commits (الرقم اللي جنب آخر Commit).
شوف آخر Commit هل اسمه مثلاً:
Remove old files
Delete files
أو أي رسالة أنت كتبتها

لو ظاهر آخر Commit جديد والملفات اختفت، يبقى كل حاجة تمام وGitHub اتحدث بنجاح ✅

ابعتلي صورة للصفحة الحالية على GitHub بعد الـ Refresh وأنا أتأكد لك 100٪.

تم

ممتاز جدًا 🎉

طالما:

مسحت الملفات من Windsurf ✅
عملت Commit ✅
عملت Push/Sync ✅
والملفات اختفت من GitHub بعد الـ Refresh ✅

يبقى الريبو دلوقتي متزامن بين Windsurf و GitHub بشكل صحيح.

نصيحة أخيرة للمشروع:

أضف ملف README.md يشرح المشروع.
أضف ملف .gitignore عشان الملفات الكبيرة وملفات النتائج ما تترفعش مرة تانية.
لو ملف yolov8n.pt أو مجلد results كبير جدًا ومش مطلوب، خليه خارج Git.

ومن الصورة السابقة واضح إن آخر Commit كان:
"First upload from Windsurf"
وده معناه إن Windsurf هو اللي بيدير الرفع حاليًا كما كنتِ عايزة.

لو عايزة أتأكد إن مشروع التخرج نفسه مرتب وجاهز للتسليم على GitHub، ابعتي رابط الريبو أو قولي الملفات اللي فضلت فيه وأنا أراجع الهيكل معاك خطوة بخطوة.

طيب يلا خطوه خطوه 

تمام. نبدأ بأول خطوة بس:

الخطوة 1: افتحي Windsurf وافتحي Terminal

من فوق اختاري:

Terminal → New Terminal

وبعدين اكتبي الأمر ده:

git status

هيطلعلك كلام.
ابعتيلي صورة أو انسخيلي النتيجة هنا، وأنا أقولك الخطوة اللي بعدها.

تمام ✅ الحالة عندك ممتازة:

nothing to commit, working tree clean

يعني Windsurf و GitHub متزامنين.

الخطوة 2: نعمل ملف .gitignore

من الشمال فوق في المشروع اضغطي يمين على اسم المشروع drowsiness_project
ثم New File
واكتبي الاسم بالضبط:

.gitignore

وحطي فيه ده:

.venv/
__pycache__/
*.pyc
results/
*.rar
.DS_Store

بعد ما تحفظي قوليلي تم.

تم

ممتاز 👌

الخطوة 3: احفظي الملف وشيكي Git

في الـ Terminal اكتبي:

git status

لو ظهر:

.gitignore
modified

أو

new file: .gitignore

يبقى تمام.

بعدها نفذي:

git add .gitignore
git commit -m "Add gitignore"
git push origin main

وابعتيلي صورة أو نتيجة الأوامر بعد ما تخلص. 🚀

ممتاز جدًا 👏

واضح من الصورة إن .gitignore اتعمل صح، ولسه Untracked فقط.

نفذي الأوامر دي بالترتيب في الـ Terminal:

git add .gitignore

ثم:

git commit -m "Add gitignore"

ثم:

git push origin main

لو كله نجح هتشوفي رسالة فيها حاجة شبه:

Enumerating objects...
Writing objects...
To github.com:...
main -> main

📌 ابعتيلي صورة بعد تنفيذ الثلاث أوامر أو أي رسالة خطأ تظهر، وأنا أكمل معاكي الخطوة التالية (عمل README احترافي للمشروع).

ممتاز جدًا 🎉🎉

من الصورة واضح إن:

main -> main

ظهر بنجاح، يعني:

.gitignore اترفع على GitHub ✅
الـ Push نجح ✅
المشروع متزامن مع GitHub ✅
الخطوة 4: نعمل README احترافي

اعملي ملف جديد باسم:

README.md

وحطي فيه النص ده:

# Driver Drowsiness Detection System

Graduation Project II – El-Sewedy University of Technology

## Project Overview
This project uses Artificial Intelligence and Computer Vision to detect driver drowsiness in real time.

The system monitors:
- Eye closure (EAR)
- Yawning detection (MAR)
- Head movement
- Facial landmarks using MediaPipe

When drowsiness is detected, an alarm is triggered to alert the driver.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- MobileNetV2
- YOLOv8
- Arduino Uno

## Project Structure

```text
data/
models/
utils/
real_time.py
train_eye.py
train_yawn.py
evaluate_visual.py
requirements.txt
Installation
pip install -r requirements.txt


Installation
pip install -r requirements.txt

Run
python real_time.py

Driver Drowsiness Project II FINAL(1).pdf
PDF
تزبيط GIT HIP
Thought for 5s

أيوه كده الرفع نجح من Windsurf ✅
ظاهر عندك الملفات كلها واسم الـ commit: First upload from Windsurf.

بس عشان تزبط GitHub صح اعمل الآتي:

افتح ملف README واكتب شرح المشروع.
يفضل تمسح الملفات الثقيلة زي:
results.rar
أي موديلات كبيرة لو مش ضرورية
تأكد إن requirements.txt موجود عشان أي حد يشغل المشروع.
خلي الملفات الأساسية واضحة:
real_time.py
train_eye.py
train_yawn.py
final_report.py

لو عايز README جاهز للمشروع ده أكتبهولك دلوقتي.

تمام مسحته في WIND كيف امسمحهم من على قيت

لو مسحت الملفات من Windsurf ولسه عايزها تتمسح من GitHub، لازم تعمل Commit وبعدها Push.

في Windsurf:

افتح Source Control (علامة Git).
هتلاقي الملفات المحذوفة مكتوب جنبها D (Deleted).

اكتب رسالة مثل:

Remove old files
اضغط Commit.
بعد الـ Commit اضغط Sync Changes أو Push.

أو من التيرمنال داخل Windsurf:

git add .
git commit -m "Remove old files"
git push origin main

بعدها اعمل Refresh لصفحة GitHub وستجد الملفات اختفت.

لو ابعتلي صورة من Source Control في Windsurf بعد الحذف، أقولك بالضبط تضغط إيه.

تم 

ممتاز 👍

دلوقتي ادخل على GitHub واعمل:

Refresh (F5) للصفحة.
تأكد أن الملفات اللي مسحتها اختفت.
اضغط على Commits (الرقم اللي جنب آخر Commit).
شوف آخر Commit هل اسمه مثلاً:
Remove old files
Delete files
أو أي رسالة أنت كتبتها

لو ظاهر آخر Commit جديد والملفات اختفت، يبقى كل حاجة تمام وGitHub اتحدث بنجاح ✅

ابعتلي صورة للصفحة الحالية على GitHub بعد الـ Refresh وأنا أتأكد لك 100٪.

تم

ممتاز جدًا 🎉

طالما:

مسحت الملفات من Windsurf ✅
عملت Commit ✅
عملت Push/Sync ✅
والملفات اختفت من GitHub بعد الـ Refresh ✅

يبقى الريبو دلوقتي متزامن بين Windsurf و GitHub بشكل صحيح.

نصيحة أخيرة للمشروع:

أضف ملف README.md يشرح المشروع.
أضف ملف .gitignore عشان الملفات الكبيرة وملفات النتائج ما تترفعش مرة تانية.
لو ملف yolov8n.pt أو مجلد results كبير جدًا ومش مطلوب، خليه خارج Git.

ومن الصورة السابقة واضح إن آخر Commit كان:
"First upload from Windsurf"
وده معناه إن Windsurf هو اللي بيدير الرفع حاليًا كما كنتِ عايزة.

لو عايزة أتأكد إن مشروع التخرج نفسه مرتب وجاهز للتسليم على GitHub، ابعتي رابط الريبو أو قولي الملفات اللي فضلت فيه وأنا أراجع الهيكل معاك خطوة بخطوة.

طيب يلا خطوه خطوه 

تمام. نبدأ بأول خطوة بس:

الخطوة 1: افتحي Windsurf وافتحي Terminal

من فوق اختاري:

Terminal → New Terminal

وبعدين اكتبي الأمر ده:

git status

هيطلعلك كلام.
ابعتيلي صورة أو انسخيلي النتيجة هنا، وأنا أقولك الخطوة اللي بعدها.

تمام ✅ الحالة عندك ممتازة:

nothing to commit, working tree clean

يعني Windsurf و GitHub متزامنين.

الخطوة 2: نعمل ملف .gitignore

من الشمال فوق في المشروع اضغطي يمين على اسم المشروع drowsiness_project
ثم New File
واكتبي الاسم بالضبط:

.gitignore

وحطي فيه ده:

.venv/
__pycache__/
*.pyc
results/
*.rar
.DS_Store

بعد ما تحفظي قوليلي تم.

تم

ممتاز 👌

الخطوة 3: احفظي الملف وشيكي Git

في الـ Terminal اكتبي:

git status

لو ظهر:

.gitignore
modified

أو

new file: .gitignore

يبقى تمام.

بعدها نفذي:

git add .gitignore
git commit -m "Add gitignore"
git push origin main

وابعتيلي صورة أو نتيجة الأوامر بعد ما تخلص. 🚀

ممتاز جدًا 👏

واضح من الصورة إن .gitignore اتعمل صح، ولسه Untracked فقط.

نفذي الأوامر دي بالترتيب في الـ Terminal:

git add .gitignore

ثم:

git commit -m "Add gitignore"

ثم:

git push origin main

لو كله نجح هتشوفي رسالة فيها حاجة شبه:

Enumerating objects...
Writing objects...
To github.com:...
main -> main

📌 ابعتيلي صورة بعد تنفيذ الثلاث أوامر أو أي رسالة خطأ تظهر، وأنا أكمل معاكي الخطوة التالية (عمل README احترافي للمشروع).

ممتاز جدًا 🎉🎉

من الصورة واضح إن:

main -> main

ظهر بنجاح، يعني:

.gitignore اترفع على GitHub ✅
الـ Push نجح ✅
المشروع متزامن مع GitHub ✅
الخطوة 4: نعمل README احترافي

اعملي ملف جديد باسم:

README.md

وحطي فيه النص ده:

# Driver Drowsiness Detection System

Graduation Project II – El-Sewedy University of Technology

## Project Overview
This project uses Artificial Intelligence and Computer Vision to detect driver drowsiness in real time.

The system monitors:
- Eye closure (EAR)
- Yawning detection (MAR)
- Head movement
- Facial landmarks using MediaPipe

When drowsiness is detected, an alarm is triggered to alert the driver.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- MobileNetV2
- YOLOv8
- Arduino Uno

## Project Structure

```text
data/
models/
utils/
real_time.py
train_eye.py
train_yawn.py
evaluate_visual.py
requirements.txt
Installation
pip install -r requirements.txt
Run
python real_time.py
Authors
Abdelhakim Nabil
Ahmed Alham
Ahd Malik
Magy Romani
Mariam Magdy
Abla Abdelmoneim
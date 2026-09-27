\# 🩻 MediScan — Scoliosis Detection \& Grading



A Streamlit application that analyses full-spine AP/PA radiographs and

classifies scoliosis severity (mild / moderate / severe) using a VGG-16 +

Vision Transformer (ViT) fusion model built in PyTorch — generating an

estimated Cobb angle, AIS grade, differential diagnosis, and a structured

clinical decision-support report.



> ⚠️ \*\*Research use only.\*\* This tool is a clinical \*decision-support aid\*,

> not a diagnostic device. All outputs — classification, Cobb angle

> estimates, and recommendations — must be reviewed and confirmed by a

> qualified radiologist or orthopaedic spine surgeon before any clinical

> decision is made. Do not initiate or modify treatment based solely on

> this tool's output.



\## What it does



1\. \*\*Patient Info\*\* — capture demographics relevant to scoliosis grading:

&#x20;  age, sex, height/weight, Risser sign, skeletal maturity, menarchal

&#x20;  status (for progression-risk estimation).

2\. \*\*Spinal X-Ray\*\* — upload a full-spine AP/PA radiograph (plus optional

&#x20;  lateral and prior-scan images for comparison), with an image quality

&#x20;  checklist (full spine visible, erect posture, iliac crests visible, no

&#x20;  rotation artefact).

3\. \*\*Detection \& Report\*\* — runs the image through the fusion model and

&#x20;  produces:

&#x20;  - Predicted severity class (Mild / Moderate / Severe) with confidence

&#x20;  - Estimated Cobb angle and AIS grade

&#x20;  - Class probabilities for all three severity levels

&#x20;  - A differential diagnosis list (AIS, Scheuermann's kyphoscoliosis,

&#x20;    congenital, neuromuscular scoliosis) derived from the model's

&#x20;    softmax output

&#x20;  - Progression risk (Low / Moderate / High), computed from Cobb angle

&#x20;    and Risser sign per standard AIS guidelines

&#x20;  - Bracing-candidate and surgical-threshold flags

&#x20;  - Clinical recommendations by severity tier

&#x20;  - A downloadable `.txt` report combining all of the above



\## Model



Inference is served by a custom \*\*VGG-16 + ViT-B/16 fusion\*\* architecture:



\- VGG-16 convolutional features → flattened → projected to a 512-dim

&#x20; vector

\- ViT-B/16 (classification head stripped) → 768-dim feature vector

\- Both vectors are concatenated (1280-dim) and passed through a fusion

&#x20; head: `Linear(1280→512) → ReLU → Dropout(0.3) → Linear(512→3)`

\- Output: softmax over 3 classes — `mild`, `moderate`, `severe`

\- Input: 224×224 RGB, ImageNet-normalised



The model is loaded once via `@st.cache\_resource` from a checkpoint file

containing either a raw `state\_dict` or a wrapped checkpoint (`{'model':

...}` or `{'state\_dict': ...}`).



\*\*Note:\*\* the source file also contains a separate, commented-out training

script (top-of-file) documenting an earlier experiment that trained a

\*\*ResNet50\*\* transfer-learning model in two phases (frozen backbone, then

fine-tuning the last 20 layers) with augmentation, early stopping, and LR

scheduling. That script does not correspond to the fusion architecture

actually loaded at inference time — it's kept as reference/history rather

than as the current training pipeline. If you retrain the model, base the

training script on the `VGGViTFusion` class defined in `app.py`, not the

ResNet50 script, so the checkpoint format matches what the app expects.



\## Requirements



```

streamlit

numpy

pillow

torch

torchvision

```



Install with:



```bash

pip install streamlit numpy pillow torch torchvision

```



(Use the correct `torch`/`torchvision` build for your machine — CPU-only

is fine for inference; see \[pytorch.org](https://pytorch.org/get-started/locally/)

for the right install command if you need GPU support.)



\## Model checkpoint



The app currently expects the trained weights at a \*\*hardcoded path\*\*:



```python

MODEL\_PATH = r"C:\\mediscan-app\\best\_fusion\_model.pth"

```



Before running, either:

\- place your trained checkpoint at exactly that path, \*\*or\*\*

\- edit `MODEL\_PATH` in `app.py` to point to wherever your `.pth` file

&#x20; actually lives (a relative path is recommended if you plan to share or

&#x20; deploy this project, so it isn't tied to one machine).



\## Running it



```bash

streamlit run app.py

```



This opens the app in your browser (typically `http://localhost:8501`).

Fill in patient info, upload an X-ray, then run the analysis from the

\*\*Detection \& Report\*\* tab.



\## Output



Clicking \*\*Export Report\*\* generates a downloadable `.txt` file

containing patient information, radiographic findings, model confidence,

class probabilities, differential diagnosis, and clinical recommendations

— named `SpineAI\_<patient\_id>\_<date>.txt`.



\## Known limitations



\- `MODEL\_PATH` is a hardcoded absolute Windows path — not portable across

&#x20; machines without editing the source.

\- Cobb angle values are \*\*fixed estimates per predicted class\*\*

&#x20; (17.5° / 32.0° / 47.0°), not measured directly from the image — they

&#x20; are a proxy tied to the classification, not an independent

&#x20; angle-measurement model.

\- Progression risk and differential diagnosis percentages are derived

&#x20; from simple rule-based heuristics layered on top of the model's

&#x20; confidence score, not learned outputs in their own right.

\- No automated tests are included in this project currently.



\## Disclaimer



This project is intended for research, portfolio, and educational

purposes. It is \*\*not\*\* a certified medical device and must not be used

for real patient diagnosis or treatment decisions without full clinical

validation and regulatory clearance.


# WellCare: Real-Time Metabolic & Skin Health Intelligence

## 📌 Problem Statement
We are currently **biologically blind** to how our daily meals trigger "silent internal heat" and vascular stress. This hidden physiological strain causes **sudden acne flare-ups** and mental fatigue before we even see the damage. Without real-time insight, we are powerless to stop this inflammation from damaging our skin and body from the inside out.

---

## 🛠️ Technical Core & Methodology

### 1. Physiological Strain Index (PSI)
The system implements a research-backed heat strain formula based on the **Moran et al. (1998)** framework. It maps internal stress on a scale of 0–10 by combining:
* **Cardiovascular Strain:** Normalized Heart Rate shifts relative to metabolic limits.
* **Vascular Resistance:** Measuring the body's effort to dump heat through blood vessel tone.

### 2. Vascular Tone Monitoring (B/A Ratio)
WellCare uses the **second derivative of the PPG signal** to calculate the **B/A Ratio**. Based on research by **Takazawa et al. (1998)**, this ratio serves as a proxy for:
* **Diet-Induced Thermogenesis (DIT):** The "heat" generated during digestion.
* **Vasoconstriction/Dilation:** Identifying internal thermal retention that triggers skin inflammation.

### 3. Predictive Impact Modeling (Random Forest)
A machine learning engine trained on 7 critical physiological factors:
* **Heart Rate & HRV** (Nervous system load)
* **B/A Shift & Temperature** (Thermal/Vascular strain)
* **Sleep & Hydration** (Recovery buffers)
* **Personal Baselines** (Individualized context)

### 4. Pre-Meal "Readiness Audit"
A proactive 5-minute baseline scan that calculates your "Biological Budget" before you eat.
* **Green Zone:** High capacity; safe for high-impact/heating meals.
* **Red Zone:** High baseline heat/stress; suggests "Cooling" foods to prevent skin flare-ups.

---

## 🔬 Research Foundations
* **Moran, D. S., et al. (1998):** *A physiological strain index to evaluate heat stress.* (American Journal of Physiology).
* **Takazawa, K., et al. (1998):** *Assessment of vasoactive agents using the second derivative of the photoplethysmogram.* (Hypertension).

---

## 🚀 Vision
To eliminate metabolic blindness by providing a real-time "internal dashboard" for the human body, turning invisible physiological spikes into actionable dietary intelligence.

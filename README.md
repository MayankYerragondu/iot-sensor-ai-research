# Personalized Anomaly Detection for Smart Home IoT Environments

## 📌 Abstract
The goal of this project is to determine how well anomaly detection models perform for **home intruder detection using PIR (Passive Infrared) data**.  
Traditional PIR-based alarm systems often trigger false alarms from non-threatening household activities (e.g., pets moving, family walking).  

To solve this, we propose a **personalized anomaly detection model** that learns historical motion patterns per device (time of day, duration, frequency, and context). By continuously adapting to a household’s routine, the system filters predictable activity and flags only unusual patterns.

We integrate multiple sensors:
- **PIR alarms** (motion detection)
- **Contact alarms** (doors/windows)
- **Environmental sensors** (temperature, humidity, luminance)

Models are trained per device using AWS Glue + SageMaker pipelines, harmonizing real-time (MongoDB) and archival (S3) data.  
Performance of state-of-the-art methods is compared to highlight improvements in **false alarm reduction, forecasting accuracy, and scalability**.

---
## 📖 Introduction
- **Anomaly Detection**: Recognizing behaviors or events deviating from learned “normal” patterns.  
- **Smart Home IoT**: Combination of PIR sensors, contact sensors, and environmental sensors.  
- **Problem**: Traditional systems → high false-positive rates, asynchronous noisy data, difficulty harmonizing MongoDB (real-time) with S3 (archival).  
- **Proposed Solution**: Device-specific anomaly detection pipeline that adapts over time, reduces false alarms, and improves security precision.

---


## ⚙️ Methodology

### PIR Anomaly Detection
- Model: **Isolation Forest (per-device personalization)**  
- Features:  
  - Time of day  
  - Day of week  
  - Time difference between events  
- Benefits: Learns household-specific motion patterns, reducing false alarms.

### Contact Alarm Anomaly Detection
- Model: **One-Class SVM (OC-SVM)**  
- Features:  
  - Hour of day  
  - Day of week  
  - Event frequency (hourly aggregation)  
- Benefits: Detects unusual access patterns (e.g., late-night entries, unusual bursts).

### Environmental Sensor Anomaly Detection
- Model: **LSTM Encoder–Decoder (multivariate forecasting)**  
- Sensors: Temperature, Humidity, Luminance  
- Windowing: **5-min average**  
- Benefits: Captures cross-sensor dependencies (e.g., door open → light + temp change).

---
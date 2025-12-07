# RL-Project
# Reinforcement Learning for Digital Advertising Optimization

본 프로젝트는 **디지털 광고 시스템에서 어떤 사용자에게 광고(treatment)를 노출할지 결정하는 정책(policy)을 강화학습으로 학습**하는 것을 목표로 합니다.  
목표는 다음 두 가지를 동시에 최적화하는 것입니다:

- **전환(conversion) 및 방문(visit) 극대화**
- **광고 비용(CPM, CPC 등) 최소화**

이를 위해 Policy Gradient(PG), Deep Q-Network(DQN), Advantage Actor-Critic(A2C) 3가지 강화학습 알고리즘을 동일 환경에서 비교 실험했습니다.

---

## 🚀 Project Overview

### ✨ Problem Definition
- 광고 노출을 할지(treat=1) 하지 않을지(treat=0)를 결정하는 **two-action RL 문제**
- 사용자별 feature를 기반으로 광고 효과(visit, conversion)를 최대화
- Conversion 발생 시:
 → 가장 높은 가치(conversion_reward = 15)를 부여
 → 실제 광고 시스템에서 전환이 가장 큰 비즈니스 가치를 가진다는 가정을 반영

- Visit 발생 시:
 → 전환은 없었지만 방문한 경우 visit_reward = 1 부여
 → 전환 가능성 증가 신호를 반영한 보상

- Treatment(광고 노출) 비용:
 → treat_cost = 0.015만큼 차감
 → CPM 기반 광고 비용을 단순화한 형태

- Baseline-centered reward:
 → 전체 reward의 평균값을 baseline으로 빼서 변동 중심화
 → 정책 학습 시 variance 감소에 도움
 → PG/A2C 학습 안정성 향상
---

## 🤖 Algorithms Implemented

### 1. **Policy Gradient (PG)**
- 확률적 정책을 직접 최적화
- baseline 대비 가장 빠르게 수렴하고 높은 validation reward 달성

### 2. **Deep Q-Network (DQN)**
- action-value 기반 모델
- replay buffer + ε-greedy exploration

### 3. **Advantage Actor-Critic (A2C)**
- PG + Value function 결합
- 안정적 학습, 마지막 성능 최고

---

## 🏗 Environment & Reward Design

### ✔ State
- 사용자 feature vector (변환 가능)

### ✔ Action
- `0`: 광고 미노출 (No-Ad)  
- `1`: 광고 노출 (Treat)

### ✔ Reward
- **treat=1** 일 때 CPM 비용 차감  
- **visit=1** 발생 시 CPC 비용 차감  
- **conversion=1** 발생 시 큰 reward 부여  
- 기존 reward 대비 개선된 버전 → 학습 안정성 및 성능 향상

---

## 📈 Training Results

### 1️⃣ **Policy Gradient (PG)**  
![PG Curve](./plots/pg_curve.png)

- reward variance가 있지만 꾸준히 증가
- validation 기준 약 **0.075~0.085** 수준 도달

---

### 2️⃣ **DQN**
![DQN Curve](./plots/dqn_curve.png)

- exploration 영향으로 초기 진동이 크지만  
- 전반적으로 **0.04~0.06** validation reward 확보  
- baseline 대비 확실한 개선

---

### 3️⃣ **A2C**
![A2C Curve](./plots/a2c_curve.png)

- 안정적으로 reward 상승  
- 최종적으로 **PG와 유사하거나 더 높은 validation reward** 도달

---

## 🧪 Final Comparison (Baseline vs RL Models)

강화학습 모델들의 성능을 다음 3가지 baseline과 비교:

- **No-Ad**: 항상 광고 안 함  
- **Always-Treat**: 모든 사용자에게 광고  
- **Random**: 임의로 노출  

### 🔹 개선된 reward 설계 이후 결과

![Final Comparison 1](./plots/final_comparison_reward_new1.png)

- **A2C(best) ≈ PG(best) > DQN(best)**  
- 기존 baseline 대비 **2~10배 높은 reward**

---

### 🔹 기존 reward 대비 성능 차이
![Final Comparison 2](./plots/final_comparison_reward_new2.png)

- reward 개선 후 모든 RL 알고리즘의 성능이 크게 상승  
- 특히 PG/A2C는 안정성과 성능 모두 개선됨

---

## 🧠 Discussion

### ✔ 어떤 알고리즘이 가장 좋았는가?
- **A2C가 가장 안정적이며 최종 reward 최고**
- PG는 빠른 수렴과 높은 reward를 보임
- DQN은 변동성은 있으나 baseline 대비 강력한 성능

### ✔ Reward 설계가 중요한 이유
- 초기 reward 설계에서는 treat 선택의 근거가 약했으나  
- 광고 비용·방문·전환을 반영한 reward로 바꾼 이후  
  → 모든 알고리즘의 reward가 크게 개선됨

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train_pg.py
python train_dqn.py
python train_a2c.py

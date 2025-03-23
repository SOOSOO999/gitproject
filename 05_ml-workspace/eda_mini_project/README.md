# 환자의 상태에 따른 폐암 진행 속도 예측하기

# 1️⃣ 팀 소개
> SK네트웍스 Family AI 캠프 11기 ML Project<br/>
> 팀 명: 3조 <br/>
> 기간: 2025.03.14 - 2025.03.21<br/>

## 👤팀원 소개

<table>
  <thead>
    <td align="center">
      <a href="">
        <img src="https://github.com/user-attachments/assets/a28f14f0-2e04-4bb9-be4f-2fe81924e523" height="150" width="150" alt="jungsu"/><br /><hr/>
        배정수
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/HybuKimo">
        <img src="https://github.com/HybuKimo.png" width="150" alt="junhee"/><br /><hr/>
        신준희
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/yugyeongh">
        <img src="https://github.com/yugyeongh.png" width="150" alt="yugyeong"/><br /><hr/>
        현유경
      </a><br />
    </td>
  </thead>
</table>

<br/><br/>

# 2️⃣프로젝트 개요
<바꾸기>
폐암은 전 세계적으로 암 사망의 주요 원인으로 뽑힐 만큼 조기 발견과 치료가 생존율을 결정짓는 중요한 질환 중 하나이다. 폐암은 조기에 발견하지 않으면 빠르게 진행되며, 생존에도 큰 영향을 미치는 치명적인 질환이다. 폐암이 얼마나 빨리 진행되는지는 다양한 변수에 의해 달라질 수 있다. 본 프로젝트에서는 환자의 환경적·유전적 요인 및 생활 습관이 폐암 진행 속도에 미치는 영향을 분석하고, 데이터 탐색적 분석(EDA)을 통해 주요 요인을 시각적으로 이해하는 것을 목표로 한다.
<br/>
<br/>
## 🚀프로젝트명
- 데이터로 보는 폐암 진행 속도의 결정 요인
<br/>

## 📂사용한 데이터
-  [Lung Cancer Prediction](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link/data?status=pending&suggestionBundleId=178)
  - 데이터 수집 기간: 정확한 기간은 명시되지 않았으나, 6년간 수집한 데이터


<br/>

# 3️⃣기술 스택
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
  <img src="https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=vscode&logoColor=white">
  <img src="https://img.shields.io/badge/numpy-%235865F2.svg?style=for-the-badge&logo=numpy&logoColor=white">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/pandas-%23000000.svg?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white">
</p>
<br/><br/>

# 4️⃣WBS
| 작업 명                | 시작일 | 종료일 | 담당자         | 산출물        |
|-------------------------|-------|-------|-----------------|---------------|
| Machine Learning 모델 적용       | 03-14 | 03-19 | ALL    |  Jupiter Notebook 코드  | 
| 문서 작업 및 발표 준비   | 03-19 | 03-20 | ALL             |  README.md     | 
| 최종 점검                | 03-20 | 03-21 | ALL             |               | 

<br/><br/>

# 5️⃣분석 방법
## 1. 폐암 예측 데이터셋에 적합한 모델 찾기
- 지도 학습 분류 모델을 선정한 이유
  - 우리의 목표: 폐암 진행 레벨을 예측하는 모델 만들기
  - EDA를 통해 분석한 결과를 바탕으로 폐암 진행 레벨 컬럼을 target 데이터로 설정
    - 폐암 진행 레벨 컬럼은 1,2,3의 값으로 나눠져 있음
  - 폐암 진행 레벨 컬럼과 상관관계를 가진 컬럼은 X 데이터로 설정
  - 입력 데이터를 주어진 라벨로 분류하는 것이므로 지도 학습 중 분류 모델을 선정 

<br/><br/>

# 6️⃣ Machine Learning 결과


<br/><br/>

# 7️⃣ 한 줄 회고
| 팀원  | 한 줄 회고                 |
|-------|----------------------------|
| 정수  |  |
| 준희  |  |
| 유경  |  |

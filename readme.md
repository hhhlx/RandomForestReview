# 랜덤 포레스트 알고리즘의 원리 분석 및 구현

## 팀원 정보
- 종열 인공지능학과yuezhong24@gmail.com
- 유운상 정보사회미디어 yoohoojessicaaa@gmail.com
- 황릉훤 정보사회미디어학과 2021037483 h2792597254@163.com
- 조로단 전자공학 CAOLUDAN0909@126.com  
Video Recording Link：https://youtu.be/nk2exzYEXhM?si=N8Gk4IOuXLMExMIs
## 주제 선정 이유 분석

### 1. 알고리즘의 이론적 가치
- 랜덤 포레스트 알고리즘은 의사 결정 트리, 앙상블 학습 등 여러 핵심 기계 학습 개념을 통합하며, 알고리즘 원리가 명확하고 수학적 파생 프로세스가 완료되어 단일 분류기에서 앙상블 학습으로의 알고리즘 진화 아이디어를 구현하는 데 적합합니다. -깊이 있는 학습.

### 2. 구현 난이도 보통
- 랜덤 포레스트의 핵심 단계는 명확한 구조를 가지고 있으며 프로그래밍으로 구현하기 쉽습니다. 기본 버전은 일주일 이내에 구현이 가능합니다.

### 3. 높은 활용가치
- 데이터 품질에 대한 요구 사항이 상대적으로 느슨한 분류 및 회귀 문제에 적합하며 누락된 값과 이상값을 처리할 수 있습니다. 높은 계산 효율성과 처리 병렬화 용이

### 4. 인증 조건이 충족되었습니다.
- UCI 기계 학습 라이브러리는 충분한 오픈 소스 데이터 세트를 제공하며 Python 환경에서 완전한 과학 컴퓨팅 도구로 지원되며 sklearn 라이브러리의 구현 결과와 비교 및 ​​검증될 수 있습니다.

## 종이 구조

### 1장 소개
1. 연구배경 및 의의
2. 연구 목적

### 2장 랜덤 포레스트 알고리즘의 원리
1. 의사결정나무의 기본
- CART 의사결정트리 알고리즘에 대한 자세한 설명
- 지니지수
- 기능 선택
- 나무 생성 및 가지치기

2. 랜덤 포레스트 구축 과정
- 부트스트랩 샘플링 원리
- 무작위 기능 선택
- 통합적 접근

3. 이론적 분석
- 일반화 오류 분석
- 분산-편차 분해
- 융합분석

### 3장 알고리즘 구현
1. 데이터 전처리
2. 핵심 알고리즘 구현
3. 주요 매개변수 설정

### 4장 실험 검증
1. 실험 설계
2. 결과 분석

### 5장 요약

## 시행계획

### 1~2일차
- 문헌 읽기 및 분류
- 알고리즘 원리 도출
- 기술 솔루션 결정

### 3~5일
- Python은 CART 결정 트리를 구현합니다.
- 랜덤 포레스트 알고리즘 구현
- 실험적 검증 수행

### 6~7일차
- 실험 결과 정리
- 논문 작성 완료

**기술 옵션**:
- 프로그래밍 언어: Python
- 주요 라이브러리: numpy, pandas, sklearn(결과 비교용)
- 기본 알고리즘: CART 의사결정 트리
- 데이터세트: UCI Machine Learning Library의 분류 데이터세트

## 예상 결과

1. 알고리즘 원리의 완전한 도출 및 수학적 분석
2. 실행 가능한 Python 구현 코드
3. 실험결과 분석보고서

## 참고자료
[1] Fayyad, U.P, Piatetsky-Shapiro, G., Smyth, P.: From data mining to knowledge discovery in databases. AI Mag. 17(3), 37 (1996)  
[2] Briem, G.J., Benediktsson, J.A., Sveinsson, J.R.: Multiple classifiers applied to multisource remote sensing data. IEEE Trans. Geosci. 40(10) (2002)  
[3] Kuncheva, L.I., Whitaker, C.J.: Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy. Mach. Learn. 51(2), 181–207 (2003)  
[4] Lam, L., Suen, C.Y.: Application of majority voting to pattern recognition: an analysis of its behavior and performance. IEEE 27(5), 553–568 (1997)  
[5] Breiman, L.: Bagging predictors. Mach. Learn. 24(2), 123–140 (1996)  
[6] Dietterich, T.G.: An experimental comparison of three methods for constructing ensembles of decision trees: bagging, boosting, and randomization. Mach. Learn. 40(2), 139–157  
[7] Azar, A.T., Elshazly, H.I., Hassanien, A.E., Elkorany, A.M.: A random forest classifier for lymph diseases. Comput. Methods Programs Biomed. 113(2), 465–473 (2014)  


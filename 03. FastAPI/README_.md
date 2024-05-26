## 구성

### Practice.jpyynb  -   딥러닝 프로세스 - 전처리,학습,결과,평가 코딩

## 서버,클라이언트로 분리
### client.py        -    API 받아오기, 전처리, 학습, 결과 도출 등
### main.py          -    요청에 대한 응답


### API 사용 필수 (우분투)
### 터미널 실행 
### 1. code 입력 > vscode 실행
### 2. vscode 실행창에서 가상환경 들어가기
### 3. conda activate so >> 가상환경 실행 
###  - conda activate so >>> 아나콘다 가상환경 AI 하실 때 무조건 실행 
### 4. 03.novel_API 폴더 접속 
### 6, 명선님 데이터 전송 API 서버 실행
### 5. uvicorn main:app --host 192.168.1.54 --port 8000 입력

### main.py > API 데이터 전송, 받아오기, client 함수 불러오기
### client > 데이터 처리, 알고리즘 생성,학습,평가

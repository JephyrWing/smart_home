# Aruba HAR GRU 튜닝 기록

검증 Macro-F1이 가장 높은 **학습 연장** 후보를 `aruba-gru-v2`로 저장했다. 같은 Aruba 테스트 창 **70,085건**에서 보류 전 Macro-F1은 **0.698250 → 0.713543** (+0.015294)였다. 확신 부족 보류율은 **3.4073% → 2.8893%**였다. 새 커널에서 체크포인트를 복원해 테스트를 다시 채점했고, 같은 설정으로 재학습해 가중치 텐서·전처리·지표가 일치하는 것을 확인했다.

## 데이터와 비교 조건

주석 Aruba 파일 `data/aruba/data.txt`의 SHA-256은 `abfd43b32361d062515320b5c28e87b0c9be5a0d4e777a8f85054abc7d9e61dc`다. 1,719,553행을 파싱했고 오류는 0건이었다. WSU CASAS의 단일 거주자 Aruba 데이터로, 이 작업은 노트북에 기록된 주석 보관본을 사용했다. [WSU CASAS 데이터 목록](https://casas.wsu.edu/datasets/) · [사용한 판본의 보관 위치](https://github.com/dbouchabou/Fully-Convolutional-Network-Smart-Homes/tree/master/datasets)

| 구간 | 원본 날짜 | 일수 | 유효 정답 창 |
| --- | --- | ---: | ---: |
| 학습 | 2010-11-04 ~ 2011-04-06 | 154 | 321,351 |
| 검증 | 2011-04-07 ~ 2011-05-09 | 33 | 77,563 |
| 테스트 | 2011-05-10 ~ 2011-06-11 | 33 | 70,085 |

날짜를 먼저 나누고 구간 안에서만 창을 만들었다. 입력은 끝 이벤트까지 최대 600초·32건이며 유효 센서 이벤트가 4건 이상이어야 한다. 원본 순서로 동시각을 정렬하며 같은 시각에서도 창 끝 이후 행을 포함하지 않는다. 센서·상태 사전과 시간 간격 99백분위수 상한(216.222239초)은 학습 구간에서 계산했다. GRU 입력은 20개 센서의 ID 임베딩, 상태 원-핫과 `log1p` 변환한 이전 이벤트와의 간격이다. 활동 주석은 창 끝의 정답을 복원하고 집계하는 데 사용했다. 미주석·중첩·미지원 활동은 정답 창에서 제외했다. 모델 간 입력 사전, 창 적격성, 날짜, 클래스 순서가 같다.

귀가의 유효 학습 창은 **11건(6일)**, 외출은 **83건(14일)**이다. 귀가 주석의 원본 1,315행 중 등록 센서 끝점은 47건이고 그중 36건은 최소 길이에 미달했다. 귀가 유효 창 중 9건은 4~11개 이벤트였고, 외출 유효 창은 모두 32개였다. 검증 지지는 귀가 9건(3일), 외출 3건(2일)이다. 학습 원본의 귀가·외출 주석 구간에 속한 `D004` 이벤트는 각각 565건과 562건이었다. 두 활동의 유효 학습 창에 시연 현관 센서 `D001`가 포함된 사례는 0건이다. 등록 센서와 시연 이벤트열은 기존 매핑을 유지했다.

## 후보 학습과 검증

후보는 다음 세 가지다. 학습 연장으로 여섯 에포크 이후의 개선 여부를 확인했다. 학습 정답 창 수를 `n`으로 두고 클래스 가중치 `n**(-power)`의 지수를 0.50에서 0.75로 올려 희소 클래스의 손실 비중을 높였다. 같은 지수에서 은닉 크기를 32에서 48로 늘려 용량 효과를 비교했다. 가중치는 평균 1로 정규화했다. 모든 후보는 단방향 단층 GRU, 임베딩 12, Adam(학습률 0.003, 기본 나머지 인자), 배치 512, CPU 4스레드, PyTorch·NumPy 시드 17과 동일한 학습 창을 사용했다. 검증 Macro-F1이 처음 최고였던 에포크를 복원했다.

| 후보 | 가중 지수 | 은닉 크기 | 최대 에포크 / 조기 종료 대기 | 선택 에포크 | 검증 Macro-F1 | 귀가 F1 | 외출 F1 | 보류 임계값 | 검증 유지율 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 v1 | 0.50 | 32 | 6 / 2 | 6 | 0.697686 | 0.400000 | 0.021053 | 0.75 | 96.34% |
| **학습 연장** | 0.50 | 32 | 12 / 3 | 9 | 0.714028 | 0.533333 | 0.024390 | 0.75 | 96.66% |
| 희소 클래스 가중 강화 | 0.75 | 32 | 12 / 3 | 12 | 0.696259 | 0.378378 | 0.022727 | 0.75 | 96.41% |
| 희소 클래스 가중 강화 + GRU 확대 | 0.75 | 48 | 12 / 3 | 6 | 0.688593 | 0.285714 | 0.016260 | 0.75 | 96.15% |

- 기존 v1의 에포크별 검증 Macro-F1은 0.6125 → 0.6375 → 0.6315 → 0.6569 → 0.6919 → 0.6977였다.
- 학습 연장의 에포크별 검증 Macro-F1은 0.6125 → 0.6375 → 0.6315 → 0.6569 → 0.6919 → 0.6977 → 0.6986 → 0.6820 → 0.7140 → 0.6932 → 0.7084 → 0.7117였다.
- 희소 클래스 가중 강화의 에포크별 검증 Macro-F1은 0.5814 → 0.6005 → 0.6118 → 0.6255 → 0.6505 → 0.6748 → 0.6629 → 0.6880 → 0.6956 → 0.6861 → 0.6922 → 0.6963였다.
- 희소 클래스 가중 강화 + GRU 확대의 에포크별 검증 Macro-F1은 0.5928 → 0.6514 → 0.6760 → 0.6248 → 0.6611 → 0.6886 → 0.6695 → 0.6816 → 0.6696였다.

저장된 v1을 같은 검증 창에서 다시 채점했다. 학습 연장 후보의 첫 6개 에포크도 v1 검증값을 재현했다. 보류 임계값은 선택 에포크의 검증 확률에서 0.00·0.45·0.55·0.65·0.75·0.85를 비교했다. 검증 창 70% 이상을 유지하는 후보 중 유지한 정답당 +1과 오답당 −2의 합을 최대화했고, 동률이면 유지율이 높은 값을 골랐다. 모델 비교에는 보류 전 Macro-F1을 사용했다. 사전에 제공된 v1 테스트 결과를 기준선으로 기록했으며, 후보 선택을 고정한 뒤 v1과 선택 모델만 테스트와 합성 프리셋에서 추론했다.

## Aruba 테스트

| 지표 | v1 | v2 |
| --- | ---: | ---: |
| 유효 정답 창 | 70,085 | 70,085 |
| 보류 전 Macro-F1 | 0.69824956 | 0.71354338 |
| 확신 부족 보류 | 2,388 (3.4073%) | 2,025 (2.8893%) |
| 입력 부족 끝점 / 지원 활동 정답이 있는 입력 부족 끝점 | 1,646 / 589 | 1,646 / 589 |

| 활동 | 학습 지지 | 검증 지지 | 테스트 지지 | v1 F1 | v2 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 식사 준비 | 98,388 | 26,799 | 23,765 | 0.960788 | 0.960909 |
| 식사 | 10,396 | 1,321 | 2,086 | 0.806398 | 0.783425 |
| 휴식 | 204,380 | 47,666 | 42,336 | 0.979186 | 0.979348 |
| 업무 | 7,556 | 1,688 | 1,786 | 0.932733 | 0.929427 |
| 침실에서 화장실로 이동 | 537 | 77 | 95 | 1.000000 | 1.000000 |
| 귀가 | 11 | 9 | 3 | 0.153846 | 0.240000 |
| 외출 | 83 | 3 | 14 | 0.054795 | 0.101695 |

테스트 귀가 F1은 0.1538에서 0.2400으로, 외출 F1은 0.0548에서 0.1017로 올랐다. 식사 F1은 0.8064에서 0.7834로, 업무 F1은 0.9327에서 0.9294로 내렸다. 귀가의 정답 3건은 두 모델 모두 맞혔고 v2에서 귀가 오탐은 33건에서 19건으로 줄었다.

혼동행렬의 행은 정답, 열은 보류 적용 전 예측이다. 순서는 식사 준비, 식사, 휴식, 업무, 침실에서 화장실로 이동, 귀가, 외출이다.

**v1**

```text
 23106    291    281     49      0     24     14
   239   1714    121      2      0      5      5
   963    156  41000    176      0      3     38
    20      0      5   1761      0      0      0
     0      0      0      0     95      0      0
     0      0      0      0      0      3      0
     5      4      0      2      0      1      2
```

**v2**

```text
 22971    394    337     48      0     11      4
   204   1711    163      2      0      4      2
   839    174  41114    172      0      3     34
    27      0     12   1745      0      0      2
     0      0      0      0     95      0      0
     0      0      0      0      0      3      0
     5      3      0      2      0      1      3
```

## 변경 없이 재생한 센서 프리셋 8개

동일 기준 시각과 이벤트열로 실제 저장 모델 두 개를 각각 실행했다. 괄호는 런타임의 최대 소프트맥스 확률이며, 입력 한 건은 확률 계산 전에 보류한다. 기대 활동은 화면 표시용이다.

| 프리셋 | 기대 활동 | 이벤트 | v1 실제 결과 | v2 실제 결과 |
| --- | --- | ---: | --- | --- |
| `meal_preparation` | 식사 준비 | 9 | 식사 준비 (0.9990) | 식사 준비 (0.9984) |
| `eating` | 식사 | 9 | 식사 (0.9956) | 식사 (0.9984) |
| `relax` | 휴식 | 8 | 휴식 (0.9984) | 휴식 (0.9999) |
| `work` | 업무 | 8 | 업무 (0.9920) | 업무 (0.9889) |
| `bed_to_toilet` | 침실에서 화장실로 이동 | 8 | 침실에서 화장실로 이동 (0.9915) | 침실에서 화장실로 이동 (0.9969) |
| `enter_home` | 귀가 | 5 | 식사 준비 (0.8262) | 휴식 (0.8213) |
| `leave_home` | 외출 | 6 | 판단 보류 · `low_confidence` (0.7164) | 휴식 (0.8238) |
| `insufficient_events` | 센서 입력 부족 | 1 | 판단 보류 · `insufficient_events` | 판단 보류 · `insufficient_events` |

두 모델 모두 정상 프리셋 7개 중 5개를 기대와 일치하게 분류했다. v2는 합성 귀가와 외출을 휴식으로 분류했다. 특히 외출은 v1에서 확신 부족으로 보류했으나 v2에서 임계값을 넘긴 오분류가 됐다. 입력 한 건 프리셋은 두 모델 모두 보류했다.

## 실행 근거와 한계

- 새 .venv 커널에서 `nbformat.validate()`, 전체 코드 셀의 Python 구문과 직전 한국어 Markdown, 새 모델 기본 로드, v1 명시적 로드, v2 누락 시 v1 복원, 손상·버전 불일치 진단을 확인했다. 시연은 원본 재학습 없이 캐시를 사용했다. 주석·경계·윈도 회귀 검증과 입력에 추가한 가짜 사용자·활동 필드가 결과에 영향을 주지 않는 검사도 통과했다.
- 별도 FastAPI 포트 18083과 Streamlit 포트 18513을 사용했다. 실제 Streamlit의 루트와 헬스 경로가 HTTP 200을 반환했다. 실제 `smart_home`의 pgvector 0.8.5에 읽기 전용으로 연결했다. Streamlit AppTest가 여덟 시나리오의 선택과 실행 버튼을 조작했고, 실제 HTTP 응답과 화면의 기대 활동·추정 또는 보류 사유·관측 시각·v2 표시를 대조했다. B로 전환한 뒤에도 최근 HAR 결과가 보였고 기기·유지 작업·기기 로그는 시연 전후 일치했다.
- HAR 추론은 실제 체크포인트로 실행했다. 격리 검증에서만 Tev1에 비어 있는 캐시 경로를 지정해 미가용 분기를 사용했다. 이번 HAR 단계의 Tev1 실가중치 추론과 브라우저 렌더링은 미검증이다. 브라우저 자동화는 Work 설정에서 비활성화되어 있었다. 기존 사용자 API, Streamlit 및 PostgreSQL 프로세스를 제어하지 않았다.
- 결과는 단일 가정의 단일 날짜 분할과 시드에 기반한다. 연속 창이 센서 이벤트를 공유하므로 창 수는 독립 활동의 횟수와 다르다. 특히 귀가 3건·외출 14건의 테스트 F1은 몇 건의 결과에 크게 움직인다. 합성 센서열과 Aruba의 실제 문 센서 분포 차이 때문에 프리셋은 화면 흐름과 개별 결과 확인에 사용한다.

## 재현

`C:\CHOI\smart_home`에서 Python 3.12.6, torch 2.15.0+cpu, NumPy 2.5.4, scikit-learn 1.9.1, nbclient 0.11.0, Streamlit 1.67.0으로 실행했다. v2를 재학습하려면 같은 가상환경의 커널에서 노트북의 공통 설정·집 레지스트리와 HAR 학습 함수 정의까지 실행한 뒤 아래 호출을 별도 셀에서 실행한다. 학습 결과는 `data/har/aruba-gru-v2.pt`에 저장된다.

```python
report = train_and_save_har(ARUBA_PATH, HAR_CHECKPOINT)
pprint(report["test"])
```

후보를 다시 비교하려면 학습 실행 셀을 건너뛰고 노트북의 HAR 정의 뒤에 아래 코드를 실행한다. 검증 함수는 각 후보의 에포크별 Macro-F1을 출력하며, 반환 보고서에 최고 에포크와 클래스별 검증 F1, 보류 임계값을 담는다.

```python
splits, preprocessing, _ = prepare_har_data(ARUBA_PATH)
candidates = [
    ("v1", 0.50, 32, 6, 2),
    ("longer", 0.50, 32, 12, 3),
    ("rare_weight", 0.75, 32, 12, 3),
    ("rare_weight_wider", 0.75, 48, 12, 3),
]
prior_threads = torch.get_num_threads()
try:
    torch.set_num_threads(HAR_TRAINING["cpu_threads"])
    for name, power, hidden_size, epochs, patience in candidates:
        training = dict(
            HAR_TRAINING, class_weight_power=power, hidden_size=hidden_size,
            max_epochs=epochs, patience=patience,
        )
        metadata = dict(
            preprocessing, training=training,
            architecture={"embedding_size": training["embedding_size"], "hidden_size": hidden_size},
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(training["seed"])
            _, validation = fit_har_gru(splits["train"], splits["validation"], metadata)
        print(name, validation)
finally:
    torch.set_num_threads(prior_threads)
```

시연은 새 커널에서 체크포인트 로드 셀을 실행한다. 두 저장 모델의 테스트와 프리셋은 아래 코드로 재확인한다.

```python
for checkpoint in (HAR_LEGACY_CHECKPOINT, HAR_CHECKPOINT):
    runtime = load_har(checkpoint)
    verify_har_test_report(runtime, ARUBA_PATH)
    verify_har_demo(runtime)
```

화면을 재현하려면 노트북의 DB 연결과 API 정의를 실행한 뒤 사용하지 않는 포트에서 새 API를 시작한다. 검증용 API가 18083이면 다음 명령으로 새 화면을 연다.

```powershell
$env:SMART_HOME_API_URL = 'http://127.0.0.1:18083'
.\.venv\Scripts\python.exe -m streamlit run streamlit_smarthome.py --server.headless true --server.address 127.0.0.1 --server.port 18513 --browser.gatherUsageStats false
```

기존 저장 출력과 메타데이터는 최신 디스크본에서 보존했고 Streamlit의 SHA-256은 시작 리비전과 같다.

이번 작업의 자동 검증 도구와 원시 출력은 작업 디렉터리 밖의 임시 경로에 남겼다. `validation.json`은 후보별 검증값, `test-and-presets.json`은 테스트·프리셋 비교, `ui-evidence.json`은 실제 API 응답과 화면 텍스트를 담는다. `sweep.py`는 처음 복사한 노트북의 공통 경로와 최종 학습 함수와 동일한 코드로 후보를 비교했다. 이후 검증은 아래 SHA-256의 구현 소스에서 실행했다.

```powershell
$work = Join-Path $env:TEMP 'smart-home-har-stage3'
$py = '.\.venv\Scripts\python.exe'
& $py -B -u (Join-Path $work 'sweep.py')
# 최초 v2 파일을 만들 때 실행한 테스트 평가와 저장
& $py -B -u (Join-Path $work 'postselection.py')
& $py -B -u (Join-Path $work 'fresh_kernel.py') eval
& $py -B -u (Join-Path $work 'fresh_kernel.py') train
& $py -B -u (Join-Path $work 'ui.py')
Get-FileHash -Algorithm SHA256 smarthome.ipynb,streamlit_smarthome.py,data/har/aruba-gru.pt,data/har/aruba-gru-v2.pt
```

| 파일 | SHA-256 |
| --- | --- |
| `data/har/aruba-gru.pt` | `13d74fc92eb6da3fb5f20794a87355074e1a539f9d4ece687912ed1a68898a2a` |
| `data/har/aruba-gru-v2.pt` | `42cc638746b7405ffac5bc709a349acfc10b7416e7c5e45df88dc9b727e32d60` |
| `smarthome.ipynb` | `f355cefb0ebc7664723870ce36d5d0c11bc35287b0838d79337b18e211b9c601` |
| `streamlit_smarthome.py` | `f4ea4d9010722f0b012d1ca0992583bfb603574370b6767de033f7ed77efa88b` |

## 5단계 · 센서 이벤트 표 검증

Streamlit의 `합성 센서 이벤트` 표가 `/state`의 공간 이름을 사용한다. 같은 공간의 움직임 센서는 레지스트리 순서대로 번호를 붙이고 `D001`은 `현관문 개폐 센서`로 표시한다. 원본 시각, 센서 ID, 상태와 선택적 출처도 표에 표시한다. 기존 API 형식만으로 표시를 구성하며 센서 이벤트와 HAR 입력은 기존 형식을 사용한다.

| 공간 | 센서 | 센서 ID | 확인한 상태 |
| --- | --- | --- | --- |
| 주방 | 움직임 센서 1 | `M015` | `ON`, `OFF` |
| 주방 | 움직임 센서 2 | `M017` | `ON`, `OFF` |
| 주방 | 움직임 센서 3 | `M018` | `ON`, `OFF` |
| 현관 | 움직임 센서 1 | `M011` | `ON`, `OFF` |
| 현관 | 현관문 개폐 센서 | `D001` | `OPEN`, `CLOSE` |

- **MVP-04 통과.** `.venv`의 새 커널에서 저장된 v2로 Aruba의 시간순 테스트 창 70,085건을 다시 평가했다. 보류 전 Macro-F1 `0.713543375927383`, 확신 부족 보류 2,025건(`2.889348648%`)이 체크포인트의 보고 값과 일치했다. 이번 표시 검증에서는 저장 가중치를 사용했다.
- **MVP-06 통과.** 별도 API 포트 `18084`, Streamlit 포트 `18514`, 테스트 전용 스키마에서 실제 pgvector 0.8.5와 v2 추론을 연결했다. Streamlit의 루트와 헬스 경로가 HTTP 200을 반환했고, AppTest로 프리셋 8개에서 센서 20종과 이벤트 54행의 순서, 시각, 공간, 이름, ID, 상태, 출처를 대조했다. 귀가·외출은 휴식으로 표시됐고 입력 한 건은 `insufficient_events`로 보류됐다. A에서 B로 바꿔도 최신 시연 결과가 표시됐다. 이 표 시험에서 Tev1은 캐시 미가용 분기를 사용했다. 출처가 없는 기존 형식과 미등록 센서 표시는 별도의 고정 입력으로 확인했다.
- **MVP-06 통과.** 임시 Docker 컨테이너와 볼륨을 DB 포트 `15442`에 생성해 입력 없이 `postgres` 자격증명을 만들고 `smart_home` 및 pgvector 0.8.5에 연결했다. 두 번째 새 커널은 같은 비밀번호 파일과 기록을 재사용했다. 검증용 컨테이너와 볼륨만 정리했다.
- `nbformat` 검사, Python 구문, 코드 셀 53개의 직전 한국어 Markdown 검사가 통과했다. 노트북의 시작·종료 SHA-256이 일치하므로 저장 출력과 메타데이터도 바이트 단위로 보존됐다. 작업 전후 루트 파일 목록이 일치했다. 검증은 격리 포트에서 실행했고 사용자 API, Streamlit, 공유 Docker DB를 중지하거나 재시작하지 않았다. 브라우저 렌더링은 AppTest와 HTTP로 확인하는 이번 범위에서 미검증이다.

기존 통합 하네스의 노트북 검증 함수를 별도 포트 `18085`와 테스트 전용 스키마에서 다시 실행했다.

| 기준 | 상태 | 확인한 결과 |
| --- | --- | --- |
| MVP-01 | 통과 | 실제 Gemma 요청으로 방 1 전등을 켜고 끄면서 커밋된 상태, 응답 문구, AppTest 표시를 확인했다. 잘못된 대상·입력과 미지원 명령으로 기기가 바뀌지 않았다. |
| MVP-02 | 통과 | 실제 PostgreSQL에서 A의 10분 유지, B의 끄기·취소 차단, 만료 후 끄기, 동일 UUID 재전송과 충돌 409를 확인했다. 제약 조건과 기억 시드 오류 주입으로 503 및 롤백 경로를 확인했다. |
| MVP-03 | 통과 | 실제 E5와 pgvector에서 A/B별 코사인 거리순 상위 3건을 조회했다. 실제 Tev1은 T04에서 A 30%, B 70%를 제안했고 T06 휴식은 `log_only`였다. 잘못된 선택 문자와 강제 타임아웃은 시험 대역으로 확인했다. 첫 통합 실행은 Tev1 첫 호출에서 `tev1_timeout`이었고, CPU 스레드 기본값으로 다시 실행한 통합 시험은 통과했다. |
| MVP-04 | 통과 | v2 저장 가중치로 Aruba 테스트를 다시 채점하고, 합성 프리셋의 추정과 입력 부족 보류를 AppTest에서 확인했다. |
| MVP-05 | 통과 | 실제 등록 모델 `gemma4-ally`의 한국어 입력은 1,076토큰이었고 Ollama 계수와 일치했다. 32,768토큰 한도 내 생성 여유 256, 기록·기억 축소, 초과 입력 보류와 격리 API의 연결 실패 표시를 확인했다. 잘못된 모델 출력은 응답 훼손을 주입했다. |
| MVP-06 | 통과 | 임시 Docker 첫 연결과 새 커널의 재사용, 실제 Streamlit HTTP와 AppTest, 노트북 구조와 파일 범위를 확인했다. |

공유 DB 중지 시험은 격리 스키마의 오류 주입으로 대체해 503을 검증했다. 읽기 전용 독립 리뷰는 저장된 두 HAR 모델의 Aruba 테스트 지표를 다시 계산하고, 최종 소스와 센서 표시의 HTTP·AppTest 근거를 대조했다. 전체 MVP 리뷰에서 확인한 기존 명령 표현 처리 문제와 브라우저 직접 렌더링은 이번 변경 범위 밖이다.

다음 명령은 `C:\CHOI\smart_home`에서 실행했다. `sensor-table.json`에 실제 API 응답과 표의 모든 행을 기록했다. `har.log`, `ui.log`, `integration.log`, `integration-retry.log`, `docker.log`에 검증 출력이 있다.

```powershell
$work = Join-Path $env:TEMP 'smart-home-sensor-display-stage5'
$py = '.\.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
& $py -B -u (Join-Path $work 'check_sensor_display.py') har *> (Join-Path $work 'har.log')
& $py -B -u (Join-Path $work 'check_sensor_display.py') ui *> (Join-Path $work 'ui.log')
& $py -B -u (Join-Path $work 'check_sensor_display.py') integration *> (Join-Path $work 'integration-retry.log')
& $py -B -u (Join-Path $work 'check_docker_lifecycle.py') *> (Join-Path $work 'docker.log')
Get-FileHash -Algorithm SHA256 smarthome.ipynb,streamlit_smarthome.py,data/har/aruba-gru-v2.pt
```

| 5단계 검증 파일 | SHA-256 |
| --- | --- |
| `smarthome.ipynb` | `f355cefb0ebc7664723870ce36d5d0c11bc35287b0838d79337b18e211b9c601` |
| `streamlit_smarthome.py` | `23f043681d4cfc0ee81442eae666ab47f88a7fe6beeffc718cd283abc51c4e57` |
| `data/har/aruba-gru-v2.pt` | `42cc638746b7405ffac5bc709a349acfc10b7416e7c5e45df88dc9b727e32d60` |

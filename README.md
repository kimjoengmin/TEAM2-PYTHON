## 🔥 TEAM 2 PYTHON

### 🚀 Cell 2 : 컬럼명 및 문자열 정규화
```python
def standardize_columns(df: pd.DataFrame)
```
- DataFrame의 컬럼명과 문자열 값을 소문자+언더스코어 형식으로 정규화하는 함수
#### 🧠 기능
- 컬럼명 정규화 (컬럼명 양쪽 공백 제거, 소문자 변환, 공백을 '_' 대체, 특수문자 제거)
```python
df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(' ', '_', regex=False)
          .str.replace(r'[^0-9a-zA-Z_]', '', regex=True)
    )
```
- 문자열 또는 카테고리 타입 컬럼 값 정규화 (값을 문자열로 변환, 값 양쪽 공백 제거, 공백 '_'으로 대체)
```python
for col in df.select_dtypes(include=['object', 'category']):
        df[col] = (
            df[col].astype(str)
                   .str.strip()
                   .str.replace(' ', '_', regex=False)
        )
```

---

### 🚀 Cell 3 : ID 및 결측치 비율 높은 컬럼 제거
```python
def drop_id_unnamed_and_missing(df: pd.DataFrame, missing_thresh: float = 0.5)
```
- ID/Unnamed 컬럼 및 결측치 비율이 지정 기준 이상인 컬럼을 제거하는 함수
#### 🧠 기능
- 'Unnamed'로 시작하거나 'id'로 끝나는 컬럼명을 찾고 제거
```python
to_drop = df.columns[df.columns.str.contains(r'^(?:unnamed)|id$', case=False, regex=True)]
df = df.drop(columns=to_drop)
```
- 컬럼별 결측치 비율 계산 및 비율이 missing_thresh(**float = 0.5**) 초과인 컬럼 제거
```python
missing = df.isnull().mean()
df = df.drop(columns=missing[missing > missing_thresh].index)
```

---

### 🚀 Cell 4 : 결측치 대치(Imputation)
```python
def impute_missing(df: pd.DataFrame)
```
- ID/Unnamed 컬럼 및 결측치 비율이 지정 기준 이상인 컬럼을 제거하는 함수
#### 🧠 기능
- 수치형 컬럼 목록 추출 및 중간값(median)으로 대치
```python
num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        df[num_cols] = pd.DataFrame(
            SimpleImputer(strategy='median').fit_transform(df[num_cols]),
            columns=num_cols
        )
```
- 범주형 컬럼 목록 추출 및 최빈값(most_frequent)으로 대치
```python
cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    if cat_cols:
        df[cat_cols] = pd.DataFrame(
            SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols]),
            columns=cat_cols
        )
```

---

### 🚀 Cell 5 : 이상치 제거 함수 (IQR 기준)
```python
def remove_outliers(df: pd.DataFrame, max_removal: float = 0.2)
```
- IQR(Interquartile Range) 기준으로 이상치를 제거하되, 전체 데이터의 max_removal 비율 이하만 제거하는 함수
#### 🧠 기능
- IRQ 계산
```python
for col in df.select_dtypes(include=np.number):
  Q1, Q3 = df[col].quantile([0.25, 0.75])
  IQR = Q3 - Q1
```
- IQR 범위 내에 있는 데이터만 True인 mask 생성
```python
mask = df[col].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
```
- 남게 될 데이터가 전체의 (1 - max_removal) 이상인 경우에만 이상치 제거 수행
```python
if mask.sum() > (1 - max_removal) * total:
  df = df[mask]
```

---

### 🚀 Cell 6 : 높은 상관관계 컬럼 제거
```python
def drop_highly_correlated(df: pd.DataFrame, threshold: float = 0.95)
```
- 수치형 컬럼 중 상관계수(절댓값)가 threshold 이상인 컬럼을 제거하는 함수
#### 🧠 기능
- 구치형 컬럼끼리의 상관계수 행렬 계산 (절댓값 기준)
```python
corr = df.select_dtypes(include=np.number).corr().abs()
```
- 상삼각행렬(Upper Triangle)만 선택 (중복 비교 방지)
```python
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
```
- threshold를 초과하는 상관관계가 있는 컬럼 리스트 추출
```python
to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
```
- 해당 컬럼들 제거 후 반환
```python
return df.drop(columns=to_drop)
```

---

### 🚀 Cell 7 : 범주형 인코딩 및 수치형 스케일링
```python
def encode_and_normalize(X: pd.DataFrame, max_onehot: int = 10)
```
- 범주형 변수는 Label/OneHot 인코딩, 수치형 변수는 StandardScaler로 정규화하는 함수
#### 🧠 기능
- 범주형(object, category, bool) 컬럼 인코딩
```python
for col in X.select_dtypes(include=['object', 'category', 'bool']):
    nun = X[col].nunique()
```
- 고유값이 2개 이하 → Label Encoding
```python
if nun <= 2:
    X[col] = LabelEncoder().fit_transform(X[col])
```
- 고유값이 max_onehot 이하 → One-Hot Encoding
```python
elif nun <= max_onehot:
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    arr = ohe.fit_transform(X[[col]])
    cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
    X[cols] = arr
    X = X.drop(columns=[col])
```
- 고유값이 많으면 해당 컬럼 삭제
```python
X = X.drop(columns=[col])
```
- 수치형 컬럼 정규화 (StandardScaler 적용)
```python
num_cols = X.select_dtypes(include=np.number).columns
X[num_cols] = StandardScaler().fit_transform(X[num_cols])
```

---

### 🚀 Cell 8 : 데이터 전처리 전체 프로세스
```python
def preprocess(input_file: str, target_col: str,
               missing_thresh: float = 0.5,
               max_removal: float = 0.2,
               corr_thresh: float = 0.95,
               max_onehot: int = 10) -> str:
    # 1. CSV 파일을 읽어와 DataFrame으로 저장
    df = pd.read_csv(input_file)

    # 2. 타겟 컬럼에 결측치가 있는 행을 제거하고 인덱스를 재설정
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # 3. 타겟 변수(y)와 피처(X) 분리
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 4. 컬럼명 및 문자열 값을 소문자 및 언더스코어 형식으로 정규화
    X = standardize_columns(X)

    # 5. 'Unnamed' 또는 'id' 컬럼 및 결측치 비율이 높은 컬럼 제거
    X = drop_id_unnamed_and_missing(X, missing_thresh)

    # 6. 결측치 대치 (수치형: 중간값, 범주형: 최빈값)
    X = impute_missing(X)

    # 7. 이상치 제거 (IQR 기준, 전체 데이터의 max_removal 비율 이하만 제거)
    X = remove_outliers(X, max_removal)

    # 8. 높은 상관관계를 가진 수치형 컬럼 제거 (상관계수 절댓값이 corr_thresh 이상)
    X = drop_highly_correlated(X, corr_thresh)

    # 9. 범주형 변수 인코딩 및 수치형 변수 정규화
    X = encode_and_normalize(X, max_onehot)

    # 10. 전처리 과정에서 제거된 행을 반영하여 y의 인덱스를 X와 일치시킴
    y = y.loc[X.index].reset_index(drop=True)

    # 11. 원본 파일명에서 확장자를 제거하고 새로운 파일명 생성
    base = os.path.splitext(os.path.basename(input_file))[0]
    out_file = f"processed_{base}.csv"

    # 12. 전처리된 X와 y를 합쳐 새로운 CSV 파일로 저장
    pd.concat([X.reset_index(drop=True), y], axis=1).to_csv(out_file, index=False)

    # 13. 저장된 파일 경로를 출력
    print(f"[INFO] Processed data saved to: {out_file}")

    # 14. 전처리된 파일 경로 반환
    return out_file



---

### 🚀 Cell 9 : 메인 실행
```python
input_file = '파일 경로'
processed_path = preprocess(input_file, '타겟 컬럼')
```

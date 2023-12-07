import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

# 데이터 불러오기
data = pd.read_csv('./df/dataset.csv', names=['MQ1', 'MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'CO2'])

# 데이터 전처리 함수
def preprocess_inputs(df):
    df = df.copy()
    y = df['CO2']
    X = df.drop('CO2', axis=1)
    return train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

# 학습용 및 테스트용 데이터 얻기
X_train, X_test, y_train, y_test = preprocess_inputs(data)

# 랜덤 포레스트 모델 생성 및 학습
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)

# 모델 성능 평가 및 출력
acc = model.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(acc * 100))

# TreeExplainer 사용
explainer = shap.TreeExplainer(model)

# X_test의 컬럼 순서를 X_train의 컬럼 순서와 동일하게 맞추기
X_test = X_test[X_train.columns]

# SHAP 값 생성
shap_values = explainer.shap_values(X_test)

# SHAP 요약 그래프 그리기
shap.summary_plot(shap_values, X_test, class_names=model.classes_)

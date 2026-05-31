from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 데이터셋 로드
data = load_breast_cancer()

X = data.data
y = data.target

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# 예측
pred = model.predict(X_test)

# 정확도 계산
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)
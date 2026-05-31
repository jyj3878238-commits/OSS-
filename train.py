from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# 데이터셋 로드
data = load_breast_cancer()

X = data.data
y = data.target

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# -----------------------------
# Decision Tree
# -----------------------------
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

# -----------------------------
# Random Forest
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

# -----------------------------
# 결과 출력
# -----------------------------
print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

print("Dataset loaded")
print("Samples:", len(data.data))
print("Features:", len(data.feature_names))
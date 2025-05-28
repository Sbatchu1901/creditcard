from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    log_reg = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)

    log_reg.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)

    return log_reg, rf_clf

from cc import load_data
from preprocessing import preprocess_data, split_data, apply_smote
from modeltraining import train_models
from evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
import joblib

def run_pipeline():
    file_path = r"C:\Users\sruja\OneDrive\Desktop\creditcard.csv"
    df = load_data(file_path)
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    log_reg, rf_clf = train_models(X_train_res, y_train_res)

    y_pred_log, y_proba_log, cm_log = evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
    y_pred_rf, y_proba_rf, cm_rf = evaluate_model(rf_clf, X_test, y_test, "Random Forest")

    plot_confusion_matrix(cm_log, "Logistic Regression")
    plot_confusion_matrix(cm_rf, "Random Forest")
    
    plot_roc_curve(y_test, {"Logistic Regression": y_proba_log, "Random Forest": y_proba_rf})

if __name__ == "__main__":
    run_pipeline()



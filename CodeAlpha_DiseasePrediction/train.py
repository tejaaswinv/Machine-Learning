from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,confusion_matrix,ConfusionMatrixDisplay
from xgboost import XGBClassifier

SEED=42

def main():
    data=load_breast_cancer(as_frame=True); X=data.data; y=data.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    models={
        "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,random_state=SEED))]),
        "SVM":Pipeline([("scale",StandardScaler()),("model",SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=SEED))]),
        "Random Forest":RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=SEED,n_jobs=-1),
        "XGBoost":XGBClassifier(n_estimators=350,max_depth=4,learning_rate=.04,subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=SEED,n_jobs=4),
    }
    results=[]; best=None
    for name,m in models.items():
        m.fit(X_train,y_train); pred=m.predict(X_test); prob=m.predict_proba(X_test)[:,1]
        r={"model":name,"accuracy":float(accuracy_score(y_test,pred)),"precision":float(precision_score(y_test,pred)),"recall":float(recall_score(y_test,pred)),"f1":float(f1_score(y_test,pred)),"roc_auc":float(roc_auc_score(y_test,prob))}
        results.append(r)
        if best is None or r["roc_auc"]>best["metrics"]["roc_auc"]: best={"name":name,"model":m,"metrics":r,"pred":pred,"prob":prob}
    out=Path("artifacts"); out.mkdir(exist_ok=True); joblib.dump(best["model"],out/"best_disease_model.joblib"); (out/"metrics.json").write_text(json.dumps(results,indent=2))
    schema={c:{"min":float(X[c].min()),"max":float(X[c].max()),"default":float(X[c].median())} for c in X.columns}; (out/"feature_schema.json").write_text(json.dumps(schema,indent=2))
    (out/"meta.json").write_text(json.dumps({"features":list(X.columns),"target_names":list(data.target_names),"best_model":best["name"]},indent=2))
    fpr,tpr,_=roc_curve(y_test,best["prob"]); fig=plt.figure(figsize=(7,5)); plt.plot(fpr,tpr,label=f"{best['name']} AUC={best['metrics']['roc_auc']:.3f}"); plt.plot([0,1],[0,1],linestyle="--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("Disease Prediction ROC Curve"); plt.legend(); plt.tight_layout(); fig.savefig(out/"roc_curve.png",dpi=180); plt.close(fig)
    cm=confusion_matrix(y_test,best["pred"]); fig,ax=plt.subplots(figsize=(5,5)); ConfusionMatrixDisplay(cm,display_labels=data.target_names).plot(ax=ax); plt.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=180); plt.close(fig)
    print(pd.DataFrame(results).sort_values("roc_auc",ascending=False).to_string(index=False)); print(f"\nBest model: {best['name']}")

if __name__=="__main__": main()

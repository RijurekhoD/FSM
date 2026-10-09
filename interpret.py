from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def shap_run(model,Xtrain,Xtest,names,out,name,bg_n=80,ex_n=300,seed=42):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(seed)
    bg=np.asarray(Xtrain)[rng.choice(len(Xtrain),min(bg_n,len(Xtrain)),replace=False)]
    ex=np.asarray(Xtest)[rng.choice(len(Xtest),min(ex_n,len(Xtest)),replace=False)]
    try:
        vals=shap.TreeExplainer(model).shap_values(ex)
        if isinstance(vals,list): vals=vals[-1]
        vals=np.asarray(vals)
        if vals.ndim==3: vals=vals[:,:,-1]
    except Exception:
        pred=lambda z:model.predict_proba(z)[:,1]
        vals=shap.KernelExplainer(pred,bg).shap_values(ex,nsamples=100)
        if isinstance(vals,list): vals=vals[-1]
        vals=np.asarray(vals)
    pd.DataFrame({"feature":names,"mean_abs_SHAP":np.mean(np.abs(vals),axis=0)}).sort_values("mean_abs_SHAP",ascending=False).to_csv(out/f"{name}_importance.csv",index=False)
    np.savez_compressed(out/f"{name}_values.npz",values=vals,X=ex,feature_names=np.asarray(names))
    try:
        shap.summary_plot(vals,ex,feature_names=names,show=False); plt.tight_layout()
        plt.savefig(out/f"{name}_summary.png",dpi=300); plt.close()
    except Exception as e: print("SHAP plot warning:",e)

def pysr_run(X,y,names,out,name,niter=100,timeout=1800,maxsize=20,select_k=6,seed=42):
    from pysr import PySRRegressor,TemplateExpressionSpec
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    # Template fixes the outer logistic link; PySR searches for the inner f(X).
    template=TemplateExpressionSpec(expressions=["f"],variable_names=list(names),
        combine="1 / (1 + exp(-f("+", ".join(names)+")))")
    model=PySRRegressor(expression_spec=template,binary_operators=["+","-","*","/"],
        unary_operators=["square","sqrt","log","exp","abs"],niterations=niter,
        timeout_in_seconds=timeout,maxsize=maxsize,select_k_features=min(select_k,len(names)),
        model_selection="best",random_state=seed,deterministic=True,parallelism="serial",
        output_directory=str(out/f"pysr_{name}"),progress=True)
    model.fit(np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.float32))
    model.equations_.to_csv(out/f"{name}_equations.csv",index=False)
    (out/f"{name}_equation.txt").write_text("Probability equation: p = 1/(1+exp(-f(X)))\\nInner symbolic expression:\\n"+str(model.sympy())+"\\nLaTeX:\\n"+str(model.latex()),encoding="utf-8")
    return model

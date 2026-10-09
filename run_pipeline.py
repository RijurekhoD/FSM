from pathlib import Path
import json, warnings
import numpy as np, pandas as pd, rasterio, joblib
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.base import clone
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from config import *
from src.core import read_target,warp_predictor,sample_at_inventory_centres,read_stack,save_map,build_models,prob,SpatialStack,threshold_train_oof,metrics
from src.interpret import shap_run,pysr_run

warnings.filterwarnings("ignore")
OUT=Path(OUTPUT_DIR); DATA=Path(DATA_DIR)
for sub in ["maps","shap","symbolic","models","harmonized"]: (OUT/sub).mkdir(parents=True,exist_ok=True)
for fn in list(PREDICTORS.values())+[TARGET_FILE]:
    if not (DATA/fn).exists(): raise FileNotFoundError(f"Missing input: {DATA/fn}")
yg,valid,profile,ttrans,tcrs=read_target(DATA/TARGET_FILE,TARGET_POSITIVE_VALUES,TARGET_NEGATIVE_VALUES)
if tcrs is None or not tcrs.is_projected: raise ValueError("Target must use projected metre-based CRS.")
with rasterio.open(DATA/TARGET_FILE) as _target_src:
    target_bounds = _target_src.bounds
r,c=np.where(valid); yall=yg[r,c].astype(int)
xx,yy=rasterio.transform.xy(ttrans,r,c,offset="center")
gx=np.floor(np.asarray(xx)/SPATIAL_BLOCK_SIZE_M).astype(int); gy=np.floor(np.asarray(yy)/SPATIAL_BLOCK_SIZE_M).astype(int)
groups=np.asarray([f"{a}_{b}" for a,b in zip(gx,gy)])
if len(np.unique(yall))<2: raise ValueError("Target has only one class; check target coding.")
sp=GroupShuffleSplit(n_splits=1,test_size=TEST_BLOCK_FRACTION,random_state=RANDOM_SEED)
itr,ite=next(sp.split(np.zeros(len(yall)),yall,groups))
if len(np.unique(yall[itr]))<2 or len(np.unique(yall[ite]))<2: raise ValueError("Spatial split has one class in train/test. Adjust block size or seed.")
names=list(PREDICTORS); results=[]
meta={"target":TARGET_FILE,"crs":str(tcrs),"seed":RANDOM_SEED,"block_size_m":SPATIAL_BLOCK_SIZE_M,
      "n_inventory_cells":len(yall),"n_train":len(itr),"n_test":len(ite),"resolutions_m":RESOLUTIONS_M}
rng=np.random.default_rng(RANDOM_SEED)

for res in RESOLUTIONS_M:
    print(f"\n--- {res} m ---")
    paths=[]
    for feat,fn in PREDICTORS.items():
        p=warp_predictor(DATA/fn,OUT/"harmonized"/f"{feat}_{res}m.tif",res,tcrs,target_bounds); paths.append(p)
    X=np.column_stack([sample_at_inventory_centres(p,valid,ttrans,tcrs) for p in paths]).astype(np.float32)
    usable=np.all(np.isfinite(X),axis=1); ids=np.flatnonzero(usable)
    tr=np.intersect1d(itr,ids); te=np.intersect1d(ite,ids)
    Xtr,ytr,gtr=X[tr],yall[tr],groups[tr]; Xte,yte=X[te],yall[te]
    if len(tr)<30 or len(te)<10: raise ValueError(f"Too few complete inventory samples at {res} m.")
    models=build_models(RANDOM_SEED,QUICK_RUN); fitted={}; oof_probs={}
    splitter=GroupKFold(n_splits=min(CV_SPLITS,len(np.unique(gtr))))
    for name,est in models.items():
        print("Fit",name)
        oof=np.full(len(ytr),np.nan)
        for a,b in splitter.split(Xtr,ytr,gtr):
            m=clone(est).fit(Xtr[a],ytr[a]); oof[b]=prob(m,Xtr[b])
        t=threshold_train_oof(ytr,oof)
        model=clone(est).fit(Xtr,ytr); fitted[name]=model; oof_probs[name]=oof
        p=prob(model,Xte); results.append(metrics(yte,p,t,name,res))
        joblib.dump(model,OUT/"models"/f"{name}_{res}m.joblib")
    stack=SpatialStack(models,CV_SPLITS,RANDOM_SEED).fit(Xtr,ytr,gtr)
    # Stack threshold from training OOF base probabilities passed through fitted meta-model.
    Z=np.column_stack([oof_probs[n] for n in models])
    ok=np.all(np.isfinite(Z),axis=1)
    p_oof_stack=stack.meta_.predict_proba(Z[ok])[:,list(stack.classes_).index(1)]
    ts=threshold_train_oof(ytr[ok],p_oof_stack)
    ptest=stack.predict_proba(Xte)[:,list(stack.classes_).index(1)]
    results.append(metrics(yte,ptest,ts,"StackedEnsemble",res))
    joblib.dump(stack,OUT/"models"/f"StackedEnsemble_{res}m.joblib")

    # Fit mapping models using all inventory samples after test metrics are frozen.
    Xfull=X[ids]; yfull=yall[ids]; gfull=groups[ids]
    fullmodels={n:clone(e).fit(Xfull,yfull) for n,e in models.items()}
    fullstack=SpatialStack(models,CV_SPLITS,RANDOM_SEED).fit(Xfull,yfull,gfull)
    joblib.dump(fullstack,OUT/"models"/f"final_StackedEnsemble_{res}m.joblib")
    for n,m in fullmodels.items(): joblib.dump(m,OUT/"models"/f"final_{n}_{res}m.joblib")
    arr,vmap,mprof=read_stack(paths); flat=arr.reshape(-1,len(names)); pos=np.flatnonzero(vmap.ravel())
    stackvals=np.empty(len(pos),dtype=np.float32); sums=np.zeros(len(pos),dtype=np.float64)
    ind={n:np.empty(len(pos),dtype=np.float32) for n in fullmodels} if WRITE_INDIVIDUAL_MAPS else {}
    for start in range(0,len(pos),PREDICTION_CHUNK_SIZE):
        loc=pos[start:start+PREDICTION_CHUNK_SIZE]; chunk=flat[loc]
        for n,m in fullmodels.items():
            pr=prob(m,chunk); sums[start:start+len(loc)]+=pr
            if WRITE_INDIVIDUAL_MAPS: ind[n][start:start+len(loc)]=pr
        stackvals[start:start+len(loc)]=fullstack.predict_proba(chunk)[:,list(fullstack.classes_).index(1)]
    meanp=sums/len(fullmodels)
    if WRITE_INDIVIDUAL_MAPS:
        for n,z in ind.items():
            outarr=np.full(vmap.size,np.nan,dtype=np.float32); outarr[pos]=z
            save_map(OUT/"maps"/f"{n}_{res}m.tif",outarr.reshape(vmap.shape),vmap,mprof)
    z=np.full(vmap.size,np.nan,dtype=np.float32); z[pos]=meanp
    save_map(OUT/"maps"/f"MeanProbabilityEnsemble_{res}m.tif",z.reshape(vmap.shape),vmap,mprof)
    z=np.full(vmap.size,np.nan,dtype=np.float32); z[pos]=stackvals
    save_map(OUT/"maps"/f"StackedEnsemble_{res}m.tif",z.reshape(vmap.shape),vmap,mprof)

    print("SHAP: interpret held-out observations")
    for n,m in fitted.items():
        shap_run(m,Xtr,Xte,names,OUT/"shap",f"{n}_{res}m",SHAP_BACKGROUND_N,SHAP_EXPLAIN_N,RANDOM_SEED)
    shap_run(stack,Xtr,Xte,names,OUT/"shap",f"StackedEnsemble_{res}m",SHAP_BACKGROUND_N,SHAP_EXPLAIN_N,RANDOM_SEED)

    print("PySR symbolic surrogate; this may take a long time")
    take=min(PYSR_TRAIN_N,len(Xtr)); si=rng.choice(len(Xtr),take,replace=False) if take<len(Xtr) else np.arange(len(Xtr))
    targetp=stack.predict_proba(Xtr[si])[:,list(stack.classes_).index(1)]
    sr=pysr_run(Xtr[si],targetp,names,OUT/"symbolic",f"StackedEnsemble_{res}m",PYSR_ITERATIONS,
                PYSR_TIMEOUT_SECONDS,PYSR_MAX_SIZE,PYSR_SELECT_K_FEATURES,RANDOM_SEED)
    try:
        srtest=np.asarray(sr.predict(Xte)).reshape(-1)
        pstack=stack.predict_proba(Xte)[:,list(stack.classes_).index(1)]
        pd.DataFrame([{"resolution_m":res,"surrogate_RMSE":mean_squared_error(pstack,srtest)**.5,
            "surrogate_MAE":mean_absolute_error(pstack,srtest),"surrogate_R2":r2_score(pstack,srtest)}]).to_csv(
            OUT/"symbolic"/f"StackedEnsemble_{res}m_fidelity.csv",index=False)
    except Exception as e: print("Symbolic fidelity could not be calculated:",e)
    meta.setdefault("valid_samples_by_resolution",{})[str(res)]=int(len(ids))

pd.DataFrame(results).to_csv(OUT/"metrics.csv",index=False)
(OUT/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
print("\nFinished. Check output/metrics.csv, maps/, shap/, symbolic/ and run_metadata.json.")

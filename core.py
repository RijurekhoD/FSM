from pathlib import Path
import numpy as np, rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import from_origin
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def read_target(path, positives, negatives):
    with rasterio.open(path) as s:
        a=s.read(1); prof=s.profile.copy(); tr=s.transform; crs=s.crs; nd=s.nodata
    valid=np.isfinite(a)
    if nd is not None: valid &= a != nd
    y=np.full(a.shape,-1,dtype=np.int8)
    for v in positives: y[a==v]=1
    for v in negatives: y[a==v]=0
    valid &= y>=0
    return y,valid,prof,tr,crs

def warp_predictor(src_path, dst_path, resolution, target_crs, reference_bounds):
    dst_path=Path(dst_path); dst_path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(src_path) as s:
        if not s.crs or not s.crs.is_projected:
            raise ValueError(f"{src_path} must have a projected metre-based CRS.")
        # All factors use the same target-inventory extent and grid origin at each resolution.
        left,bottom,right,top=reference_bounds
        w=int(np.ceil((right-left)/resolution)); h=int(np.ceil((top-bottom)/resolution))
        tr=from_origin(left,top,resolution,resolution)
        out=np.full((h,w),np.nan,dtype=np.float32)
        reproject(rasterio.band(s,1),out,src_transform=s.transform,src_crs=s.crs,
                  src_nodata=s.nodata,dst_transform=tr,dst_crs=target_crs,
                  dst_nodata=np.nan,resampling=Resampling.bilinear)
        prof=s.profile.copy()
    prof.update(driver="GTiff",height=h,width=w,transform=tr,crs=target_crs,count=1,
                dtype="float32",nodata=-9999.0,compress="lzw",tiled=True)
    with rasterio.open(dst_path,"w",**prof) as d: d.write(np.where(np.isfinite(out),out,-9999).astype("float32"),1)
    return dst_path

def sample_at_inventory_centres(path, target_valid, target_transform, target_crs):
    r,c=np.where(target_valid); x,y=rasterio.transform.xy(target_transform,r,c,offset="center")
    with rasterio.open(path) as s:
        if s.crs!=target_crs:
            x,y=rasterio.warp.transform(target_crs,s.crs,x,y)
        vals=[]
        for start in range(0,len(x),100000):
            a=np.ma.asarray(list(s.sample(list(zip(x[start:start+100000],y[start:start+100000])),masked=True)))
            v=np.asarray(a[:,0].filled(np.nan) if np.ma.isMaskedArray(a) else a[:,0],dtype=np.float32)
            vals.extend(v.tolist())
    return np.asarray(vals,dtype=np.float32)

def read_stack(paths):
    arr=[]; prof=[]
    for p in paths:
        with rasterio.open(p) as s:
            a=s.read(1).astype(np.float32)
            if s.nodata is not None: a[a==s.nodata]=np.nan
            arr.append(a); prof.append(s.profile.copy())
    if len({a.shape for a in arr})!=1: raise ValueError("Map rasters have different grids.")
    st=np.stack(arr,axis=-1); return st,np.all(np.isfinite(st),axis=-1),prof[0]

def save_map(path, values, valid, profile):
    p=profile.copy(); p.update(driver="GTiff",count=1,dtype="float32",nodata=-9999.0,compress="lzw",tiled=True,BIGTIFF="IF_SAFER")
    a=np.full(valid.shape,-9999,dtype=np.float32); a[valid]=values
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(path,"w",**p) as d: d.write(a,1)

def build_models(seed=42, quick=False):
    n=100 if quick else 400
    return {
      "RF":RandomForestClassifier(n_estimators=100 if quick else 400,class_weight="balanced",min_samples_leaf=2,n_jobs=-1,random_state=seed),
      "SVM":make_pipeline(StandardScaler(),SVC(C=1,kernel="rbf",gamma="scale",probability=True,class_weight="balanced",random_state=seed)),
      "XGB":XGBClassifier(n_estimators=n,max_depth=3,learning_rate=.04,subsample=.8,colsample_bytree=.8,objective="binary:logistic",eval_metric="logloss",n_jobs=-1,random_state=seed),
      "KNN":make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=9 if quick else 25,weights="distance",p=2,n_jobs=-1)),
      "NB":GaussianNB(),
      "AdaBoost":AdaBoostClassifier(n_estimators=n,learning_rate=.05,random_state=seed),
      "LGBM":LGBMClassifier(n_estimators=n,learning_rate=.03,num_leaves=31,class_weight="balanced",verbosity=-1,n_jobs=-1,random_state=seed),
    }

def prob(m,X):
    if hasattr(m,"predict_proba"):
        classes=getattr(m,"classes_",None)
        if classes is None and hasattr(m,"named_steps"): classes=m.named_steps[list(m.named_steps)[-1]].classes_
        return m.predict_proba(X)[:,list(classes).index(1)]
    from scipy.special import expit
    return expit(m.decision_function(X))

class SpatialStack:
    """GroupKFold out-of-fold base predictions train the logistic meta-model."""
    def __init__(self, estimators, n_splits=5, seed=42): self.estimators=estimators; self.n_splits=n_splits; self.seed=seed
    def fit(self,X,y,groups):
        X=np.asarray(X); y=np.asarray(y); groups=np.asarray(groups)
        k=min(self.n_splits,len(np.unique(groups)))
        if k<2: raise ValueError("Need at least two spatial groups for stacking.")
        oof=np.full((len(y),len(self.estimators)),np.nan)
        for ti,vi in GroupKFold(n_splits=k).split(X,y,groups):
            for j,(name,est) in enumerate(self.estimators.items()):
                m=clone(est).fit(X[ti],y[ti]); oof[vi,j]=prob(m,X[vi])
        ok=np.all(np.isfinite(oof),axis=1)
        self.meta_=LogisticRegression(class_weight="balanced",max_iter=2000,random_state=self.seed).fit(oof[ok],y[ok])
        self.models_={n:clone(e).fit(X,y) for n,e in self.estimators.items()}
        self.classes_=self.meta_.classes_
        return self
    def predict_proba(self,X):
        z=np.column_stack([prob(m,X) for m in self.models_.values()])
        return self.meta_.predict_proba(z)

def threshold_train_oof(y,p):
    from sklearn.metrics import roc_curve
    ok=np.isfinite(p)
    if len(np.unique(y[ok]))<2:return .5
    fpr,tpr,ts=roc_curve(y[ok],p[ok]); t=ts[np.argmax(tpr-fpr)]
    return float(t) if np.isfinite(t) else .5

def metrics(y,p,t,name,res):
    from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score,roc_auc_score,average_precision_score,precision_score,recall_score,confusion_matrix
    pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"resolution_m":res,"model":name,"threshold":t,"n_test":len(y),
    "accuracy":accuracy_score(y,pred),"F1":f1_score(y,pred,zero_division=0),
    "kappa":cohen_kappa_score(y,pred),"ROC_AUC":roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
    "PR_AUC":average_precision_score(y,p) if len(np.unique(y))==2 else np.nan,
    "precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),
    "specificity":tn/(tn+fp) if tn+fp else np.nan,"TN":tn,"FP":fp,"FN":fn,"TP":tp}

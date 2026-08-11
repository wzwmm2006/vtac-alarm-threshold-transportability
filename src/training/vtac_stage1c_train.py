"""TRAIN-only frozen VTaC Stage 1C materialization, CV epoch selection, and final models."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import wfdb
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'03_code'/'preprocessing'));import vtac_stage1b_pipeline_v3 as p
SEED=20260811;FINAL=[20260811,20260812,20260813,20260814,20260815]
def sh(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def forbid(*paths):
 if any(any(x in str(q).lower() for x in ('validation','test','external','challenge')) for q in paths):raise RuntimeError('scope_guard_rejected_path')
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
class FCN(nn.Module):
 def __init__(self):
  super().__init__();self.c=nn.Sequential(nn.Conv1d(4,128,51,padding=25),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0),nn.Conv1d(128,256,25,padding=12),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0),nn.Conv1d(256,128,13,padding=6),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0),nn.AdaptiveMaxPool1d(1));self.f=nn.Sequential(nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0));self.o=nn.Sequential(nn.Dropout(0),nn.Linear(64,1))
 def forward(self,x):return self.o(self.f(self.c(x).squeeze(-1))).squeeze(1)
def materialize(a,cache):
 split=pd.read_csv(a/'benchmark_data_split.csv');lab=pd.read_csv(a/'event_labels.csv');z=split[split.split=='train'].merge(lab,on=['record','event'],validate='one_to_one').sort_values(['record','event']).reset_index(drop=True);X=np.empty((len(z),4,2500),np.float32)
 for i,x in enumerate(z.itertuples()):
  b=a/'waveforms'/x.record/x.event;h=wfdb.rdheader(str(b));m=p.mapping(h.sig_name);r=wfdb.rdrecord(str(b),sampfrom=72500,sampto=75000,physical=True);X[i]=p.preprocess(np.asarray(r.p_signal,dtype=float).T,m)[0]
 y=z.decision.astype(bool).astype(np.int64).to_numpy();cache.mkdir(parents=True,exist_ok=True);np.save(cache/'X.npy',X);np.save(cache/'y.npy',y);z[['record','event']].assign(y_true=y).to_csv(cache/'index.csv',index=False);return X,y,z[['record','event']].assign(y_true=y)
def fit(X,y,tr,va,epochs,device,seedval,collect=False):
 seed(seedval);m=FCN().to(device);o=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=.005);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.54,device=device));bs=32;metrics=[]
 for e in range(1,epochs+1):
  m.train();order=np.random.default_rng(seedval+e).permutation(tr)
  losses=[]
  for j in range(0,len(order),bs):
   ix=order[j:j+bs];xb=torch.from_numpy(X[ix]).to(device);yb=torch.from_numpy(y[ix]).float().to(device);o.zero_grad();loss=lossfn(m(xb),yb)
   if not torch.isfinite(loss):raise RuntimeError('nonfinite_training_loss')
   loss.backward();o.step();losses.append(float(loss.detach().cpu()))
  if collect:
   m.eval();pr=[]
   with torch.no_grad():
    for j in range(0,len(va),bs):pr.extend(torch.sigmoid(m(torch.from_numpy(X[va[j:j+bs]]).to(device))).cpu().numpy())
   auc=roc_auc_score(y[va],pr)
   if not np.isfinite(auc):raise RuntimeError('nan_inner_auc')
   metrics.append((e,auc,float(np.mean(losses))))
 return m,metrics,float(np.mean(losses))
def main():
 q=argparse.ArgumentParser();q.add_argument('--archive',type=Path,required=True);q.add_argument('--cache',type=Path,required=True);q.add_argument('--models',type=Path,required=True);q.add_argument('--manifests',type=Path,required=True);q.add_argument('--qc',type=Path,required=True);a=q.parse_args();forbid(a.cache,a.models,a.manifests,a.qc);torch.use_deterministic_algorithms(False);device='cuda' if torch.cuda.is_available() else 'cpu';X,y,index=materialize(a.archive,a.cache)
 if X.shape!=(4060,4,2500) or not np.isfinite(X).all() or y.sum()!=1163 or index.record.nunique()!=1808:raise RuntimeError('train_tensor_integrity_fail')
 sg=StratifiedGroupKFold(3,shuffle=True,random_state=SEED);fold=np.empty(len(y),int)
 for k,(_,v) in enumerate(sg.split(X,y,index.record)):fold[v]=k
 fdf=index.assign(fold=fold);a.manifests.mkdir(parents=True,exist_ok=True);fpath=a.manifests/'train_inner_cv_folds.csv';fdf.to_csv(fpath,index=False)
 if fdf.groupby('record').fold.nunique().max()!=1:raise RuntimeError('record_overlap')
 cfg=ROOT/'00_protocol/frozen/PRIMARY_FCN_CONFIG_LOCK_v3.yaml';am=ROOT/'00_protocol/amendments/STAGE1B_PREPROCESSING_AMENDMENT_002.md';pp=ROOT/'03_code/preprocessing/vtac_stage1b_pipeline_v3.py';pre={'amendment':sh(am),'config':sh(cfg),'preprocess':sh(pp),'fold_hash':sh(fpath),'tensor_hash':sh(a.cache/'X.npy')};(a.manifests/'stage1c_preflight.json').write_text(json.dumps(pre,indent=2))
 allm=[]
 for k in range(3):
  _,met,_=fit(X,y,np.where(fold!=k)[0],np.where(fold==k)[0],500,device,SEED,True)
  allm.extend({'fold':k+1,'epoch':e,'heldout_auc':u,'training_loss':l} for e,u,l in met)
 mdf=pd.DataFrame(allm);a.qc.mkdir(parents=True,exist_ok=True);mp=a.qc/'inner_cv_epoch_metrics.csv';mdf.to_csv(mp,index=False);mean=mdf.groupby('epoch').heldout_auc.mean();E=int(mean[mean==mean.max()].index.min());sel=mdf[mdf.epoch==E];lock={'selected_epoch':E,'selection_metric':'mean held-out TRAIN-fold AUROC','fold_assignment_hash':sh(fpath),'preprocessing_hash':sh(pp),'fcn_config_v3_hash':sh(cfg),'cv_seed':SEED,'fold_auc':sel[['fold','heldout_auc']].to_dict('records'),'mean_auc':float(mean[E]),'timestamp':datetime.now(timezone.utc).isoformat()};lp=ROOT/'00_protocol/frozen/SELECTED_TRAINING_EPOCH_v1.json';lp.write_text(json.dumps(lock,indent=2));
 if E in (1,500) or (sel.heldout_auc<.5).any():raise RuntimeError('pathological_epoch_selection_stop')
 (a.qc/'training_duration_selection.md').write_text(json.dumps(lock,indent=2));a.models.mkdir(parents=True,exist_ok=True);models=[]
 for s in FINAL:
  t=time.time();m,_,loss=fit(X,y,np.arange(len(y)),np.array([],int),E,device,s,False);path=a.models/f'fcn_realtime_seed_{s}.pt';torch.save(m.state_dict(),path);models.append({'seed':s,'path':str(path),'sha256':sh(path),'final_training_loss':loss,'runtime_seconds':time.time()-t})
 ens={'rule':'arithmetic_mean_of_five_prespecified_seed_sigmoid_probabilities','seeds':FINAL};ep=ROOT/'00_protocol/frozen/PRIMARY_MODEL_ENSEMBLE_LOCK_v1.yaml';ep.write_text('rule: arithmetic_mean_of_five_prespecified_seed_sigmoid_probabilities\nseeds: [20260811, 20260812, 20260813, 20260814, 20260815]\n');(ROOT/'00_protocol/frozen/PRIMARY_MODEL_ENSEMBLE_LOCK_v1.md').write_text('# Primary Model Ensemble Lock v1\n\nArithmetic mean of the five frozen seed sigmoid probabilities. No weighting, selection, calibration, or stacking.\n')
 manifest={'models':models,'epoch_lock_hash':sh(lp),'config_hash':sh(cfg),'amendment_hash':sh(am),'preprocessing_hash':sh(pp),'train_tensor_hash':sh(a.cache/'X.npy'),'fold_hash':sh(fpath),'ensemble_hash':sh(ep),'environment':{'python':sys.version,'torch':torch.__version__,'device':device},'timestamp':datetime.now(timezone.utc).isoformat()};(ROOT/'09_reproducibility/manifests/model_manifest_v1.json').write_text(json.dumps(manifest,indent=2));(ROOT/'09_reproducibility/manifests/stage1c_model_freeze.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'epoch':E,'mean_auc':float(mean[E]),'models':models},indent=2))
if __name__=='__main__':main()

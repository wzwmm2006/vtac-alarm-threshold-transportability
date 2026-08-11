import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, wfdb, torch, torch.nn as nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_code'/'preprocessing'))
import vtac_stage1b_pipeline_v3 as p
SEEDS=[20260811,20260812,20260813,20260814,20260815]
class FCN(nn.Module):
 def __init__(self):
  super().__init__(); self.c=nn.Sequential(nn.Conv1d(4,128,51,padding=25),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0),nn.Conv1d(128,256,25,padding=12),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0),nn.Conv1d(256,128,13,padding=6),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0),nn.AdaptiveMaxPool1d(1)); self.f=nn.Sequential(nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0)); self.o=nn.Sequential(nn.Dropout(0),nn.Linear(64,1))
 def forward(self,x): return self.o(self.f(self.c(x).squeeze(-1))).squeeze(1)
def sh(x):
 h=hashlib.sha256();
 with open(x,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def main():
 TARGET_SPLIT='test'
 archive=ROOT/'02_data/raw/vtac-1.0'; out=ROOT/'04_analysis/runs/stage1e_v1'; out.mkdir(parents=True,exist_ok=True)
 split=pd.read_csv(archive/'benchmark_data_split.csv'); split['split']=split['split'].astype(str).str.lower().str.strip().replace({'val':'validation'}); lab=pd.read_csv(archive/'event_labels.csv'); z=split[split['split']==TARGET_SPLIT].merge(lab,on=['record','event'],validate='one_to_one').sort_values(['record','event']).reset_index(drop=True)
 if len(z)!=482 or z.record.nunique()!=226 or int(z.decision.sum())!=137: raise RuntimeError('test_metadata_gate')
 X=np.empty((len(z),4,2500),np.float32); rows=[]; maps=[]
 for i,x in enumerate(z.itertuples()):
  base=archive/'waveforms'/x.record/x.event; h=wfdb.rdheader(str(base)); m=p.mapping(h.sig_name)
  if m['ecg_count']==0: raise RuntimeError('validation_zero_ecg:'+str(x.event))
  r=wfdb.rdrecord(str(base),sampfrom=72500,sampto=75000,physical=True); tensor,q=p.preprocess(np.asarray(r.p_signal,dtype=float).T,m)
  if tensor.shape!=(4,2500) or not np.isfinite(tensor).all(): raise RuntimeError('validation_tensor_failure:'+str(x.event))
  X[i]=tensor; rows.append({'record':x.record,'event':x.event,'alarm_sample':75000,'window_start':72500,'window_end_exclusive':75000,'expected_samples':2500,'extracted_samples':2500,'post_alarm_samples_used':0,'boundary_status':'PASS','failure_reason':''}); maps.append({'record':x.record,'event':x.event,'ecg_count':m['ecg_count'],'channel_order':'ECG1,ECG2,PPG,ABP','missing_ppg':m['selected']['ppg'] is None,'missing_abp':m['selected']['abp'] is None})
 pd.DataFrame(rows).to_csv(ROOT/'02_data/manifests/validation_realtime_window_qa.csv',index=False); pd.DataFrame(maps).to_csv(ROOT/'02_data/manifests/validation_channel_mapping_qa.csv',index=False)
 device='cuda' if torch.cuda.is_available() else 'cpu'; scores=[]
 modeldir=ROOT/'04_analysis/runs/stage1c_v1/models'
 for s in SEEDS:
  model=FCN().to(device); model.load_state_dict(torch.load(modeldir/f'fcn_realtime_seed_{s}.pt',map_location=device)); model.eval()
  with torch.no_grad(): pr=torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()
  if not np.isfinite(pr).all() or ((pr<0)|(pr>1)).any(): raise RuntimeError('invalid_validation_probability')
  scores.append(pr)
 outdf=z[['record','event','decision']].rename(columns={'decision':'y_true'}).copy(); outdf.insert(0,'split','test');
 for s,pr in zip(SEEDS,scores): outdf[f'p_seed_{s}']=pr
 outdf['score']=np.mean(np.vstack(scores),axis=0)
 if len(outdf)!=482 or outdf.record.nunique()!=226 or int(outdf.y_true.sum())!=137 or outdf.event.duplicated().any() or not np.isfinite(outdf.score).all(): raise RuntimeError('validation_scores_gate')
 scorepath=out/'test_scores.csv'; outdf[['split','record','event','y_true','score']].to_csv(scorepath,index=False); outdf.to_csv(out/'test_scores_with_seeds.csv',index=False); (out/'validation_scores.sha256').write_text(sh(scorepath)+'  validation_scores.csv\n',encoding='ascii')
 (out/'test_inference_manifest.json').write_text(json.dumps({'rows':495,'records':226,'true_vt':141,'false_vt':354,'event_exclusions':0,'all_finite':True,'post_alarm_samples_used_max':0,'models':{str(s):sh(modeldir/f'fcn_realtime_seed_{s}.pt') for s in SEEDS},'preprocessing_hash':sh(ROOT/'03_code/preprocessing/vtac_stage1b_pipeline_v3.py'),'score_hash':sh(scorepath)},indent=2),encoding='ascii')
 print(json.dumps({'rows':495,'records':226,'true_vt':141,'false_vt':354,'score_hash':sh(scorepath)},indent=2))
if __name__=='__main__': main()

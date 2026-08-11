"""Frozen release-native, leakage-safe VTaC preprocessing v3."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
FS, WINDOW_SAMPLES, ALARM_INDEX=250,2500,75000
CANONICAL_CHANNELS=('ecg_1','ecg_2','ppg','abp')
ECG={'I','II','III','AVR','AVL','AVF','V','V1','V2','V3','V4','V5','V6','MCL'}
def norm(x): return ''.join(c for c in str(x).upper().strip() if c.isalnum())
def classify(x):
 n=norm(x)
 if n in ECG or n in {'ECG','EKG'} or n.startswith('ECG'): return 'ecg'
 if n=='PLETH': return 'ppg'
 if n=='ABP': return 'abp'
 return None
def mapping(names):
 c={'ecg':[],'ppg':[],'abp':[]}
 for i,x in enumerate(names):
  k=classify(x)
  if k:c[k].append({'source_index':i,'label':str(x)})
 if not c['ecg']: raise ValueError('structural_zero_ecg')
 s={'ecg_1':c['ecg'][0],'ecg_2':c['ecg'][1] if len(c['ecg'])>1 else None,'ppg':c['ppg'][0] if c['ppg'] else None,'abp':c['abp'][0] if c['abp'] else None}
 return {'selected':s,'ecg_count':len(c['ecg']),'source_labels':list(map(str,names)),'unmapped':[str(x) for x in names if classify(x) is None]}
def filts(k):
 notch=iirnotch(60,30,fs=FS)
 if k=='ecg':return [butter(2,1,btype='high',fs=FS),butter(2,30,btype='low',fs=FS),notch]
 if k=='ppg':return [notch,butter(1,[.5,5],btype='band',fs=FS)]
 return [notch,butter(2,16,btype='low',fs=FS)]
def preprocess(w,m):
 w=np.asarray(w,dtype=np.float64)
 if w.shape[1]!=2500:raise ValueError('window_not_2500')
 out=np.zeros((4,2500),np.float32); q=[]
 for i,slot in enumerate(CANONICAL_CHANNELS):
  src=m['selected'][slot]; z={'canonical':slot,'raw_nonfinite_count':0,'filtered_nonfinite_count':0,'pre_zscore_std':None,'zero_variance':False,'final_zero_filled_due_missing':False,'final_zeroed_due_invalidity':False,'final_finite':True}
  if src is None:z['final_zero_filled_due_missing']=True;q.append(z);continue
  x=w[src['source_index']];z['raw_nonfinite_count']=int((~np.isfinite(x)).sum())
  if z['raw_nonfinite_count']:z['final_zeroed_due_invalidity']=True;q.append(z);continue
  y=x
  for b,a in filts('ecg' if slot.startswith('ecg') else slot):y=filtfilt(b,a,y)
  z['filtered_nonfinite_count']=int((~np.isfinite(y)).sum());sd=float(np.std(y));z['pre_zscore_std']=sd;z['zero_variance']=not np.isfinite(sd) or sd==0
  if z['filtered_nonfinite_count'] or z['zero_variance']:z['final_zeroed_due_invalidity']=True;q.append(z);continue
  y=(y-float(np.mean(y)))/sd
  if not np.isfinite(y).all():z['final_zeroed_due_invalidity']=True
  else:out[i]=y.astype(np.float32)
  q.append(z)
 out=np.nan_to_num(out,nan=0.,posinf=0.,neginf=0.)
 if not np.isfinite(out).all():raise ValueError('surviving_inf_nan')
 return out,{'channel_order':list(CANONICAL_CHANNELS),'channel_validity':q}

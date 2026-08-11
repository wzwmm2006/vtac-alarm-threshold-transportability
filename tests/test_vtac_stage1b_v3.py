import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.append(str(ROOT/'03_code'/'preprocessing'));import vtac_stage1b_pipeline_v3 as p
class V3(unittest.TestCase):
 def test_mcl_ecg(self):self.assertEqual(p.classify('MCL'),'ecg')
 def test_one_ecg(self):
  m=p.mapping(['II','PLETH']);t,q=p.preprocess(np.ones((2,2500)),m);self.assertTrue(np.all(t[1]==0));self.assertEqual(m['selected']['ecg_1']['label'],'II')
 def test_first_two(self):self.assertEqual([p.mapping(['V3','II','I'])['selected'][x]['source_index'] for x in ('ecg_1','ecg_2')],[0,1])
 def test_neither_retained(self):
  t,q=p.preprocess(np.ones((2,2500)),p.mapping(['II','V']));self.assertTrue(np.all(t[2:]==0))
 def test_zero_ecg_hard_fail(self):
  with self.assertRaises(ValueError):p.mapping(['PLETH','ABP'])
if __name__=='__main__':unittest.main()

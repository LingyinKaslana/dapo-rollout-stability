"""Run portable synthetic contracts. No model, API, download, or GPU work."""
import json
import os
from pathlib import Path
import platform
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


if __name__=='__main__':
    import numpy
    import torch
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert not torch.cuda.is_initialized()
    print(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,
        'CPU_only':True,'fixture_only':True,'scientific_result':False,
        'model_downloads':False,'API_calls':False,'CUDA_initialized':False,
        'python':platform.python_version(),'numpy':numpy.__version__,'torch':torch.__version__}))
    raise SystemExit(0 if result.wasSuccessful()else 1)

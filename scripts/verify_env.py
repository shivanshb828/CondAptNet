import sys
print(f"Python version: {sys.version}")
print("Step 1 done")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"MPS: {torch.backends.mps.is_available()}")
except Exception as e:
    print(f"PyTorch FAILED: {e}")

try:
    import esm
    print("ESM-2: OK")
except Exception as e:
    print(f"ESM-2 FAILED: {e}")

try:
    import pandas as pd
    print(f"Pandas: {pd.__version__}")
except Exception as e:
    print(f"Pandas FAILED: {e}")

try:
    import RNA
    print("ViennaRNA: OK")
except Exception as e:
    print(f"ViennaRNA: NOT available")

print("Done")
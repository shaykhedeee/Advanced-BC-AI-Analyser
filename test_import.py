import sys
out = open('import_test.txt', 'w')
sys.stdout = out
sys.stderr = out

try:
    import warnings
    warnings.filterwarnings('ignore')
    from dashboard import EdgeTrackerDashboard
    print("Dashboard import OK!")
except Exception as e:
    print(f"Dashboard import FAILED: {e}")
    import traceback
    traceback.print_exc()

out.close()

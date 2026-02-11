import subprocess, sys

cmd = ["python3","/opt/multi-strat-engine/reports/health_check.py"]
try:
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    print(out.strip())
except subprocess.CalledProcessError as e:
    print("Health check failed:\n" + e.output)
    sys.exit(1)

import os
import subprocess
import sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
os.execv(sys.executable, [sys.executable, '-u', 'serve.py'])

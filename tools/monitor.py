import sys
import subprocess
import time
import psutil

def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor.py <command>")
        sys.exit(1)

    cmd = sys.argv[1:]
    start_time = time.time()
    process = subprocess.Popen(cmd)
    
    peak_rss = 0
    try:
        ps_proc = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                # Include children in case the command spawns subprocesses
                current_rss = ps_proc.memory_info().rss
                for child in ps_proc.children(recursive=True):
                    current_rss += child.memory_info().rss
                if current_rss > peak_rss:
                    peak_rss = current_rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.1)
    except Exception as e:
        print(f"Monitor error: {e}")

    end_time = time.time()
    wall_time = end_time - start_time
    
    print("\n" + "="*40)
    print("MONITOR REPORT")
    print("="*40)
    print(f"Command   : {' '.join(cmd)}")
    print(f"Peak RSS  : {peak_rss / (1024**3):.2f} GB ({peak_rss / (1024**2):.2f} MB)")
    print(f"Wall Time : {wall_time:.2f} seconds")
    print("="*40)

if __name__ == "__main__":
    main()

import subprocess

def get_git_commit():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return rev.decode().strip()
    except Exception:
        return None

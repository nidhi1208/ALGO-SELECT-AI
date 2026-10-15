"""
Master Entry Point for AI-Based Algorithm Selection System.
Launches the FastAPI backend and serves the high-performance glassmorphism dashboard.
"""

import os
import sys
import socket
import webbrowser
import uvicorn

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def find_available_port(start_port: int = 8000) -> int:
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

def main():
    # Ensure project root is in sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=" * 65)
    print("  ALGO·SELECT AI — DAA Algorithm Selection System")
    print("  Time Complexity & Machine Learning Engine")
    print("=" * 65)

    port = find_available_port(8000)
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print(f"\n[+] Serving interactive dashboard on: {url}")
    print(f"[+] API docs available on:            {url}/docs")
    print("[+] Press Ctrl+C to terminate the server.\n")

    # Automatically launch user browser
    import threading
    def _open_browser():
        import time
        time.sleep(1.0)
        webbrowser.open(url)
    threading.Thread(target=_open_browser, daemon=True).start()

    # Start FastAPI server via uvicorn
    uvicorn.run("backend.app:app", host=host, port=port, log_level="info", reload=False)

if __name__ == "__main__":
    main()

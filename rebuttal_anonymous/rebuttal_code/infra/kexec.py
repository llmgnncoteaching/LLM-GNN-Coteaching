"""
kexec.py — Execute code on a remote Jupyter kernel via the REST/WebSocket API.

Usage:
    python kexec.py --host HOST:PORT --token TOKEN --kid KERNEL_ID --code 'print("hi")'
    python kexec.py --host HOST:PORT --token TOKEN --kid KERNEL_ID < script.py

Can also be imported:
    from kexec import kexec
    output = kexec(host, token, kid, code)
"""

import argparse
import json
import sys
import time
import uuid

import requests
import websocket


def kexec(host: str, token: str, kid: str, code: str, timeout: int = 120) -> dict:
    """Execute code on a remote Jupyter kernel and return {'stdout': ..., 'stderr': ..., 'error': ...}."""
    base_url = f"http://{host}"
    ws_url = f"ws://{host}"
    headers = {"Authorization": f"token {token}"}

    # Open websocket to kernel
    ws_endpoint = f"{ws_url}/api/kernels/{kid}/channels?token={token}"
    ws = websocket.create_connection(ws_endpoint, timeout=timeout)

    msg_id = str(uuid.uuid4())
    execute_request = {
        "header": {
            "msg_id": msg_id,
            "username": "kexec",
            "session": str(uuid.uuid4()),
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
    }

    ws.send(json.dumps(execute_request))

    stdout_parts = []
    stderr_parts = []
    error_parts = []
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            ws.settimeout(deadline - time.time())
            raw = ws.recv()
        except Exception as e:
            error_parts.append(f"WebSocket recv error: {e}")
            break

        msg = json.loads(raw)
        parent_id = msg.get("parent_header", {}).get("msg_id", "")
        if parent_id != msg_id:
            continue

        msg_type = msg.get("msg_type", "")
        content = msg.get("content", {})

        if msg_type == "stream":
            if content.get("name") == "stdout":
                stdout_parts.append(content.get("text", ""))
            elif content.get("name") == "stderr":
                stderr_parts.append(content.get("text", ""))

        elif msg_type == "execute_result":
            data = content.get("data", {})
            stdout_parts.append(data.get("text/plain", ""))

        elif msg_type == "display_data":
            data = content.get("data", {})
            stdout_parts.append(data.get("text/plain", ""))

        elif msg_type == "error":
            tb = content.get("traceback", [])
            error_parts.append("\n".join(tb))

        elif msg_type == "status":
            if content.get("execution_state") == "idle":
                break  # Done

        elif msg_type in ("execute_reply",):
            pass  # Wait for idle status

    ws.close()

    return {
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
        "error": "".join(error_parts),
    }


def main():
    parser = argparse.ArgumentParser(description="Execute code on a remote Jupyter kernel")
    parser.add_argument("--host", required=True, help="host:port of the Jupyter server")
    parser.add_argument("--token", required=True, help="Jupyter token")
    parser.add_argument("--kid", required=True, help="Kernel ID")
    parser.add_argument("--code", default=None, help="Code to execute (or stdin)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    args = parser.parse_args()

    code = args.code if args.code is not None else sys.stdin.read()

    result = kexec(args.host, args.token, args.kid, code, timeout=args.timeout)

    if result["stdout"]:
        print("STDOUT:", result["stdout"])
    if result["stderr"]:
        print("STDERR:", result["stderr"])
    if result["error"]:
        print("ERROR:", result["error"])
    if not any(result.values()):
        print("(no output)")


if __name__ == "__main__":
    main()

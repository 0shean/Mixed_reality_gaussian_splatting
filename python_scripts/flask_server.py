#!/usr/bin/env python3
"""
Flask server to receive messages from Meta Quest 3 VR application.
Runs on local network and prints received messages to console.
"""

from flask import Flask, request, jsonify
import socket

app = Flask(__name__)

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

@app.route('/message', methods=['POST'])
def receive_message():
    """Endpoint to receive messages from Quest."""
    try:
        # Debug: print raw request data
        print(f"[DEBUG] Content-Type: {request.content_type}")
        print(f"[DEBUG] Raw data: {request.data}")
        print(f"[DEBUG] Headers: {dict(request.headers)}")

        data = request.get_json(force=True)

        if data and 'message' in data:
            message = data['message']
            print(f"[RECEIVED] {message}")
            return jsonify({"status": "success", "received": message}), 200
        else:
            print("[ERROR] No 'message' field in request")
            print(f"[ERROR] Received data: {data}")
            return jsonify({"status": "error", "message": "No 'message' field"}), 400

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        print(f"[ERROR] Request data: {request.data}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    """Simple endpoint to test if server is running."""
    print("[PING] Server pinged")
    return jsonify({"status": "alive"}), 200

if __name__ == '__main__':
    local_ip = get_local_ip()
    port = 5001  # Using 5001 instead of 5000 to avoid conflict with macOS AirPlay Receiver

    print("=" * 60)
    print("FLASK SERVER FOR META QUEST 3")
    print("=" * 60)
    print(f"Server running on: http://{local_ip}:{port}")
    print(f"\nEndpoints:")
    print(f"  POST http://{local_ip}:{port}/message  - Receive messages")
    print(f"  GET  http://{local_ip}:{port}/ping     - Health check")
    print(f"\nUse this IP address in Unity: {local_ip}")
    print("=" * 60)
    print("\nWaiting for messages from Quest...\n")

    # Run server on all network interfaces
    app.run(host='0.0.0.0', port=port, debug=True)

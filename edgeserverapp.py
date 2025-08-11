#!/usr/bin/env python3
# edgeserverapp.py
"""
Edge server for federated learning.

Behavior:
- When first client uploads weights, start an acceptance window (CLIENT_ACCEPT_WINDOW seconds).
- Accept any further uploads while window is open.
- When window expires, automatically forward whatever was collected to central server,
  then clear buffer and reset the window.
- Provides health/status endpoints and a manual aggregation trigger for testing.

Environment variables:
- CENTRAL_SERVER_URL (default: http://localhost:8001)
- SERVER_AES_KEY (32 hex characters recommended; default shown for local testing)
- SERVER_AES_IV  (32 hex characters recommended; default shown for local testing)
- CLIENT_ACCEPT_WINDOW (seconds, default 60)
- PORT (default 8000)
"""

import os
import json
import base64
import logging
import requests
from datetime import datetime, timedelta
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
from typing import Optional

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edge-server")

# --- Flask app ---
app = Flask(__name__)
CORS(app)  # allow all origins; tighten in production if needed

# --- Config from environment ---
CENTRAL_SERVER_URL = os.getenv("CENTRAL_SERVER_URL", "http://localhost:8001")
SERVER_AES_KEY = bytes.fromhex(os.getenv("SERVER_AES_KEY", "000102030405060708090a0b0c0d0e0f"))
SERVER_AES_IV = bytes.fromhex(os.getenv("SERVER_AES_IV", "101112131415161718191a1b1c1d1e1f"))
CLIENT_ACCEPT_WINDOW = int(os.getenv("CLIENT_ACCEPT_WINDOW", "60"))  # seconds

# --- Global state ---
client_weights_buffer = []   # list of dicts: {client_id, weights, timestamp}
first_client_time: Optional[datetime] = None
_accept_lock = threading.Lock()
_window_timer: Optional[threading.Timer] = None

# Helper: reset state (used after forwarding)
def _reset_window_state():
    global client_weights_buffer, first_client_time, _window_timer
    with _accept_lock:
        client_weights_buffer = []
        first_client_time = None
        if _window_timer:
            try:
                _window_timer.cancel()
            except Exception:
                pass
            _window_timer = None
        logger.info("Acceptance window state reset.")


# --- AES helper functions ---
def encrypt_aes(plaintext: str) -> str:
    try:
        cipher = AES.new(SERVER_AES_KEY, AES.MODE_CBC, SERVER_AES_IV)
        padded = pad(plaintext.encode("utf-8"), AES.block_size)
        ct = cipher.encrypt(padded)
        return base64.b64encode(ct).decode("utf-8")
    except Exception as e:
        logger.error(f"AES encryption failed: {e}")
        raise

def decrypt_aes(encrypted_text: str) -> str:
    try:
        ct = base64.b64decode(encrypted_text)
        cipher = AES.new(SERVER_AES_KEY, AES.MODE_CBC, SERVER_AES_IV)
        pt = unpad(cipher.decrypt(ct), AES.block_size).decode("utf-8")
        return pt
    except Exception as e:
        logger.error(f"AES decryption failed: {e}")
        raise

# --- Timer callback: called when acceptance window expires ---
def _on_window_expire():
    """
    Called by a background Timer when the acceptance window ends.
    It will attempt to forward collected weights to central server.
    """
    global _window_timer
    logger.info("Acceptance window expired. Triggering aggregation/forward.")
    try:
        # Acquire lock and snapshot buffer
        with _accept_lock:
            if not client_weights_buffer:
                logger.info("No client weights to forward at window expiry.")
                _reset_window_state()
                return

            # snapshot to local var
            snapshot = list(client_weights_buffer)
            # clear state now to allow new window after we begin forwarding
            client_weights_buffer = []
            first_client_time_local = first_client_time  # for logging
            _window_timer = None

        # Build payload and forward outside lock
        payload = {
            "clients": snapshot,
            "total_clients": len(snapshot)
        }
        aes_encrypted = encrypt_aes(json.dumps(payload))
        response = requests.post(
            f"{CENTRAL_SERVER_URL}/uploadWeights",
            json={"encrypted_data": aes_encrypted},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        if response.status_code == 200:
            logger.info(f"Successfully forwarded {len(snapshot)} client updates to central server.")
        else:
            logger.error(f"Central server returned {response.status_code}: {response.text}")
            # Option: you could retry or store to disk. For now just log.

    except Exception as e:
        logger.exception(f"Error during forward on window expiry: {e}")

    finally:
        # Ensure global state is reset and ready for next window
        with _accept_lock:
            # already cleared buffer above; ensure first_client_time reset
            _reset_window_state()


# Helper to (re)start the window timer; expected to be called while holding lock or safely
def _start_window_timer():
    global _window_timer
    if _window_timer:
        try:
            _window_timer.cancel()
        except Exception:
            pass
    _window_timer = threading.Timer(CLIENT_ACCEPT_WINDOW, _on_window_expire)
    _window_timer.daemon = True
    _window_timer.start()
    logger.info(f"Started acceptance window timer for {CLIENT_ACCEPT_WINDOW}s.")


# --- Flask endpoints ---
@app.errorhandler(Exception)
def handle_error(error):
    logger.exception(f"Unhandled error: {error}")
    return jsonify({"error": "Internal server error"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "edge-server"}), 200


@app.route("/status", methods=["GET"])
def get_status():
    with _accept_lock:
        now = datetime.now()
        time_elapsed = None
        time_left = None
        if first_client_time:
            time_elapsed = (now - first_client_time).total_seconds()
            time_left = max(0, CLIENT_ACCEPT_WINDOW - time_elapsed)
        return jsonify({
            "edge_server": "running",
            "central_url": CENTRAL_SERVER_URL,
            "clients_buffered": len(client_weights_buffer),
            "first_client_time": first_client_time.isoformat() if first_client_time else None,
            "accept_window_seconds": CLIENT_ACCEPT_WINDOW,
            "time_left_seconds": int(time_left) if time_left is not None else None
        }), 200


@app.route("/uploadWeights", methods=["POST"])
def upload_weights():
    """
    Primary endpoint used by clients to upload weights.
    Behavior:
     - If no first client yet: set first_client_time, start timer, accept.
     - If within window: accept.
     - If window expired: reject (client should start a new round later).
    """
    global client_weights_buffer, first_client_time

    try:
        data = request.get_json()
        if not data or "weights" not in data:
            return jsonify({"error": "Missing 'weights' field"}), 400

        now = datetime.now()

        with _accept_lock:
            # If first client, initialize window
            if first_client_time is None:
                first_client_time = now
                logger.info(f"First client upload at {first_client_time.isoformat()}; opening {CLIENT_ACCEPT_WINDOW}s window.")
                # start timer outside lock to avoid holding lock during timer creation? it's okay here
                _start_window_timer()

            # Check if still within acceptance window
            elapsed = (now - first_client_time).total_seconds()
            if elapsed > CLIENT_ACCEPT_WINDOW:
                logger.info("Upload rejected: acceptance window expired.")
                return jsonify({"status": "rejected", "message": "Client registration window has closed"}), 403

            # Accept the weights
            client_id = len(client_weights_buffer) + 1
            client_entry = {
                "client_id": client_id,
                "weights": data["weights"],
                "timestamp": now.isoformat()
            }
            client_weights_buffer.append(client_entry)
            logger.info(f"Accepted weights from client {client_id}. Buffer size: {len(client_weights_buffer)}")

            # For convenience, return how many seconds left in window
            time_left = max(0, CLIENT_ACCEPT_WINDOW - int(elapsed))

            # If you want: you could immediately aggregate if a min number reached. Not done here.
            return jsonify({
                "status": "success",
                "message": f"Weights received. Window open for {time_left}s more.",
                "client_id": client_id,
                "buffered_clients": len(client_weights_buffer),
                "aggregation_triggered": False
            }), 200

    except Exception as e:
        logger.exception(f"Error in upload_weights: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/forceAggregate", methods=["POST"])
def force_aggregate():
    """
    Manual aggregation trigger (useful for testing).
    This will forward whatever is buffered right now to the central server.
    """
    global client_weights_buffer

    with _accept_lock:
        if not client_weights_buffer:
            return jsonify({"status": "no-op", "message": "No buffered client updates to aggregate"}), 200

        snapshot = list(client_weights_buffer)
        client_weights_buffer = []

        # Cancel timer and reset first_client_time
        global first_client_time, _window_timer
        if _window_timer:
            try:
                _window_timer.cancel()
            except Exception:
                pass
            _window_timer = None
        first_client_time = None

    try:
        payload = {"clients": snapshot, "total_clients": len(snapshot)}
        aes_encrypted = encrypt_aes(json.dumps(payload))
        response = requests.post(
            f"{CENTRAL_SERVER_URL}/uploadWeights",
            json={"encrypted_data": aes_encrypted},
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        if response.status_code == 200:
            logger.info("Force-aggregated and forwarded to central server successfully.")
            return jsonify({"status": "success", "message": "Aggregated and forwarded"}), 200
        else:
            logger.error(f"Central server error during forceAggregate: {response.status_code} - {response.text}")
            return jsonify({"status": "error", "message": "Failed to forward to central server", "code": response.status_code}), 500

    except Exception as e:
        logger.exception(f"Error forwarding on forceAggregate: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/receiveGlobalModel", methods=["POST"])
def receive_global_model():
    """
    Called by central server to deliver the aggregated/global model.
    It expects {"encrypted_data": "<base64-ciphertext>"} where decryption yields JSON.
    """
    try:
        data = request.get_json()
        if not data or "encrypted_data" not in data:
            return jsonify({"error": "Missing 'encrypted_data' field"}), 400

        global_model_json = decrypt_aes(data["encrypted_data"])
        global_model_data = json.loads(global_model_json)

        # Store model in memory for clients to download (simple approach)
        # In production you might store to S3 or DB and serve via URL
        global global_model_storage
        global_model_storage = global_model_data

        logger.info(f"Received global model from central server. Weights count: {len(global_model_data.get('weights', []))}")
        return jsonify({"status": "success", "message": "Global model received and stored"}), 200

    except Exception as e:
        logger.exception(f"Error in receive_global_model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/getGlobalModel", methods=["GET"])
def get_global_model():
    """
    Clients poll this endpoint to retrieve the latest global model.
    """
    try:
        global global_model_storage
        if 'global_model_storage' not in globals() or global_model_storage is None:
            return jsonify({"error": "Global model not available"}), 404

        return jsonify({"status": "success", "model": global_model_storage}), 200

    except Exception as e:
        logger.exception(f"Error in get_global_model: {e}")
        return jsonify({"error": str(e)}), 500


# --- Run server ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Edge Server on port {port}")
    logger.info(f"Central Server URL: {CENTRAL_SERVER_URL}")
    app.run(host="0.0.0.0", port=port, debug=False)

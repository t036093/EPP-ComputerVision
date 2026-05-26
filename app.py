# ==============================
# IMPORTACIONES
# ==============================
# FastAPI: framework para crear el servidor web
# WebSocket: permite comunicación en tiempo real con el frontend
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# Para servir archivos (HTML, JS, etc.)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
# ONNX Runtime: ejecuta el modelo de IA
import onnxruntime as ort
# NumPy: manejo de arreglos (imágenes, tensores)
import numpy as np
# OpenCV: procesamiento de imágenes
import cv2
# Base64: decodificar imágenes enviadas desde el navegador
import base64
# JSON: comunicación entre frontend y backend
import json
# Serial: comunicación con Arduino (control de acceso físico)
import serial
import time
from typing import List
import asyncio
import sys
import os
import urllib.request
import urllib.parse

# Variable global para persistir estadísticas durante la sesión
stats_global = {
    "PASS": 0,
    "DENIED": 0,
    "TOTAL": 0
}

# Variables de Estado Global (Sincronización Endpoint - WebSocket)
evaluation_active = False
current_attempt = 0
pass_votes = 0
denied_votes = 0
target_user_id = None
target_full_name = None
evaluation_result = None
evaluation_event = asyncio.Event()

#==============================
# CONFIGURACIÓN DE ARDUINO
#==============================
# try:
#     arduino = serial.Serial("COM5", 9600, timeout=1)
#     time.sleep(2)
#     print("Arduino conectado")
# except Exception as e:
#     arduino = None
#     print("Arduino no conectado:", e)

# def mandar_acceso():
#     if arduino:
#         arduino.write(b"A")

# def mandar_denegado():
#     if arduino:
#         arduino.write(b"D")

# ==============================
# CARGAR VARIABLES DEL .ENV (Sin dependencias externas)
# ==============================
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"\'')
    print("✅ Archivo .env cargado exitosamente.")

# ==============================
# CONFIGURACIÓN DE SUPABASE (REST API DIRECTA)
# ==============================
# Usamos HTTP directo (urllib) para evitar dependencias C++ como pyiceberg
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()

if SUPABASE_URL and SUPABASE_KEY:
    print("✅ Supabase REST API URL detectada.")
    
    # Diagnóstico Rápido: Las llaves anon/service_role son JWT y deben empezar con 'ey'
    if not SUPABASE_KEY.startswith("ey"):
        print("❌ ERROR DE DIAGNÓSTICO: Tu SUPABASE_KEY no parece ser una llave válida.")
        print("   -> Asegúrate de estar usando la 'anon public key' o la 'service_role key' (Ambas empiezan con 'ey...').")
    else:
        print(f"✅ Supabase Key válida (Longitud: {len(SUPABASE_KEY)} caracteres).")
else:
    print("⚠️ Supabase NO inicializado. Configura SUPABASE_URL y SUPABASE_KEY en tus variables de entorno o en tu archivo .env.")

# ==============================
# INICIALIZACIÓN DEL SERVIDOR
# ==============================
app = FastAPI()

# Ruta del modelo ONNX entrenado (YOLO en tu caso)
MODEL_PATH = "bestest.onnx"

# Se carga el modelo en memoria (solo una vez)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Nombre de la entrada del modelo (input tensor)
input_name = session.get_inputs()[0].name

# Nombres de las salidas del modelo
output_names = [o.name for o in session.get_outputs()]

# Debug: imprime info del modelo
print("INPUT:", session.get_inputs()[0].shape, session.get_inputs()[0].type)

for out in session.get_outputs():
    print("OUTPUT:", out.name, out.shape, out.type)

# Monta carpeta "static" para servir archivos (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==============================
# RUTA PRINCIPAL (SERVIR HTML)
# ==============================
@app.get("/")
def root():
    # Devuelve el archivo index.html al entrar a la página
    return FileResponse("templates/index.html")

@app.get("/admin")
def get_admin():
    """Vista completa: Estadísticas y controles de flujo"""
    return FileResponse("templates/admin.html")

# ==============================
# API DEL DASHBOARD (DATOS EN VIVO)
# ==============================
@app.get("/api/dashboard")
def get_dashboard_data():
    """Obtiene métricas globales y últimos logs desde Supabase"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"error": "Supabase no configurado", "metrics": None, "logs": []}
    
    try:
        # 1. Obtener métricas de la vista
        endpoint_metrics = f"{SUPABASE_URL}/rest/v1/view_global_metrics?select=*"
        req_m = urllib.request.Request(endpoint_metrics, method="GET")
        req_m.add_header("apikey", SUPABASE_KEY)
        req_m.add_header("Authorization", f"Bearer {SUPABASE_KEY}")
        
        metrics = {"total_attempts": 0, "total_passed": 0, "total_denied": 0, "compliance_percentage": 0}
        with urllib.request.urlopen(req_m) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data and len(data) > 0:
                metrics = data[0]
                
        # 2. Obtener los últimos 10 logs de acceso (Haciendo JOIN con la tabla users)
        endpoint_logs = f"{SUPABASE_URL}/rest/v1/access_logs?select=id,timestamp,status,missing_hardhat,missing_vest,users(full_name)&order=timestamp.desc&limit=10"
        req_l = urllib.request.Request(endpoint_logs, method="GET")
        req_l.add_header("apikey", SUPABASE_KEY)
        req_l.add_header("Authorization", f"Bearer {SUPABASE_KEY}")
        
        logs = []
        with urllib.request.urlopen(req_l) as response:
            logs = json.loads(response.read().decode("utf-8"))
            
        # 3. Obtener tendencia semanal (Últimos 7 días)
        endpoint_trend = f"{SUPABASE_URL}/rest/v1/access_logs?select=timestamp,status&order=timestamp.desc&limit=1000"
        req_t = urllib.request.Request(endpoint_trend, method="GET")
        req_t.add_header("apikey", SUPABASE_KEY)
        req_t.add_header("Authorization", f"Bearer {SUPABASE_KEY}")
        
        trend_data = []
        with urllib.request.urlopen(req_t) as response:
            trend_data = json.loads(response.read().decode("utf-8"))
            
        import datetime
        # Supabase guarda en UTC, así que calculamos "hoy" en UTC para que coincida con los strings
        today = datetime.datetime.now(datetime.timezone.utc).date()
        last_7_days = [(today - datetime.timedelta(days=i)) for i in range(6, -1, -1)]
        labels = [d.strftime("%b %d") for d in last_7_days] # Ej. May 26
        
        pass_counts = {d: 0 for d in last_7_days}
        denied_counts = {d: 0 for d in last_7_days}
        
        for row in trend_data:
            try:
                date_str = row["timestamp"][:10]
                row_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                if row_date in pass_counts:
                    if row["status"] == "PASS":
                        pass_counts[row_date] += 1
                    elif row["status"] == "DENIED":
                        denied_counts[row_date] += 1
            except Exception:
                pass
                
        trend = {
            "labels": labels,
            "pass": [pass_counts[d] for d in last_7_days],
            "denied": [denied_counts[d] for d in last_7_days]
        }
            
        return {"metrics": metrics, "logs": logs, "trend": trend}
        
    except Exception as e:
        print(f"Error fetching dashboard data: {e}")
        return {"error": str(e), "metrics": None, "logs": [], "trend": None}

# ==============================
# ANALYTICS ENDPOINTS
# ==============================
@app.get("/analytics", response_class=HTMLResponse)
def analytics_view():
    return FileResponse("templates/analytics.html")

@app.get("/api/analytics/users")
def get_analytics_users():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    try:
        endpoint = f"{SUPABASE_URL}/rest/v1/users?select=id,full_name,risk_classification&order=full_name.asc"
        req = urllib.request.Request(endpoint, method="GET")
        req.add_header("apikey", SUPABASE_KEY)
        req.add_header("Authorization", f"Bearer {SUPABASE_KEY}")
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print("Error fetching users:", e)
        return []

@app.get("/api/analytics/data")
def get_analytics_data(start_date: str = None, end_date: str = None, user_id: str = None, status: str = None, risk: str = None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"error": "Supabase no configurado"}
        
    try:
        # Check if we need to filter by risk (requires inner join to discard non-matching users)
        if risk and risk != "ALL":
            query = f"select=id,timestamp,status,missing_hardhat,missing_vest,users!inner(id,full_name,risk_classification)&users.risk_classification=eq.{risk}&order=timestamp.desc&limit=1000"
        else:
            # Default left join to include logs with missing or anonymous users
            query = "select=id,timestamp,status,missing_hardhat,missing_vest,users(id,full_name,risk_classification)&order=timestamp.desc&limit=1000"
        
        if start_date:
            query += f"&timestamp=gte.{start_date}T00:00:00"
        if end_date:
            query += f"&timestamp=lte.{end_date}T23:59:59"
        if user_id and user_id != "ALL":
            query += f"&user_id=eq.{user_id}"
        if status and status != "ALL":
            query += f"&status=eq.{status}"
            
        endpoint = f"{SUPABASE_URL}/rest/v1/access_logs?{query}"
        req = urllib.request.Request(endpoint, method="GET")
        req.add_header("apikey", SUPABASE_KEY)
        req.add_header("Authorization", f"Bearer {SUPABASE_KEY}")
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data
            
    except Exception as e:
        print("Error fetching analytics data:", e)
        return {"error": str(e)}

# ==============================
# BASE DE DATOS HELPER
# ==============================
def insert_access_log(user_id, status, missing_hardhat, missing_vest):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        endpoint = f"{SUPABASE_URL}/rest/v1/access_logs"
        log_data = {
            "user_id": user_id,
            "status": status,
            "missing_hardhat": missing_hardhat,
            "missing_vest": missing_vest
        }
        req = urllib.request.Request(endpoint, data=json.dumps(log_data).encode("utf-8"), method="POST")
        req.add_header("apikey", SUPABASE_KEY)
        req.add_header("Authorization", f"Bearer {SUPABASE_KEY}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Prefer", "return=minimal")
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print("Error insertando en BD:", e)

# ==============================
# ENDPOINT DE ESCANEO NFC (TRIGGER)
# ==============================
@app.get("/api/v1/scan")
async def scan_nfc(credential_id: str):
    """Filter 1: Escaneo NFC y Activación de la Ráfaga"""
    global evaluation_active, current_attempt, pass_votes, denied_votes, target_user_id, target_full_name, evaluation_result, evaluation_event
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"status": "error", "reason": "Supabase no configurado"}
        
    if evaluation_active:
        return {"status": "error", "reason": "Sistema ocupado evaluando"}

    # -------------------------
    # FILTER 1: VALIDACIÓN DB
    # -------------------------
    def check_user():
        endpoint = f"{SUPABASE_URL}/rest/v1/users?credential_id=eq.{urllib.parse.quote(credential_id)}&select=id,full_name"
        req = urllib.request.Request(endpoint, method="GET")
        req.add_header("apikey", SUPABASE_KEY)
        req.add_header("Authorization", f"Bearer {SUPABASE_KEY}")
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        user_data = await asyncio.to_thread(check_user)
    except Exception as e:
        print("Error en Filter 1:", e)
        return {"status": "error", "reason": "Error de conexión a BD"}

    if not user_data or len(user_data) == 0:
        return {"status": "denied", "reason": "Invalid Credential"}

    # -------------------------
    # PREPARAR FILTER 2
    # -------------------------
    target_user_id = user_data[0]["id"]
    target_full_name = user_data[0]["full_name"]
    current_attempt = 0
    pass_votes = 0
    denied_votes = 0
    evaluation_result = None
    evaluation_event.clear()
    
    # Activar la evaluación en el WebSocket
    evaluation_active = True
    
    # Esperar el resultado pasivamente con Timeout de 10 seg
    try:
        await asyncio.wait_for(evaluation_event.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        evaluation_active = False
        return {"status": "error", "reason": "Timeout (¿Cámara desconectada?)"}
        
    return evaluation_result

# ==============================
# DECODIFICAR IMAGEN BASE64
# ==============================
def decode_base64_image(data_url: str):
    """
    Convierte una imagen enviada desde el navegador (base64)
    a formato OpenCV (BGR).
    """

    # Separa encabezado "data:image/jpeg;base64," del contenido real
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    # Decodifica base64 → bytes
    image_bytes = base64.b64decode(encoded)

    # Convierte bytes a arreglo NumPy
    np_arr = np.frombuffer(image_bytes, np.uint8)

    # Decodifica a imagen OpenCV
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return image

# ==============================
# PREPROCESAMIENTO + INFERENCIA
# ==============================
def run_model(image_bgr):
    """
    Prepara la imagen y ejecuta el modelo ONNX
    """

    # Convierte a float32 (requerido por el modelo)
    img = image_bgr.astype(np.float32)

    # Normaliza valores [0,255] → [0,1]
    img = img / 255.0

    # Redimensiona a 640x640 (input esperado por el nuevo modelo)
    img = cv2.resize(img, (640, 640))

    # Cambia formato HWC → CHW (canales primero)
    img = np.transpose(img, (2, 0, 1))

    # Agrega dimensión de batch → (1, C, H, W)
    img = np.expand_dims(img, axis=0)

    # Ejecuta el modelo
    outputs = session.run(output_names, {input_name: img})

    # Debug: imprime outputs
    print("=== OUTPUTS DEL MODELO ===")
    for i, out in enumerate(outputs):
        if isinstance(out, np.ndarray):
            print(f"Output {i}: shape={out.shape}, dtype={out.dtype}")
            print(out[:3] if out.ndim > 0 else out)
        else:
            print(f"Output {i}: tipo {type(out)}")

    print("OUTPUT SHAPE REAL:", outputs[0].shape)

    return outputs


# ==============================
# POSTPROCESAMIENTO (DETECCIONES)
# ==============================
def parse_model_output(outputs, conf_threshold=0.75):
    """
    Convierte la salida cruda del modelo en detecciones legibles:
    bounding boxes + clase + score
    """

    raw = outputs[0]  # Ejemplo: (1, 29, 13125)
    detections = []

    # Validación de tipo
    if not isinstance(raw, np.ndarray):
        return detections

    # Quita batch → (29, 13125)
    raw = np.squeeze(raw, axis=0)

    # Transpone → (13125, 29)
    raw = raw.T

    # Mapeo de clases del modelo nuevo (4 clases)
    class_map = {
        0: "HARDHAT",
        1: "NO_HARDHAT",
        2: "NO_VEST",
        3: "VEST"
    }

    # Guarda solo la mejor detección por etiqueta
    best_by_label = {}

    for row in raw:
        # Coordenadas del bounding box (centro + tamaño)
        cx, cy, w, h = row[:4]

        # Probabilidades por clase
        class_scores = row[4:]

        # Clase con mayor probabilidad
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])

        # Filtrado por confianza
        if score < conf_threshold:
            continue

        # Solo clases relevantes
        if class_id not in class_map:
            continue

        # Convertir a formato esquina (x1,y1,x2,y2)
        x1 = int(cx - (w / 2))
        y1 = int(cy - (h / 2))
        x2 = int(cx + (w / 2))
        y2 = int(cy + (h / 2))

        label = class_map[class_id]

        det = {
            "label": label,
            "score": score,
            "box": [x1, y1, x2, y2]
        }

        # Mantener solo la mejor detección por clase
        if label not in best_by_label or score > best_by_label[label]["score"]:
            best_by_label[label] = det

    detections = list(best_by_label.values())

    print("=== DETECTIONS PARSEADAS ===")
    print(detections)

    return detections


# ==============================
# LÓGICA DE DECISIÓN (EPP)
# ==============================
def evaluate_status(detections):
    global stats_global
    
    # Si no hay absolutamente nada en pantalla (nadie frente a la cámara), no evaluamos.
    if not detections:
        return "WAITING"
        
    labels = [d["label"] for d in detections]
    
    # Lógica de decisión
    if "HARDHAT" in labels and "VEST" in labels:
        current_status = "PASS"
    else:
        current_status = "DENIED"
    
    # Actualizamos el contador global
    stats_global["TOTAL"] += 1
    stats_global[current_status] += 1
    
    return current_status


# ==============================
# WEBSOCKET (TIEMPO REAL)
# ==============================
active_connections: List[dict] = []

@app.on_event("shutdown")
async def shutdown_event():
    """Cierra todas las conexiones WebSocket para que Uvicorn se detenga inmediatamente"""
    print("🛑 Cerrando conexiones WebSocket de forma segura...")
    for client in list(active_connections):
        try:
            await client["ws"].close()
        except Exception:
            pass
    active_connections.clear()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_type: str = "kiosk"):
    global evaluation_active, current_attempt, pass_votes, denied_votes, target_user_id, target_full_name, evaluation_result, evaluation_event
    
    # Acepta conexión del cliente
    await websocket.accept()
    
    client_info = {
        "ws": websocket,
        "client_type": client_type,
        "frame_counter": 0
    }
    active_connections.append(client_info)

    try:
        while True:
            # Recibe mensaje del frontend
            message = await websocket.receive_text()
            data = json.loads(message)

            credential_id = data.get("credential_id") # Capturar la credencial (NFC/Barcode) si existe

            # Verifica si es un comando (ej. start_kiosk)
            action = data.get("action")
            if action:
                # Retransmitir a todos los clientes asincrónicamente
                payload = json.dumps({"action": action})
                tasks = [client["ws"].send_text(payload) for client in active_connections]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                continue

            # Extrae imagen enviada
            image_data = data.get("image")

            # Validación
            if not image_data:
                await websocket.send_text(json.dumps({
                    "error": "No se recibió imagen"
                }))
                continue

            # Decodifica imagen
            image_bgr = decode_base64_image(image_data)

            if image_bgr is None:
                await websocket.send_text(json.dumps({
                    "error": "No se pudo decodificar la imagen"
                }))
                continue

            # Inferencia
            try:
                outputs = run_model(image_bgr)
                detections = parse_model_output(outputs, conf_threshold=0.20)
                status = evaluate_status(detections)

                # Si hay una evaluación activa solicitada por el Endpoint NFC
                if evaluation_active:
                    current_attempt += 1
                    
                    labels = [d["label"] for d in detections]
                    missing_hardhat = "HARDHAT" not in labels
                    missing_vest = "VEST" not in labels
                    
                    if status == "PASS":
                        pass_votes += 1
                    else:
                        denied_votes += 1
                    
                    # Condición de éxito: Evaluamos al final de la ráfaga (10 frames)
                    if current_attempt >= 10:
                        # IMPORTANTE: Desactivar inmediatamente para evitar duplicados por concurrencia
                        evaluation_active = False
                        
                        # Decisión final: Threshold de 3 pases positivos en la ráfaga
                        final_status = "PASS" if pass_votes >= 3 else "DENIED"
                        
                        # 1. Insertar BD (Una sola vez)
                        asyncio.create_task(asyncio.to_thread(insert_access_log, target_user_id, final_status, missing_hardhat, missing_vest))
                        
                        # 2. Señal Arduino
                        if final_status == "PASS" and arduino and arduino.is_open:
                            arduino.write(b"A")
                            
                        # 3. Preparar resultado y avisar al endpoint HTTP
                        if final_status == "PASS":
                            evaluation_result = {"status": "allowed", "user": target_full_name}
                        else:
                            evaluation_result = {"status": "denied", "reason": "Missing PPE"}
                            
                        evaluation_event.set()

            except Exception as e:
                print("ERROR EN INFERENCIA:", e)
                detections = []
                status = "ERROR"

            # 1. Payload base para todos
            payload_data = {
                "detections": detections,
                "status": status,
                "stats": stats_global,
                "evaluation_active": evaluation_active,
                "target_user": target_full_name if evaluation_active else None
            }
            
            # 2. Generar imagen optimizada para Admin solo si hay admins conectados
            admin_b64 = None
            has_admin = any(c["client_type"] == "admin" for c in active_connections)
            if has_admin:
                # Reducir a 400x400
                admin_img = cv2.resize(image_bgr, (400, 400))
                # Comprimir a 40% JPEG
                _, admin_buffer = cv2.imencode('.jpg', admin_img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                admin_b64 = base64.b64encode(admin_buffer).decode('utf-8')

            # 3. Distribuir a los clientes
            tasks = []
            for client in active_connections:
                ws = client["ws"]
                c_type = client["client_type"]
                
                if c_type == "admin":
                    admin_payload = dict(payload_data)
                    if admin_b64:
                        admin_payload["image"] = admin_b64
                    tasks.append(ws.send_text(json.dumps(admin_payload)))
                else:
                    # Kiosk no recibe imagen, solo data
                    tasks.append(ws.send_text(json.dumps(payload_data)))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    except (WebSocketDisconnect, RuntimeError, Exception) as e:
        # Cliente se desconectó o la conexión fue cerrada por el shutdown
        if client_info in active_connections:
            active_connections.remove(client_info)
        print(f"Cliente {client_type} desconectado. Motivo: {type(e).__name__}")

# ==============================
# EJECUTOR PRINCIPAL
# ==============================
if __name__ == "__main__":
    import uvicorn
    
    # Soporte para HTTPS (Requerido para activar la cámara en otros dispositivos)
    cert_path = "192.168.1.74+2.pem"
    key_path = "192.168.1.74+2-key.pem"
    ssl_kwargs = {}
    
    if os.path.exists(cert_path) and os.path.exists(key_path):
        ssl_kwargs["ssl_certfile"] = cert_path
        ssl_kwargs["ssl_keyfile"] = key_path
        print(f"🔒 Iniciando servidor con soporte HTTPS en el puerto 8000...")
    else:
        print("⚠️ Iniciando servidor HTTP sin seguridad (solo funcionará la cámara en localhost).")

    # En Windows, forzar a uvicorn a usar asyncio (selector) en vez de proactor desde la raíz
    uvicorn.run("app:app", host="0.0.0.0", port=8000, loop="asyncio", reload=False, **ssl_kwargs)
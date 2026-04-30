# ==============================
# IMPORTACIONES
# ==============================
# FastAPI: framework para crear el servidor web
# WebSocket: permite comunicación en tiempo real con el frontend
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Para servir archivos (HTML, JS, etc.)
from fastapi.responses import FileResponse
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

    # Redimensiona a 800x800 (input esperado por el modelo)
    img = cv2.resize(img, (800, 800))

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

    # Mapeo de clases del modelo
    class_map = {
        2: "HARDHAT",
        5: "NO_HARDHAT",
        7: "NO_VEST",
        11: "VEST"
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
    """
    Determina si el usuario cumple con el EPP:
    - PASS: tiene casco y chaleco
    - DENIED: falta alguno o hay violación
    """

    labels = [d["label"] for d in detections]

    # Si detecta incumplimiento → DENIED
    if "NO_HARDHAT" in labels or "NO_VEST" in labels:
        # mandar_denegado()
        return "DENIED"

    # Si tiene ambos → PASS
    if "HARDHAT" in labels and "VEST" in labels:
        # mandar_acceso()
        return "PASS"

    # Caso intermedio → DENIED
    # mandar_denegado()
    return "DENIED"

# ==============================
# WEBSOCKET (TIEMPO REAL)
# ==============================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Acepta conexión del cliente
    await websocket.accept()

    try:
        while True:
            # Recibe mensaje del frontend
            message = await websocket.receive_text()
            data = json.loads(message)

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

            except Exception as e:
                print("ERROR EN INFERENCIA:", e)
                detections = []
                status = "ERROR"

            # Debug
            print("=== MENSAJE ENVIADO AL FRONTEND ===")
            print(detections)

            # Envía resultados al frontend
            await websocket.send_text(json.dumps({
                "detections": detections,
                "status": status
            }))

    except WebSocketDisconnect:
        # Cliente se desconectó
        print("Cliente desconectado")
# EPP Computer Vision & Access Control 👷‍♂️🛡️

Sistema integral de visión por computadora para el control de acceso en áreas industriales. El sistema detecta en tiempo real si el trabajador porta su Equipo de Protección Personal (EPP: Casco y Chaleco) mediante un modelo YOLOv8 exportado a ONNX. 

El flujo de acceso se activa mediante un **Trigger NFC / RFID** (desde un dispositivo móvil o gafete), evaluando ráfagas de video en tiempo real e interactuando con un microcontrolador (Arduino) para la apertura de torniquetes.

## 🚀 Requisitos Previos

- Python 3.10 o superior.
- Una cuenta en [Supabase](https://supabase.com/) con el esquema configurado.
- Un dispositivo de lectura RFID/NFC que pueda disparar una petición HTTP GET.
- (Opcional) Arduino conectado vía USB/Serial.

## ⚙️ Instalación (Paso a Paso)

### 1. Clonar el repositorio
```bash
git clone https://github.com/t036093/EPP-ComputerVision.git
cd EPP-ComputerVision
```

### 2. Crear y activar Entorno Virtual
Es muy recomendable instalar las dependencias aisladas del sistema:

**En Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**En Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar Dependencias
Instala todas las librerías necesarias con un solo comando:
```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno
Crea un archivo llamado `.env` en la raíz del proyecto. Este archivo **jamás** se subirá a GitHub por seguridad. Pega lo siguiente y reemplaza con tus propios datos:

```env
SUPABASE_URL="https://tu-proyecto.supabase.co"
# OJO: Usa la Service Role Key si necesitas hacer bypass al RLS de base de datos
SUPABASE_KEY="tu-super-secreta-service-role-key"
```

### 5. Certificados SSL (HTTPS Local)
Para que el Kiosko pueda acceder a la cámara del navegador desde otros dispositivos en tu red local (como un iPad), es necesario usar HTTPS.

Asegúrate de generar certificados usando `mkcert` para la IP de tu servidor (ej. `192.168.1.74`) y coloca los archivos `.pem` en la raíz del proyecto.

---

## 🏃‍♂️ Cómo arrancar el Servidor

Para iniciar el servidor FastAPI usando Uvicorn con los certificados SSL, ejecuta:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --ssl-certfile=tu_certificado.pem --ssl-keyfile=tu_llave-key.pem
```
*(Cambia el nombre de los archivos `.pem` a los que hayas generado en tu máquina).*

### 🌐 Rutas Principales

- **Kiosko (Cámara y Espejo):** `https://<TU_IP>:8000/`
- **Dashboard de Administrador:** `https://<TU_IP>:8000/admin`
- **Trigger NFC (Para celulares):** `https://<TU_IP>:8000/api/v1/scan?credential_id=RFID-001`

---

## 🏗 Arquitectura del Sistema

1. **Estado de Standby:** El Kiosko transmite video continuamente sin procesar nada en la base de datos (ahorro de recursos).
2. **Activación:** El trabajador escanea su tarjeta NFC, lo que dispara una petición a `/api/v1/scan`.
3. **Filtro 1 (Supabase):** Verifica instantáneamente si la tarjeta pertenece a un empleado activo.
4. **Filtro 2 (Visión Artificial):** Activa una "Ráfaga" de 10 frames para evaluar si el empleado tiene puesto el casco y el chaleco.
5. **Decisión y Registro:** Responde en milisegundos, inyecta el log a Supabase y (si es aprobado) envía una señal `b"A"` por Serial al Arduino para abrir la puerta.

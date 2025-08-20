# transcriptor_openai_whisper_full.py
import os
import shutil
import subprocess
import tempfile
from datetime import datetime

from openai import OpenAI
import openai as openai_lib  # para capturar excepciones específicas

# ── FFmpeg check (para conversiones/segmentado) ───────────────────────────────
try:
    subprocess.run(["ffmpeg", "-version"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ FFmpeg se ejecuta correctamente")
except Exception:
    print("⚠️ No se pudo verificar FFmpeg. Si falla la conversión/segmentado, revisá el PATH.")

# ── Config OpenAI: usa variable de entorno OPENAI_API_KEY ────────────────────
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("⚠️ Falta OPENAI_API_KEY en el entorno. Configúrala antes de continuar.")
client = OpenAI(api_key=API_KEY, timeout=120, max_retries=3)

# ── Input de usuario (igual que tu script) ───────────────────────────────────
def obtener_datos_usuario():
    print("\n=== Configuración Inicial ===")
    while True:
        carpeta = input("👉 Ruta completa de la carpeta con archivos: ").strip()
        if os.path.isdir(carpeta):
            break
        print("❌ ¡La carpeta no existe! Intenta nuevamente")
    while True:
        idioma = input("👉 Idioma para transcripción (ej: es, en, fr): ").strip().lower()
        if len(idioma) == 2 and idioma.isalpha():
            break
        print("❌ ¡Formato de idioma incorrecto! Usa código de 2 letras")
    return carpeta, idioma

def inicializar_txt(carpeta):
    txt_path = os.path.join(carpeta, f"resumenes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== Resúmenes Analizados ===\n\n")
    return txt_path

# ── Utilidades de tamaño/formatos ────────────────────────────────────────────
def es_archivo_valido(archivo):
    extensiones = ('.mp3', '.wav', '.mp4', '.avi', '.mov', '.m4a', '.flac', '.mkv', '.aac')
    return archivo.lower().endswith(extensiones)

def mb(path):  # tamaño en MB
    return os.path.getsize(path) / (1024 * 1024)

def convertir_a_m4a_mono(src_path: str, bitrate="96k") -> str:
    """Convierte a M4A (AAC) mono con bitrate dado. Devuelve ruta temporal."""
    tmp = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
    tmp.close()
    out = tmp.name
    cmd = [
        "ffmpeg", "-y", "-i", src_path, "-vn",
        "-c:a", "aac", "-b:a", bitrate, "-ac", "1",
        out, "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)
    return out

def segmentar_a_m4a_chunks(src_path: str, segment_sec: int = 600, bitrate="96k"):
    """
    Parte el audio en tramos M4A mono de `segment_sec` segundos.
    Devuelve (carpeta_temporal, [rutas_de_partes]).
    """
    tmpdir = tempfile.mkdtemp(prefix="chunks_")
    pattern = os.path.join(tmpdir, "part_%03d.m4a")
    cmd = [
        "ffmpeg", "-y", "-i", src_path, "-vn",
        "-c:a", "aac", "-b:a", bitrate, "-ac", "1",
        "-f", "segment", "-segment_time", str(segment_sec),
        "-reset_timestamps", "1",
        pattern, "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)
    parts = sorted(os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".m4a"))
    return tmpdir, parts

# ── Transcripción robusta con límites (>25 MB) ───────────────────────────────
MAX_MB = 25
SAFE_MB = 24   # umbral antes del límite

def _transcribir_path(path, idioma):
    with open(path, "rb") as f:
        txt = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",   # SOLO texto plano
            language=idioma,
            temperature=0
        )
    return (txt or "").strip()

def transcribir_con_fallback(src_path: str, idioma: str, target_bitrate="96k", segment_sec=600) -> str:
    """
    1) Si <=24 MB: intenta directo.
    2) Si es grande o falla: convertir a M4A mono (96 kbps) y reintentar.
    3) Si aún supera 25 MB o vuelve a fallar: segmentar y concatenar.
    """
    # 1) Intento directo si no es demasiado grande
    if mb(src_path) <= SAFE_MB:
        try:
            return _transcribir_path(src_path, idioma)
        except (openai_lib.BadRequestError, openai_lib.APIError):
            pass  # seguimos al fallback

    # 2) Convertir a M4A mono
    m4a = None
    try:
        m4a = convertir_a_m4a_mono(src_path, bitrate=target_bitrate)
        if mb(m4a) <= MAX_MB:
            try:
                return _transcribir_path(m4a, idioma)
            except (openai_lib.BadRequestError, openai_lib.APIError):
                # pasamos a segmentado
                pass
        # 3) Segmentar (desde el archivo original para mejorar cortes) y concatenar
        tmpdir, parts = segmentar_a_m4a_chunks(src_path, segment_sec=segment_sec, bitrate=target_bitrate)
        try:
            textos = []
            for i, p in enumerate(parts, 1):
                print(f"   ⤷ Parte {i}/{len(parts)} ({mb(p):.1f} MB)")
                textos.append(_transcribir_path(p, idioma))
            return " ".join(textos).strip()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        if m4a and os.path.exists(m4a):
            try:
                os.remove(m4a)
            except Exception:
                pass

# ── Proceso principal (transcribe + analiza, y guarda en TXT combinado) ──────
def procesar_archivos(carpeta, idioma, txt_path):
    total = sum(1 for f in os.listdir(carpeta) if es_archivo_valido(f))
    procesados = 0
    errores = 0

    print("\n=== Proceso de Transcripción y Análisis (OpenAI whisper-1) ===")
    print(f"📂 Carpeta seleccionada: {carpeta}")
    print(f"🌐 Idioma seleccionado:  {idioma}")
    print(f"💾 Archivo TXT:         {txt_path}\n")

    for archivo in os.listdir(carpeta):
        if not es_archivo_valido(archivo):
            continue

        ruta_completa = os.path.join(carpeta, archivo)
        try:
            print(f"🔍 Procesando {procesados+1}/{total}: {archivo}")

            # 1) Transcribir (con manejo de >25 MB)
            texto = transcribir_con_fallback(ruta_completa, idioma, target_bitrate="96k", segment_sec=600)

            # 2) Analizar con GPT (mismo formato que tu código original)
            prompt = f"""
Analiza al 100% la transcripción del video titulado "{archivo}" de Kallaway.
Dame un análisis detallado y super completo del video basado en la siguiente Transcripción Completa:
{texto}
""".strip()

            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": """Actúa como un experto en generacion de contenido viral. Tu tarea es analizar profundamente el contenido, identificar los métodos que enseña, las herramientas que recomienda, los tonos de comunicación que sugiere usar, y cualquier otro patrón o estrategia clave que mencione para lograr viralidad. Organiza la información con claridad, separando los siguientes puntos:
1) Resumen general de la clase.
2) Métodos y estrategias específicas para lograr viralidad (con explicación y ejemplo si corresponde).
3) Tonos de comunicación recomendados (cuándo y por qué).
4) Herramientas mencionadas o utilizadas (software, apps, técnicas de edición, etc.).
5) Consejos prácticos y frases clave (literal si aportan valor directo).
6) Errores comunes que recomienda evitar (si se mencionan).
7) Cualquier otra recomendación relevante o insight inesperado.
Sé exhaustivo y fiel al texto."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.7,
                frequency_penalty=0.4,
                presence_penalty=0.25
            )

            resumen = response.choices[0].message.content.strip()

            # 3) Escribir análisis en el TXT combinado (igual que antes)
            with open(txt_path, 'a', encoding='utf-8') as f:
                f.write(f"📁 Archivo: {archivo}\n")
                f.write(f"📝 Resumen: {resumen}\n")
                f.write("-"*50 + "\n\n")

            print(f"✅ Análisis exitoso: {archivo}")
            procesados += 1

        except subprocess.CalledProcessError:
            errores += 1
            print(f"❌ Error de conversión/segmentado previo en {archivo}")
        except Exception as e:
            errores += 1
            print(f"❌ Error procesando {archivo}: {e}")

    return procesados, errores

def mostrar_resumen(procesados, errores, txt_path):
    print("\n=== Resumen Final ===")
    print(f"✅ Archivos procesados correctamente: {procesados}")
    print(f"❌ Archivos con errores: {errores}")
    print(f"💾 Archivo TXT generado: {txt_path}")
    print("\n⚠️ Revisa los resultados en el archivo TXT generado")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        carpeta, idioma = obtener_datos_usuario()
        txt_path = inicializar_txt(carpeta)
        procesados, errores = procesar_archivos(carpeta, idioma, txt_path)
        mostrar_resumen(procesados, errores, txt_path)
    except Exception as e:
        print(f"❌¡Error crítico! {e}")

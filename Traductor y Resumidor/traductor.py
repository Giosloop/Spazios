# transcribe.py
import os, sys, argparse,subprocess, getpass
from datetime import datetime
import openai
from openai import OpenAI


def _ffmpeg_to_m4a(src_path: str, bitrate="96k") -> str:
    base, _ = os.path.splitext(src_path)
    out = base + ".transcode.m4a"
    cmd = [
        "ffmpeg", "-y", "-i", src_path, "-vn",
        "-c:a", "aac", "-b:a", bitrate, "-ac", "1", out,
        "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)
    return out

def transcribir_con_retry(client: OpenAI, ruta: str, idioma: str | None) -> str:
    """
    Intenta directo. Si el servidor responde 400 (lectura fallida),
    convierte a .m4a (AAC 96k mono) y reintenta.
    """
    def _call(path):
        with open(path, "rb") as f:
            return client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language=idioma or None,
                temperature=0
            )

    try:
        return (_call(ruta) or "").strip()
    except openai.BadRequestError as e:
        # sólo retry si es realmente un 400 de parsing
        if getattr(e, "status_code", None) == 400 or "reading your request" in str(e).lower():
            print("⚠️ 400 al leer el archivo. Convirtiendo a M4A y reintentando...")
            try:
                m4a = _ffmpeg_to_m4a(ruta)
                return (_call(m4a) or "").strip()
            except Exception as e2:
                raise RuntimeError(f"Falló el retry con M4A: {e2}") from e
        raise
    

# --- Opcional: usar llavero del sistema para la API key ---
try:
    import keyring
except ImportError:
    keyring = None

SERVICE_NAME = "transcriber_openai"

def get_api_key():
    k = os.getenv("OPENAI_API_KEY")
    if k: return k
    if keyring:
        k = keyring.get_password(SERVICE_NAME, getpass.getuser())
        if k: return k
    # Fallback: pedir por consola (sin eco)
    return input("Pegá tu OPENAI_API_KEY (visible): ").strip()
def es_archivo_valido(nombre):
    return nombre.lower().endswith((
        ".mp3",".wav",".mp4",".avi",".mov",".m4a",".flac",".mkv",".aac"
    ))

def transcribir_archivo(client: OpenAI, ruta: str, idioma: str|None) -> str:
    with open(ruta, "rb") as f:
        txt = transcribir_con_retry(client, ruta, idioma)
    return (txt or "").strip()

def transcribir_carpeta(client: OpenAI, carpeta: str, idioma: str|None,
                        mode: str, outfile: str|None, headers: bool):
    archivos = [f for f in os.listdir(carpeta) if es_archivo_valido(f)]
    archivos.sort()
    total = len(archivos)
    if total == 0:
        print("⚠️ No se encontraron archivos de audio/video en la carpeta.")
        return 0, 0

    ok = err = 0
    combined_chunks = []

    # Si es single y no pasaron outfile, generamos uno timestamped
    if mode == "single":
        if not outfile:
            outfile = os.path.join(
                carpeta,
                f"transcripciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
        # Crear/truncar archivo desde el inicio
        with open(outfile, "w", encoding="utf-8") as _:
            pass

    print(f"\n=== Transcripción ({mode}) ===")
    print(f"📂 Carpeta: {carpeta}")
    print(f"🌐 Idioma: {idioma or 'autodetección'}")
    if mode == "single":
        print(f"📝 Archivo combinado: {outfile}")
    print("")

    for i, nombre in enumerate(archivos, 1):
        ruta = os.path.join(carpeta, nombre)
        print(f"🔍 {i}/{total} {nombre}")
        try:
            texto = transcribir_archivo(client, ruta, idioma)

            if mode == "split":
                base, _ = os.path.splitext(nombre)
                out_path = os.path.join(carpeta, f"{base}.txt")
                with open(out_path, "w", encoding="utf-8") as out:
                    out.write(texto + ("\n" if texto and not texto.endswith("\n") else ""))
            else:  # single
                if headers:
                    combined_chunks.append(f"# {nombre}\n{texto}\n")
                else:
                    # Sin headers: solo concatenar con una línea en blanco entre archivos
                    combined_chunks.append(texto + "\n")

            ok += 1
        except Exception as e:
            err += 1
            print(f"❌ Error en {nombre}: {e}")

    if mode == "single":
        with open(outfile, "a", encoding="utf-8") as out:
            # Dos saltos finales por prolijidad
            out.write("\n".join(combined_chunks).rstrip() + "\n")

    print(f"\n✅ OK: {ok}   ❌ Errores: {err}")
    if mode == "split":
        print("📄 Se creó un .txt por cada archivo de audio/video.")
    else:
        print(f"📄 Todo quedó en: {outfile}")
    return ok, err

def main():
    ap = argparse.ArgumentParser(
        description="Transcribe audio/video con whisper-1 (solo texto)."
    )
    ap.add_argument("--dir", help="Ruta a la carpeta con archivos")
    ap.add_argument("--lang", default="", help="Idioma (es, en, fr). Vacío = autodetección")
    ap.add_argument("--mode", choices=["split", "single"], help="Guardar por archivo (split) o juntas (single)")
    ap.add_argument("--outfile", help="Ruta del archivo combinado (solo si --mode single)")
    ap.add_argument("--no-headers", action="store_true",
                    help="En modo single, no agregar nombres de archivo como separadores")
    args = ap.parse_args()

    # Fallback interactivo si faltan args
    carpeta = args.dir or input("👉 Ruta completa de la carpeta: ").strip()
    if not os.path.isdir(carpeta):
        raise SystemExit("❌ La carpeta no existe.")
    idioma = (args.lang or input("👉 Idioma (ej: es, en) o ENTER: ").strip().lower()) or None
    mode = args.mode or (input("👉 Guardado [split=separado / single=juntas]: ").strip().lower() or "split")
    if mode not in ("split", "single"):
        raise SystemExit("❌ Valor de --mode inválido. Usa 'split' o 'single'.")
    headers = not args.no_headers
    outfile = args.outfile

    client = OpenAI(api_key=get_api_key())
    transcribir_carpeta(client, carpeta, idioma, mode, outfile, headers)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🟡 Cancelado por el usuario.")
    except Exception as e:
        print(f"❌ ¡Error crítico! {e}", file=sys.stderr)

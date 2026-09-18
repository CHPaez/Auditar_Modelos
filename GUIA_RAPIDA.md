# Guía rápida: cómo usar este kit

Este documento explica, paso a paso y sin tecnicismos, cómo poner en marcha el kit de auditoría en tu computadora. Es lo mínimo necesario para verlo funcionando.

## ¿Qué hace esto?

Le da un modelo que reconoce imágenes y le hace una serie de pruebas para ver qué tan bien funciona: si acierta, si aguanta fotos de mala calidad, si su código está bien hecho, y si empieza a fallar con el tiempo. Al final, muestra los resultados en una página que se abre en el navegador, fácil de leer.

## Paso 1 — Instalar (una sola vez)

Abre una terminal (PowerShell) dentro de la carpeta del proyecto y copia estos comandos uno por uno:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Esto puede tardar unos minutos la primera vez.

## Paso 2 — Correr la auditoría básica

```powershell
python scripts\run_audit.py
python scripts\robustness_test.py
```

La primera vez, esto descarga automáticamente un modelo de ejemplo (unos 350 MB) y las fotos de prueba — no hay que bajar nada a mano, solo esperar.

## Paso 3 — Ver el resultado, de forma visual

```powershell
python scripts\generate_report.py
start report.html
```

Se te abre una página en el navegador con los resultados: qué tan bien acertó el modelo (con números grandes y una tabla de errores) y qué tanto empeora con fotos de mala calidad (borrosas, giradas, con mal brillo).

**Con estos 3 pasos ya tenés lo esencial funcionando.** Lo que sigue es opcional, para profundizar.

---

## Opcional — Ver las imágenes una por una

Si querés navegar foto por foto cuáles acertó y cuáles no (no solo la tabla de resultados):

```powershell
pip install -r requirements-visual.txt
python scripts\browse_results.py
```

Ojo: esto instala un programa más pesado y te abre su propia ventana en el navegador (distinta a `report.html`).

## Opcional — Ver si el modelo "se desgasta" con el tiempo

Compara cómo responde el modelo a fotos normales contra fotos con ruido, para simular qué pasaría si las fotos reales empiezan a verse distinto con el tiempo:

```powershell
pip install -r requirements-drift.txt
python scripts\drift_check.py
start drift_report.html
```

## Opcional — Revisar si el código de un modelo propio está bien hecho

Esto es distinto a todo lo anterior: en vez de mirar cómo responde el modelo, mira si el código que lo entrena o lo sirve está bien escrito y sin dependencias con fallas de seguridad conocidas. Se usa apuntando a la carpeta de código de tu propio proyecto (no aplica a un modelo público descargado):

```powershell
pip install -r requirements-dev.txt
python scripts\code_quality_check.py --path "C:\ruta\a\tu\proyecto"
```

---

## ¿Y para auditar un modelo propio de la empresa, en vez del ejemplo?

Los mismos comandos del Paso 2 y 3, agregando `--model` y `--dataset` con los datos de tu modelo real:

```powershell
python scripts\run_audit.py --model tu-modelo --dataset tu-dataset
python scripts\robustness_test.py --model tu-modelo --dataset tu-dataset
python scripts\generate_report.py
start report.html
```

Si algo no funciona o un paso da error, copiá el mensaje completo — con eso se puede diagnosticar rápido qué pasó.

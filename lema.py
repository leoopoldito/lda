import spacy
import glob  # Para encontrar archivos que coincidan con un patrón, buscar los txt de un directorio
import os  # Para construir rutas de archivos
import pyximport
import numpy
numpy.random.seed(42)
pyximport.install(setup_args={"include_dirs": numpy.get_include()})

import matrices
# ruta al corpus, ruta que contiene los archivos .txt
CORPUS_PATH = "./corpus"

# Cargar el modelo de spaCy
print("Cargando modelo de spaCy 'es_core_news_lg'...")
# bloque try-except por si el usuario no ha descargado el modelo
try:
    # Deshabilitar "parser" y "ner" (Análisis y Reconocimiento de Entidades)
    # hace que el proceso sea MUCHO más rápido, ya que solo necesitamos
    # el tokenizador y el lematizador.
    nlp = spacy.load("es_core_news_lg", disable=["parser", "ner"])
    nlp.max_length = 2500000
except IOError:
    print("\n--- ERROR ---")
    print("Modelo 'es_core_news_lg' no encontrado.")
    print("Por favor, ejecútalo en tu terminal:")
    print("python -m spacy download es_core_news_lg")
    exit()  # Salir del script si no se encuentra el modelo

print("¡Lematizador cargado exitosamente!")
#lista donde se almacenaran los subvocabularios, lista de palabras de cada documento del corpus
subvocabularios = {}
# Inicializar el vocabulario
# Usamos un 'set', esto garantiza que no se repitan palabras en el vocabulario.
vocabulario = set()

# Encontrar todos los archivos .txt en la ruta del corpus
# glob.glob() crea una lista de todos los archivos que terminan en ".txt" dentro de la carpeta CORPUS_PATH.
# os.path.join() construye la ruta de forma segura
archivos_txt = glob.glob(os.path.join(CORPUS_PATH, "*.txt"))

if not archivos_txt:# si no hay archivos .txt
    print(f"\n--- ADVERTENCIA ---")
    print(f"No se encontraron archivos .txt en la carpeta: {CORPUS_PATH}")
    print("Asegúrate de que la variable CORPUS_PATH sea correcta.")
    exit()

print(f"\nSe encontraron {len(archivos_txt)} archivos .txt. Procesando...")

# Procesar cada archivo
for filepath in archivos_txt:
    print(f"  Procesando: {filepath}")

    #crear diccionario del documento n
    lemas_de_este_archivo = []

    # Abrimos el archivo en modo lectura ('r') con codificación 'utf-8' (muy importante para texto en español con acentos y 'ñ')
    with open(filepath, 'r', encoding='utf-8') as f:
        # Leemos todo el contenido del archivo en una sola variable
        texto_completo = f.read()

        # Procesar el texto con spaCy
        # Pasamos el texto completo al objeto 'nlp'
        # spaCy se encarga de tokenizar (dividir en palabras)
        # y aplicar los componentes del pipeline (en este caso, lematizar)
        doc = nlp(texto_completo)

        # Lematizar y filtrar tokens, iteramos sobre cada 'token' (palabra) que spaCy encontró
        for token in doc:
            # Filtros de limpieza
            # Eliminamos:
            # token.is_stop: "Stop words" (palabras comunes como "el", "de", "y")
            # token.is_punct: Signos de puntuación (",", ".", ";")
            # token.is_space: Espacios en blanco o saltos de línea
            # not token.is_alpha: Tokens que no son alfabéticos (ignora números)

            if (not token.is_stop and not token.is_punct and not token.is_space ):
                # Lematización, obtenemos el lema del token y lo convertimos a minuscualas para estandarizar.
                lema = token.lemma_.lower()
                # Agregamos el lema al set, vocabulario general
                vocabulario.add(lema)
                # Agregamos el lema al subvocabulario del archivo actual, en este caso no se filtrar repeticiones
                lemas_de_este_archivo.append(lema)
        #agregamos el subvocabulario a la lista de subvocabularios
        subvocabularios[filepath] = lemas_de_este_archivo
#asignar un id a cada palabra del vocabulario
vocabulario_a_id = {palabra: id for id, palabra in enumerate(vocabulario)}
# el mapa inverso (ID -> Palabra)
id_a_vocabulario = {id: palabra for palabra, id in vocabulario_a_id.items()}

listas_de_ids = []
for archivo, lemas in subvocabularios.items():
    id_de_la_palabras = [vocabulario_a_id[lema] for lema in lemas]
    listas_de_ids.append(id_de_la_palabras)

# Cython necesita una lista de arrays de NumPy,
# Usamos dtype=numpy.int32 porque tu Cython espera 'int'.
corpus_para_cython = [
    numpy.array(doc, dtype=numpy.int32) for doc in listas_de_ids
]
# Cuantas palabras hay en el vocabulario
print("\n--- Proceso Terminado ---")
print(f"El vocabulario final tiene {len(vocabulario)} palabras únicas.")


K = 50

#llamada a la función para obtener las matrices
phig, thetag, ids_20 = matrices.llenarMatrices(corpus_para_cython, len(corpus_para_cython), len(vocabulario), K, 1500)
#pasar los ids a palabras
lista_topicos_palabras = []
for i in range(K):
    lista_ids = ids_20[i]
    lista_palabras = [id_a_vocabulario[id] for id in lista_ids]
    lista_topicos_palabras.append(lista_palabras)
    #imprimir las 20 palabras del topico
    print(f"Top 20 palabras del topico {i}: {lista_palabras}")


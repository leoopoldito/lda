import numpy as np
cimport numpy as np

# Importamos funciones de C para generar números aleatorios
from libc.stdlib cimport rand, RAND_MAX

#from Lematizador.LDA import n_k_t

#list corpus es la bolsa de palabras, de cada documento del corpus
# K es el numero de topicos en el documento, M el nuemro total de documentos, V el tamaño del vocabulario, It es el numero de iteraciones
cpdef llenarMatrices(list corpus, int M, int V, int K, int it):
    cdef int i, m, n, k, w, topico
    cdef double p_total, r_num, proMay #probabilidad mayor
    # Declaramos los arrays de probabilidad (para el muestreo)
    cdef double[:] prob_topico = np.empty(K, dtype=np.float64)

    # Inicializar Matrices (usando NumPy)
    # Estas son las matrices que querías llenar
    # N_mk: Matriz Documento-Tópico (M x K), núumero de palabras del documento m en el topico k
    cdef int[:, :] N_mk_c = np.zeros((M, K), dtype=np.int32)
    # N_kv: Matriz Tópico-Palabra (K x V), número de veces que la palabra v esta en el topico k
    cdef int[:, :] N_kv_c = np.zeros((K, V), dtype=np.int32)
    # N_k: Vector de conteo total por Tópico (K)
    cdef int[:] N_k_c = np.zeros(K, dtype=np.int32)

    #guardar la asignacion del topico de cada palabra del documento m
    cdef list Z = []

    #primer llenado de matrices, con la que la probabilidad de K para cualquier topico es igual
    for m in range(M):
        # Obtenemos el array de numpy del documento 'm'
        doc = corpus[m]
        cdef int[::1] doc_c = doc  # Creamos un "memoryview" rápido
        cdef int N_d = doc.shape[0]  # Palabras en este doc
        cdef int[:] z_m = np.empty(N_d, dtype=np.int32)

        for n in range(N_d):
            w = doc_c[n]  # El ID de la palabra
            # Asignar un tópico aleatorio (0 a K-1)
            topico = rand() % K
            N_mk_c[m][topico] += 1
            N_kv_c[topico][w] += 1
            N_k_c[topico] += 1
            z_m[n] = topico

        Z.append(z_m)#se añade la asignacion de cada una de las palabras del documento m a una lista con

    #posteriores llenados de matrices

    for i in range(it):
        #calcular la probabilidad de cada una de las palabras
        for m in range(M):
            # Obtenemos el array de numpy del documento 'm'
            doc = corpus[m]
            cdef int[::1] doc_c = doc  # Creamos un "memoryview" rápido
            cdef int N_d = doc.shape[0]  # Palabras en este doc
            cdef int[:] z_m = Z[m]
            cdef float beta = 0.01
            cdef float alpha = 50/K

            for n in range(N_d):
                w = doc_c[n]  # El ID de la palabra
                topico = z_m[n]
                N_mk_c[m][topico] -= 1
                N_kv_c[topico][w] -= 1
                N_k_c[topico] -= 1

                for k in range(K):
                    prob= ((N_kv_c[k][topico]+beta-1)/(N_k_c+beta-1))*((N_mk_c[m][topico]+alpha-1)/(N_d+alpha-1))
                    prob_topico[k] = prob

                proMay = 0.0
                for k in range(K):
                    if prob_topico[k]>proMay:
                        proMay = prob_topico[k]
                        topico = k

                N_mk_c[m][topico] += 1
                N_kv_c[topico][w] += 1
                N_k_c[topico] += 1
                z_m[n] = topico

    #calculo de los estimadores de bayes para asignar las 20 palabras a dicho documento
    #for k in range(K):
    #   phi = 1


import math
import numpy as np
cimport numpy as np

# Importamos funciones de C para generar números aleatorios
from libc.stdlib cimport rand, RAND_MAX, srand
# --- NUEVO: Importamos la librería Estándar de C para I/O (Archivos) ---
cimport libc.stdio as stdio

#list corpus es la bolsa de palabras, de cada documento del corpus, M es el numero de documentos, V el tamaño del vocabulario
# K es el número de topicos en el documento, It es el número de iteraciones

cpdef llenarMatrices(list corpus, int M, int V, int K, int it):
    srand(42)
#variables que nos permitiran iterar en los ciclos for, funcionan para su letra mayuscula, ejemplo m para interar en los docuemntos M
    # w es para obtener el id de la palabra, topico para un topico en especifico, t para asignar id de una palabra
    #n_d_t para el numero de veces que aparece la palabra t en el documento m
    cdef int i, m, n, k, w, topico, t, n_d_t
    cdef double p_total, r_num#p_total para calcular la probabilidad de la palabra w para cada topico
    #array de probabilidad del topico para los posteriores llenados de matrices
    cdef double[:] prob_topico = np.empty(K, dtype=np.float64)

    # Inicializar Matrices (usando NumPy)
    # N_mk: Matriz Documento-Tópico (M x K), núumero de palabras del documento m en el topico k
    cdef int[:, :] N_mk_c = np.zeros((M, K), dtype=np.int32)
    # N_kv: Matriz Tópico-Palabra (K x V), número de veces que la palabra v esta en el topico k
    cdef int[:, :] N_kv_c = np.zeros((K, V), dtype=np.int32)
    # N_k: Vector de conteo total por Tópico (K)
    cdef int[:] N_k_c = np.zeros(K, dtype=np.int32)
    #matriz phi
    cdef double[:,:] phi = np.zeros((K, V), dtype=np.float64)
    #matriz theta
    cdef double[:,:] theta = np.zeros((M, K), dtype=np.float64)


    #guardar la asignacion del topico de cada palabra del documento m
    cdef list Z = []

    cdef int[::1] doc_c  # Creamos un "memoryview", es como pasarle la dirección de memoria en la que se encuentra
    cdef int N_d  # Contar cuantas palabras tiene un doc
    cdef int[:] z_m #array que almacena el topico asignado a una palabra del documento m
    #definición de beta y alpha
    cdef float beta = 0.01
    cdef float alpha = 50 / K

    #primer llenado de matrices, con la que la probabilidad de K para cualquier topico es igual
    for m in range(M):
        # Obtenemos el array de numpy del documento 'm'
        doc = corpus[m] # tomar el documento m
        doc_c = doc  # Creamos un "memoryview", es como pasarle la dirección de memoria en la que se encuentra
        N_d = doc.shape[0]  # Palabras en este doc
        z_m = np.empty(N_d, dtype=np.int32) #array de numpy con un tamaño igual a N_d
        for n in range(N_d):
            w = doc_c[n]  # El ID de la palabra
            # Asignar un tópico aleatorio (0 a K-1)
            topico = rand() % K
            #llenar las matrices
            N_mk_c[m][topico] += 1
            N_kv_c[topico][w] += 1
            N_k_c[topico] += 1
            z_m[n] = topico

        Z.append(z_m)#se añade la asignacion del topico de cada una de las palabras del documento m a una lista

    #reasignacion de topicos, "llenado" de matrices 2 hasta it
    for i in range(it):
        # pasar por cada uno de los documentos
        for m in range(M):
            # Obtenemos el array de numpy del documento 'm'
            doc = corpus[m] # tomar el documento m
            doc_c = doc  # Creamos un "memoryview", es como pasarle la dirección de memoria en la que se encuentra
            N_d = doc.shape[0]  # Palabras en este doc
            z_m = Z[m] # se le asigna el numero de topico de cada palabra del docummento m en el array

            # pasar por cada una de la palabra del documento m
            for n in range(N_d):
                w = doc_c[n]  # El ID de la palabra
                topico = z_m[n]# recibimos el topico que tiene asignada la palabra n del documento m
                #decrementamos las matrices
                N_mk_c[m][topico] -= 1
                N_kv_c[topico][w] -= 1
                N_k_c[topico] -= 1

                #hacer el array de probabilidad de la palabra w para cada topico, empezara en 0 y se le sumaran las probabilidades
                prob_total=0
                for k in range(K):
                    #formula para la probabilidad
                    prob= ((N_kv_c[k][w]+beta)/(N_k_c[k]+beta*V))*((N_mk_c[m][k]+alpha)/(N_d-1+K*alpha))
                    prob_total += prob
                    #asignar la probabilidad de la palabra w en el topico k en el array
                    prob_topico[k] = prob_total

                #probabilidad aleatoria entre 0 y la probabilidad total para reasignar la palabra a un topico
                pwk = np.random.rand() * prob_total
                for k in range(K):
                    if pwk<prob_topico[k]:
                        topico = k
                        break
                #llenar las matrices
                N_mk_c[m][topico] += 1
                N_kv_c[topico][w] += 1
                N_k_c[topico] += 1
                z_m[n] = topico


    #Calculo de phi, tamaño de topicos x vocabulario(KxV)
    #calculo de los estimadores de bayes para asignar las 20 palabras a dicho documento
    for k in range(K):
        #denominador de la formula de phi
        phi_den=N_k_c[k] + beta * V
        for w in range(V):
            phi_num=N_kv_c[k][w] + beta#numerador de la formula de phi
            phi[k][w] = phi_num / phi_den#asignar a la matriz el valor de la probabilidad

    #calculo de theta, tamaño documentos x topicos(MxK)
    for m in range(M):
        #numerador y denominador de phi
        #para calcular el numero de palabras en el documento m
        doc = corpus[m]
        doc_c = doc
        N_d = doc.shape[0]  # Palabras en este doc
        #denomindar de theta
        theta_den=N_d + K*alpha
        for k in range(K):
            theta_num=N_mk_c[m][k]+alpha#numerador de theta
            theta[m][k] = theta_num / theta_den#asignar a la matriz el valor de la probabilidad
    #tomar los 20 idices con mayor probabilidad de phi, nos dara los 20 ids mas probables por topico
    top_20_indices_por_fila = np.argpartition(phi, -20, axis=1)[:, -20:]

    #inicio para calcular la entropia
    #primero calculamos en número total de palabras en el corpus, denominador de la entropia
    totalPalabrasCorpus = 0
    for m in range(M):
        totalPalabrasCorpus += len(corpus[m])


    entropia = 0#basicamento toda la sumatoria del numerador, sum del documento m=1 hasta M
    #calcular el numerador
    #recorrer cada documento
    for m in range(M):
        doc = corpus[m]#asignarle a doc el documento de la posición m
        IDUnicos, conteo = np.unique(doc, return_counts=True)#obtener arrays que tienen palabras(ids) unicos y el conteo por id
        sumatoriaV = 0# sum de v=1 hasta V n_d_o_v, pero solo tomamos en cuenta las palabras que si estan en el documente m
        #recorrer cada palabra unica del documento
        for i in range(len(IDUnicos)):
            t = IDUnicos[i]#t toma el id de la palabra
            n_d_t = conteo[i]# cuantas veces esta la palabra en el documento
            #iniciar la sumatorio de phi * theta, la sumatoria de k=1 hasta K
            sumaPhiTheta = 0.0
            #recorrer cada uno de los topicos
            for k in range(K):
                #phi[k][t] --> probabilidad de la palabra t en el topico k
                #theta[m][k] --> probabilidad del topico k en el documento m
                sumaPhiTheta += phi[k][t] * theta[m][k]
            sumatoriaV += math.log(sumaPhiTheta) * n_d_t#multipicar n_d_o_v por el ln de la suma de phi * theta
        entropia += sumatoriaV#hacer la sumatoria general

    entropia = -entropia / totalPalabrasCorpus
    print(entropia)
    print(np.exp(entropia))

    return theta, phi, top_20_indices_por_fila

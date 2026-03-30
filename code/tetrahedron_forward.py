import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import verde as vd
import choclo
import harmonica as hm
import math
from numba import njit, jit, prange
import pandas as pd
from scipy.interpolate import griddata

constant = choclo.constants.VACUUM_MAGNETIC_PERMEABILITY / (4 * np.pi)


def data_importer_paraview(file, 
                           x_grid = 100, 
                           y_grid = 100, 
                           var_names = ["Points:0", "Points:1", "Points:2", "Bz"]):
    
    data=pd.read_csv(file, delimiter=',')
    # Hd_x,Hd_y,Hd_z=data.get('Hd_x'),data.get('Hd_y'),data.get('Hd_z')
    X,Y,Z=data.get(var_names[0]),data.get(var_names[1]),data.get(var_names[2])
    bz_comp = np.asarray(data.get(var_names[3]))

    xi = np.linspace(min(X), max(X), x_grid)
    yi = np.linspace(min(Y), max(Y), y_grid)
    xi, yi = np.meshgrid(xi, yi)
    
    zi = griddata((X, Y), Z, (xi, yi), method='cubic')
    Bz = griddata((X, Y), bz_comp, (xi, yi), method='cubic')

    return xi, yi, zi, Bz, X, Y, Z, bz_comp


def read_tecplot_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content


def read_tecplot_header(file_content):
    lines = file_content.strip().split('\n')
    title = lines[0].split('=')[1].strip().strip('"')
    variables = [v.strip().strip('"') for v in lines[1].split('=')[1].strip().split(',')]
    
    # Procura pela linha que contém informações de ZONE
    zone_line = [line for line in lines if 'ZONE' in line][0]
    N = int(zone_line.split('N=')[1].split(',')[0].strip())
    E = int(zone_line.split('E=')[1].strip())
    
    return title, variables, N, E

def organize_data(file_content, N, E):
    lines = file_content.strip().split('\n')
    
    # Inicializar um grande array para armazenar todos os valores
    all_values = []
    
    # Procurar a partir da linha 4 até 6*N + E
    start_index = 4
    end_index = start_index + 6 * N + E
    
    # Verifica se o end_index não ultrapassa o tamanho das linhas
    if end_index > len(lines):
        end_index = len(lines)
    
    for i in range(start_index, end_index):
        values = lines[i].strip().split()
        all_values.extend(values)
    
    # Dividir os valores conforme as especificações
    variables_values = {}
    variables = ["X", "Y", "Z", "Mx", "My", "Mz"]
    for i, var in enumerate(variables):
        variables_values[var] = [float(value) for value in all_values[i*N:(i+1)*N]]
    
    sd_values = [float(value) for value in all_values[6*N:6*N + E]]
    vertices_indexes = [int(value)-1 for value in all_values[6*N + E:]] # -1 to make the first index be 0
    
    # Dividir vertices_indexes em arrays de 4 valores cada
    vertices_indexes_array = [vertices_indexes[i:i+4] for i in range(0, len(vertices_indexes), 4)]
    
    return variables_values, sd_values, vertices_indexes_array


# Função para calcular a normal de uma face
def calculate_normal(face):
    v1, v2, v3 = face
    
    # Subtração dos vetores manualmente
    vector1_x = v2[0] - v1[0]
    vector1_y = v2[1] - v1[1]
    vector1_z = v2[2] - v1[2]
    
    vector2_x = v3[0] - v1[0]
    vector2_y = v3[1] - v1[1]
    vector2_z = v3[2] - v1[2]
    
    # Produto vetorial manual
    normal_x = vector1_y * vector2_z - vector1_z * vector2_y
    normal_y = vector1_z * vector2_x - vector1_x * vector2_z
    normal_z = vector1_x * vector2_y - vector1_y * vector2_x
    
    # Normalização
    norm = math.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    
    return np.asarray([normal_x, normal_y, normal_z])

# Função para garantir que a normal está apontando para fora do tetraedro
def ensure_normal_outward(face, normal, point_inside, epsilon=1e-15):
    v1 = face[0]
    
    # Subtração manual
    sub_x = point_inside[0] - v1[0]
    sub_y = point_inside[1] - v1[1]
    sub_z = point_inside[2] - v1[2]
    
    # Produto escalar manual
    dot_product = normal[0] * sub_x + normal[1] * sub_y + normal[2] * sub_z
    
    # Se o produto escalar for positivo, a normal está apontando para dentro
    if dot_product > epsilon:
        normal = [-normal[0], -normal[1], -normal[2]]  # Inverte a normal para apontar para fora
    
    return normal


def sort_tetrahedrons(tetrahedrons, tetrahedrons_centroid):
    tetrahedrons_sorted = np.copy(tetrahedrons)
    epsilon = 1e-15
    for i, (faces, centroid_tetrahedron) in enumerate(zip(tetrahedrons, tetrahedrons_centroid)):
       
        for j, face in enumerate(faces):
            n_f = calculate_normal(face)
            n_f_copy = n_f
            
            n_f = ensure_normal_outward(face, n_f, centroid_tetrahedron)
            
            # Verificar se a normal está invertida sem usar are_vectors_close
            if not (math.isclose(n_f_copy[0], n_f[0], abs_tol=epsilon) and
                    math.isclose(n_f_copy[1], n_f[1], abs_tol=epsilon) and
                    math.isclose(n_f_copy[2], n_f[2], abs_tol=epsilon)):
                r1, r2, r3 = face[::-1]
            else:
                r1, r2, r3 = face
            
            tetrahedrons_sorted[i, j, :, :] = r1, r2, r3
    return tetrahedrons_sorted

def define_tetrahedrons(X, Y, Z, vertices_indexes):
    tetrahedrons = []
    tetrahedrons_centroids = []
    
    for vertices_array in vertices_indexes:
        vertices = np.array([
            [X[vertices_array[0]], Y[vertices_array[0]], Z[vertices_array[0]]],
            [X[vertices_array[1]], Y[vertices_array[1]], Z[vertices_array[1]]],
            [X[vertices_array[2]], Y[vertices_array[2]], Z[vertices_array[2]]],
            [X[vertices_array[3]], Y[vertices_array[3]], Z[vertices_array[3]]],
        ]) * 1.0e-6  # converte microm to meter

        centroid_tetrahedron = np.mean(vertices, axis=0)
        # Define as faces do tetraedro
        faces = [
            [vertices[0], vertices[1], vertices[2]],
            [vertices[0], vertices[1], vertices[3]],
            [vertices[0], vertices[2], vertices[3]],
            [vertices[1], vertices[2], vertices[3]]
        ]
        
        tetrahedrons.append(faces)
        tetrahedrons_centroids.append(centroid_tetrahedron)
    
    return tetrahedrons, tetrahedrons_centroids


def define_tetrahedrons_magnetizations(Mx, My, Mz, vertices_indexes, Ms=480000):
    # Inicializa o array com a quantidade de elementos esperada
    n = len(vertices_indexes)
    tetrahedrons_magnetization = np.empty((n, 3))

    for i in range(n):
        # Acessa os índices diretamente usando vertices_indexes[i]
        mag_x = (Mx[vertices_indexes[i][0]] + Mx[vertices_indexes[i][1]] + 
                 Mx[vertices_indexes[i][2]] + Mx[vertices_indexes[i][3]]) / 4
        mag_y = (My[vertices_indexes[i][0]] + My[vertices_indexes[i][1]] + 
                 My[vertices_indexes[i][2]] + My[vertices_indexes[i][3]]) / 4
        mag_z = (Mz[vertices_indexes[i][0]] + Mz[vertices_indexes[i][1]] + 
                 Mz[vertices_indexes[i][2]] + Mz[vertices_indexes[i][3]]) / 4

        # Multiplicação por Ms
        tetrahedrons_magnetization[i, 0] = mag_x * Ms
        tetrahedrons_magnetization[i, 1] = mag_y * Ms
        tetrahedrons_magnetization[i, 2] = mag_z * Ms

    return tetrahedrons_magnetization


def define_tetrahedrons_magnetizations_weighted(X, Y, Z, Mx, My, Mz, vertices_indexes, Ms=480000):
    # Pré-alocar arrays para armazenar as magnetizações dos tetraedros
    n_tetrahedrons = len(vertices_indexes)
    tetrahedrons_magnetization_x = [0.0] * n_tetrahedrons
    tetrahedrons_magnetization_y = [0.0] * n_tetrahedrons
    tetrahedrons_magnetization_z = [0.0] * n_tetrahedrons

    # Loop paralelizado usando prange
    for idx in range(n_tetrahedrons):
        vertices_array = vertices_indexes[idx]
        
        # Pré-alocar coordenadas dos vértices e converter de microm para metro
        v0_x, v0_y, v0_z = X[vertices_array[0]] * 1.0e-6, Y[vertices_array[0]] * 1.0e-6, Z[vertices_array[0]] * 1.0e-6
        v1_x, v1_y, v1_z = X[vertices_array[1]] * 1.0e-6, Y[vertices_array[1]] * 1.0e-6, Z[vertices_array[1]] * 1.0e-6
        v2_x, v2_y, v2_z = X[vertices_array[2]] * 1.0e-6, Y[vertices_array[2]] * 1.0e-6, Z[vertices_array[2]] * 1.0e-6
        v3_x, v3_y, v3_z = X[vertices_array[3]] * 1.0e-6, Y[vertices_array[3]] * 1.0e-6, Z[vertices_array[3]] * 1.0e-6
        
        # Calcular o centróide do tetraedro
        centroid_x = (v0_x + v1_x + v2_x + v3_x) / 4.0
        centroid_y = (v0_y + v1_y + v2_y + v3_y) / 4.0
        centroid_z = (v0_z + v1_z + v2_z + v3_z) / 4.0
        
        # Inicializar variáveis de magnetização e total_weight
        mag_tetrahedron_x, mag_tetrahedron_y, mag_tetrahedron_z = 0.0, 0.0, 0.0
        total_weight = 0.0

        # Calcular distâncias, pesos e somar as contribuições ponderadas de cada vértice
        for i, (vx, vy, vz) in enumerate([(v0_x, v0_y, v0_z), (v1_x, v1_y, v1_z), (v2_x, v2_y, v2_z), (v3_x, v3_y, v3_z)]):
            distance = math.sqrt((vx - centroid_x) ** 2 + (vy - centroid_y) ** 2 + (vz - centroid_z) ** 2)
            weight = 1 / distance
            total_weight += weight

            # Somar as componentes ponderadas de magnetização
            mag_tetrahedron_x += Mx[vertices_array[i]] * Ms * weight
            mag_tetrahedron_y += My[vertices_array[i]] * Ms * weight
            mag_tetrahedron_z += Mz[vertices_array[i]] * Ms * weight

        # Normalizar as contribuições e armazenar nos arrays pré-alocados
        tetrahedrons_magnetization_x[idx] = mag_tetrahedron_x / total_weight
        tetrahedrons_magnetization_y[idx] = mag_tetrahedron_y / total_weight
        tetrahedrons_magnetization_z[idx] = mag_tetrahedron_z / total_weight

    # Retornar os resultados em forma de lista de tuplas
    return list(zip(tetrahedrons_magnetization_x, tetrahedrons_magnetization_y, tetrahedrons_magnetization_z))


    

@jit(nopython=True, parallel=True)
def calculate_Omega_f_and_Delta_W_f(tetrahedrons, tetrahedrons_mag, coordinates, Bx, By, Bz):
    
    
    easting = coordinates[0].ravel() 
    northing = coordinates[1].ravel()
    upward = coordinates[2].ravel()

    for i in prange(len(easting)):
        r_x, r_y, r_z = easting[i], northing[i], upward[i]
 
        for j in range(len(tetrahedrons_mag)):
            faces = tetrahedrons[j]
            M = tetrahedrons_mag[j]

            for k in range(len(faces)):
                ############### Calculate the normal vector of the face ################
                r1_x, r1_y, r1_z = faces[k][0][0], faces[k][0][1], faces[k][0][2]
                r2_x, r2_y, r2_z = faces[k][1][0], faces[k][1][1], faces[k][1][2]
                r3_x, r3_y, r3_z = faces[k][2][0], faces[k][2][1], faces[k][2][2]
                
                v1_x, v1_y, v1_z = r2_x - r1_x, r2_y - r1_y, r2_z - r1_z 
                v2_x, v2_y, v2_z = r3_x - r1_x, r3_y - r1_y, r3_z - r1_z

                n_f_x = v1_y * v2_z - v1_z * v2_y
                n_f_y = v1_z * v2_x - v1_x * v2_z
                n_f_z = v1_x * v2_y - v1_y * v2_x

                norm = math.sqrt(n_f_x**2 + n_f_y**2 + n_f_z**2)

                n_f_x, n_f_y, n_f_z = n_f_x / norm, n_f_y / norm, n_f_z / norm
                ############### Calculate the normal vector of the face ################
                
                ############### Calculate Omega_f ###################################### 
                v1_x, v1_y, v1_z = r1_x - r_x, r1_y - r_y, r1_z - r_z
                v2_x, v2_y, v2_z = r2_x - r_x, r2_y - r_y, r2_z - r_z
                v3_x, v3_y, v3_z = r3_x - r_x, r3_y - r_y, r3_z - r_z

                v1_norm = math.sqrt(v1_x**2 + v1_y**2 + v1_z**2)
                v2_norm = math.sqrt(v2_x**2 + v2_y**2 + v2_z**2)
                v3_norm = math.sqrt(v3_x**2 + v3_y**2 + v3_z**2)

                dot_v1_v2 = v1_x * v2_x + v1_y * v2_y + v1_z * v2_z
                dot_v1_v3 = v1_x * v3_x + v1_y * v3_y + v1_z * v3_z
                dot_v2_v3 = v2_x * v3_x + v2_y * v3_y + v2_z * v3_z

                cross_v2_v3_x, cross_v2_v3_y, cross_v2_v3_z = v2_y * v3_z - v2_z * v3_y, v2_z * v3_x - v2_x * v3_z, v2_x * v3_y - v2_y * v3_x

                D = (v1_norm * v2_norm * v3_norm +
                     v3_norm * dot_v1_v2 +
                     v2_norm * dot_v1_v3 +
                     v1_norm * dot_v2_v3)

                numerator = v1_x * cross_v2_v3_x + v1_y * cross_v2_v3_y + v1_z * cross_v2_v3_z

                Omega_f = 2 * math.atan2(numerator, D)
                ############### Calculate Omega_f ###################################### 

                ############### Calculate W_f for each edge ############################
                W_e_acum_x, W_e_acum_y, W_e_acum_z = 0.0, 0.0, 0.0
                # Calculate W_f for edge r1-r2
                v1_norm = math.sqrt((r1_x - r_x)**2 + (r1_y - r_y)**2 + (r1_z - r_z)**2)
                v2_norm = math.sqrt((r2_x - r_x)**2 + (r2_y - r_y)**2 + (r2_z - r_z)**2)

                v3_x, v3_y, v3_z = r2_x - r1_x, r2_y - r1_y, r2_z - r1_z
                v3_norm = math.sqrt(v3_x**2 + v3_y**2 + v3_z**2)

                u_e_x, u_e_y, u_e_z = v3_x / v3_norm, v3_y / v3_norm, v3_z / v3_norm

                W_e = math.log((v2_norm + v1_norm + v3_norm) / (v2_norm + v1_norm - v3_norm))

                cross_product_x, cross_product_y, cross_product_z = n_f_y * u_e_z - n_f_z * u_e_y, n_f_z * u_e_x - n_f_x * u_e_z, n_f_x * u_e_y - n_f_y * u_e_x

                W_e_acum_x += cross_product_x * W_e
                W_e_acum_y += cross_product_y * W_e
                W_e_acum_z += cross_product_z * W_e

                # Calculate W_f for edge r2-r3
                v1_norm = math.sqrt((r2_x - r_x)**2 + (r2_y - r_y)**2 + (r2_z - r_z)**2)
                v2_norm = math.sqrt((r3_x - r_x)**2 + (r3_y - r_y)**2 + (r3_z - r_z)**2)

                v3_x, v3_y, v3_z = r3_x - r2_x, r3_y - r2_y, r3_z - r2_z
                v3_norm = math.sqrt(v3_x**2 + v3_y**2 + v3_z**2)

                u_e_x, u_e_y, u_e_z = v3_x / v3_norm, v3_y / v3_norm, v3_z / v3_norm

                W_e = math.log((v2_norm + v1_norm + v3_norm) / (v2_norm + v1_norm - v3_norm))

                cross_product_x, cross_product_y, cross_product_z = n_f_y * u_e_z - n_f_z * u_e_y, n_f_z * u_e_x - n_f_x * u_e_z, n_f_x * u_e_y - n_f_y * u_e_x

                W_e_acum_x += cross_product_x * W_e
                W_e_acum_y += cross_product_y * W_e
                W_e_acum_z += cross_product_z * W_e

                # Calculate W_f for edge r3-r1
                v1_norm = math.sqrt((r3_x - r_x)**2 + (r3_y - r_y)**2 + (r3_z - r_z)**2)
                v2_norm = math.sqrt((r1_x - r_x)**2 + (r1_y - r_y)**2 + (r1_z - r_z)**2)

                v3_x, v3_y, v3_z = r1_x - r3_x, r1_y - r3_y, r1_z - r3_z
                v3_norm = math.sqrt(v3_x**2 + v3_y**2 + v3_z**2)

                u_e_x, u_e_y, u_e_z = v3_x / v3_norm, v3_y / v3_norm, v3_z / v3_norm

                W_e = math.log((v2_norm + v1_norm + v3_norm) / (v2_norm + v1_norm - v3_norm))

                cross_product_x, cross_product_y, cross_product_z = n_f_y * u_e_z - n_f_z * u_e_y, n_f_z * u_e_x - n_f_x * u_e_z, n_f_x * u_e_y - n_f_y * u_e_x

                W_e_acum_x += cross_product_x * W_e
                W_e_acum_y += cross_product_y * W_e
                W_e_acum_z += cross_product_z * W_e
                ############### Calculate W_f for each edge ############################

                ############### Main formula ###########################################
                Delta_W_f_x, Delta_W_f_y, Delta_W_f_z = W_e_acum_x + n_f_x * Omega_f, W_e_acum_y + n_f_y * Omega_f, W_e_acum_z + n_f_z * Omega_f

                cross_product_x, cross_product_y, cross_product_z = M[1] * n_f_z - M[2] * n_f_y, M[2] * n_f_x - M[0] * n_f_z, M[0] * n_f_y - M[1] * n_f_x

                bx = cross_product_y * Delta_W_f_z - cross_product_z * Delta_W_f_y
                by = cross_product_z * Delta_W_f_x - cross_product_x * Delta_W_f_z
                bz = cross_product_x * Delta_W_f_y - cross_product_y * Delta_W_f_x
                ############### Main formula ###########################################
                
                Bx[i] += bx
                By[i] += by
                Bz[i] += bz

    Bx *= -constant
    By *= -constant
    Bz *= -constant


def calculate_grain_magnetic_moments(X, Y, Z, Mx, My, Mz, vertices_indexes, Ms=480000):
    # Pré-alocar arrays para armazenar as magnetizações dos tetraedros
    n_tetrahedrons = len(vertices_indexes)
    tetrahedrons_magnetization_x = [0.0] * n_tetrahedrons
    tetrahedrons_magnetization_y = [0.0] * n_tetrahedrons
    tetrahedrons_magnetization_z = [0.0] * n_tetrahedrons

    # Loop paralelizado usando prange
    for idx in range(n_tetrahedrons):
        vertices_array = vertices_indexes[idx]
        
        # Pré-alocar coordenadas dos vértices e converter de microm para metro
        v0_x, v0_y, v0_z = X[vertices_array[0]] * 1.0e-6, Y[vertices_array[0]] * 1.0e-6, Z[vertices_array[0]] * 1.0e-6
        v1_x, v1_y, v1_z = X[vertices_array[1]] * 1.0e-6, Y[vertices_array[1]] * 1.0e-6, Z[vertices_array[1]] * 1.0e-6
        v2_x, v2_y, v2_z = X[vertices_array[2]] * 1.0e-6, Y[vertices_array[2]] * 1.0e-6, Z[vertices_array[2]] * 1.0e-6
        v3_x, v3_y, v3_z = X[vertices_array[3]] * 1.0e-6, Y[vertices_array[3]] * 1.0e-6, Z[vertices_array[3]] * 1.0e-6

        ## Calcular volume do tetraedro
        # Calcular os vetores v1_v0, v2_v0 e v3_v0
        v1_v0_x, v1_v0_y, v1_v0_z = v1_x - v0_x, v1_y - v0_y, v1_z - v0_z
        v2_v0_x, v2_v0_y, v2_v0_z = v2_x - v0_x, v2_y - v0_y, v2_z - v0_z
        v3_v0_x, v3_v0_y, v3_v0_z = v3_x - v0_x, v3_y - v0_y, v3_z - v0_z
        # Calcular o produto vetorial de v2_v0 e v3_v0 (termo a termo)
        cross_x = v2_v0_y * v3_v0_z - v2_v0_z * v3_v0_y
        cross_y = v2_v0_z * v3_v0_x - v2_v0_x * v3_v0_z
        cross_z = v2_v0_x * v3_v0_y - v2_v0_y * v3_v0_x
        # Calcular o produto escalar entre v1_v0 e o vetor cruzado (cross_x, cross_y, cross_z)
        dot_product = v1_v0_x * cross_x + v1_v0_y * cross_y + v1_v0_z * cross_z
        # Calcular o volume do tetraedro
        volume = abs(dot_product) / 6
        
        # Calcular o centróide do tetraedro
        centroid_x = (v0_x + v1_x + v2_x + v3_x) / 4.0
        centroid_y = (v0_y + v1_y + v2_y + v3_y) / 4.0
        centroid_z = (v0_z + v1_z + v2_z + v3_z) / 4.0
        
        # Inicializar variáveis de magnetização e total_weight
        mag_tetrahedron_x, mag_tetrahedron_y, mag_tetrahedron_z = 0.0, 0.0, 0.0
        total_weight = 0.0

        # Calcular distâncias, pesos e somar as contribuições ponderadas de cada vértice
        for i, (vx, vy, vz) in enumerate([(v0_x, v0_y, v0_z), (v1_x, v1_y, v1_z), (v2_x, v2_y, v2_z), (v3_x, v3_y, v3_z)]):
            distance = math.sqrt((vx - centroid_x) ** 2 + (vy - centroid_y) ** 2 + (vz - centroid_z) ** 2)
            weight = 1 / distance
            total_weight += weight

            # Somar as componentes ponderadas de magnetização
            mag_tetrahedron_x += Mx[vertices_array[i]] * Ms * weight
            mag_tetrahedron_y += My[vertices_array[i]] * Ms * weight
            mag_tetrahedron_z += Mz[vertices_array[i]] * Ms * weight

        # Normalizar as contribuições e armazenar nos arrays pré-alocados
        tetrahedrons_magnetization_x[idx] = mag_tetrahedron_x / total_weight * volume
        tetrahedrons_magnetization_y[idx] = mag_tetrahedron_y / total_weight * volume
        tetrahedrons_magnetization_z[idx] = mag_tetrahedron_z / total_weight * volume

    # Retornar os resultados em forma de lista de tuplas
    return np.array([np.sum(tetrahedrons_magnetization_x), 
                     np.sum(tetrahedrons_magnetization_y), 
                     np.sum(tetrahedrons_magnetization_z)
                    ])

    
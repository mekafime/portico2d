# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

#Datos en memoria
nodos = []  # [(id, x, y)]
miembros = []  # [(id, nodo_inicio, nodo_fin, material_id)]
cargas = []  # [(tipo, nodo_id, Fx, Fy, M, q)]
restricciones = []  # [(nodo_id, restriccion_x, restriccion_y, rotacion)]
materiales = []  # [(id, E, A, I)]
reacciones = []  # [(nodo_id, Rx, Ry, Mz)]


def generar_imagen():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.clear()

    if nodos:
     #Obtener rangos de los nodos para la autoescala
     x_min = min(n[1] for n in nodos) if nodos else -5
     x_max = max(n[1] for n in nodos) if nodos else 5
     y_min = min(n[2] for n in nodos) if nodos else 0
     y_max = max(n[2] for n in nodos) if nodos else 10
     
    #Incluir el tamano de las cargas en la escala
     carga_max = max([abs(c[2]) + abs(c[3]) for c in cargas if c[0] == "puntual"], default=0)
     carga_dist_max = max([abs(c[5]) for c in cargas if c[0] == "distribuida"], default=0)
     
    #Agregamos margen basado en nodos y cargas
     margen_x = max((x_max - x_min) * 0.2, 2)  # 20% del tama?o en X
     margen_y = max((y_max - y_min) * 0.2, 2) + carga_max + carga_dist_max  # Incluye cargas
   
     #Ajustar margenes de los ejes para mantener la proporcion correcta
     ax.set_aspect('equal', adjustable='datalim')  # Mantiene la proporcion de la estructura
     ax.set_xlim(x_min - margen_x, x_max + margen_x)
     ax.set_ylim(y_min - margen_y, y_max + margen_y)

     
    else:
        #Si no hay nodos, mostrar una grilla con los ejes visibles
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 10)
     
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Estructura 2D')    
   
    #Dibujar nodos
    for nodo in nodos:
      ax.plot(nodo[1], nodo[2], 'ro', markersize=8)
      ax.text(nodo[1], nodo[2], f'N{nodo[0]}', fontsize=12, verticalalignment='bottom')

    #Dibujar miembros
    for miembro in miembros:
      n1 = next(n for n in nodos if n[0] == miembro[1])
      n2 = next(n for n in nodos if n[0] == miembro[2])
      ax.plot([n1[1], n2[1]], [n1[2], n2[2]], 'k-', linewidth=2)
     
    #Dibujar cargas
    for carga in cargas:
        if carga[0] == "puntual":
            nodo = next(n for n in nodos if n[0] == carga[1])
            ax.arrow(nodo[1], nodo[2], -carga[2] * 0.2 if carga[2] < 0 else carga[2] * 0.2,
                 -carga[3] * 0.2 if carga[3] < 0 else carga[3] * 0.2, 
                 head_width=0.3, head_length=0.3, fc='blue', ec='blue')


        elif carga[0] == "distribuida":
         n1 = next(n for n in nodos if n[0] == carga[1])
         n2 = next(n for n in nodos if n[0] == carga[2])
         q = carga[5]  # Magnitud de la carga
         tipo_carga = carga[6]  # "global" o "local"

         x_pos = np.linspace(n1[1], n2[1], 5)
         y_pos = np.linspace(n1[2], n2[2], 5)

         if tipo_carga == "global":
          sentido = -1 if carga[5] > 0 else 1
         #Carga vertical hacia abajo (global Y)
          for i in range(len(x_pos)):
           ax.arrow(x_pos[i], y_pos[i], 0, q * 0.5, 
                         head_width=0.3, head_length=0.3, fc='green', ec='green')

         elif tipo_carga == "local":
          #Calcular vector unitario perpendicular al miembro
          dx = n2[1] - n1[1]
          dy = n2[2] - n1[2]
          longitud = (dx**2 + dy**2)**0.5
          normal_x = -dy / longitud  # Perpendicular en X
          normal_y = dx / longitud   # Perpendicular en Y
          
          sentido = -1 if carga[5] > 0 else 1

          for i in range(len(x_pos)):
           ax.arrow(x_pos[i], y_pos[i], normal_x * q * 0.5, normal_y * q * 0.5, 
                         head_width=0.3, head_length=0.3, fc='orange', ec='orange')

    #Dibujar restricciones
    for restriccion in restricciones:
        nodo = next(n for n in nodos if n[0] == restriccion[0])
        if restriccion[1]:  # Restricción en X
            ax.plot(nodo[1], nodo[2], 'bs', markersize=10)
        if restriccion[2]:  # Restricción en Y
            ax.plot(nodo[1], nodo[2], 'rs', markersize=10)
        if restriccion[3]:  # Restricción en Rotación
            ax.plot(nodo[1], nodo[2], 'gs', markersize=10)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_data = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return img_data
    
@app.route('/generar_imagen', methods=['POST'])
def generar_imagen_endpoint():
    return jsonify({"img_data": generar_imagen()})
    
@app.route('/reiniciar', methods=['POST'])
def reiniciar():
    global nodos, miembros, cargas, restricciones, materiales, reacciones
    nodos.clear()
    miembros.clear()
    cargas.clear()
    restricciones.clear()
    materiales.clear()
    reacciones.clear()
    
    #Generar una imagen vacia inmediatamente despues de reiniciar
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 10)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Estructura 2D')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_data = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return jsonify({"mensaje": "Datos reiniciados", "img_data": img_data}) 

@app.route('/')
def index():
    img_data = generar_imagen()
    return render_template('index.html', nodos=nodos, miembros=miembros, cargas=cargas, 
                           restricciones=restricciones, materiales=materiales, reacciones=reacciones, img_data=img_data)


@app.route('/agregar_nodo', methods=['POST'])
def agregar_nodo():
    x = float(request.form['x'])
    y = float(request.form['y'])
    nodos.append((len(nodos) + 1, x, y))
    return jsonify({"img_data": generar_imagen()})
    
@app.route('/borrar_nodo', methods=['POST'])
def borrar_nodo():
    nodo_id = int(request.form['nodo_id'])
    global nodos, miembros, cargas, restricciones
    nodos = [n for n in nodos if n[0] != nodo_id]
    miembros = [m for m in miembros if m[1] != nodo_id and m[2] != nodo_id]
    cargas = [c for c in cargas if c[1] != nodo_id]
    restricciones = [r for r in restricciones if r[0] != nodo_id]
    return jsonify({"img_data": generar_imagen()})    

@app.route('/agregar_miembro', methods=['POST'])
def agregar_miembro():
    nodo_inicio = int(request.form['nodo_inicio'])
    nodo_fin = int(request.form['nodo_fin'])
    material_id = int(request.form['material_id'])
    miembros.append((len(miembros) + 1, nodo_inicio, nodo_fin, material_id))
    return jsonify({"img_data": generar_imagen()})
    
@app.route('/borrar_miembro', methods=['POST'])
def borrar_miembro():
    miembro_id = int(request.form['miembro_id'])
    global miembros
    miembros = [m for m in miembros if m[0] != miembro_id]
    return jsonify({"img_data": generar_imagen()})    

@app.route('/agregar_material', methods=['POST'])
def agregar_material():
    E = float(request.form['modulo_elasticidad'])
    A = float(request.form['area'])
    I = float(request.form['inercia'])
    material_id = len(materiales) + 1
    materiales.append((len(materiales) + 1, E, A, I))
    return jsonify({"materiales": materiales})
    
@app.route('/borrar_material', methods=['POST'])
def borrar_material():
    material_id = int(request.form['material_id'])
    global materiales, miembros
    materiales = [m for m in materiales if m[0] != material_id]
    miembros = [m for m in miembros if m[3] != material_id]
    return jsonify({"materiales": materiales})    

@app.route('/agregar_carga', methods=['POST'])
def agregar_carga():
    tipo = request.form['tipo']
    
    if tipo == "puntual":
        nodo_id = int(request.form['nodo_id'])
        Fx = float(request.form['Fx'])
        Fy = float(request.form['Fy'])
        M = float(request.form['M'])
        cargas.append((tipo, nodo_id, Fx, Fy, M, 0, ""))
    
    elif tipo == "distribuida":
        nodo_inicio = int(request.form['nodo_inicio'])
        nodo_fin = int(request.form['nodo_fin'])
        q = float(request.form['q'])
        tipo_carga = request.form['tipo_carga']  # "global" o "local"
        cargas.append((tipo, nodo_inicio, nodo_fin, 0, 0, q, tipo_carga))
    
    return jsonify({"img_data": generar_imagen()})
    
@app.route('/borrar_carga_puntual', methods=['POST'])
def borrar_carga_puntual():
    nodo_id = int(request.form['nodo_id'])
    global cargas
    cargas = [c for c in cargas if not (c[0] == "puntual" and c[1] == nodo_id)]
    return jsonify({"img_data": generar_imagen()})   
    
@app.route('/borrar_carga_distribuida', methods=['POST'])
def borrar_carga_distribuida():
    nodo_inicio = int(request.form['nodo_inicio'])
    nodo_fin = int(request.form['nodo_fin'])
    global cargas
    cargas = [c for c in cargas if not (c[0] == "distribuida" and c[1] == nodo_inicio and c[2] == nodo_fin)]
    return jsonify({"img_data": generar_imagen()})

@app.route('/agregar_restriccion', methods=['POST'])
def agregar_restriccion():
    nodo_id = int(request.form['nodo_id'])
    restriccion_x = 'restriccion_x' in request.form
    restriccion_y = 'restriccion_y' in request.form
    rotacion = 'rotacion' in request.form
    restricciones.append((nodo_id, restriccion_x, restriccion_y, rotacion))
    return jsonify({"img_data": generar_imagen()})
    
@app.route('/borrar_restriccion', methods=['POST'])
def borrar_restriccion():
    nodo_id = int(request.form['nodo_id'])
    global restricciones
    restricciones = [r for r in restricciones if r[0] != nodo_id]
    return jsonify({"img_data": generar_imagen()})    

@app.route('/ejecutar_analisis', methods=['POST'])
def ejecutar_analisis():
    global reacciones
    reacciones.clear()

    print(f"📌 Ejecutando análisis... Total nodos: {len(nodos)}, Total miembros: {len(miembros)}")

    if not nodos or not miembros:
        print("⚠️ No hay nodos o miembros definidos para el análisis.")
        return jsonify({"error": "No hay nodos o miembros definidos para el análisis"}), 400

    # Asignar reacciones ficticias si hay restricciones
    for restriccion in restricciones:
        nodo_id = restriccion[0]
        reacciones.append((nodo_id, np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-5, 5)))

    print(f"📌 Se calcularon {len(reacciones)} reacciones.")

    try:
        esfuerzos = calcular_esfuerzos()
        
        print(f"📌 Esfuerzos calculados: {esfuerzos}")  # 🔹 VERIFICAR SI REALMENTE SE CALCULAN

        if not esfuerzos:
            print("⚠️ No se calcularon esfuerzos, posiblemente faltan datos.")
            return jsonify({"error": "No se calcularon esfuerzos, verifica los datos de entrada."}), 400

        print(f"📌 Se calcularon esfuerzos para {len(esfuerzos)} miembros.")

        #Generar gráficos de esfuerzos
        img_axial = generar_grafico_esfuerzo(esfuerzos, "axial")
        img_cortante = generar_grafico_esfuerzo(esfuerzos, "cortante")
        img_momento = generar_grafico_esfuerzo(esfuerzos, "momento")
        img_deformada = generar_grafico_deformaciones()

        print("✅ Análisis completado exitosamente.")

        return jsonify({
            "mensaje": "Análisis completado",
            "img_axial": img_axial,
            "img_cortante": img_cortante,
            "img_momento": img_momento,
            "img_deformada": img_deformada,
            "reacciones": reacciones
        })
    except Exception as e:
        print(f"❌ Error en el análisis: {str(e)}")
        return jsonify({"error": f"Error en el análisis: {str(e)}"}), 500

        
def calcular_esfuerzos():
    """Calcula esfuerzos Axial, Cortante y Momento para cada miembro."""
    esfuerzos = {}

    if not miembros:
        print("⚠️ No hay miembros en la estructura.")
        return esfuerzos  # Devolver vacío si no hay miembros

    print(f"📌 Calculando esfuerzos para {len(miembros)} miembros.")

    for miembro in miembros:
        try:
            nodo_inicio = next((n for n in nodos if n[0] == miembro[1]), None)
            nodo_fin = next((n for n in nodos if n[0] == miembro[2]), None)
            material = next((m for m in materiales if m[0] == miembro[3]), None)

            if not nodo_inicio or not nodo_fin:
                print(f"⚠️ No se encontraron los nodos o material para el miembro {miembro[0]}")
                continue  # Saltar este miembro si no se encuentran sus nodos

            L = ((nodo_fin[1] - nodo_inicio[1])**2 + (nodo_fin[2] - nodo_inicio[2])**2) ** 0.5  
            E, A, I = material[1], material[2], material[3]

            # Obtener carga aplicada en el nodo
            carga = next((c for c in cargas if c[1] == nodo_inicio[0]), None)
            P = carga[2] if carga else 0  # Tomamos la carga en X (podría ser en Y también)

            # Cálculo de esfuerzos teóricos
            axial = np.linspace(-P / A, P / A, 10)  # Esfuerzo normal = P / A
            cortante = np.linspace(-P, P, 10)  # Esfuerzo cortante (aproximado)
            momento = np.linspace(-P * L / 2, P * L / 2, 10)  # Momento flector simple

            # Cálculo de deformaciones usando la ecuación de flexión
            deformaciones = np.linspace(0, (P * L**3) / (3 * E * I) if E > 0 and I > 0 else 0, 10)

            esfuerzos[miembro[0]] = {
                "axial": axial,
                "cortante": cortante,
                "momento": momento,
                "deformaciones": deformaciones,
                "longitud": L
            }
        except Exception as e:
            print(f"❌ Error en cálculo de esfuerzos para miembro {miembro[0]}: {str(e)}")

    print(f"✅ Esfuerzos generados: {esfuerzos}")  # 🔹 REVISAR QUE NO ESTÉ VACÍO
    return esfuerzos

def generar_grafico_deformaciones():
    """Genera el gráfico de la estructura deformada."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.clear()

    if nodos:
        x_min = min(n[1] for n in nodos)
        x_max = max(n[1] for n in nodos)
        y_min = min(n[2] for n in nodos)
        y_max = max(n[2] for n in nodos)

        margen_x = (x_max - x_min) * 0.2
        margen_y = (y_max - y_min) * 0.2

        ax.set_xlim(x_min - margen_x, x_max + margen_x)
        ax.set_ylim(y_min - margen_y, y_max + margen_y)

    else:
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 10)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Estructura Deformada')

    # Dibujar nodos originales
    for nodo in nodos:
        ax.plot(nodo[1], nodo[2], 'ro', markersize=8)
        ax.text(nodo[1], nodo[2], f'N{nodo[0]}', fontsize=12, verticalalignment='bottom')

    # Dibujar estructura deformada
    escala_deformacion = 500  # Factor de escala
    for miembro in miembros:
        n1 = next(n for n in nodos if n[0] == miembro[1])
        n2 = next(n for n in nodos if n[0] == miembro[2])

        dx1, dy1 = obtener_deformacion(n1[0])
        dx2, dy2 = obtener_deformacion(n2[0])

        ax.plot([n1[1] + dx1 * escala_deformacion, n2[1] + dx2 * escala_deformacion],
                [n1[2] + dy1 * escala_deformacion, n2[2] + dy2 * escala_deformacion],
                'b--', linewidth=2, label="Deformada" if miembro[0] == 1 else "")

    ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_data = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return img_data
  
def generar_grafico_esfuerzo(esfuerzos, tipo):
    """Genera el gráfico de esfuerzos sobre la estructura."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if not esfuerzos:
        print(f"⚠️ No hay datos de esfuerzos para {tipo}.")
        ax.text(0.5, 0.5, "No hay datos de esfuerzos", fontsize=12, ha='center')
    else:
        print(f"📌 Generando gráfico de {tipo} para {len(esfuerzos)} miembros.")
        for miembro_id, datos in esfuerzos.items():
            x_vals = np.linspace(0, datos["longitud"], len(datos.get(tipo, [])))  # Usa la misma cantidad de puntos
            if tipo in datos:
                esfuerzos_esc = datos[tipo] * 0.02  
                ax.plot(x_vals, esfuerzos_esc, label=f'Miembro {miembro_id}')
        
        ax.set_xlabel("Longitud (m)")
        ax.set_ylabel(f"{tipo.capitalize()} (kN)")
        ax.set_title(f"Diagrama de {tipo.capitalize()}")
        ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_data = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_data


if __name__ == '__main__':
    app.run(debug=True)

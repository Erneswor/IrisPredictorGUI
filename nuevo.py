from tkinter import font, messagebox
import matplotlib
from sklearn.calibration import cross_val_predict
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
from matplotlib.colors import BoundaryNorm, ListedColormap
from mlxtend.plotting import plot_decision_regions
import tkinter as tk
from tkinter import simpledialog

# -------------------- Configuración global --------------------
ruta_csv = 'datos/archive/Iris.csv'  # Ajustar esta ruta

def cargar_datos_csv(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={
        'sepallengthcm': 'sepal_length',
        'sepalwidthcm': 'sepal_width',
        'petallengthcm': 'petal_length',
        'petalwidthcm': 'petal_width',
        'species': 'species'
    }, inplace=True)
    return df

#---------------------- Interfaz de Regresión Lineal ------------#
def multi_input_dialog_lineal():
    dialog = tk.Toplevel()
    dialog.title("Inputs Regresión Lineal")
    dialog.geometry("350x150")
    
    species_keys = ["Iris-versicolor", "Iris-setosa", "Iris-virginica"]
    entries = {}
    
    for i, key in enumerate(species_keys):
        tk.Label(dialog, text=f"Ingrese Ancho del pétalo para {key}:", anchor="w")\
            .grid(row=i, column=0, padx=5, pady=5, sticky="w")
        entry = tk.Entry(dialog)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries[key] = entry

    result = {}
    def on_ok():
        try:
            for key in species_keys:
                result[key] = float(entries[key].get())
        except ValueError:
            for key in species_keys:
                result[key] = None
        dialog.destroy()
        
    tk.Button(dialog, text="OK", command=on_ok)\
        .grid(row=len(species_keys), column=0, columnspan=2, pady=10)
    dialog.wait_window()
    return result

# -------------------- Regresión Lineal --------------------
def regresion_lineal():
    df = cargar_datos_csv(ruta_csv)
    
    # Obtener los tres valores en un solo cuadro
    x_usuario_dict = multi_input_dialog_lineal()
    
    # Usamos el estilo clásico de Matplotlib
    plt.style.use('classic')
    
    species = df['species'].unique()
    palette = ['blue', 'green', 'red']
    
    plt.figure(figsize=(10, 6))
    
    for idx, species_name in enumerate(species):
        df_species = df[df['species'] == species_name]
        # Se utiliza 'petal_width' para X y 'petal_length' para Y
        X_species = df_species[['petal_width']].values
        y_species = df_species['petal_length'].values
        
        modelo = LinearRegression()
        modelo.fit(X_species, y_species)
        
        # Obtener el valor ingresado para la especie actual
        x_valor = x_usuario_dict.get(species_name)
        if x_valor is None:
            # Si no se encontró, usar el valor del primer input
            x_valor = list(x_usuario_dict.values())[0]
        
        y_pred = modelo.predict([[x_valor]])[0]
        
        # Calcular R² y RMSE
        r2 = modelo.score(X_species, y_species)
        rmse = np.sqrt(mean_squared_error(y_species, modelo.predict(X_species)))
        print(f'Especie: {species_name}, X = {x_valor:.2f}, Predicción Y = {y_pred:.2f}')
        print(f"R²: {r2:.3f} | Error: {rmse:.3f}")
        
        # Graficar los datos y la línea de regresión
        plt.scatter(X_species.ravel(), y_species.ravel(), 
                    color=palette[idx], 
                    label=f"{species_name}\nR²: {r2:.3f}\nRMSE: {rmse:.3f}", s=80)
        X_range = np.linspace(X_species.min(), X_species.max(), 100).reshape(-1, 1)
        plt.plot(X_range, modelo.predict(X_range), color=palette[idx], linewidth=2)
        plt.scatter(x_valor, y_pred, color='black', marker='X', s=120)
        plt.annotate(f"{species_name}: {y_pred:.2f}", (x_valor, y_pred),
                     textcoords="offset points", xytext=(5,5), ha="left",
                     fontsize=9, color=palette[idx])
    
    plt.xlabel('Ancho del pétalo (cm)', labelpad=10)
    plt.ylabel('Largo del pétalo (cm)', labelpad=10)
    plt.title('Regresión Lineal por Especie', pad=15)
    plt.legend(title="Especies")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#-------------------- Interfaz de Regresión Polinomial ------------#
def multi_input_dialog_polinomial():
    dialog = tk.Toplevel()
    dialog.title("Inputs Regresión Polinomial")
    dialog.geometry("400x200")
    
    keys = ["grado", "Iris-versicolor", "Iris-setosa", "Iris-virginica"]
    labels_text = ["Grado de la regresión:",
                   "Ancho del pétalo para Iris-versicolor:",
                   "Ancho del pétalo para Iris-setosa:",
                   "Ancho del pétalo para Iris-virginica:"]
    
    entries = {}
    for i, key in enumerate(keys):
        tk.Label(dialog, text=labels_text[i], anchor="w")\
            .grid(row=i, column=0, padx=5, pady=5, sticky="w")
        entry = tk.Entry(dialog)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries[key] = entry

    result = {}
    def on_ok():
        try:
            result["grado"] = int(entries["grado"].get())
        except ValueError:
            result["grado"] = None
        for key in keys[1:]:
            try:
                result[key] = float(entries[key].get())
            except ValueError:
                result[key] = None
        dialog.destroy()
        
    tk.Button(dialog, text="OK", command=on_ok)\
        .grid(row=len(keys), column=0, columnspan=2, pady=10)
    dialog.wait_window()
    return result

# -------------------- Regresión Polinomial --------------------
def regresion_polinomial():
    df = cargar_datos_csv(ruta_csv)
    
    # Obtener grado y entradas para cada especie
    inputs = multi_input_dialog_polinomial()
    grado = inputs.get("grado")
    if grado is None:
        print("El grado ingresado es inválido.")
        return

    plt.style.use('classic')
    
    species = df['species'].unique()
    palette = ['blue', 'green', 'red']
    
    plt.figure(figsize=(10, 6))
    
    for idx, species_name in enumerate(species):
        df_species = df[df['species'] == species_name]
        X_species = df_species[['petal_width']].values
        y_species = df_species['petal_length'].values
        
        poly = PolynomialFeatures(degree=grado)
        X_poly = poly.fit_transform(X_species)
        modelo = LinearRegression()
        modelo.fit(X_poly, y_species)
        
        x_usuario = inputs.get(species_name)
        if x_usuario is None:
            x_usuario = list(inputs.values())[1]  # valor por defecto del primer input de especie
        
        X_usuario_poly = poly.transform([[x_usuario]])
        y_pred = modelo.predict(X_usuario_poly)[0]
        print(f'Especie: {species_name}, X = {x_usuario:.2f}, Predicción Y = {y_pred:.2f}')
        
        r2 = modelo.score(X_poly, y_species)
        rmse = np.sqrt(mean_squared_error(y_species, modelo.predict(X_poly)))
        print(f"R²: {r2:.3f} | Error: {rmse:.3f}")
        
        plt.scatter(X_species.ravel(), y_species.ravel(), 
                    color=palette[idx], 
                    label=f"{species_name}\nR²: {r2:.3f}\nRMSE: {rmse:.3f}", s=80)
        X_range = np.linspace(X_species.min(), X_species.max(), 100).reshape(-1, 1)
        X_range_poly = poly.transform(X_range)
        plt.plot(X_range, modelo.predict(X_range_poly), color=palette[idx], linewidth=2)
        plt.scatter(x_usuario, y_pred, color='black', marker='X', s=120)
        plt.annotate(f"{species_name}: {y_pred:.2f}", (x_usuario, y_pred),
                     textcoords="offset points", xytext=(5,5), ha="left",
                     fontsize=9, color=palette[idx])
    
    plt.xlabel('Ancho del pétalo (cm)', labelpad=10)
    plt.ylabel('Largo del pétalo (cm)', labelpad=10)
    plt.title(f'Regresión Polinómica (Grado {grado})', pad=15)
    plt.legend(title="Especies")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------- Interfaz de Naive Bayes --------------------
def multi_input_dialog_bayes():
    dialog = tk.Toplevel()
    dialog.title("Bayes Coordenadas")
    dialog.geometry("350x150")

    entries = {}
    tk.Label(dialog, text="Ingrese Ancho del pétalo (X):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    entry_x = tk.Entry(dialog)
    entry_x.grid(row=0, column=1, padx=5, pady=5)
    entries["X"] = entry_x

    tk.Label(dialog, text="Ingrese Largo del pétalo (Y):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    entry_y = tk.Entry(dialog)
    entry_y.grid(row=1, column=1, padx=5, pady=5)
    entries["Y"] = entry_y

    result = {}
    def on_ok():
        try:
            result["X"] = float(entries["X"].get())
            result["Y"] = float(entries["Y"].get())
        except ValueError:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos.")
            result["X"], result["Y"] = None, None
        dialog.destroy()

    tk.Button(dialog, text="OK", command=on_ok).grid(row=2, column=0, columnspan=2, pady=10)
    dialog.wait_window()
    return result


def bayes(): 
    df = cargar_datos_csv(ruta_csv)
    inputs = multi_input_dialog_bayes()

    x_usuario = inputs.get("X")
    y_usuario = inputs.get("Y")
    if x_usuario is None or y_usuario is None:
        print("Entrada no válida. Intente nuevamente.")
        return

    # Preparar datos
    df["species"] = df["species"].astype("category")
    species_names = df["species"].cat.categories  
    y = df["species"].cat.codes
    X = df[['petal_width', 'petal_length']].values

    # Normalización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_scaled = scaler.transform([[x_usuario, y_usuario]])

    cv = KFold(n_splits=5, shuffle=True, random_state=50)
    gnb = GaussianNB(var_smoothing=1e-8)
    cv_scores = cross_val_score(gnb, X_scaled, y, cv=cv)
    acc_cv = np.mean(cv_scores)
    print(f"Cross-Validation Accuracy: {acc_cv:.3f}")

    # Entrenar el modelo sobre todo el dataset para visualización y reporte
    gnb.fit(X_scaled, y)
    y_pred_full = gnb.predict(X_scaled)
    print("\nReporte de Clasificación (Todos los datos):")
    print(classification_report(y, y_pred_full, target_names=species_names))

    # Matriz de Confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y, y_pred_full), annot=True, fmt='d', cmap='Blues', 
                xticklabels=species_names, yticklabels=species_names)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - Naive Bayes')
    plt.show()

    # Predicción para el dato de entrada
    prediccion = gnb.predict(input_scaled)[0]
    especie_predicha = species_names[prediccion]
    probas = gnb.predict_proba(input_scaled)[0]
    confianza = max(probas) * 100
    print(f"Clasificación Naive Bayes: {especie_predicha}")
    print(f"Confianza: {confianza:.1f}%")

    # --- Frontera de Decisión en Espacio Original ---
    # Crear malla en el espacio original
    x_min_orig = X[:, 0].min() - 0.1
    x_max_orig = X[:, 0].max() + 0.1
    y_min_orig = X[:, 1].min() - 0.1
    y_max_orig = X[:, 1].max() + 0.1

    xx_orig, yy_orig = np.meshgrid(np.linspace(x_min_orig, x_max_orig, 300),
                                   np.linspace(y_min_orig, y_max_orig, 300))
    mesh_orig = np.c_[xx_orig.ravel(), yy_orig.ravel()]
    mesh_scaled = scaler.transform(mesh_orig)
    Z = gnb.predict(mesh_scaled).reshape(xx_orig.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    # Convertir los conjuntos de entrenamiento y prueba a espacio original:
    X_train_orig = scaler.inverse_transform(X_train)
    X_test_orig = scaler.inverse_transform(X_test)
    datasets = [(X_train_orig, y_train), (X_test_orig, y_test), (X, y)]
    titles = ['Entrenamiento', 'Prueba', 'Todos los datos']

    # Paleta personalizada para cada especie
    custom_palette = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}

    fig, axs = plt.subplots(1, 3, figsize=(22, 6))
    for ax, (X_data, y_data), title in zip(axs, datasets, titles):
        # Graficar frontera de decisión (en espacio original)
        ax.contourf(xx_orig, yy_orig, Z, alpha=0.4,
                    cmap=ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF']))
        ax.set_aspect('equal')
        # Graficar puntos para cada clase
        for sp in np.unique(y_data):
            idx = np.where(y_data == sp)
            ax.scatter(X_data[idx, 0], X_data[idx, 1],
                       color=custom_palette[species_names[sp]],
                       edgecolor='k', s=80, label=species_names[sp])
        # Marcar el punto de entrada (en espacio original)
        input_orig = scaler.inverse_transform(input_scaled)
        ax.scatter(input_orig[0, 0], input_orig[0, 1], color='black', marker='X', s=250,
                   label=f'Entrada: {especie_predicha} ({confianza:.1f}%)', zorder=10)
        ax.set_xlabel('Ancho del pétalo (cm)', fontsize=12)
        ax.set_ylabel('Largo del pétalo (cm)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()

    plt.suptitle(f'Clasificación Naive Bayes (CV Accuracy: {acc_cv:.3f})', fontsize=16)
    plt.tight_layout()
    plt.show()


#-------------------- Interfaz de SVM --------------------
def multi_input_dialog_SVM():
    dialog = tk.Toplevel()
    dialog.title("SVM Coordenadas")
    dialog.geometry("350x150")

    entries = {}
    tk.Label(dialog, text="Ingrese Ancho del pétalo (X):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    entry_x = tk.Entry(dialog)
    entry_x.grid(row=0, column=1, padx=5, pady=5)
    entries["X"] = entry_x

    tk.Label(dialog, text="Ingrese Largo del pétalo (Y):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    entry_y = tk.Entry(dialog)
    entry_y.grid(row=1, column=1, padx=5, pady=5)
    entries["Y"] = entry_y

    result = {}
    def on_ok():
        try:
            result["X"] = float(entries["X"].get())
            result["Y"] = float(entries["Y"].get())
        except ValueError:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos.")
            return
        dialog.destroy()

    tk.Button(dialog, text="OK", command=on_ok).grid(row=2, column=0, columnspan=2, pady=10)
    dialog.wait_window()
    return result

def svm():
    df = cargar_datos_csv(ruta_csv)
    inputs = multi_input_dialog_SVM()
    x_usuario = inputs.get("X")
    y_usuario = inputs.get("Y")
    if x_usuario is None or y_usuario is None:
        print("Entrada no válida. Intente nuevamente.")
        return

    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])
    species_names = le.classes_
    X = df[["petal_width", "petal_length"]].values
    y = df["species"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_data = scaler.transform([[x_usuario, y_usuario]])
    punto_usuario_x = input_data[0][0]
    punto_usuario_y = input_data[0][1]

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    svms = {kernel: SVC(kernel=kernel).fit(X_train, y_train) for kernel in ["linear", "poly", "rbf", "sigmoid"]}
    
    prediccion = svms["linear"].predict(input_data)[0]
    especie_predicha = species_names[prediccion]
    print(f"Clasificación: {especie_predicha}")

    # Calcular accuracy
    y_pred = svms["linear"].predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")

    def inverse_transform(scaled, feature_idx):
        return (scaled * scaler.scale_[feature_idx]) + scaler.mean_[feature_idx]

    x_min = min(X_scaled[:, 0].min(), punto_usuario_x) - 0.5
    x_max = max(X_scaled[:, 0].max(), punto_usuario_x) + 0.5
    y_min = min(X_scaled[:, 1].min(), punto_usuario_y) - 0.5
    y_max = max(X_scaled[:, 1].max(), punto_usuario_y) + 0.5

    x_ticks = np.linspace(x_min, x_max, 5)
    y_ticks = np.linspace(y_min, y_max, 5)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    kernels = ["linear", "poly", "rbf", "sigmoid"]

    for i, kernel in enumerate(kernels):
        model = svms[kernel]
        ax = axes[i]
        plot_decision_regions(X_scaled, y, clf=model, ax=ax, legend=2)
        ax.scatter(punto_usuario_x, punto_usuario_y, 
                   color="red", marker="X", s=200,
                   edgecolor="black", linewidth=1.5,
                   label=f"Entrada: {especie_predicha}" )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{inverse_transform(x, 0):.1f}" for x in x_ticks], rotation=30, fontsize=10)
        ax.set_yticklabels([f"{inverse_transform(y, 1):.1f}" for y in y_ticks], fontsize=10)
        ax.set_title(f"Kernel: {kernel.upper()}", pad=15, fontsize=14, fontweight="bold")
        ax.set_xlabel("Ancho del pétalo (cm)", labelpad=12, fontsize=12)
        ax.set_ylabel("Largo del pétalo (cm)", labelpad=12, fontsize=12)

        handles, labels = ax.get_legend_handles_labels()
        labels = [species_names[int(l)] if l.isdigit() else l for l in labels]
        ax.legend(handles, labels, loc="upper right", fontsize=10, title_fontsize=11, frameon=True, framealpha=0.9, edgecolor="black")

    plt.suptitle(f"Clasificación SVM - Diferentes Kernels (Accuracy: {acc:.3f})", y=0.98, fontsize=12, fontweight="bold")
    plt.tight_layout(pad=4.5, w_pad=3.5, h_pad=3.5)
    plt.show()


# -------------------- Interfaz --------------------

# Crear ventana principal
root = tk.Tk()
root.title("Análisis de Iris")
root.geometry("350x300")
root.resizable(False, False)  

# Centrar ventana en la pantalla
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = (screen_width // 2) - (350 // 2)
y_position = (screen_height // 2) - (300 // 2)
root.geometry(f"350x300+{x_position}+{y_position}")

# Estilos de botones
btn_style = {
    "font": ("Times New Roman", 12, "bold"),
    "fg": "white",
    "bg": "#DC64DE",  # Morado claro
    "activebackground": "#0056b3",  # Azul más oscuro al presionar
    "activeforeground": "white",
    "bd": 3,
    "width": 22,
    "height": 2
}

# Título
titulo_font = font.Font(family="Arial", size=14, weight="bold")
tk.Label(root, text="Seleccione un Análisis", font=titulo_font, pady=10).grid(row=0, column=0, columnspan=2)

# Botones
tk.Button(root, text="Regresión Lineal", command=regresion_lineal, **btn_style).grid(row=1, column=0, pady=5, padx=10)
tk.Button(root, text="Regresión Polinomial", command=regresion_polinomial, **btn_style).grid(row=2, column=0, pady=5, padx=10)
tk.Button(root, text="Naive Bayes", command=bayes, **btn_style).grid(row=3, column=0, pady=5, padx=10)
tk.Button(root, text="SVM", command=svm, **btn_style).grid(row=4, column=0, pady=5, padx=10)



root.mainloop()
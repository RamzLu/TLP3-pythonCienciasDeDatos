#importo la libreria fastapi con todas las herramientas de la clase 
#principal que es FastAPI que seria como el molde para construir este servidor web
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
#joblib es una libr para empaquetar y desemp objet pesados de py (como el modelo de vinos)
import joblib
from pydantic import BaseModel
import pandas as pd #p/darle forma correcta al modelo

#--------------------------------------creacion modelo---------------------------------------------------
#creo un esquema
class VinoData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

#-------------------------------------inicio de la app--------------------------------------------------
#creo el servidor con el molde del FastAPI y lo guardo. Doy un titulo y descripcion
# en una variable llamada "app". Todas las rutas, recepcion de datos y demas se 
# conectaran a esta variable
app = FastAPI(
    #detalles personalizados
    title= "API para la Prediccion de Vinos",
    description="Backend para el modelo de Machine Learning del TP final.",
    version= "1.0.0" #etiqueta p/llevar el control de versiones
)

#--------------------------------config de cors-------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permite todas las origenes
    allow_credentials=True,
    allow_methods=["*"],  # permite todos los metodos
    allow_headers=["*"],  # permite todos los encabezados
)

#---------------------------------carga de modelo y escalador-------------------------------------------
#funcion try...catch pero de python
try:
    #los .pkl son archivos binarios cerrados. No puedo abrirlos con un bloc de notas
    #la herramienta joblib.load() sirve para "abrir y leer" estos archivos especiales y traerlos a la memoria de tu servidor
    modelo = joblib.load("models/modelo_vino.pkl") # modelo es para cargar el cerebro que entreno Lu, lo q predice 
    escalador = joblib.load("models/scaler_vino.pkl") #escalador es para cargar la regla matmtica. Cuando el front mande datos de un nuevo vino
    #este escalador ajustara los numeros antes de pasarlo al cerebro para evitar confusiones
    print("cerebro conectado: modelo y escalador conectados")
except Exception as e:
    print("error al cargar los archivos de ML")

#------------------------------------------endpoint p/prediccion-------------------------------------------
@app.post("/predecir") #aca esperan los datos
#post pq desde el front nos mandaran la info de las caracteristicas de los vinos
#cuando llegue una peticion devuelve este diccionario para que FastAPI empaquete esto
#y lo traduzca automat. a JSON para el nav
def hacer_prediccion(datos: VinoData): #configuracion. VinoData es el molde q creamos arriba (BaseModel), es una lista con las caracteristicas del vino q mandan del front
    #datos: nombre de la caja etiquetada donde guardo la info q mandan
    
    df_usuario = pd.DataFrame([{
        
        #estos nombres son los mismos q del CSV q/entreno Lu, si no coinciden entonces no funcionara el modelo
       #no estoy repitiendo codigo, estoy mapeando/trasladando los datos de un formato a otro. De la caja de datos q mandan del front (datos) a un dataframe (df_usuario) q es
       #el formato q necesita el modelo. Si paso el objeto de datos directo, el modelo no lo va a entender y va a tirar error. Al hacer pd.DataFrame([{ ... }]) agarro la info y las 
       #hago una mini tabla q/tiene todo en una sola fila. Saco los valores de la caja y los acomodo en columnas de una tabla p/q el cerebro matematico pueda leer
       "fixed acidity": datos.fixed_acidity,
        "volatile acidity": datos.volatile_acidity,
        "citric acid": datos.citric_acid,
        "residual sugar": datos.residual_sugar,
        "chlorides": datos.chlorides,
        "free sulfur dioxide": datos.free_sulfur_dioxide,
        "total sulfur dioxide": datos.total_sulfur_dioxide,
        "density": datos.density,
        "pH": datos.pH,
        "sulphates": datos.sulphates,
        "alcohol": datos.alcohol 
    }])
    
    #trycatch de js pero en python. Si el modelo no puede hacer la prediccion entonces devuelve un error 400 y un mensaje de error
    try:
        datos_escalados = escalador.transform(df_usuario)
        #el modelo no recibe numeros crudos sino q primero lo pasa por el escalador para q los ajuste y no se confunda. El escalador 
        #es como un traductor q ajusta los numeros a la escala q el modelo entiende. EJ si mando un ph de 3.5 y alcohol de 11.2 se 
        #puede confundir pq el modelo entreno con numeros mas grandes. El escalador ajusta esos numeros a la escala q el modelo entiende
        prediccion = modelo.predict(datos_escalados)
        #aca entran los datos ya ajustados al modelo y el modelo devuelve la prediccion de calidad del vino. La prediccion es un array 
        #con un solo valor, por eso accedo al primer elemento con [0]
        
        #saco el numero exacto de la prediccion (q viene en el array) y me aseguro q sea un entero (0 o 1)
        resultado_ia = int(prediccion[0])
        
        #como el modelo devuelve 0 o 1 (lenguaje maquina), lo traduzco a lenguaje humano p/q el front pueda mostrar algo lindo en la web
        #si el modelo devuelve 1 entonces es vino premium, si devuelve 0 es vino regular
        if resultado_ia == 1:
            etiqueta_humana = "Vino de Alta Calidad"
        else:
            etiqueta_humana = "Vino de Calidad Regular"
        
        #devuelvo un diccionario con el msj de exito, el codigo de la ia y el texto traducido. 
        #FastAPI empaqueta todo esto, lo pasa a JSON y lo manda directo al front listito p/usar
        return{
            "mensaje": "Predicción realizada con éxito",
            "codigo_calidad": resultado_ia,
            "resultado": etiqueta_humana
        }
    
    #es el catch de js pero en py. Si el modelo falla aca lo atrapamos
    #Exception as e: atrapa los errores y los guarda en la variable e. Luego uso str(e) para convertir el error a string y mostrarlo 
    #en el mensaje de error
    #raise HTTPException(...) es para devolver un error HTTP (ej 400) oficial. Si no uso raise 
    #HTTPException el front no recibe un codigo de error real y se queda colgado sin saber q paso
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en el modelo: {str(e)}")
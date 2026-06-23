#importo la libreria fastapi con todas las herramientas de la clase 
#principal que es FastAPI que seria como el molde para construir este servidor web
from fastapi import FastAPI, HTTPException
#jolib es una libr para empaquetar y desemp objet pesados de py (como el modelo de vinos)
import joblib
from pydantic import BaseModel
import pandas as pd #p/darle forma correcta al modleo

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

#---------------------------------icarga de modelo y escalador-------------------------------------------
#funcion try...catch pero de python
try:
    #los .pkl son archivos binarios cerrados. No puedo abrirlos con un bloc de notas
    #la herramienta joblib.load() sirve para "abrir y leer" estos archivos especiales y traerlos a la memoria de tu servedr
    modelo = joblib.load("models/modelo_vino.pkl") # modelo es para cargar el cerebro que entreno Lu, lo q predice 
    escalador = joblib.load("models/scaler_vino.pkl") #escalador es para cargar la regla matmtica. Cuando el front mande datos de un nuevo vino
    #este escalador ajustara los numeros antes de pasarlo al brocere para evitar confuciones
    print("cerebro conectado: modelo y escalador conectados")
except Exception as e:
    print("error al cargar los archivos de ML")

#------------------------------------------endpoint p/prediccion-------------------------------------------
@app.post("/predecir") #aca esperan los datos
#post pq desde el front nos mandaran la info de las caracteristicas de los vinos
#cuando llegue una eticion devuelve este diccionario para que FastAPI empaquete esto
#y lo traduzca automat. a JSONpara el nav
def hacer_prediccion(datos: VinoData): #configuracion. VinoData es el molde q creamos arriba (BaseModel), es una lista con las caracteristicas del vino q mandan del front
    #datos: npmbre de la caja etiquetada donde guardo la info q mandan
    
    df_usuario = pd.DataFrame([{
        
        #estos nombres son los mismos q del CSV q/entreno Lu, si no coinciden entonces no funcionara el modelo
       #no estoy repitiendo codigo, estoy mapeando/trasladando los datos de un formato a otro. De la caja de datos q mandan del front (datos) a un dataframe (df_usuario) q es
       #el formato q necesita el modelo. Si paso el objer de datos directo, el modelo no lo va a entender y va a tirar erro. Al hacer pd.DataFrame([{ ... }]) agarro la info y las 
       #hago una mini tabal q/tiene todo en una sola fila. Saco los valores de la caja y los acomodo en columnas de una tabla p/q el cerebro matemac pueda leer
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
        
        #devuelvo un diccionario con un mensaje de exito y la prediccion redondeada a 2 decimales. FastAPI lo traduce a JSON y lo envia al front
        #con prediccion[0] le digo que saque el numero de la prediccion 
        #con float(...) lo convierto en decimal de py normal (pq la web no entiende los decimales de la libreria)
        #con round(..., 2) lo redondeo a 2 decimales para q el front reciba un 5.49 en uvez de un 5.4899999...
        return{
            "mensaje": "Predicción realizada con éxito",
            "calidad_predicha": round(float(prediccion[0]), 2)
        }
    
    #es el catch de js pero en py. Si el modelo no puede hacer la prediccion entonces devuelve un error 400 y un mensaje de error
    #Exception as e: atrapa los errores y los guarda en la variable e. Luego uso str(e) para convertir el error a string y mostrarlo 
    #en el mensaje de error
    #raise HTTPException(...) es para devolver un error 400 al front con un mensaje de error. FastAPI lo traduce a JSON y lo envia al front
    # y al usar raise HTTPException(status_code=400...) le mando el msj al nav del user, asi el front puede mostrarlo en la web. Si no uso raise 
    #HTTPException(...) el front no recibe el msj y no sabe q paso
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en el modelo: {str(e)}")
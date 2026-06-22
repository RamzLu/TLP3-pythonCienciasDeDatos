#importo la libreria fastapi con todas las herramientas de la clase 
#principal que es FastAPI que seria como el molde para construir este servidor web
from fastapi import FastAPI

#creo el servidor con el molde del FastAPI y lo guardo. Doy un titulo y descripcion
# en una variable llamada "app". Todas las rutas, recepcion de datos y demas se 
# conectaran a esta variable
app = FastAPI(
    #detalles personalizados
    title= "API para la Prediccion de Vinos",
    description="Backend para el modelo de Machine Learning del TP final."
)

@app.get("/")
#cuando llegue una eticion devuelve este diccionario para que FastAPI empaquete esto
#y lo traduzca automat. a JSONpara el nav
def ruta_raiz():
    return{
        "mensaje" : "API del back funcionando"
    }
import requests
import pandas as pd
import urllib3
from io import StringIO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://sdmx.oecd.org/public/rest/data/OECD.DAF.CM,DSD_FP@DF_FPS,1.0/IRL.A.....?dimensionAtObservation=AllDimensions&format=csvfilewithlabels"

response = requests.get(
    url,
    timeout=30,
    verify=False
)

df = pd.read_csv(StringIO(response.text))

df.to_csv("irlanda_oecd.csv", index=False)

print("Archivo guardado correctamente")
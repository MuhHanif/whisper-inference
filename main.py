import nest_asyncio
from pyngrok import ngrok
import uvicorn
import json

# read file config
with open('config_file.json') as f:
    config = json.load(f)

use_ngrok = config["enable_tunnel"]

# run API server
if use_ngrok:
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
uvicorn.run(app, port=8000)
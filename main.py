from typing import Optional
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from cv2 import cv2
from fastapi.responses import FileResponse
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable
import matplotlib.image as mpimg
app = FastAPI()
import subprocess
import os
import base64
import re
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   
    allow_methods=["*"],      
    allow_headers=["*"]       
)


@app.get("/", status_code=201)
def read_root():
    return {"status": "ready"}


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image



def remove_img( path, img_name):
    os.remove(path + '/' + img_name)
# check if file exists or not
    if not (os.path.exists(path + '/' + img_name)):
        # file did not exists
        return True
@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    try:
        suffix = file.filename.split(".")[-1] 
        extension = suffix in ("jpg", "jpeg", "png")
        print(suffix)
       
        file.filename = f"{uuid.uuid4()}.jpg"
        if not extension:
            return {"error":"Wrong Image Format - Image must be jpg or png format!"}
        image=read_imagefile(await file.read())
        img_path='input_temp_img'
        save_path='output_temp_img'
        image.save(f'{img_path}/{file.filename}', suffix)
    except:
        raise HTTPException(status_code=415, detail="Unexpected Image Format")
        remove_img(img_path,file.filename)    
    else: 
        try:
            p = subprocess.Popen(f"python eval.py --trained_model=weights/yolact_plus_resnet50_papaya_weight.pth --config=yolact_resnet50_papaya_config --display_masks=False --score_threshold=0.8 --top_k=15 --image={img_path}/{file.filename}", stdout=subprocess.PIPE)
        
            (out,_)  = p.communicate()
        except:
            remove_img(img_path,file.filename)
            raise HTTPException(status_code=500, detail='Model Error')
        else:
            try:
                confident = out.decode("utf-8")
                confident = confident[confident.find("(")+1:confident.find(")")]
                ripeness = re.findall(r'@(\w+)', out.decode("utf-8"))
                img = open(save_path+"/"+file.filename,'rb')
            except:
                remove_img(img_path,file.filename)
                return {"ripeness": 'Not Papaya'}    
            else:
                img_read = img.read()
                img_64_encode = base64.encodebytes(img_read)
                img.close()
                print(ripeness)
                print(confident)
                remove_img(img_path,file.filename)
                remove_img(save_path,file.filename)
            
            return {"ripeness": ripeness, "img": img_64_encode,"confident": confident}
    # imgplot = plt.imshow(image)
    # plt.show()

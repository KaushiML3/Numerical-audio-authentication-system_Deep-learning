from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,FileResponse , JSONResponse,HTMLResponse


import uvicorn 
import tempfile
import shutil
import os
import warnings


from app.src.voice_authentication import inference_voice_authent


warnings.filterwarnings("ignore")


app=FastAPI(title="Project s",
    description="FastAPI",
    version="0.115.4")

# Allow all origins (replace * with specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

@app.get("/")
async def root():
  return {"Fast API":"API is woorking"}


# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = filter out info, 2 = filter out warnings, 3 = filter out errors
warnings.filterwarnings("ignore")


@app.post("/voice_authenticate")    
async def voice_authenticate(audio_file1: UploadFile = File(...),audio_file2: UploadFile = File(...),cut_off=1):
    disease_cls=[]
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Create a temporary file path
        temp_file_path1 = os.path.join(temp_dir,audio_file1.filename)

        # Write the uploaded file content to the temporary file
        with open(temp_file_path1, "wb") as temp_file:
            shutil.copyfileobj(audio_file1.file, temp_file)


                    # Create a temporary file path
        temp_file_path2 = os.path.join(temp_dir,audio_file2.filename)

        # Write the uploaded file content to the temporary file
        with open(temp_file_path2, "wb") as temp_file:
            shutil.copyfileobj(audio_file2.file, temp_file)

        status, message=inference_voice_authent(cut_off,temp_file_path1,temp_file_path2)
        
        shutil.rmtree(temp_dir)

        if status ==1:
            return {"status":1,"Message":message}
        else:
            return {"status":0,"Message":message}
    

    except Exception as e:
        return {"status":0,"Message":e}


    
    


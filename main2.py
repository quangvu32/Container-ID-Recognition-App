from fastapi import FastAPI, APIRouter
app = FastAPI()
router = APIRouter()

@app.get('/')
async def welcome():
    return {'message': 'Welcome to the Container Recognition API!'}

@router.get('/hello')
async def say_hello():
    return {'message': 'Hello from the Container Recognition API!'}
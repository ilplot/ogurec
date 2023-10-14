from fastapi import FastAPI, File, UploadFile, Response, Depends
import uvicorn
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
from sqlalchemy.orm import Session
from schemas import *
from database import get_db
from models import *
import random
import string
import datetime


def generate_token(length=256):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))


model = YOLO('bestn.pt')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_numbers(
        path_to_photo,
        filename,
        conf=0.5,  # Ниже какого уровня уверенности не включаем детектированную цифру в итоговый результат
        w_coef=2,
        # Коэффициент, отвечающий за максимальное горизонтальное расстояние между цифрами, чем больше, тем больше может быть расстояние
        h_coef=0.25,
        # Коэффициент, отвечающий за максимальное отклонение цифры от предыдущей по вертикали, чем больше, тем больше возможно отклонение
) :
    result = model.predict(path_to_photo, conf=conf, save=True)[0]
    nums = [result.names[int(num)] for num in result.boxes.cls]
    probs = result.boxes.conf.tolist()
    xywh = result.boxes.xywhn.tolist()
    data = {tuple(coord): (num, prob) for coord, num, prob in zip(xywh, nums, probs)}

    x_sorted_coords = sorted(xywh, key=lambda x: x[0])
    metadata = [[x_sorted_coords[0]]]

    digits = []

    json_ans = {
        "source_image": filename,
        "date":datetime.datetime.now()
    }

    for x, y, w, h in x_sorted_coords[1:]:
        for arr in metadata:
            last_x, last_y, last_w, last_h = arr[-1]
            if abs(y - last_y) < (h + last_h) * h_coef and abs(x - last_x) < (w + last_w) * w_coef:
                arr.append([x, y, w, h])
                break
        else:
            metadata.append([[x, y, w, h]])


    kord = [[tuple(coord) for coord in arr] for arr in metadata]
    #print(kord[0][1])
    metadata = [[data[tuple(coord)] for coord in arr] for arr in metadata]
    #print(num)


    ans = []

    for arr in metadata:
        num = ''
        full_prob = 1
        l = 0
        for digit, prob in arr:
            new = {}
            num += digit
            full_prob *= prob
            new['x0'] = kord[0][l][0]
            new['y0'] = kord[0][l][1]
            new['x1'] = kord[0][l][0] + kord[0][l][2]
            new['y1'] = kord[0][l][1]
            new['x2'] = kord[0][l][0] + kord[0][l][2]
            new['y2'] = kord[0][l][1] + kord[0][l][3]
            new['x3'] = kord[0][l][0]
            new['y4'] = kord[0][l][1] + kord[0][l][3]
            new['value'] = digit
            new['confidence'] = prob
            digits.append(new)
            l += 1
            #new.append([digit, prob,  kord[0][l]])

        json_ans["confidence"] = prob
        json_ans["full_number"] = num
        ans.append((num, prob))

    json_ans['digits'] = digits
    json_ans['position_number'] = {
        'x0' : digits[0]['x0'],
        'y0': digits[0]['y0'],
        'x1': digits[0]['x0'] + digits[-1]['x1'],
        'y1': digits[0]['y0'],
        'x2': digits[0]['x0'] + digits[-1]['x1'],
        'y2': digits[0]['y0'] + digits[-1]['y2'],
        'x3': digits[0]['x0'],
        'y3': digits[0]['y0'] + digits[-1]['y2']
    }

    json_ans["unrecognized_digits"] = 0

    #print(new)
    #print(ans)
    return json_ans

@app.post("/authtorization")
async def auth(details: CreateUserRequest, db: Session = Depends(get_db)):
    token = generate_token()
    to_create = User(
        surname=details.surname,
        name=details.name,
        role=details.role,
        token=token
    )
    db.add(to_create)
    db.commit()
    return {'token': token}


@app.post("/image")
async def im_post(token : str, image: UploadFile = File(...), db: Session = Depends(get_db)):
    #if db.query(User).filter(User.token == token).first():
    contents = await image.read()
    filename = image.filename
    with open(f"predict.jpg", 'wb') as file:
        file.write(contents)
    #print(image_to_numbers(path_to_photo='predict.jpg'))
    return image_to_numbers(filename=filename, path_to_photo='predict.jpg')
    # else:
    #     return {'message': 'Вы не авторизованы'}


if __name__ == '__main__':
    uvicorn.run(app, port=8000)
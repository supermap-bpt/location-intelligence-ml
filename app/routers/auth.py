from fastapi import APIRouter, HTTPException
from app.services.auth import decode_jwt
from fastapi import APIRouter, Request, HTTPException
from app.services.auth import decode_jwt

router = APIRouter()

@router.post("/decode")
def decode_token(req: Request):
    result = decode_jwt(req)
    if not result["valid"]:
        raise HTTPException(status_code=401, detail=result["error"])
    return {"message": "Token is valid", "user": result["payload"]}

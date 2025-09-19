# app/services/jwt_service.py
import os, jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SECRET_KEY = (os.getenv("JWT_SECRET") or "").strip()
ALGORITHM = (os.getenv("JWT_ALGORITHM") or "HS256").strip()

def decode_jwt(req):
    """
    Extracts, decodes, and verifies a JWT from the Authorization header.
    Expects: Authorization: Bearer <token>
    """
    try:
        auth_header = getattr(req, "headers", {}).get("Authorization")
        if not auth_header:
            raise ValueError("Missing Authorization header")

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise ValueError("Invalid Authorization header format")

        token = parts[1].strip()
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"valid": True, "payload": payload}

    except ExpiredSignatureError:
        return {"valid": False, "error": "Token has expired"}
    except InvalidTokenError:
        return {"valid": False, "error": "Invalid token"}
    except Exception as e:
        return {"valid": False, "error": str(e)}
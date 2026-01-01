import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def encode_image(image_path: Path) -> str:
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def delete_point(url: str) -> Dict[str, Any]:
    response = requests.delete(url, timeout=30)
    response.raise_for_status()
    return response.json()


def require_image(path: Optional[Path]) -> Path:
    if path is None:
        print("--image is required for this action", file=sys.stderr)
        sys.exit(1)
    if not path.exists():
        print(f"Image not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample client for face service.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="FastAPI server base URL")
    parser.add_argument("--image", type=Path, help="Path to face image input")
    parser.add_argument("--employee-id", default="EMP001", help="Unique employee id for registration")
    parser.add_argument("--employee-name", default="Sample Employee", help="Employee name")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold override for recognition")
    parser.add_argument(
        "--action",
        choices=["REGISTER", "RECOGNIZE", "DELETE"],
        default="REGISTER",
        help="Choose which API to call",
    )
    parser.add_argument("--point-id", help="Required when using --action DELETE")
    args = parser.parse_args()

    action = args.action.upper()
    if action == "DELETE":
        if not args.point_id:
            print("--point-id is required for DELETE action", file=sys.stderr)
            sys.exit(1)
        delete_url = f"{args.base_url}/face/{args.point_id}"
        print(f"Deleting point {args.point_id} via {delete_url} ...")
        result = delete_point(delete_url)
        print("Delete response:", json.dumps(result, indent=2))
        return

    image_path = require_image(args.image)
    image_b64 = encode_image(image_path)

    if action == "REGISTER":
        register_payload = {
            "employee_id": args.employee_id,
            "employee_name": args.employee_name,
            "image_base64": image_b64,
            "metadata": {
                "source": str(image_path),
            },
        }
        register_url = f"{args.base_url}/face/register"
        print(f"Registering face via {register_url} ...")
        register_response = post_json(register_url, register_payload)
        print("Register response:", json.dumps(register_response, indent=2))
        point_id: Optional[str] = register_response.get("point_id")
        if point_id:
            print(f"Point ID for future operations: {point_id}")
        return

    if action == "RECOGNIZE":
        recognize_payload = {
            "image_base64": image_b64,
            "threshold": args.threshold,
        }
        recognize_url = f"{args.base_url}/face/recognize"
        print(f"Recognizing face via {recognize_url} ...")
        recognize_response = post_json(recognize_url, recognize_payload)
        print("Recognize response:", json.dumps(recognize_response, indent=2))
        return

    print(f"Unsupported action {action}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()


# Face Recognition demo with Qdrant and AdaFace
FastAPI backend that accepts base64 photos, crops faces via MTCNN, generates embeddings with AdaFace, and stores/queries them in Qdrant. A companion CLI lets user register, recognize, and delete in the end‑to‑end flow quickly.

<div style="width: 100%;">
  <img src="./resources/demo.jpg" alt="anh" style="width: 100%;">
</div>
<br/>

### Run Qdrant locally
```
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY=my-secret-key \
  qdrant/qdrant
```


### Show Qdrant dashboard
```
http://localhost:6333/dashboard#/collections/faces
```
-> Enter key `my-secret-key`


### Setup Environments 
- Install `uv`
``` bash
curl -LsSf https://astral.sh/uv/install.sh | s
```
- Create environment
```
uv venv
```
- Install dependencies
```
uv sync
```


### Examples
You can **run pre-defined  auto-test script** or **test using your own images**.

- Run auto-test script
``` bash
sh ./script.sh
```

- Register Ronaldo face
``` bash
uv run python sample.py --action REGISTER --image ./resources/ronaldo1.jpg --employee-id 1 --employee-name "Ronaldo"
```
```
Registering face via http://localhost:8000/face/register ...
Register response: {
  "status": "registered",
  "employee_id": "1",
  "point_id": "1ecec7a6-9a1d-4011-b883-d63d636fcec7"
}
Point ID for future operations: 1ecec7a6-9a1d-4011-b883-d63d636fcec7
```

- Register Messi face 
``` bash
uv run python sample.py --action REGISTER --image ./resources/messi1.jpg --employee-id 2 --employee-name "Messi"
```
```
Registering face via http://localhost:8000/face/register ...
Register response: {
  "status": "registered",
  "employee_id": "2",
  "point_id": "06a4323f-010b-4e50-b82c-f3e4e79a0ac9"
}
Point ID for future operations: 06a4323f-010b-4e50-b82c-f3e4e79a0ac9
```

- Recognize Ronaldo face with another image
``` bash
uv run python sample.py --action RECOGNIZE --image ./resources/ronaldo2.jpg --threshold 0.5
```
```
Recognizing face via http://localhost:8000/face/recognize ...
Recognize response: {
  "matched": false,
  "confidence": 0.19501653,
  "threshold": 0.5,
  "candidate": {
    "point_id": "1ecec7a6-9a1d-4011-b883-d63d636fcec7",
    "payload": {
      "employee_id": "1",
      "employee_name": "Nguyen Huy Ronaldo",
      "metadata": {
        "source": "ronaldo1.jpg"
      }
    }
  }
}
```

- Delete Ronaldo face data
``` bash
uv run python sample.py --action DELETE --point-id 1ecec7a6-9a1d-4011-b883-d63d636fcec7
```
```
Deleting point 1ecec7a6-9a1d-4011-b883-d63d636fcec7 via http://localhost:8000/face/1ecec7a6-9a1d-4011-b883-d63d636fcec7 ...
Delete response: {
  "deleted": true,
  "point_id": "1ecec7a6-9a1d-4011-b883-d63d636fcec7"
}
```
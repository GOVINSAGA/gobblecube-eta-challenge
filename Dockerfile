# Dockerfile for the Gobblecube ETA Challenge submission.
#
# Build:
#   docker build -t my-eta .
# Test the grader pathway:
#   docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Submission surface: predict.py + grade.py + trained lookup tables.
COPY predict.py grade.py ./
COPY model.pkl ./

# Grader invokes: python grade.py <input.parquet> <output.csv>
ENTRYPOINT ["python", "grade.py"]

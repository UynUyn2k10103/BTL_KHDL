# Tạo môi trường ảo:

python -m venv env

# Mở môi trường ảo:

.\env\Scripts\activate

# Update pip:

python.exe -m pip install --upgrade pip

# Cài đặt các lib cần thiết

pip install -r requirement.txt

# run api

uvicorn main:app

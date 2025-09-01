import requests

# =========================
# Função para obter token
# =========================
def get_token(username, password, client_id="string", client_secret="********"):
    """
    Faz login na API e retorna o token de acesso.
    """
    url = "http://34.193.187.218:5000/users/token/retrieve"

    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "scope": "",
        "client_id": client_id,
        "client_secret": client_secret
    }

    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(url, data=data, headers=headers)
    response.raise_for_status()

    return response.json()

# =========================
# Função para enviar CSV
# =========================
def send_csv(csv_path, username, password):
    """
    Obtém o token e envia o CSV para a API.
    Retorna o JSON de resposta.
    """
    # =========================
    # Obter token
    # =========================
    token_response = get_token(username, password)
    access_token = token_response.get("token")

    # =========================
    # Configurar request
    # =========================
    url = "http://34.193.187.218:5000/evaluate/multilabel_metrics?threshold=0.5"
    headers = {"X-API-Key": access_token}

    # =========================
    # Enviar CSV
    # =========================
    with open(csv_path, "rb") as f:
        files = {"file": (csv_path, f, "text/csv")}
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        return response.json()

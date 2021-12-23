import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app.backend.app:app",
        # ssl_certfile="/etc/apache2/certificate/apache-certificate.crt",
        # ssl_keyfile="/etc/apache2/certificate/apache.key",
        # timeout_keep_alive=30,
        # host='127.0.0.1',
        # port=8000,
        reload=True)

"""Default settings for the Trismik client."""

evaluation_settings = {
    "max_iterations": 150,
}

client_settings = {"endpoint": "https://api.trismik.com/adaptive-testing"}

# Environment variable names used by the Trismik client
environment_settings = {
    # URL of the Trismik service
    "trismik_service_url": "TRISMIK_SERVICE_URL",
    # API key for authentication
    "trismik_api_key": "TRISMIK_API_KEY",
}
